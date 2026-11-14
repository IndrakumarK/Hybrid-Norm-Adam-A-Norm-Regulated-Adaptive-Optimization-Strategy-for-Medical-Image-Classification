
"""
Hybrid Norm Adam (HNAdam) Optimizer
===================================

A hybrid adaptive optimization algorithm combining Adam and AMSGrad
through an adaptive-norm-based switching mechanism.

Manuscript formulation
----------------------

A(t) = clip(
    K_t0 * max(abs(g_t)) /
    (max(abs(m_(t-1))) + epsilon),
    1, 4
)

Switching mechanism:
    If A(t) < tau:
        Use AMSGrad
    Else:
        Use Adam

Default threshold:
    tau = 2.0

K_t0:
    Randomly initialized once per optimizer instance from U(2, 4).
    The initialized value remains fixed throughout that training run.

Notes
-----
The adaptive norm is calculated separately for each model variable
because TensorFlow invokes the dense update method for each variable.
"""

import random
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Optimizer


class HNAdam(Optimizer):
    """
    Hybrid Norm Adam (HNAdam).

    Combines Adam and AMSGrad using an adaptive norm and a
    threshold-based switching mechanism.
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        K_t0=None,
        threshold=2.0,
        name="HNAdam",
        **kwargs
    ):
        super().__init__(name, **kwargs)

        # Optimizer hyperparameters
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)

        self.epsilon = float(epsilon)
        self.threshold = float(threshold)

        # Validate hyperparameters
        if not 0.0 <= beta_1 < 1.0:
            raise ValueError("beta_1 must be in the range [0, 1).")

        if not 0.0 <= beta_2 < 1.0:
            raise ValueError("beta_2 must be in the range [0, 1).")

        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")

        if self.threshold != 2.0:
            raise ValueError(
                "The manuscript specifies a fixed switching threshold "
                "of 2.0."
            )

        # Select K_t0 once during optimizer initialization.
        # The same value is retained throughout the training run.
        self.K_t0 = (
            random.uniform(2.0, 4.0)
            if K_t0 is None
            else float(K_t0)
        )

        if not 2.0 <= self.K_t0 <= 4.0:
            raise ValueError("K_t0 must be between 2.0 and 4.0.")

    def _create_slots(self, var_list):
        """
        Create the first-moment, second-moment, and AMSGrad slots.
        """
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            self.add_slot(var, "vhat")

    @tf.function
    def _resource_apply_dense(self, grad, var):
        """
        Apply the HNAdam update to a dense gradient.

        The adaptive norm is evaluated using the current gradient
        and the previous first-moment estimate.
        """

        dtype = var.dtype.base_dtype

        # --------------------------------------------------
        # Step 1: Retrieve hyperparameters
        # --------------------------------------------------

        lr = self._get_hyper("learning_rate", dtype)
        beta_1 = self._get_hyper("beta_1", dtype)
        beta_2 = self._get_hyper("beta_2", dtype)

        epsilon = tf.cast(self.epsilon, dtype)
        threshold = tf.cast(self.threshold, dtype)
        K_t0 = tf.cast(self.K_t0, dtype)

        # --------------------------------------------------
        # Step 2: Retrieve optimizer slots
        # --------------------------------------------------

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        vhat = self.get_slot(var, "vhat")

        # Preserve the previous first moment for A(t).
        m_previous = tf.identity(m)

        # --------------------------------------------------
        # Step 3: Update first and second moments
        # --------------------------------------------------

        m_new = (
            beta_1 * m_previous
            + (1.0 - beta_1) * grad
        )

        v_new = (
            beta_2 * v
            + (1.0 - beta_2) * tf.square(grad)
        )

        m_update = m.assign(m_new)
        v_update = v.assign(v_new)

        # --------------------------------------------------
        # Step 4: Calculate bias-correction factor
        # --------------------------------------------------

        t = tf.cast(self.iterations + 1, dtype)

        beta_1_power = tf.pow(beta_1, t)
        beta_2_power = tf.pow(beta_2, t)

        m_hat = m_new / (1.0 - beta_1_power)
        v_hat = v_new / (1.0 - beta_2_power)

        # --------------------------------------------------
        # Step 5: Calculate adaptive norm A(t)
        # --------------------------------------------------

        max_gradient = tf.reduce_max(tf.abs(grad))

        max_previous_moment = tf.reduce_max(
            tf.abs(m_previous)
        )

        adaptive_norm = (
            K_t0 * max_gradient
            / (max_previous_moment + epsilon)
        )

        # Restrict A(t) to the manuscript-defined range [1, 4].
        adaptive_norm = tf.clip_by_value(
            adaptive_norm,
            tf.cast(1.0, dtype),
            tf.cast(4.0, dtype)
        )

        # --------------------------------------------------
        # Step 6: Update AMSGrad maximum second moment
        # --------------------------------------------------

        vhat_new = tf.maximum(vhat, v_new)
        vhat_update = vhat.assign(vhat_new)

        # Bias-corrected AMSGrad second moment.
        vhat_corrected = (
            vhat_new / (1.0 - beta_2_power)
        )

        # --------------------------------------------------
        # Step 7: Calculate Adam update
        # --------------------------------------------------

        # Adam uses the current bias-corrected second moment.
        adam_update = (
            lr * m_hat
            / (tf.sqrt(v_hat) + epsilon)
        )

        # --------------------------------------------------
        # Step 8: Calculate AMSGrad update
        # --------------------------------------------------

        # AMSGrad uses the maximum historical second moment.
        amsgrad_update = (
            lr * m_hat
            / (tf.sqrt(vhat_corrected) + epsilon)
        )

        # --------------------------------------------------
        # Step 9: Hybrid switching mechanism
        # --------------------------------------------------

        # A(t) < 2: AMSGrad
        # A(t) >= 2: Adam

        use_amsgrad = adaptive_norm < threshold

        update = tf.where(
            use_amsgrad,
            amsgrad_update,
            adam_update
        )

        # --------------------------------------------------
        # Step 10: Update model parameters
        # --------------------------------------------------

        with tf.control_dependencies([
            m_update,
            v_update,
            vhat_update
        ]):
            return var.assign_sub(update)

    def _resource_apply_sparse(self, grad, var, indices):
        """
        Handle sparse gradients by converting them to dense form.

        This preserves the same update logic as the dense-gradient
        implementation.
        """

        dense_grad = tf.convert_to_tensor(
            tf.IndexedSlices(
                values=grad,
                indices=indices,
                dense_shape=tf.shape(var)
            )
        )

        return self._resource_apply_dense(
            dense_grad,
            var
        )

    def get_config(self):
        """
        Return optimizer configuration for serialization.
        """

        config = super().get_config()

        config.update({
            "learning_rate": self._serialize_hyperparameter(
                "learning_rate"
            ),
            "beta_1": self._serialize_hyperparameter(
                "beta_1"
            ),
            "beta_2": self._serialize_hyperparameter(
                "beta_2"
            ),
            "epsilon": self.epsilon,
            "K_t0": self.K_t0,
            "threshold": self.threshold
        })

        return config
