
import random
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Optimizer


class HNAdam(Optimizer):
    """
    Hybrid Norm Adam (HNAdam)

    Combines Adam and AMSGrad using a dynamically calculated
    adaptive norm and a threshold-based switching mechanism.

    Manuscript formulation:
        A(t) = clip(
            K_t0 * max(abs(g_t)) /
            (max(abs(m_(t-1))) + epsilon),
            1, 4
        )

        If A(t) < 2:
            Use AMSGrad
        Else:
            Use Adam
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

        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)

        self.epsilon = epsilon
        self.threshold = threshold

        # Select the norm scaling parameter once at initialization.
        # A fixed value can also be supplied for reproducibility.
        self.K_t0 = (
            random.uniform(2.0, 4.0)
            if K_t0 is None
            else float(K_t0)
        )

        if not 2.0 <= self.K_t0 <= 4.0:
            raise ValueError("K_t0 must be between 2 and 4.")

    def _create_slots(self, var_list):
        """Initialize Adam and AMSGrad moment variables."""
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            self.add_slot(var, "vhat")

    @tf.function
    def _resource_apply_dense(self, grad, var):

        dtype = var.dtype.base_dtype

        lr = self._get_hyper("learning_rate", dtype)
        beta_1 = self._get_hyper("beta_1", dtype)
        beta_2 = self._get_hyper("beta_2", dtype)

        epsilon = tf.cast(self.epsilon, dtype)
        threshold = tf.cast(self.threshold, dtype)
        K_t0 = tf.cast(self.K_t0, dtype)

        # Retrieve optimizer slots
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        vhat = self.get_slot(var, "vhat")

        # ------------------------------------------------
        # Step 1: Preserve the previous first moment
        # ------------------------------------------------
        m_previous = tf.identity(m)

        # ------------------------------------------------
        # Step 2: Update first and second moments
        # ------------------------------------------------
        m.assign(
            beta_1 * m_previous +
            (1.0 - beta_1) * grad
        )

        v.assign(
            beta_2 * v +
            (1.0 - beta_2) * tf.square(grad)
        )

        # ------------------------------------------------
        # Step 3: Bias correction
        # ------------------------------------------------
        t = tf.cast(self.iterations + 1, dtype)

        m_hat = m / (1.0 - tf.pow(beta_1, t))
        v_hat = v / (1.0 - tf.pow(beta_2, t))

        # ------------------------------------------------
        # Step 4: Calculate adaptive norm A(t)
        # Using the previous first-moment estimate
        # ------------------------------------------------
        max_gradient = tf.reduce_max(tf.abs(grad))
        max_previous_moment = tf.reduce_max(
            tf.abs(m_previous)
        )

        adaptive_norm = (
            K_t0 * max_gradient /
            (max_previous_moment + epsilon)
        )

        # Restrict adaptive norm to [1, 4]
        adaptive_norm = tf.clip_by_value(
            adaptive_norm,
            tf.cast(1.0, dtype),
            tf.cast(4.0, dtype)
        )

        # ------------------------------------------------
        # Step 5: Update AMSGrad maximum second moment
        # ------------------------------------------------
        vhat.assign(
            tf.maximum(vhat, v)
        )

        vhat_corrected = (
            vhat / (1.0 - tf.pow(beta_2, t))
        )

        # ------------------------------------------------
        # Step 6: Calculate Adam update
        # ------------------------------------------------
        adam_update = (
            lr * m_hat /
            (tf.sqrt(v_hat) + epsilon)
        )

        # ------------------------------------------------
        # Step 7: Calculate AMSGrad update
        # ------------------------------------------------
        amsgrad_update = (
            lr * m_hat /
            (tf.sqrt(vhat_corrected) + epsilon)
        )

        # ------------------------------------------------
        # Step 8: Hybrid switching mechanism
        # A(t) < 2 : AMSGrad
        # A(t) >= 2: Adam
        # ------------------------------------------------
        use_amsgrad = adaptive_norm < threshold

        update = tf.where(
            use_amsgrad,
            amsgrad_update,
            adam_update
        )

        # ------------------------------------------------
        # Step 9: Update model parameters
        # ------------------------------------------------
        return var.assign_sub(update)

    def _resource_apply_sparse(self, grad, var, indices):

        # Convert sparse gradients into dense gradients
        dense_grad = tf.convert_to_tensor(
            tf.IndexedSlices(
                grad,
                indices,
                tf.shape(var)
            )
        )

        return self._resource_apply_dense(
            dense_grad,
            var
        )

    def get_config(self):

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
