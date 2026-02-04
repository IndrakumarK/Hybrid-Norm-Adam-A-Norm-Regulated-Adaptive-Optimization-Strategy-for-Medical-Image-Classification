import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer


class HNAdam(Optimizer):
    """
    Hybrid Norm Adam (HNAdam)

    Implements the proposed Hybrid Norm Adam optimizer described in the manuscript.
    The optimizer dynamically adjusts the norm K(t) based on gradient statistics
    and switches between Adam and AMSGrad update rules to balance exploration
    and exploitation, improving convergence and generalization.
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        alpha=0.01,   # gradient norm coefficient
        beta=0.001,  # weight norm coefficient
        name="HNAdam",
        **kwargs
    ):
        super().__init__(name, **kwargs)

        # Hyperparameters
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("alpha", alpha)
        self._set_hyper("beta", beta)

        self.epsilon = epsilon

    def _create_slots(self, var_list):
        """
        Create optimizer slots:
        m     : first moment vector
        v     : second moment vector
        vhat  : maximum second moment (AMSGrad)
        """
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            self.add_slot(var, "vhat")

    @tf.function
    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype

        # Retrieve hyperparameters
        lr = self._get_hyper("learning_rate", var_dtype)
        beta_1 = self._get_hyper("beta_1", var_dtype)
        beta_2 = self._get_hyper("beta_2", var_dtype)
        alpha = self._get_hyper("alpha", var_dtype)
        beta = self._get_hyper("beta", var_dtype)

        # Slots
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        vhat = self.get_slot(var, "vhat")

        # -------------------------------------------------
        # Eq. (19): First moment estimate
        # m_t = ß1 * m_{t-1} + (1 - ß1) * g_t
        # -------------------------------------------------
        m.assign(beta_1 * m + (1.0 - beta_1) * grad)

        # -------------------------------------------------
        # Eq. (20): Second moment estimate
        # v_t = ß2 * v_{t-1} + (1 - ß2) * g_t^2
        # -------------------------------------------------
        v.assign(beta_2 * v + (1.0 - beta_2) * tf.square(grad))

        # -------------------------------------------------
        # AMSGrad second moment tracking
        # v^_t = max(v^_{t-1}, v_t)
        # -------------------------------------------------
        vhat.assign(tf.maximum(vhat, v))

        # -------------------------------------------------
        # Bias correction
        # -------------------------------------------------
        t = tf.cast(self.iterations + 1, var_dtype)
        m_hat = m / (1.0 - tf.pow(beta_1, t))
        v_hat = v / (1.0 - tf.pow(beta_2, t))
        vhat_corr = vhat / (1.0 - tf.pow(beta_2, t))

        # -------------------------------------------------
        # Eq. (22): Dynamic norm computation
        # K(t) = |g_t| / (|m_{t-1}| + e)
        # -------------------------------------------------
        K_t = tf.abs(grad) / (tf.abs(m) + self.epsilon)

        # -------------------------------------------------
        # Eq. (34): Hybrid switching condition
        # If mean(K(t)) < 2 ? AMSGrad
        # Else ? Adam
        # -------------------------------------------------
        use_amsgrad = tf.reduce_mean(K_t) < 2.0

        # -------------------------------------------------
        # Adam update rule
        # -------------------------------------------------
        adam_update = lr * m_hat / (tf.sqrt(v_hat) + self.epsilon)

        # -------------------------------------------------
        # AMSGrad update rule
        # -------------------------------------------------
        amsgrad_update = lr * m_hat / (tf.sqrt(vhat_corr) + self.epsilon)

        # -------------------------------------------------
        # Select update based on hybrid norm condition
        # -------------------------------------------------
        update = tf.where(use_amsgrad, amsgrad_update, adam_update)

        # -------------------------------------------------
        # Hybrid Norm Stabilization (Manuscript Section 3.6)
        # -------------------------------------------------
        grad_norm = tf.norm(grad, ord=2)
        weight_l1 = tf.norm(var, ord=1)

        stabilizer = 1.0 / (alpha * grad_norm + beta * weight_l1 + 1.0)

        # -------------------------------------------------
        # Parameter update
        # -------------------------------------------------
        var.assign_sub(update * stabilizer)

    def _resource_apply_sparse(self, grad, var, indices):
        """
        Convert sparse gradients to dense form to enable
        norm-based computations.
        """
        dense_grad = tf.convert_to_tensor(
            tf.IndexedSlices(grad, indices, tf.shape(var))
        )
        return self._resource_apply_dense(dense_grad, var)

    def get_config(self):
        """
        Required for optimizer serialization.
        """
        config = super().get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta_1": self._serialize_hyperparameter("beta_1"),
            "beta_2": self._serialize_hyperparameter("beta_2"),
            "epsilon": self.epsilon,
            "alpha": self._serialize_hyperparameter("alpha"),
            "beta": self._serialize_hyperparameter("beta"),
        })
        return config
