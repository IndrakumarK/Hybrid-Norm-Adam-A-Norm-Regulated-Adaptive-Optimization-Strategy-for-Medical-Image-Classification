"""
Hybrid Norm Adam (HNAdam) optimizer for TensorFlow/Keras.
"""

import random
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Optimizer


class HNAdam(Optimizer):
    """
    Hybrid Norm Adam optimizer.

    Adaptive norm:
        A(t) = clip(
            K_t0 * max(abs(g_t)) / (max(abs(m_{t-1})) + epsilon),
            1, 4
        )

    Switching mechanism:
        A(t) < threshold  -> AMSGrad update
        A(t) >= threshold -> Adam update

    K_t0 is sampled uniformly from [2, 4] when not explicitly supplied.
    """

    def __init__(
        self,
        learning_rate=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        K_t0=None,
        threshold=2.0,
        name="HNAdam",
        **kwargs
    ):
        super().__init__(name, **kwargs)

        if not 0.0 < beta_1 < 1.0:
            raise ValueError("beta_1 must be between 0 and 1.")

        if not 0.0 < beta_2 < 1.0:
            raise ValueError("beta_2 must be between 0 and 1.")

        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        if K_t0 is not None and not 2.0 <= K_t0 <= 4.0:
            raise ValueError("K_t0 must be within [2, 4].")

        if not 1.0 <= threshold <= 4.0:
            raise ValueError("threshold must be within [1, 4].")

        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)

        self.epsilon = float(epsilon)
        self.K_t0 = float(
            random.uniform(2.0, 4.0) if K_t0 is None else K_t0
        )
        self.threshold = float(threshold)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            self.add_slot(var, "vhat")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype

        coefficients = (apply_state or {}).get(
            (var.device, var_dtype), {}
        )

        lr_t = coefficients.get(
            "lr_t", self._decayed_lr(var_dtype)
        )

        beta_1_t = coefficients.get(
            "beta_1_t", self._get_hyper("beta_1", var_dtype)
        )

        beta_2_t = coefficients.get(
            "beta_2_t", self._get_hyper("beta_2", var_dtype)
        )

        epsilon_t = tf.cast(self.epsilon, var_dtype)
        threshold_t = tf.cast(self.threshold, var_dtype)
        K_t0 = tf.cast(self.K_t0, var_dtype)

        step = tf.cast(self.iterations + 1, var_dtype)

        beta1_power = tf.pow(beta_1_t, step)
        beta2_power = tf.pow(beta_2_t, step)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        vhat = self.get_slot(var, "vhat")

        m_prev = tf.identity(m)

        m_t = m.assign(
            beta_1_t * m + (1.0 - beta_1_t) * grad
        )

        v_t = v.assign(
            beta_2_t * v + (1.0 - beta_2_t) * tf.square(grad)
        )

        m_corr = m_t / (1.0 - beta1_power)
        v_corr = v_t / (1.0 - beta2_power)

        norm_g = tf.reduce_max(tf.abs(grad))
        norm_m = tf.reduce_max(tf.abs(m_prev))

        adaptive_norm = (
            K_t0 * norm_g / (norm_m + epsilon_t)
        )

        adaptive_norm = tf.clip_by_value(
            adaptive_norm,
            tf.cast(1.0, var_dtype),
            tf.cast(4.0, var_dtype),
        )

        vhat_t = vhat.assign(
            tf.maximum(vhat, v_t)
        )

        vhat_corr = vhat_t / (1.0 - beta2_power)

        adam_update = (
            lr_t * m_corr /
            (tf.sqrt(v_corr) + epsilon_t)
        )

        amsgrad_update = (
            lr_t * m_corr /
            (tf.sqrt(vhat_corr) + epsilon_t)
        )

        update = tf.where(
            adaptive_norm < threshold_t,
            amsgrad_update,
            adam_update,
        )

        return var.assign_sub(update)

    def _resource_apply_sparse(
        self, grad, var, indices, apply_state=None
    ):
        dense_grad = tf.scatter_nd(
            indices=tf.expand_dims(indices, axis=1),
            updates=grad,
            shape=tf.shape(var),
        )

        return self._resource_apply_dense(
            dense_grad, var, apply_state
        )

    def get_config(self):
        config = super().get_config()

        config.update({
            "learning_rate": self._serialize_hyperparameter(
                "learning_rate"
            ),
            "beta_1": self._serialize_hyperparameter("beta_1"),
            "beta_2": self._serialize_hyperparameter("beta_2"),
            "epsilon": self.epsilon,
            "K_t0": self.K_t0,
            "threshold": self.threshold,
        })

        return config