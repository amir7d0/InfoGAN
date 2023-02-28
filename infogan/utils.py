import tensorflow as tf
from math import pi


class GaussianNLLLoss:
    def __call__(self, x, z_mean, z_log_var):
        r"""Gaussian negative log likelihood loss.

        Compute negative log-likelihood for Gaussian distribution
        L = 0.5 * [log(2 * pi * var) + ((x - mu)**2)/var]
        mu: the mean (μ), var: the variance (σ^2)
        """
        mu, var = z_mean, tf.math.exp(z_log_var)
        eps = 1e-6
        loss = 0.5 * (tf.math.log(2 * pi * var + eps) + (x - mu) ** 2 / (var + eps))
        return tf.reduce_mean(loss)
