import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

TINY = 1e-6


class Categorical(object):
    def __init__(self, dim):
        self.dim = dim

    def sample(self, batch_size):
        prob = tf.ones([self.dim]) * 1.0 / self.dim
        cat = tfd.Categorical(probs=prob).sample([batch_size,])
        cat_onehot = tf.one_hot(cat, self.dim)
        return cat_onehot

    def sample_test(self, cat, batch_size):
        # for sampling specific class, e.g. (cat=2, batch_size=1) -> [0, 0, 1, 0, ..., 0]
        z_cat = np.array([cat] * batch_size)
        z_cat = tf.one_hot(z_cat, self.dim)
        return z_cat


class Uniform(object):
    def __init__(self, dim):
        self.dim = dim

    def sample(self, batch_size):
        return tfd.Uniform(low=-1.0, high=1.0).sample([batch_size, self.dim])

    def sample_test(self, c, batch_size):
        # for sampling specific number, e.g. c=0.5
        return np.array([c] * batch_size).reshape([batch_size, 1])


class GaussianNLLLoss:
    def __call__(self, x, z_mean, z_log_var):
        r"""Gaussian negative log likelihood loss.

        Compute negative log-likelihood for Gaussian distribution
        L = 0.5 * [log(2 * pi * var) + ((x - mu)**2)/var]
        mu: the mean (μ), var: the variance (σ^2)
        """
        mu, var = z_mean, tf.math.exp(z_log_var)
        loss = 0.5 * (tf.math.log(2 * np.pi * var + TINY) + (x - mu) ** 2 / (var + TINY))
        return tf.reduce_mean(loss)


class LogProb:
    def __call__(self, x, z_mean, z_log_var):
        r"""
        Returns the log of the probability density/mass function evaluated at value.
        """
        mu, var = z_mean, tf.math.exp(z_log_var)
        neg_l = -0.5 * (tf.math.log(2 * np.pi * var + TINY) + (x - mu) ** 2 / (var + TINY))
        return tf.reduce_sum(neg_l, axis=1)

