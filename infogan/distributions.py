import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

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

