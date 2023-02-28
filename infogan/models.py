import tensorflow as tf

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def get_generator_model(shape):
    noise = tf.keras.layers.Input(shape)
    x = tf.keras.layers.Dense(1024, )(noise)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(7*7*128, )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Reshape([7, 7, 128])(x)
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='tanh')(x)
    return tf.keras.models.Model(inputs=noise, outputs=x)


def get_discriminator_model(shape):
    input_image = tf.keras.layers.Input(shape)
    x = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same')(input_image)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    mid = x
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.models.Model(inputs=input_image, outputs=[x, mid])


def get_recognition_model(shape, latent_spec):

    mid = tf.keras.layers.Input(shape)
    x = tf.keras.layers.Dense(128, )(mid)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    disc_outputs = []
    for i, dist in enumerate(latent_spec['discrete-latent-codes']):
        disc_outputs.append(tf.keras.layers.Dense(dist.dim, name=f'categorical_code_{i}')(x))
    cont_outputs = []
    for i, dist in enumerate(latent_spec['continuous-latent-codes']):
        cont_mu = tf.keras.layers.Dense(1, name=f'z_mean_continuous_code_{i}')(x)
        cont_var = tf.keras.layers.Dense(1, name=f'z_log_var_continuous_code_{i}')(x)
        cont_outputs.append([Sampling(name=f'continuous_code_{i}')([cont_mu, cont_var]),
                             cont_mu, cont_var])

    return tf.keras.models.Model(inputs=mid, outputs=[disc_outputs, cont_outputs])
