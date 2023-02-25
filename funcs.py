import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from infogan.distributions import Uniform, Categorical
from infogan.config import config


def sample(latent_spec, batch_size):
    z = latent_spec['noise-variables'][0].sample(batch_size)
    cont_latent_dist = latent_spec['continuous-latent-codes']
    disc_latent_dist = latent_spec['discrete-latent-codes']
    cont_samples = [dist.sample(batch_size) for dist in cont_latent_dist]
    #     cont_total = tf.concat(cont_samples, axis=-1)
    disc_samples = [dist.sample(batch_size) for dist in disc_latent_dist]
    #     disc_total = tf.concat(disc_samples, axis=-1)
    noise = tf.concat([z, *disc_samples, *cont_samples], axis=1)
    return noise, disc_samples, cont_samples


class InfoGANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, n_row=10, latent_spec={}, log_dir=''):
        self.n_row = n_row
        self.latent_spec = latent_spec
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        noise, _, _ = sample(self.latent_spec, self.n_row ** 2)
        generated_images = self.model.generator(noise, training=False)
        generated_images = (generated_images * 127.5) + 127.5

        predictions = generated_images.numpy()
        fig = plt.figure(figsize=(self.n_row, self.n_row))
        for i in range(predictions.shape[0]):
            plt.subplot(self.n_row, self.n_row, i+1)
            plt.imshow(predictions[i, :, :] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        # tf.keras.utils.save_img(f'{self.log_dir}/img_epoch_{epoch}.png', generated_images.reshape(self.n_row, self.n_row, 1))
        fig.savefig(f'{self.log_dir}/img_epoch_{epoch}.png')
        # plt.title("Epoch " + str(epoch))
        # plt.show()
        return None



def sample_test(latent_spec, batch_size,
                discrete_idx_c=(0, 0), continuous_idx_c=(0, 0)):
    z = latent_spec['noise-variables'][0].sample(batch_size)
    # cont_latent_dist = latent_spec['continuous-latent-codes']
    # disc_latent_dist = latent_spec['discrete-latent-codes']

    # cont_samples = [dist.sample_test(cont, batch_size) for dist in cont_latent_dist]
    # disc_samples = [dist.sample_test(cat, batch_size) for dist in disc_latent_dist]
    _, disc_samples, cont_samples = sample(latent_spec, batch_size)

    idx, cat = discrete_idx_c
    disc_samples[idx] = latent_spec['discrete-latent-codes'][idx].sample_test(cat, batch_size)
    idx, c = continuous_idx_c
    cont_samples[idx] = latent_spec['continuous-latent-codes'][idx].sample_test(c, batch_size)
    noise = tf.concat([z, *disc_samples, *cont_samples], axis=1)

    return noise, disc_samples, cont_samples


def plot_test(generator, latent_spec, idx_of_varting_disc=0, idx_of_varting_cont=0):
    output_image = []
    for cat in range(10):
        var_cont_images = []
        for cont in np.arange(-2, 2, 0.4):
            noise, _, _ = sample_test(latent_spec, batch_size=1,
                                      discrete_idx_c=(0, cat), continuous_idx_c=(0, cont))
            imgs = generator(noise)
            imgs = (imgs.numpy() + 1.) / 2.
            var_cont_images.append(imgs.reshape([28, 28]))

        output_image.append(np.concatenate(var_cont_images, 0))

    output_image = np.concatenate(output_image, 1)
    plt.figure(figsize=(20, 10))
    plt.title(f"varying discrete latent code {idx_of_varting_disc}, varying continuous latent code {idx_of_varting_cont}")
    plt.imshow(output_image, cmap="gray")
    plt.axis("off")
    plt.savefig(f'varying-discrete-{idx_of_varting_disc}_varying-continuous-{idx_of_varting_cont}')
    plt.show()
