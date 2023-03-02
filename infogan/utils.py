import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv


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


def sample_test(latent_spec, batch_size,
                discrete_idx_c=(0, 0),
                continuous_idx_c=(0, 0)):
    _, disc_samples, cont_samples = sample(latent_spec, batch_size)
    z = latent_spec['noise-variables'][0].sample(batch_size)

    idx, cat = discrete_idx_c
    disc_samples[idx] = latent_spec['discrete-latent-codes'][idx].sample_test(cat, batch_size)
    idx, c = continuous_idx_c
    cont_samples[idx] = latent_spec['continuous-latent-codes'][idx].sample_test(c, batch_size)
    noise = tf.concat([z, *disc_samples, *cont_samples], axis=1)

    return noise, disc_samples, cont_samples


def plot_test(generator, latent_spec,
              n_row=10, n_col=10,
              idx_range_varying_disc=(0, [0, 10]),
              idx_range_varying_cont=(0, [-2, 2])):

    disc_idx, disc_range = idx_range_varying_disc
    cont_idx, cont_range = idx_range_varying_cont

    cat_codes = np.random.choice(np.arange(disc_range[0], disc_range[1], 1), size=n_row, replace=False)
    output_images = []
    for cat in cat_codes:
        var_cont_images = []
        for cont in np.linspace(cont_range[0], cont_range[1], n_col):
            noise, _, cc = sample_test(latent_spec, batch_size=1,
                                       discrete_idx_c=(disc_idx, cat),
                                       continuous_idx_c=(cont_idx, cont))
            imgs = generator(noise, training=False)
            imgs = (imgs.numpy() * 127.5) + 127.5
            var_cont_images.append(imgs.reshape([28, 28]))
        output_images.append(np.concatenate(var_cont_images, 1))

    output_image = np.concatenate(output_images, 0)
    plt.figure(figsize=(20, 10))
    plt.title(f"varying discrete latent code {disc_idx} in [{disc_range[0]}, {disc_range[1]}], varying continuous latent code {cont_idx} in [{cont_range[0]}, {cont_range[1]}]")
    plt.imshow(output_image, cmap="gray")
    plt.axis("off")
    plt.savefig(f'varying-discrete-{disc_idx}_varying-continuous-{cont_idx}')
    plt.show()


class InfoGANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, n_row=10, latent_spec=None, log_dir=''):
        self.n_row = n_row
        self.latent_spec = latent_spec
        self.log_dir = log_dir
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        noise, _, _ = sample(self.latent_spec, self.n_row ** 2)
        generated_images = self.model.generator(noise, training=False)
        generated_images = (generated_images * 127.5) + 127.5

        predictions = generated_images.numpy()
        fig = plt.figure(figsize=(self.n_row, self.n_row))
        for i in range(predictions.shape[0]):
            plt.subplot(self.n_row, self.n_row, i+1)
            plt.imshow(predictions[i, :, :], cmap='gray')
            plt.axis('off')
        fig.savefig(f'{self.log_dir}/img_epoch_{epoch+1:04d}.png')
        plt.close(fig)
        return None


class InfoGANCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir=''):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = None
        self.save_path = ''
        super().__init__()

    def on_train_begin(self, logs=None):
        self.checkpoint = tf.train.Checkpoint(self.model)

    def on_epoch_end(self, epoch, logs=None):
        path = f'{self.checkpoint_dir}/training_checkpoints'
        self.save_path = self.checkpoint.save(path)
        return None

    def on_train_end(self, logs=None):
        print(f"Model saved in {self.save_path}")


class InfoGANCSVLogger(tf.keras.callbacks.Callback):
    def __init__(self, filename, separator=",", append=True):
        self.filename = filename
        self.sep = separator
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.csv_file = None
        super().__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if tf.io.gfile.exists(self.filename):
                with tf.io.gfile.GFile(self.filename, "r") as f:
                    self.append_header = not bool(len(f.readline()))
            mode = "a"
        else:
            mode = "w"
        self.csv_file = tf.io.gfile.GFile(self.filename, mode)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if not self.writer:
            fieldnames = ["batch"] + self.keys
            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames)
            if self.append_header:
                self.writer.writeheader()

        row_dict = {"batch": batch}
        row_dict.update((key, logs[key]) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None


# import wandb
# from typing import Any, Dict, Optional, Union
#
#
# class InfoGANWBLogger(tf.keras.callbacks.Callback):
#     def __init__(
#         self,
#         log_freq: int = 1,
#         initial_global_step: int = 0,
#         *args: Any,
#         **kwargs: Any,
#     ) -> None:
#         super().__init__(*args, **kwargs)
#         if wandb.run is None:
#             raise wandb.Error("You must call `wandb.init()` before WandbMetricsLogger()")
#         self.global_batch = 0
#         self.global_step = initial_global_step
#         self.log_freq: Any = log_freq
#         # define custom x-axis for batch logging.
#         wandb.define_metric("batch/batch_step")
#         # set all batch metrics to be logged against batch_step.
#         wandb.define_metric("batch/*", step_metric="batch/batch_step")
#
#
#     def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
#         self.global_step += 1
#         """An alias for `on_train_batch_end` for backwards compatibility."""
#         if batch % self.log_freq == 0:
#             logs = {f"batch/{k}": v for k, v in logs.items()} if logs else {}
#             logs["batch/batch_step"] = self.global_batch
#
#             lr = self._get_lr()
#             if lr is not None:
#                 logs["batch/learning_rate"] = lr
#
#             wandb.log(logs)
#
#             self.global_batch += self.log_freq