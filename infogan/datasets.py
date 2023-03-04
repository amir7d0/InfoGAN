import tensorflow as tf


def get_dataset(dataset_name, batch_size):
    if dataset_name == 'mnist':
        return MNIST(batch_size)
    if dataset_name == 'fashion_mnist':
        return FashionMNIST(batch_size)


class MNIST(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
        print(f"Number of examples: {len(train_images)}")
        print(f"Shape of the images in the dataset: {train_images.shape[1:]}")
        self.image_shape = (*train_images.shape[1:], 1)
        # Reshape each sample to (28, 28, 1) and normalize the pixel values in the [-1, 1] range
        train_images = train_images.reshape(train_images.shape[0], *self.image_shape).astype("float32")
        train_images = (train_images - 127.5) / 127.5

        ds = tf.data.Dataset.from_tensor_slices(train_images)
        ds = ds.shuffle(30000).batch(self.batch_size)
        self.dataset = ds

    def get_dataset(self):
        return self.dataset


class FashionMNIST(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        (train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
        print(f"Number of examples: {len(train_images)}")
        print(f"Shape of the images in the dataset: {train_images.shape[1:]}")
        self.image_shape = (*train_images.shape[1:], 1)
        # Reshape each sample to (28, 28, 1) and normalize the pixel values in the [-1, 1] range
        train_images = train_images.reshape(train_images.shape[0], *self.image_shape).astype("float32")
        train_images = (train_images - 127.5) / 127.5

        ds = tf.data.Dataset.from_tensor_slices(train_images)
        ds = ds.shuffle(30000).batch(self.batch_size)
        self.dataset = ds

    def get_dataset(self):
        return self.dataset


class SVHN:
    def __init__(self, batch_size):
        pass
    