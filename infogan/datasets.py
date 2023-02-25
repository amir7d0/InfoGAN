import tensorflow as tf
from config import config

class MnistDataset(object):
    def __init__(self):
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        print(f"Number of examples: {len(train_images)}")
        print(f"Shape of the images in the dataset: {train_images.shape[1:]}")
        self.image_shape = (*train_images.shape[1:], 1)
        # Reshape each sample to (28, 28, 1) and normalize the pixel values in the [-1, 1] range
        train_images = train_images.reshape(train_images.shape[0], *self.image_shape).astype("float32")
        train_images = (train_images - 127.5) / 127.5

        ds = tf.data.Dataset.from_tensor_slices(train_images)
        ds = ds.shuffle(30000).batch(config.batch_size)
        self.dataset = ds
    def get_dataset(self):
        return self.dataset


class MyDataset(object):
    def __init__(self):
        pass