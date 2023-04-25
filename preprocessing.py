from os.path import join

import cv2
import imageio
import matplotlib.pyplot as plt
from keras.datasets import cifar10, fashion_mnist, mnist
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def permute(arr, perm):
    res = np.zeros(arr.shape)
    for c in range(arr.shape[-1]):
        channel = arr[:, :, c]
        p = perm[c]
        res[:, :, c] = channel.ravel()[p].reshape(channel.shape)
    return res


def pad(img, dims=None):
    old_image_height, old_image_width, channels = img.shape
    new_image_width, new_image_height, _ = dims
    color = (0, 0, 0)
    result = np.full(dims, color, dtype=np.uint8)

    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2
    result[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = img
    return result


def resize_img(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    return np.array(Image.fromarray(img).resize(dim, resample=Image.NEAREST))


class PermutationGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, Y, augmenter: ImageDataGenerator, subinput_shape,
                 shuffle_dataset=True, batch_size=None, permutations=None, debug_path=None):
        self.Y = Y.copy()
        self.X = X.copy()
        self.n = len(X)
        self.datagen = augmenter
        self.shuffle = shuffle_dataset
        self.batch_size = batch_size
        self.subinput_shape = subinput_shape
        self.gen = self.datagen.flow(self.X, self.Y, batch_size=self.batch_size, shuffle=shuffle_dataset)
        self.permutations = permutations
        self.n_frames = len(permutations)
        self.debug_path = debug_path

    def run_debug(self):
        scale = 20
        max_imgs = 16
        if self.debug_path is None:
            return
        xb, yb = self.gen.next()
        sr, sc, channels = self.subinput_shape
        for index, x in enumerate(xb):
            x = (x * 255).astype(np.uint8)

            subimages = self.generate_frames(np.array([x]))
            if channels == 1:
                x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
            imgs = []
            x = resize_img(x, scale=scale)
            for i, ((row, col), subimg) in enumerate(zip(self.permutations, subimages)):
                if int(row) == row and int(col) == col:
                    color = (0, 255, 0)
                    width = 2
                elif int(row) == row or int(col) == col:
                    color = (0, 0, 255)
                    width = 5
                else:
                    color = (255, 0, 0)
                    width = 7
                x = cv2.rectangle(x,
                                  (int(row * sr) * scale, int(col * sc) * scale),
                                  (int((row + 1) * sr * scale), int((col + 1) * sc) * scale),
                                  color, width)
                subimg = subimg[0, ...].astype('uint8')
                if channels == 1:
                    subimg = cv2.cvtColor(subimg, cv2.COLOR_GRAY2RGB)
                subimg = resize_img(subimg, scale=scale)
                padded = pad(subimg, x.shape)
                res = np.hstack((x, padded))
                imgs.append(res)
            gif_path = join(self.debug_path, f'frames_{index}.gif')
            imageio.mimsave(gif_path, imgs, fps=55, duration=0.5)
            # os.system(f'xdg-open {gif_path}')

    def next(self):
        x, y = self.gen.next()
        xp = self.generate_frames(x)
        return xp, y

    def on_epoch_end(self):
        pass

    def __getitem__(self, index):
        return self.next()

    def __len__(self):
        return self.n // self.batch_size

    def generate_frames(self, x_batch):
        sr, sc, _ = self.subinput_shape
        x_frames = []
        for (row, col), perm in self.permutations.items():
            xb = np.zeros((x_batch.shape[0], *self.subinput_shape))
            for i, x in enumerate(x_batch):
                r_s = slice(int(row * sr), int((row + 1) * sr))
                c_s = slice(int(col * sc), int((col + 1) * sc))
                sub_img = x[r_s, c_s, :]
                xb[i, ...] = permute(sub_img, perm)
            x_frames.append(xb)
        return x_frames


def convert_ds_to_numpy(ds):
    x = np.asarray(list(map(lambda v: v[0], tfds.as_numpy(ds))))
    y = np.asarray(list(map(lambda v: v[1], tfds.as_numpy(ds))))
    return x, y


def to_categorical_n_classes(x_train, y_train, x_test, y_test):
    classes = np.unique(y_train)
    n_classes = len(classes)
    y_train = to_categorical(y_train, num_classes=n_classes)
    y_test = to_categorical(y_test, num_classes=n_classes)
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
    return (x_train, y_train), (x_test, y_test), n_classes


def load_data(dataset):
    if dataset in ['mnist', 'fashion_mnist', 'cifar10']:
        ds = {
            'mnist': mnist,
            'fashion_mnist': fashion_mnist,
            'cifar10': cifar10,
        }[dataset]
        (x_train, y_train), (x_test, y_test) = ds.load_data()
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        return to_categorical_n_classes(x_train, y_train, x_test, y_test)
    if dataset == 'emnist-letters':
        dataset = dataset.replace('-', '/')

        def transpose(x):
            return tf.image.flip_left_right(tf.image.rot90(x, -1))

        trainDataset = tfds.load(name=dataset, split='train', as_supervised=True).map(
            lambda x, y: (transpose(x), y - 1))
        testDataset = tfds.load(name=dataset, split='test', as_supervised=True).map(
            lambda x, y: (transpose(x), y - 1))
        x_train, y_train = convert_ds_to_numpy(trainDataset)
        x_test, y_test = convert_ds_to_numpy(testDataset)
        return to_categorical_n_classes(x_train, y_train, x_test, y_test)

    elif dataset == 'eurosat':
        split = ['train[:80%]', 'train[80%:]']
        trainDataset, testDataset = tfds.load(name=dataset, split=split, as_supervised=True)
        x_train, y_train = convert_ds_to_numpy(trainDataset)
        x_test, y_test = convert_ds_to_numpy(testDataset)
        return to_categorical_n_classes(x_train, y_train, x_test, y_test)
    elif dataset == 'cats_vs_dogs':
        HEIGHT = 100
        WIDTH = 100

        def preprocess(img, label):
            return tf.cast(tf.image.resize(img, [HEIGHT, WIDTH]), tf.uint8), tf.cast(label, tf.float32)

        split = ['train[:80%]', 'train[80%:]']
        trainDataset, testDataset = tfds.load(name=dataset, split=split, as_supervised=True)
        testDataset = testDataset.map(preprocess)
        trainDataset = trainDataset.map(preprocess)
        x_train, y_train = convert_ds_to_numpy(trainDataset)
        x_test, y_test = convert_ds_to_numpy(testDataset)
        classes = np.unique(y_train)
        n_classes = len(classes)
        return (x_train, y_train), (x_test, y_test), n_classes

    elif dataset == 'kmnist':
        trainDataset = tfds.load(name=dataset, split='train', as_supervised=True)
        testDataset = tfds.load(name=dataset, split='test', as_supervised=True)
        x_train, y_train = convert_ds_to_numpy(trainDataset)
        x_test, y_test = convert_ds_to_numpy(testDataset)
        return to_categorical_n_classes(x_train, y_train, x_test, y_test)
    else:
        raise Exception("No dataset with name " + dataset)


def get_classes_names_for_dataset(ds_name):
    classes = None
    if ds_name == 'mnist':
        classes = [str(i) for i in range(10)]
    elif ds_name == 'fashion_mnist':
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']
    elif ds_name == 'cifar10':
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif ds_name == 'cats_vs_dogs':
        classes = ['cat', 'dog']
    return classes


if __name__ == '__main__':
    ...
