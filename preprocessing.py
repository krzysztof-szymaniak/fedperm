from os.path import join

import cv2
import imageio
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
        res[:, :, c] = channel.ravel()[perm].reshape(channel.shape)
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
                 shuffle_dataset=True, batch_size=128, permutations=None, debug_path=None):
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

    def run_debug(self, index):
        scale = 20
        if self.debug_path is None:
            return
        x = self.X[index]
        y = self.Y[index]
        sr, sc, channels = self.subinput_shape

        subimages = self.generate_frames(np.array([x]))
        if channels == 1:
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        imgs = []
        x = resize_img(x, scale=scale)
        for i, ((row, col), subimg) in enumerate(zip(self.permutations, subimages)):
            if int(row) == row and int(col) == col:
                color = (0, 255, 0)
                width = 5
            elif int(row) == row or int(col) == col:
                color = (0, 0, 255)
                width = 3
            else:
                color = (255, 0, 0)
                width = 1
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


def load_data(dataset):
    if type(dataset) != str:
        (x_train, y_train), (x_test, y_test) = dataset.load_data()
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        classes = np.unique(y_train)
        n_classes = len(classes)
        y_train = to_categorical(y_train, num_classes=n_classes)
        y_test = to_categorical(y_test, num_classes=n_classes)
        if len(x_train.shape) == 3:
            x_train = np.expand_dims(x_train, axis=-1)
            x_test = np.expand_dims(x_test, axis=-1)
        return (x_train, y_train), (x_test, y_test), n_classes
    else:
        if dataset == 'kmnist':
            trainDataset = tfds.load(name=dataset, split='train', as_supervised=True)
            testDataset = tfds.load(name=dataset, split='test', as_supervised=True)
            x_train = np.asarray(list(map(lambda x: x[0], tfds.as_numpy(trainDataset))))
            y_train = np.asarray(list(map(lambda x: x[1], tfds.as_numpy(trainDataset))))

            x_test = np.asarray(list(map(lambda x: x[0], tfds.as_numpy(testDataset))))
            y_test = np.asarray(list(map(lambda x: x[1], tfds.as_numpy(testDataset))))

            classes = np.unique(y_train)
            n_classes = len(classes)
            y_train = to_categorical(y_train, num_classes=n_classes)
            y_test = to_categorical(y_test, num_classes=n_classes)
            return (x_train, y_train), (x_test, y_test), n_classes
        else:
            HEIGHT = 200
            WIDTH = 200

            def preprocess(img, label):
                return tf.cast(tf.image.resize(img, [HEIGHT, WIDTH]), tf.uint8), tf.cast(label, tf.float32)

            split = ['train[:70%]', 'train[70%:]']

            trainDataset, testDataset = tfds.load(name=dataset, split=split, as_supervised=True)

            testDataset = testDataset.map(preprocess)
            trainDataset = trainDataset.map(preprocess)
            x_train = np.asarray(list(map(lambda x: x[0], tfds.as_numpy(trainDataset))))
            y_train = np.asarray(list(map(lambda x: x[1], tfds.as_numpy(trainDataset))))

            x_test = np.asarray(list(map(lambda x: x[0], tfds.as_numpy(testDataset))))
            y_test = np.asarray(list(map(lambda x: x[1], tfds.as_numpy(testDataset))))

            classes = np.unique(y_train)
            n_classes = len(classes)
            return (x_train, y_train), (x_test, y_test), n_classes


if __name__ == '__main__':
    ...
