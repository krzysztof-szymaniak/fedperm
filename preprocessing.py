import os
import pathlib
from os.path import join

import imageio
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt, patches
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


def permute(arr, perm):
    res = np.zeros(arr.shape)
    for c in range(arr.shape[-1]):
        channel = arr[:, :, c]
        res[:, :, c] = channel.ravel()[perm].reshape(channel.shape)
    return res


def get_gen(augmented=True):
    return ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    ) if augmented else ImageDataGenerator(rescale=1. / 255)


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
        if self.debug_path is None:
            return
        x = self.X[index]
        y = self.Y[index]
        sr, sc, channels = self.subinput_shape

        subimages = self.generate_frames(np.array([x]))
        pathlib.Path("tmp").mkdir(exist_ok=True)
        pathes = []
        imgs = []
        for i, (row, col) in enumerate(self.permutations):
            if int(row) == row and int(col) == col:
                color = 'g'
                width = 3
            elif int(row) == row or int(col) == col:
                color = 'b'
                width = 2
            else:
                color = 'r'
                width = 1
            rect = patches.Rectangle((int(row * sr), int(col * sc)), sr, sc, linewidth=width, edgecolor=color,
                                     facecolor='none')
            pathes.append(rect)

        fig, ax = plt.subplots(1, 2, )
        if channels == 1:
            x = np.array(Image.fromarray(x[..., 0], mode='L').convert('RGB'))
        ax[0].imshow(x)
        for i, (patch, subimg) in enumerate(zip(pathes, subimages)):
            subimg = subimg[0, ...]
            if channels == 1:
                subimg = np.array(Image.fromarray(subimg[..., 0], mode='L').convert('RGB'))
            im_path = f'tmp/{i}.png'
            ax[0].add_patch(patch)
            ax[1].clear()
            ax[1].imshow(subimg)
            fig.savefig(im_path)
            imgs.append(plt.imread(im_path))
            if i == len(pathes) - 1:
                extent = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(join(self.debug_path, f'grid_{index}.png'), bbox_inches=extent.expanded(1.1, 1.2))
        gif_path = join(self.debug_path, f'frames_{index}.gif')
        imageio.mimsave(gif_path, imgs, fps=55, duration=0.5)
        os.startfile(gif_path)

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
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    classes = np.unique(y_train)
    n_classes = len(classes)
    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
    return (x_train, y_train), (x_test, y_test), n_classes


if __name__ == '__main__':
    ...
