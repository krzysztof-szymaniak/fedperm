import pathlib
import sys
import time
from os.path import join
from pprint import pprint

import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

info_dir = "training_info"


def generate_permutation(shape, random_state):
    return shuffle(np.arange(shape[0] * shape[1]), random_state=random_state)


def permute(arr, perm):
    res = np.zeros(arr.shape)
    for c in range(arr.shape[-1]):
        channel = arr[:, :, c]
        res[:, :, c] = channel.ravel()[perm].reshape(channel.shape)
    return res


class PermutationGenerator(keras.utils.Sequence):
    def __init__(self, X, Y, augmenter: ImageDataGenerator, input_shape, grid_shape,
                 shuffle_dataset=True, debug=False, batch_size=128, seed=None):
        self.Y = Y
        self.X = X
        self.n = len(X)
        self.datagen = augmenter
        self.debug = debug
        self.shuffle = shuffle_dataset
        self.random_states = init_states(seed, grid_shape)
        self.grid_shape = grid_shape
        self.input_shape = input_shape
        self.batch_size = batch_size

        self.permuted = True if seed is not None else False
        self.subinput_shape = (input_shape[0] // grid_shape[0], input_shape[1] // grid_shape[1], input_shape[2])
        self.gen = self.datagen.flow(self.X, self.Y, batch_size=self.batch_size, shuffle=shuffle_dataset)
        self.permutations = {(row, col): generate_permutation(self.subinput_shape, self.random_states[(row, col)]) for
                             (row, col) in self.random_states}
        self.n_frames = len(self.random_states)

    def next(self):
        x, y = self.gen.next()
        xp = self.generate_frames(x)
        if self.debug:
            rows = 5
            fig, ax = plt.subplots(rows, self.n_frames + 1, figsize=(10, 10))
            for j in range(rows):
                ax[j, 0].imshow(x[j])
                ax[j, 0].set_title(f"{np.argmax(y[j])}")
                for i in range(self.n_frames):
                    ax[j, i + 1].imshow(xp[i][j])
            plt.tight_layout()
            plt.show()
        return xp, y

    def on_epoch_end(self):
        ...

    def __getitem__(self, index):
        return self.next()

    def __len__(self):
        return self.n // self.batch_size

    def generate_frames(self, x_batch):
        sr = self.subinput_shape[0]
        sc = self.subinput_shape[1]
        x_frames = []
        for f, ((row, col), perm) in enumerate(self.permutations.items()):
            xb = np.zeros((x_batch.shape[0], *self.subinput_shape))
            for i, x in enumerate(x_batch):
                r_s = slice(int(row * sr), int((row + 1) * sr))
                c_s = slice(int(col * sc), int((col + 1) * sc))
                sub_img = x[r_s, c_s, :]
                xb[i, ...] = permute(sub_img, perm) if self.permuted else sub_img
            x_frames.append(xb)
        return x_frames


def bar(i, n, label=None):
    bar_width = 50
    progress = (i + 1) / n
    sys.stdout.write('\r')
    sys.stdout.write(
        f"[{'=' * int(bar_width * progress):{bar_width}s}] {int(100 * progress)}% ( {i + 1}/{n} )  {label if label is not None else ''}")
    sys.stdout.flush()


def init_states(seed, grid_shape, overlap=True, reduce_overlap=True):
    np.random.seed(seed)
    if overlap or reduce_overlap:
        if reduce_overlap:
            r_range = range(grid_shape[0])
            c_range = range(grid_shape[1])
            random_states = {(r, c): np.random.randint(1, 10000) for r in r_range for c in c_range}
            random_states[(0.5, 0.5)] = np.random.randint(1, 10000)
            np.random.seed(int(time.time()))
            return random_states
        else:
            r_range = np.arange(0, grid_shape[0] - 0.5, 0.5)
            c_range = np.arange(0, grid_shape[1] - 0.5, 0.5)
    else:
        r_range = range(grid_shape[0])
        c_range = range(grid_shape[1])
    random_states = {(r, c): np.random.randint(1, 10000) for r in r_range for c in c_range}
    np.random.seed(int(time.time()))
    return random_states


def load_data(dataset, input_shape):
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    classes = np.unique(y_train)
    n_classes = len(classes)
    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)
    x_train = x_train.reshape(x_train.shape[0], *input_shape)
    x_test = x_test.reshape(x_test.shape[0], *input_shape)
    return (x_train, y_train), (x_test, y_test), n_classes


def save_training_info(model, history, info_path=None, show=False, seed=None):
    plt.clf()
    metrics = set([m.replace('val_', '') for m in history.history.keys()])
    for met in metrics:
        plt.plot(history.history[met])
        if f"val_{met}" in history.history:
            plt.plot(history.history[f"val_{met}"])
        plt.title(f"{met}")
        plt.ylabel(met)
        plt.xlabel('epoch')
        if f"val_{met}" in history.history:
            plt.legend(['train', 'validate'], loc='right')
        else:
            plt.legend(['train'], loc='right')
        plt.savefig(join(info_path, met))
        if show:
            plt.show()
        plt.clf()
    with open(join(info_path, "model_config"), 'w') as f:
        pprint(model.get_config(), f)


def get_seed(model_name):
    with open(join(model_name, info_dir, "seed"), 'r') as f:
        seed = int(f.readline())
        return seed



if __name__ == '__main__':
    ...
