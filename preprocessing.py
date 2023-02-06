from os.path import join
from pprint import pprint

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

info_dir = "training_info"


def permute(arr, perm):
    res = np.zeros(arr.shape)
    for c in range(arr.shape[-1]):
        channel = arr[:, :, c]
        res[:, :, c] = channel.ravel()[perm].reshape(channel.shape)
    return res


class PermutationGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, Y, augmenter: ImageDataGenerator, subinput_shape,
                 shuffle_dataset=True, debug=False, batch_size=128, permutations=None):
        self.Y = Y
        self.X = X
        self.n = len(X)
        self.datagen = augmenter
        self.debug = debug
        self.shuffle = shuffle_dataset
        self.batch_size = batch_size

        self.permuted = True if permutations is not None else False
        self.subinput_shape = subinput_shape
        self.gen = self.datagen.flow(self.X, self.Y, batch_size=self.batch_size, shuffle=shuffle_dataset)
        self.permutations = permutations
        self.n_frames = len(permutations)

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
        sr, sc, _ = self.subinput_shape
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


def save_training_info(model, history, info_path=None, show=False):
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
