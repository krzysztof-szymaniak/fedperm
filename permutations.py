import numpy as np
from sklearn.utils import shuffle

from enums import Overlap
from os.path import join

import cv2
import imageio
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MAX_SEED = 10000000


def cross(r, c, size=None, ctr=None):
    return (r == ctr[0] - 0.5 and abs(c - ctr[1] - 0.5 <= size)) or (
            c == ctr[1] - 0.5 and abs(r - ctr[0] - 0.5 <= size))


def center(r, c, radius=None, ctr=None):
    return np.sqrt((r - ctr[0] + 0.5) ** 2 + (c - ctr[1] + 0.5) ** 2) <= radius


def init_keys(seed, grid_shape, overlap, channels, block_ratio, base_grid=True):
    if seed is not None:
        np.random.seed(seed)
    keys = {}
    if base_grid:
        for r in range(grid_shape[0]):
            for c in range(grid_shape[1]):
                if (r, c) not in keys:
                    keys[(r, c)] = [[np.random.randint(1, MAX_SEED) for _ in range(block_ratio ** 2 + 1)] for _ in
                                    range(channels)]

    def add_overlap(condition, **kwargs):
        for r in np.arange(0, grid_shape[0] - 0.5, 0.5):
            for c in np.arange(0, grid_shape[1] - 0.5, 0.5):
                if (r, c) not in keys and condition(r, c, **kwargs):
                    keys[(r, c)] = [[np.random.randint(1, MAX_SEED) for _ in range(block_ratio ** 2 + 1)] for _ in
                                    range(channels)]

    if overlap == Overlap.CENTER.value:
        add_overlap(condition=center, radius=0, ctr=(grid_shape[0] / 2, grid_shape[1] / 2))

    elif overlap == Overlap.CROSS.value:
        add_overlap(condition=cross, size=grid_shape[0] // 2, ctr=(grid_shape[0] / 2, grid_shape[1] / 2))

    elif overlap == Overlap.EDGES.value:
        add_overlap(condition=lambda r, c: (int(r) != r and int(c) == c) or (int(r) == r and int(c) != c))

    elif overlap == Overlap.CORNERS.value:
        add_overlap(condition=lambda r, c: int(r) != r and int(c) != c)

    elif overlap == Overlap.FULL.value:
        add_overlap(condition=lambda r, c: True)

    else:  # no overlap
        pass
    np.random.seed()  # restore randomness
    if seed is None:  # identity mode
        for key in keys:
            keys[key] = None
    return keys


def perm(shape, block_seeds=None, scheme=None):
    if scheme and block_seeds and scheme.value[0]:
        block_ratio, perm_blocks = scheme.value
        sr, sc = (shape[0] // block_ratio, shape[1] // block_ratio)
        slices = []
        for row in range(block_ratio):
            for col in range(block_ratio):
                r_s = slice(int(row * sr), int((row + 1) * sr))
                c_s = slice(int(col * sc), int((col + 1) * sc))
                slices.append((r_s, c_s))
        indexes = np.arange(shape[0] * shape[1]).reshape((shape[0], shape[1]))
        if perm_blocks:
            slices = shuffle(slices, random_state=block_seeds[0])
            new_ind = np.arange(shape[0] * shape[1]).reshape((shape[0], shape[1]))
            i = 0
            for row in range(block_ratio):
                for col in range(block_ratio):
                    r_s = slice(int(row * sr), int((row + 1) * sr))
                    c_s = slice(int(col * sc), int((col + 1) * sc))
                    s = slices[i]
                    b = block_seeds[i + 1]
                    new_ind[r_s, c_s] = shuffle(indexes[s[0], s[1]].ravel(), random_state=b).reshape(sr, sc)
                    i += 1
            return new_ind
        else:
            for i, (s, b) in enumerate(zip(slices, block_seeds[1:])):
                indexes[s[0], s[1]] = shuffle(indexes[s[0], s[1]].ravel(), random_state=b).reshape(sr, sc)
            return indexes

    indexes = np.arange(shape[0] * shape[1])
    if block_seeds is None:  # identity
        return indexes
    return shuffle(indexes, random_state=block_seeds[0])


def generate_permutations(seed, grid_shape, subinput_shape, overlap, scheme):
    channels = subinput_shape[-1]
    random_states = init_keys(seed, grid_shape, overlap, channels, scheme.value[0] if scheme else 0)
    permutations = {}
    for (row, col), frame_seeds in random_states.items():
        if seed is None:
            permutations[(row, col)] = [perm(subinput_shape) for _ in range(channels)]
        else:
            permutations[(row, col)] = [perm(subinput_shape, scheme=scheme, block_seeds=fr) for fr in frame_seeds]
    return permutations


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
            subimages = self.generate_frames(np.array([x]))
            x = cv2.normalize(x, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            x = x.astype(np.uint8)

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
                subimg = subimg[0, ...]
                subimg = cv2.normalize(subimg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                subimg = subimg.astype(np.uint8)
                if channels == 1:
                    subimg = cv2.cvtColor(subimg, cv2.COLOR_GRAY2RGB)
                subimg = resize_img(subimg, scale=scale)
                padded = pad(subimg, x.shape)
                res = np.hstack((x, padded))
                imgs.append(res)
            gif_path = join(self.debug_path, f'frames_{index + 1}.gif')
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
