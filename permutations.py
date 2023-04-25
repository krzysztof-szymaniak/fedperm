import numpy as np
from sklearn.utils import shuffle

from enum import Enum

from enums import Overlap

MAX_STATE = 10000000


def cross(r, c, size=None, ctr=None):
    return (r == ctr[0] - 0.5 and abs(c - ctr[1] - 0.5 <= size)) or (
            c == ctr[1] - 0.5 and abs(r - ctr[0] - 0.5 <= size))


def center(r, c, radius=None, ctr=None):
    return np.sqrt((r - ctr[0] + 0.5) ** 2 + (c - ctr[1] + 0.5) ** 2) <= radius


def init_states(seed, grid_shape, overlap, channels):
    if seed is not None:
        np.random.seed(seed)
    random_states = {}
    for r in range(grid_shape[0]):
        for c in range(grid_shape[1]):
            if (r, c) not in random_states:
                random_states[(r, c)] = [np.random.randint(1, MAX_STATE) for _ in range(channels)]

    def add_overlap(condition, **kwargs):
        for r in np.arange(0, grid_shape[0] - 0.5, 0.5):
            for c in np.arange(0, grid_shape[1] - 0.5, 0.5):
                if (r, c) not in random_states and condition(r, c, **kwargs):
                    random_states[(r, c)] = [np.random.randint(1, MAX_STATE) for _ in range(channels)]

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
        for key in random_states:
            random_states[key] = None
    return random_states


def perm(shape, random_state):
    indexes = np.arange(shape[0] * shape[1])
    if random_state is None:  # identity
        return indexes
    return shuffle(indexes, random_state=random_state)


def generate_permutations(seed, grid_shape, subinput_shape, overlap):
    channels = subinput_shape[-1]
    random_states = init_states(seed, grid_shape, overlap, channels)
    permutations = {}
    for (row, col), frame_seeds in random_states.items():
        permutations[(row, col)] = [perm(subinput_shape, fr) for fr in frame_seeds]
    return permutations

