import time

import numpy as np
from sklearn.utils import shuffle

MAX_RANGE = 10000000


def init_states(seed, grid_shape, overlap=True, full_overlap=False):
    np.random.seed(seed)
    if overlap:
        if not full_overlap:
            r_range = range(grid_shape[0])
            c_range = range(grid_shape[1])
            random_states = {(r, c): np.random.randint(1, MAX_RANGE) for r in r_range for c in c_range}
            random_states[(0.5, 0.5)] = np.random.randint(1, MAX_RANGE)
            np.random.seed(int(time.time()))
            return random_states
        else:
            r_range = np.arange(0, grid_shape[0] - 0.5, 0.5)
            c_range = np.arange(0, grid_shape[1] - 0.5, 0.5)
    else:
        r_range = range(grid_shape[0])
        c_range = range(grid_shape[1])
    random_states = {(r, c): np.random.randint(1, MAX_RANGE) for r in r_range for c in c_range}
    np.random.seed(int(time.time()))
    return random_states


def perm(shape, random_state):
    return shuffle(np.arange(shape[0] * shape[1]), random_state=random_state)


def generate_permutations(seed, grid_shape, subinput_shape, overlap=True, full_overlap=False):
    random_states = init_states(seed, grid_shape, overlap=overlap, full_overlap=full_overlap)
    return {(row, col): perm(subinput_shape, state) for (row, col), state in random_states.items()}
