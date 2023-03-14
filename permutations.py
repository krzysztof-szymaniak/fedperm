import numpy as np
from sklearn.utils import shuffle

MAX_SEED = 10000000


def init_states(seed, grid_shape, shape):
    np.random.seed(seed)
    assert grid_shape == (2, 2)
    if shape == 'overlap_center_2x2':
        r_range = range(grid_shape[0])
        c_range = range(grid_shape[1])
        random_states = {(r, c): np.random.randint(1, MAX_SEED) for r in r_range for c in c_range}
        random_states[(0.5, 0.5)] = np.random.randint(1, MAX_SEED)

    elif shape == 'overlap_cross_2x2':
        r_range = np.arange(0, grid_shape[0] - 0.5, 0.5)
        c_range = np.arange(0, grid_shape[1] - 0.5, 0.5)
        random_states = {(r, c): np.random.randint(1, MAX_SEED) for r in r_range for c in c_range}
        random_states[(0.5, 0.5)] = np.random.randint(1, MAX_SEED)  # center
        random_states[(0, 0.5)] = np.random.randint(1, MAX_SEED)  # top
        random_states[(1, 0.5)] = np.random.randint(1, MAX_SEED)  # bot
        random_states[(0.5, 0)] = np.random.randint(1, MAX_SEED)  # left
        random_states[(0.5, 1)] = np.random.randint(1, MAX_SEED)  # right

    elif shape == 'overlap_full_2x2':
        r_range = np.arange(0, grid_shape[0] - 0.5, 0.5)
        c_range = np.arange(0, grid_shape[1] - 0.5, 0.5)
        random_states = {(r, c): np.random.randint(1, MAX_SEED) for r in r_range for c in c_range}

    else:  # no overlap
        r_range = range(grid_shape[0])
        c_range = range(grid_shape[1])
        random_states = {(r, c): np.random.randint(1, MAX_SEED) for r in r_range for c in c_range}
    np.random.seed()
    if seed == 0:
        for key in random_states:
            random_states[key] = 0
    return random_states


def perm(shape, random_state):
    indexes = np.arange(shape[0] * shape[1])
    if random_state == 0:  # identity
        return indexes
    return shuffle(indexes, random_state=random_state)


def generate_permutations(seed, grid_shape, subinput_shape, shape):
    random_states = init_states(seed, grid_shape, shape)
    return {(row, col): perm(subinput_shape, state) for (row, col), state in random_states.items()}
