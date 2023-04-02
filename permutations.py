import numpy as np
from sklearn.utils import shuffle

MAX_STATE = 10000000


def is_multiple(x, of, eps=1e-6):
    # returns true if a is a multiple of float b up to precision eps
    div = x / of
    return abs(div - round(div)) < eps


def cross(r, c, size=None, ctr=None):
    return (r == ctr[0] - 0.5 and abs(c - ctr[1] - 0.5 <= size)) or (
            c == ctr[1] - 0.5 and abs(r - ctr[0] - 0.5 <= size))


def center(r, c, radius=None, ctr=None):
    return np.sqrt((r - ctr[0]+0.5) ** 2 + (c - ctr[1] + 0.5) ** 2) <= radius


def init_states(seed, grid_shape, shape):
    if seed is not None:
        np.random.seed(seed)
    # base grid
    random_states = {(r, c): np.random.randint(1, MAX_STATE) for r in range(grid_shape[0]) for c in
                     range(grid_shape[1])}

    def add_overlap(condition, **kwargs):
        for r in np.arange(0, grid_shape[0] - 0.5, 0.5):
            for c in np.arange(0, grid_shape[1] - 0.5, 0.5):
                if (r, c) not in random_states and condition(r, c, **kwargs):
                    random_states[(r, c)] = np.random.randint(1, MAX_STATE)

    if shape == 'overlap_center':
        add_overlap(condition=center, radius=0, ctr=(grid_shape[0] / 2, grid_shape[1] / 2))

    elif shape == 'overlap_cross':
        add_overlap(condition=cross, size=grid_shape[0]//2, ctr=(grid_shape[0] / 2, grid_shape[1] / 2))

    elif shape == 'overlap_edges':
        add_overlap(condition=lambda r, c: (int(r) != r and int(c) == c) or (int(r) == r and int(c) != c))

    elif shape == 'overlap_corners':
        add_overlap(condition=lambda r, c: int(r) != r and int(c) != c)

    elif shape == 'overlap_full':
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


def generate_permutations(seed, grid_shape, subinput_shape, shape):
    random_states = init_states(seed, grid_shape, shape)
    return {(row, col): perm(subinput_shape, state) for (row, col), state in random_states.items()}
