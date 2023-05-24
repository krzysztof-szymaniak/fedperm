from keras_preprocessing.image import ImageDataGenerator

from model.train_configs import BATCH_SIZE, augmentation
from permutation.permutations import PermutationGenerator


def get_train_valid_gens(data, permutations, sub_input_shape, examples_path, apply_flip=False):
    x_train, y_train, x_val, y_val = data
    train_ds = get_generator(x_train, y_train,
                             batch_size=BATCH_SIZE,
                             permutations=permutations,
                             sub_input_shape=sub_input_shape,
                             augmented=True,
                             examples_path=examples_path,
                             debug=False,
                             shuffle=True,
                             apply_flip=apply_flip)
    valid_ds = get_generator(x_val, y_val,
                             batch_size=BATCH_SIZE,
                             permutations=permutations,
                             examples_path=examples_path,
                             sub_input_shape=sub_input_shape)
    return train_ds, valid_ds


def get_generator(x, y,
                  apply_flip=False,
                  permutations=None,
                  examples_path=None,
                  augmented=None,
                  sub_input_shape=None,
                  shuffle=False,
                  debug=False,
                  batch_size=None,
                  ):
    aug = ImageDataGenerator(**augmentation(apply_flip=apply_flip)) if augmented else ImageDataGenerator()
    gen = PermutationGenerator(x, y, aug,
                               subinput_shape=sub_input_shape, permutations=permutations, batch_size=batch_size,
                               examples_path=examples_path, shuffle_dataset=shuffle)
    if debug:
        gen.run_debug()
    return gen

