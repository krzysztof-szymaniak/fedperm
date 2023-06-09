import pathlib
from collections import defaultdict
from contextlib import redirect_stdout

import numpy as np
import visualkeras
from PIL import ImageFont
from matplotlib import colors
from tensorflow.keras.layers import Dense, Conv2D, SpatialDropout2D, Dropout, MaxPooling2D, BatchNormalization, \
    GlobalAveragePooling2D, Add, Multiply, DepthwiseConv2D, Concatenate, Activation
from tensorflow.keras import utils

SAVE_VIZ = False


def plot_model(save_folder, model, filename):
    pathlib.Path(save_folder).mkdir(exist_ok=True, parents=True)
    if SAVE_VIZ:
        utils.plot_model(
            model, show_layer_names=False, show_shapes=True, to_file=f'{save_folder}/{filename}_summary.png'
        )
    with open(f'{save_folder}/{filename}.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    # font = ImageFont.truetype("arial.ttf", 14)
    if SAVE_VIZ:
        visualkeras.layered_view(
            model, to_file=f'{save_folder}/{filename}_layers.png',
            legend=True,
            color_map=get_color_map()
        )


def get_color_map():
    layers = [
        Dense, Conv2D, SpatialDropout2D, Dropout, MaxPooling2D, BatchNormalization,
        GlobalAveragePooling2D, Add, Multiply, DepthwiseConv2D, Concatenate, Activation
    ]

    np.random.seed(1234)
    palette = np.random.choice(list(colors.CSS4_COLORS), len(layers))
    np.random.seed()
    color_map = defaultdict(dict)
    for i, l in enumerate(layers):
        color_map[l]['fill'] = palette[i]
    return color_map
