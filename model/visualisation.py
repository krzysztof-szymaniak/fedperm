import pathlib
from collections import defaultdict
from contextlib import redirect_stdout

import visualkeras
from PIL import ImageFont
from tensorflow.keras.layers import Dense, Conv2D, SpatialDropout2D, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Add, Multiply
from tensorflow.keras import utils

VISUALIZE_IN_SEGMENTS = True

color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = 'orange'
color_map[BatchNormalization]['fill'] = 'gray'
color_map[SpatialDropout2D]['fill'] = 'pink'
color_map[Dropout]['fill'] = 'pink'
color_map[MaxPooling2D]['fill'] = 'red'
color_map[Dense]['fill'] = 'green'
color_map[GlobalAveragePooling2D]['fill'] = 'teal'
color_map[Add]['fill'] = 'purple'


def plot_model(folder, model, filename):
    pathlib.Path(folder).mkdir(exist_ok=True, parents=True)
    utils.plot_model(model, show_layer_names=True, show_shapes=True, to_file=f'{folder}/{filename}.png')
    with open(f'{folder}/{filename}.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    # font = ImageFont.truetype("arial.ttf", 16)  # using comic sans is strictly prohibited!
    visualkeras.layered_view(model, to_file=f'{folder}/{filename}_layered.png', legend=True)
