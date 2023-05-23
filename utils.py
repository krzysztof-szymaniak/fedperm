import pathlib
from contextlib import redirect_stdout
from tensorflow.keras import utils


def save_model_info(i_dir, model, filename):
    pathlib.Path(i_dir).mkdir(exist_ok=True, parents=True)
    utils.plot_model(model, show_layer_names=True, show_shapes=True, to_file=f'{i_dir}/{filename}.png')
    with open(f'{i_dir}/{filename}.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
