import os
import pathlib
from os.path import join

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.saving.save import load_model
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib

from config import DATASET, permuted, augmented, grid_shape, dataset, seed, input_shape, single_batch_mode
from model import get_composite_model, PlotProgress
from preprocessing import save_training_info, PermutationGenerator, load_data, info_dir, get_seed

print(device_lib.list_local_devices())

batch_size = 64
epochs = 1000
base_name = f"models/{DATASET}-{'perm-' if permuted else ''}{'aug-' if augmented else ''}{grid_shape[0]}x{grid_shape[1]}"

resume_training = False
version = 1
model_name = f"{base_name}-v{version}"
if not (resume_training and os.path.exists(model_name)):
    version = 1
    model_name = f"{base_name}-v{version}"
    while os.path.exists(model_name):
        version += 1
        model_name = f"{base_name}-v{version}"


def train():
    train_seed = seed
    (x_train, y_train), _, n_classes = load_data(dataset, input_shape)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, shuffle=True,
                                                      random_state=train_seed)
    if single_batch_mode:
        x_train = x_train[:batch_size]
        y_train = y_train[:batch_size]
        x_val = x_val[:batch_size]
        y_val = y_val[:batch_size]
    train_gen = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.05,  # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    ) if augmented else ImageDataGenerator(rescale=1. / 255)

    valid_gen = ImageDataGenerator(rescale=1. / 255)

    train_ds = PermutationGenerator(x_train, y_train, train_gen, input_shape, grid_shape, seed=train_seed,
                                    batch_size=batch_size, debug=False)

    valid_ds = PermutationGenerator(x_val, y_val, valid_gen, input_shape, grid_shape, seed=train_seed,
                                    batch_size=batch_size)
    training_info_dir = os.path.join(model_name, info_dir)
    pathlib.Path(training_info_dir).mkdir(exist_ok=True, parents=True)
    if resume_training:
        model = load_model(model_name)
        train_seed = get_seed(model_name)
    else:
        model = get_composite_model(train_ds.subinput_shape, n_classes, train_ds.n_frames, training_info_dir)
    print(f"Model name: {model_name}")
    try:
        history = model.fit(train_ds, epochs=epochs, verbose=1, validation_data=valid_ds,
                            steps_per_epoch=len(x_train) // batch_size,
                            validation_steps=len(x_val) // batch_size,
                            callbacks=[
                                EarlyStopping(monitor="val_loss", verbose=1, patience=32, restore_best_weights=True),
                                ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, verbose=1, min_lr=1e-6),
                                PlotProgress(training_info_dir),
                                TensorBoard(log_dir=f'./{model_name}/graph', histogram_freq=1, write_graph=True,
                                            write_images=True)
                            ])
        save_training_info(model, history, info_path=training_info_dir, show=False, seed=train_seed)
    except KeyboardInterrupt:
        print("\nInterrupted!")
    print(f"Saving model {model_name}")
    model.save(model_name)
    with open(join(training_info_dir, "seed"), 'w') as f:
        print(train_seed, file=f)


if __name__ == '__main__':
    train()
