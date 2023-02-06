import os
import pathlib
from os.path import join

from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib

from config import DATASET, grid_shape, dataset, seed, input_shape, single_batch_mode, \
    subinput_shape, augmented
from model import PlotProgress, get_aggregate_model, get_single_model
from permutations import generate_permutations
from preprocessing import save_training_info, PermutationGenerator, load_data, info_dir

print(device_lib.list_local_devices())

batch_size = 128
epochs = 1000
base_name = f"models/{DATASET}-{grid_shape[0]}x{grid_shape[1]}-seq"

resume_training = False
version = 1
model_name = f"{base_name}-v{version}"

if not (resume_training and os.path.exists(model_name)):
    version = 1
    model_name = f"{base_name}-v{version}"
    while os.path.exists(model_name):
        version += 1
        model_name = f"{base_name}-v{version}"


def train_models_sequence():
    train_seed = seed
    permutations = generate_permutations(seed, grid_shape,
                                         subinput_shape, overlap=True, full_overlap=True)
    (x_train, y_train), _, n_classes = load_data(dataset, input_shape)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, shuffle=True,
                                                      random_state=train_seed)
    if single_batch_mode:
        x_train = x_train[:batch_size]
        y_train = y_train[:batch_size]
        x_val = x_val[:batch_size]
        y_val = y_val[:batch_size]

    for i, (coords, perm) in enumerate(permutations.items()):
        sub_model = join(model_name, "models", str(i))
        train_gen = ImageDataGenerator(
            rescale=1 / 255,
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.05,  # Randomly zoom image
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        ) if augmented else ImageDataGenerator(rescale=1. / 255)
        valid_gen = ImageDataGenerator(rescale=1. / 255)
        train_ds = PermutationGenerator(x_train, y_train, train_gen, subinput_shape,
                                        permutations={coords: perm},
                                        batch_size=batch_size, debug=False)
        valid_ds = PermutationGenerator(x_val, y_val, valid_gen, subinput_shape,
                                        permutations={coords: perm},
                                        batch_size=batch_size)

        training_info_dir = os.path.join(sub_model, info_dir)
        pathlib.Path(training_info_dir).mkdir(exist_ok=True, parents=True)

        model = get_single_model(train_ds.subinput_shape, n_classes, i, training_info_dir)
        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(learning_rate=1e-3),
                      metrics=['accuracy'])

        print(f"Model name: {sub_model}")
        try:
            model.fit(train_ds, epochs=epochs, verbose=1, validation_data=valid_ds,
                      steps_per_epoch=len(x_train) // batch_size,
                      validation_steps=len(x_val) // batch_size,
                      callbacks=[
                          EarlyStopping(monitor="val_loss", verbose=1, patience=16, restore_best_weights=True),
                          ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, verbose=1, min_lr=1e-6),
                          PlotProgress(training_info_dir),
                          TensorBoard(log_dir=f'./{sub_model}/graph', histogram_freq=1, write_graph=True,
                                      write_images=True)
                      ])
        except KeyboardInterrupt:
            print("\nInterrupted!")
        print(f"Saving model {sub_model}")
        model.save(sub_model)
        save_training_info(model, model.history, info_path=training_info_dir, show=False)

    with open(join(model_name, "seed"), 'w') as f:
        print(train_seed, file=f)

    training_info_dir = os.path.join(model_name, info_dir)
    pathlib.Path(training_info_dir).mkdir(exist_ok=True, parents=True)
    aggr_model = get_aggregate_model(join(model_name, "models"), subinput_shape, n_classes, training_info_dir)
    aggr_model.compile(loss=categorical_crossentropy,
                       optimizer=Adam(learning_rate=1e-3),
                       metrics=['accuracy'])
    train_gen = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.05,  # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    ) if augmented else ImageDataGenerator(rescale=1. / 255)
    valid_gen = ImageDataGenerator(rescale=1. / 255)
    train_ds = PermutationGenerator(x_train, y_train, train_gen, subinput_shape, permutations=permutations,
                                    batch_size=batch_size, debug=False)
    valid_ds = PermutationGenerator(x_val, y_val, valid_gen, subinput_shape, permutations=permutations,
                                    batch_size=batch_size)

    print(f"Model name: {model_name}")
    try:
        aggr_model.fit(train_ds, epochs=epochs, verbose=1, validation_data=valid_ds,
                       steps_per_epoch=len(x_train) // batch_size,
                       validation_steps=len(x_val) // batch_size,
                       callbacks=[
                           EarlyStopping(monitor="val_loss", verbose=1, patience=16, restore_best_weights=True),
                           ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, verbose=1, min_lr=1e-6),
                           PlotProgress(training_info_dir),
                           TensorBoard(log_dir=f'./{model_name}/graph', histogram_freq=1, write_graph=True,
                                       write_images=True)
                       ])
    except KeyboardInterrupt:
        print("\nInterrupted!")
    print(f"Saving model {model_name}")
    aggr_model.save(model_name)
    save_training_info(aggr_model, aggr_model.history, info_path=training_info_dir, show=False)


if __name__ == '__main__':
    train_models_sequence()
