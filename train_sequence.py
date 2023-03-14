import os
import pathlib
from os.path import join

from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib

from config import grid_shape, dataset, seed, input_shape, single_batch_mode, \
    subinput_shape, augmented, shape, base_name, batch_size, epochs, LR, split_seed
from model import PlotProgress, get_aggregate_model, get_single_model
from permutations import generate_permutations
from preprocessing import save_training_info, PermutationGenerator, load_data, info_dir, get_gen

print(device_lib.list_local_devices())

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
                                         subinput_shape, shape)
    (x_train, y_train), _, n_classes = load_data(dataset, input_shape)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True,
                                                      random_state=split_seed)
    if single_batch_mode:
        x_train = x_train[:batch_size]
        y_train = y_train[:batch_size]
        x_val = x_val[:batch_size]
        y_val = y_val[:batch_size]

    for i, (coords, perm) in enumerate(permutations.items()):
        sub_model = join(model_name, "models", str(i))
        train_gen = get_gen(augmented=augmented)
        valid_gen = get_gen(augmented=False)
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
                      optimizer=Adam(learning_rate=LR),
                      metrics=['accuracy'])

        print(f"Model name: {sub_model}")
        try:
            model.fit(train_ds, epochs=epochs, verbose=1, validation_data=valid_ds,
                      steps_per_epoch=len(x_train) // batch_size,
                      validation_steps=len(x_val) // batch_size,
                      callbacks=[
                          EarlyStopping(monitor="val_loss", verbose=1, patience=20, restore_best_weights=True),
                          # ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, verbose=1, min_lr=1e-6),
                          PlotProgress(training_info_dir, plot_lr=False),
                          TensorBoard(log_dir=f'./{sub_model}/graph', histogram_freq=1, write_graph=True,
                                      write_images=True)
                      ])
        except KeyboardInterrupt:
            print("\nInterrupted!")
        print(f"Saving model {sub_model}")
        model.save(sub_model)
        save_training_info(model, model.history, info_path=training_info_dir, show=False)

    training_info_dir = os.path.join(model_name, info_dir)
    pathlib.Path(training_info_dir).mkdir(exist_ok=True, parents=True)
    aggr_model = get_aggregate_model(join(model_name, "models"), subinput_shape, n_classes, training_info_dir)
    aggr_model.compile(loss=categorical_crossentropy,
                       optimizer=Adam(learning_rate=LR),
                       metrics=['accuracy'])
    train_gen = get_gen(augmented=augmented)
    valid_gen = get_gen(augmented=False)
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
                           EarlyStopping(monitor="val_loss", verbose=1, patience=20, restore_best_weights=True),
                           # ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10, verbose=1, min_lr=1e-6),
                           PlotProgress(training_info_dir, plot_lr=False),
                           TensorBoard(log_dir=f'./{model_name}/graph', histogram_freq=1, write_graph=True,
                                       write_images=True)
                       ])
    except KeyboardInterrupt:
        print("\nInterrupted!")
    print(f"Saving model {model_name}")
    aggr_model.save(model_name)
    save_training_info(aggr_model, aggr_model.history, info_path=training_info_dir, show=False)
    with open(join(training_info_dir, "seed"), 'w') as f:
        print(train_seed, file=f)


if __name__ == '__main__':
    train_models_sequence()
