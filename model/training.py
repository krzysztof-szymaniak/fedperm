import pathlib
import pickle
from os.path import join, exists

import numpy as np
from matplotlib import pyplot as plt
from pretty_confusion_matrix import pp_matrix_from_data
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from keras.utils.generic_utils import CustomMaskWarning

from enums import Aggregation
from model.architectures.build_model import get_model, aggregate
from model.generators import get_train_valid_gens, get_generator
from model.train_configs import compile_opts, MAX_EPOCHS, BATCH_SIZE, callbacks
from model.utils import save_training_info, set_up_dirs
from model.visualisation import plot_model
from permutation.permutations import generate_permutations

import warnings
from sklearn.exceptions import UndefinedMetricWarning
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress logging
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=CustomMaskWarning)

skip_all_training = True


def train_model(x_train, y_train, x_val, y_val, model_path, permutations, sub_input_shape, n_classes, ds_name, arch,
                mode, aggr_scheme=None, m_id=None):
    training_info_dir, examples_info_dir, arch_info_dir, checkpoints_dir = set_up_dirs(model_path)
    train_dirs = (model_path, checkpoints_dir, training_info_dir)
    generators = get_train_valid_gens(x_train, y_train, x_val, y_val,
                                      permutations=permutations,
                                      sub_input_shape=sub_input_shape,
                                      apply_flip=ds_name != 'mnist' and ds_name != 'emnist-letters',
                                      examples_path=examples_info_dir)
    save_permutation(model_path, permutations)
    if mode == 'single':
        model = get_model(arch, arch_info_dir, sub_input_shape, n_classes, m_id=m_id, )
        model.compile(**compile_opts(n_classes))
        name = f'{ds_name}-{arch.name.lower()}-{mode}-{m_id}'
        fit_model(model, generators, train_dirs, name)
        return model

    models = []
    if mode == 'parallel':
        for i, _ in enumerate(permutations):
            sub_model_path = join(model_path, "subs", str(i))
            model = get_model(sub_model_path, arch_info_dir, sub_input_shape, n_classes, m_id=i, )
            model = strip_last_layer(model)
            models.append(model)

    if mode == 'composite':
        sub_model_paths = []
        for i, (coords, perm) in enumerate(permutations.items()):
            sub_perm = {coords: perm}
            sub_model_path = join(model_path, "subs", str(i))
            sub_model_paths.append(sub_model_path)
            if not skip_training(sub_model_path):
                train_model(x_train, y_train, x_val, y_val, sub_model_path, sub_perm, sub_input_shape, n_classes,
                            ds_name, arch, mode='single',
                            m_id=i, )

        for sub_path in sub_model_paths:
            model = load_model(sub_path)
            if aggr_scheme == Aggregation.STRIP_CONCAT:
                model = strip_last_layer(model)
            model.trainable = False
            models.append(model)

    inputs = [Input(shape=sub_input_shape) for _ in models]
    models_outputs = [model(inpt) for model, inpt in zip(models, inputs)]
    outputs = aggregate(models_outputs, n_classes, aggr_scheme)
    aggregated_model = Model(inputs=inputs, outputs=outputs, name=mode)
    aggregated_model.summary()
    plot_model(arch_info_dir, aggregated_model, mode)
    aggregated_model.compile(**compile_opts(n_classes))
    name = f'{ds_name}-{arch.name.lower()}-{mode}'
    fit_model(aggregated_model, generators, train_dirs, name)
    return aggregated_model


def fit_model(model, data, dirs, name):
    print("Training ", name)
    model_path, checkpoints_dir, training_info_dir = dirs
    train_ds, valid_ds = data
    try:
        if not skip_all_training:
            model.fit(train_ds, epochs=MAX_EPOCHS, verbose=1, validation_data=valid_ds,
                      steps_per_epoch=train_ds.n // BATCH_SIZE,
                      validation_steps=valid_ds.n // BATCH_SIZE,
                      callbacks=callbacks(checkpoints_dir, training_info_dir, name))
    except KeyboardInterrupt:
        print("\nInterrupted!")
    weights_path = join(checkpoints_dir, 'weights.h5')
    if exists(weights_path):
        model.load_weights(weights_path)
    print(f"Saving model {model_path}...")
    model.save(model_path)
    save_training_info(model, training_info_dir)
    print("Done ")
    return model


def predict(model_path, x_test, y_test, sub_input_shape, classes_names, mode=None, test_dir_name=None,
            invalid_test=None):
    print("Predicting ", model_path)
    if test_dir_name is None:
        test_dir_name = 'test'
    permutations = load_permutation(model_path)
    if type(invalid_test) == dict:
        permutations = generate_permutations(
            invalid_test['seed'],
            invalid_test['grid'],
            sub_input_shape,
            invalid_test['overlap'],
            invalid_test['scheme']
        )
    if mode == 'composite':
        for i, _ in enumerate(permutations):
            sub_model_path = join(model_path, "subs", str(i))
            predict(sub_model_path, x_test, y_test, sub_input_shape, classes_names, mode='single',
                    test_dir_name=test_dir_name, invalid_test=invalid_test)

    model = load_model(model_path)
    testing_path = join(model_path, test_dir_name)
    pathlib.Path(testing_path).mkdir(exist_ok=True, parents=True)
    test_gen = get_generator(x_test, y_test,
                             batch_size=len(x_test),
                             permutations=permutations,
                             sub_input_shape=sub_input_shape)
    x_test, y_test = test_gen.next()
    prediction = model.predict(x_test)

    actual_classes = np.argmax(y_test, axis=1)
    predicted_classes = np.argmax(prediction, axis=1)
    plt.ion()
    pp_matrix_from_data(actual_classes, predicted_classes, columns=classes_names,
                        figsize=[20, 20] if len(classes_names) > 10 else [8, 8])
    plt.savefig(join(testing_path, 'conf_matrix.png'))
    plt.close('all')
    plt.ioff()
    cr = classification_report(actual_classes, predicted_classes, target_names=classes_names)
    with open(join(testing_path, "report.txt"), 'w') as f:
        print(cr, file=f)
    return accuracy_score(actual_classes, predicted_classes)


def save_permutation(folder, perm):
    with open(join(folder, "permutations"), 'wb') as f:
        pickle.dump(perm, f)


def load_permutation(folder):
    with open(join(folder, "permutations"), 'rb') as f:
        return pickle.load(f)


def skip_training(model_path):
    if exists(join(model_path, 'saved_model.pb')):
        print("Model already trained, skipping")
        return True
    return False


def strip_last_layer(model):
    return Model(inputs=model.input, outputs=model.layers[-2].output, name=model.name)
