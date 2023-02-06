from os.path import join

import numpy as np
from pretty_confusion_matrix import pp_matrix_from_data
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib

from config import dataset, input_shape, grid_shape, show, model_name, info_dir, classes, subinput_shape
from permutations import generate_permutations
from preprocessing import load_data, PermutationGenerator, get_seed

print(device_lib.list_local_devices())


def test():
    seed = get_seed(model_name)
    permutations = generate_permutations(seed, grid_shape,
                                         subinput_shape, overlap=True, full_overlap=True)
    _, (x_test_o, y_test_o), _ = load_data(dataset, input_shape)
    test_ds_perm = PermutationGenerator(x_test_o, y_test_o, ImageDataGenerator(rescale=1. / 255), subinput_shape,
                                        permutations=permutations,
                                        batch_size=len(x_test_o), shuffle_dataset=False, debug=show)
    test_ds_o = PermutationGenerator(x_test_o, y_test_o, ImageDataGenerator(rescale=1. / 255), subinput_shape,
                                     permutations=permutations,
                                     batch_size=len(x_test_o), shuffle_dataset=False, debug=show)

    _ = test_ds_o.next()
    batch_test_p, y_test_p = test_ds_perm.next()
    model = load_model(model_name)

    prediction = model.predict(batch_test_p)
    y_test = np.argmax(y_test_p, axis=1)
    y_pred = np.argmax(prediction, axis=1)

    pp_matrix_from_data(y_test, y_pred, columns=classes)
    cr = classification_report(y_test, y_pred)
    with open(join(model_name, info_dir, "classification_scores"), 'w') as f:
        print(cr, file=f)


if __name__ == '__main__':
    test()
