from os.path import join

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from pretty_confusion_matrix import pp_matrix_from_data
from sklearn.metrics import classification_report

from config import dataset, input_shape, grid_shape, show, model_name, info_dir, classes
from model import get_seed
from preprocessing import load_data, PermutationGenerator


def test():
    seed = get_seed(model_name)
    _, (x_test_o, y_test_o), _ = load_data(dataset, input_shape)
    test_ds_perm = PermutationGenerator(x_test_o, y_test_o, ImageDataGenerator(rescale=1. / 255), input_shape,
                                        grid_shape,
                                        seed=seed,
                                        batch_size=len(x_test_o), shuffle_dataset=False, debug=show)
    test_ds_o = PermutationGenerator(x_test_o, y_test_o, ImageDataGenerator(rescale=1. / 255), input_shape, grid_shape,
                                     batch_size=len(x_test_o), shuffle_dataset=False, debug=show)

    batch_test_o, y_test_o = test_ds_o.next()
    batch_test_p, y_test_p = test_ds_perm.next()
    model = load_model(model_name)

    prediction = model.predict(batch_test_p)
    y_test = np.argmax(y_test_p, axis=1)
    y_pred = np.argmax(prediction, axis=1)

    # show_images(x_test_o, batch_test_o, batch_test_p, y_test_o, y_test_p)
    pp_matrix_from_data(y_test, y_pred, columns=classes)
    cr = classification_report(y_test, y_pred)
    with open(join(model_name, info_dir, "confusion_matrix"), 'w') as f:
        print(cr, file=f)


if __name__ == '__main__':
    test()


def show_images(x_o, x_batch_o, x_batch_perm, y_o, y_pred, rows=4):
    c = x_batch_o.shape[1]
    fig = plt.figure(figsize=(15, 15))
    subfigs = fig.subfigures(rows, 2)
    for i in range(rows):
        axl = subfigs[i, 0].subplots(1, 1)
        axs = subfigs[i, 1].subplots(2, c)
        axl.imshow(x_o[i])
        for j in range(c):
            axs[0, j].imshow(x_batch_o[i][j])
            axs[1, j].imshow(x_batch_perm[i][j])
    plt.show()
