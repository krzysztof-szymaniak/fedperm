import string

from keras.datasets import cifar10, fashion_mnist, mnist, cifar100
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from keras.utils.np_utils import to_categorical


def convert_ds_to_numpy(ds):
    x = np.asarray(list(map(lambda v: v[0], tfds.as_numpy(ds))))
    y = np.asarray(list(map(lambda v: v[1], tfds.as_numpy(ds))))
    return x, y


def to_categorical_n_classes(x_train, y_train, x_test, y_test, subtract_pixel_mean=False):
    classes = np.unique(y_train)
    n_classes = len(classes)
    y_train = to_categorical(y_train, num_classes=n_classes)
    y_test = to_categorical(y_test, num_classes=n_classes)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
    return (x_train, y_train), (x_test, y_test), n_classes


def load_data(dataset):
    if dataset in ['mnist', 'fashion_mnist', 'cifar10']:
        ds = {
            'mnist': mnist,
            'fashion_mnist': fashion_mnist,
            'cifar10': cifar10,
        }[dataset]
        (x_train, y_train), (x_test, y_test) = ds.load_data()
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        return to_categorical_n_classes(x_train, y_train, x_test, y_test, subtract_pixel_mean=False)
    if dataset == 'emnist-letters':
        dataset = dataset.replace('-', '/')

        def transpose(x):
            return tf.image.flip_left_right(tf.image.rot90(x, -1))

        split = ['train[:85%]', 'train[85%:]']
        trainDataset, testDataset = tfds.load(name=dataset, split=split, as_supervised=True)
        trainDataset = trainDataset.map(lambda x, y: (transpose(x), y - 1))
        testDataset = testDataset.map(lambda x, y: (transpose(x), y - 1))
        x_train, y_train = convert_ds_to_numpy(trainDataset)
        x_test, y_test = convert_ds_to_numpy(testDataset)
        return to_categorical_n_classes(x_train, y_train, x_test, y_test)

    elif dataset == 'eurosat':
        split = ['train[:80%]', 'train[80%:]']
        trainDataset, testDataset = tfds.load(name=dataset, split=split, as_supervised=True)
        x_train, y_train = convert_ds_to_numpy(trainDataset)
        x_test, y_test = convert_ds_to_numpy(testDataset)
        return to_categorical_n_classes(x_train, y_train, x_test, y_test, subtract_pixel_mean=False)
    elif dataset == 'cats_vs_dogs':
        HEIGHT = 128
        WIDTH = 128

        def preprocess(img, label):
            return tf.cast(tf.image.resize(img, [HEIGHT, WIDTH]), tf.uint8), tf.cast(label, tf.float32)

        split = ['train[:80%]', 'train[80%:]']
        trainDataset, testDataset = tfds.load(name=dataset, split=split, as_supervised=True)
        testDataset = testDataset.map(preprocess)
        trainDataset = trainDataset.map(preprocess)
        x_train, y_train = convert_ds_to_numpy(trainDataset)
        x_test, y_test = convert_ds_to_numpy(testDataset)
        classes = np.unique(y_train)
        n_classes = len(classes)
        return (x_train, y_train), (x_test, y_test), n_classes

    elif dataset == 'kmnist':
        trainDataset = tfds.load(name=dataset, split='train', as_supervised=True)
        testDataset = tfds.load(name=dataset, split='test', as_supervised=True)
        x_train, y_train = convert_ds_to_numpy(trainDataset)
        x_test, y_test = convert_ds_to_numpy(testDataset)
        return to_categorical_n_classes(x_train, y_train, x_test, y_test)
    else:
        raise Exception("No dataset with name " + dataset)


def get_classes_names_for_dataset(ds_name):
    classes = None
    if ds_name == 'mnist':
        classes = [str(i) for i in range(10)]
    elif ds_name == 'fashion_mnist':
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']
    elif ds_name == 'cifar10':
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif ds_name == 'cats_vs_dogs':
        classes = ['cat', 'dog']
    elif ds_name == 'emnist-letters':
        classes = list(string.ascii_lowercase)
    return classes


if __name__ == '__main__':
    ...
