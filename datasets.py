import string

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.datasets import cifar10, fashion_mnist, mnist, cifar100
from keras.utils.np_utils import to_categorical


def convert_tfds_to_numpy(ds):
    x = np.asarray(list(map(lambda v: v[0], tfds.as_numpy(ds))))
    y = np.asarray(list(map(lambda v: v[1], tfds.as_numpy(ds))))
    return x, y


def to_categorical_n_classes(x_train, y_train, x_test, y_test, upscale=False):
    classes = np.unique(y_train)
    n_classes = len(classes)
    y_train = to_categorical(y_train, num_classes=n_classes)
    y_test = to_categorical(y_test, num_classes=n_classes)

    if upscale:
        shape = (64, 64)
        x_train = np.array([reshape(img, shape) for img in x_train])
        x_test = np.array([reshape(img, shape) for img in x_test])

    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

    print(f"{len(x_train)=}")
    print(f"{len(x_test)=}")
    return (x_train, y_train), (x_test, y_test), n_classes


def reshape(img, shape):
    return tf.cast(tf.image.resize(img, shape), tf.uint8)


def load_data(dataset):
    upscale = dataset in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'emnist-letters']
    if dataset in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:
        ds = {
            'mnist': mnist,
            'fashion_mnist': fashion_mnist,
            'cifar10': cifar10,
            'cifar100': cifar100,
        }[dataset]
        (x_train, y_train), (x_test, y_test) = ds.load_data()
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        return to_categorical_n_classes(x_train, y_train, x_test, y_test, upscale=upscale)
    if dataset == 'emnist-letters':
        dataset = dataset.replace('-', '/')

        def transpose(x):
            return tf.image.flip_left_right(tf.image.rot90(x, -1))

        split = ['train[:85%]', 'train[85%:]']
        trainDataset, testDataset = tfds.load(name=dataset, split=split, as_supervised=True)
        trainDataset = trainDataset.map(lambda x, y: (transpose(x), y - 1))
        testDataset = testDataset.map(lambda x, y: (transpose(x), y - 1))
        x_train, y_train = convert_tfds_to_numpy(trainDataset)
        x_test, y_test = convert_tfds_to_numpy(testDataset)
        return to_categorical_n_classes(x_train, y_train, x_test, y_test, upscale=upscale)

    elif dataset == 'eurosat':
        split = ['train[:80%]', 'train[80%:]']
        trainDataset, testDataset = tfds.load(name=dataset, split=split, as_supervised=True)
        x_train, y_train = convert_tfds_to_numpy(trainDataset)
        x_test, y_test = convert_tfds_to_numpy(testDataset)
        return to_categorical_n_classes(x_train, y_train, x_test, y_test)
    elif dataset == 'cats_vs_dogs':
        HEIGHT = 128
        WIDTH = 128

        def preprocess(img, label):
            return reshape(img, (HEIGHT, WIDTH)), tf.cast(label, tf.float32)

        split = ['train[:80%]', 'train[80%:]']
        trainDataset, testDataset = tfds.load(name=dataset, split=split, as_supervised=True)
        testDataset = testDataset.map(preprocess)
        trainDataset = trainDataset.map(preprocess)
        x_train, y_train = convert_tfds_to_numpy(trainDataset)
        x_test, y_test = convert_tfds_to_numpy(testDataset)
        classes = np.unique(y_train)
        n_classes = len(classes)
        return (x_train, y_train), (x_test, y_test), n_classes

    elif dataset == 'kmnist':
        trainDataset = tfds.load(name=dataset, split='train', as_supervised=True)
        testDataset = tfds.load(name=dataset, split='test', as_supervised=True)
        x_train, y_train = convert_tfds_to_numpy(trainDataset)
        x_test, y_test = convert_tfds_to_numpy(testDataset)
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
    elif ds_name == 'cifar100':
        classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                   'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                   'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 'cup', 'dinosaur',
                   'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo',
                   'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree',
                   'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree',
                   'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit',
                   'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper',
                   'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
                   'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',
                   'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    return classes

#
# def main():
#     datasets = [
#         'cifar10',
#         'cifar100',
#         'fashion_mnist',
#         'emnist-letters',
#         # 'cats_vs_dogs',
#         'mnist',
#         # 'eurosat',
#     ]
#     for dataset in datasets:
#         (x_train, y_train), (x_test, y_test), n_classes = load_data(dataset)
#         classes = get_classes_names_for_dataset(dataset)
#         print(f'{len(classes)=}')
#
#         rand_ids = np.random.choice(range(len(x_train)), 3 * 5)
#         # fig, axs = plt.subplots(3, 5, figsize=[12, 12])
#         #
#         # for i, ax in enumerate(axs.flatten()):
#         #     ind = rand_ids[i]
#         #     if 'mnist' in dataset:
#         #         ax.imshow(x_train[ind], cmap='gray')
#         #     else:
#         #         ax.imshow(x_train[ind])
#         #     ax.set_title(classes[np.argmax(y_train[ind])], size=16)
#         #     ax.axis('off')
#         # fig.savefig(fr'C:\Users\Krzysztof\Desktop\teczka\Praca Magisterska\praca\źródła\zasoby\datasets\{dataset}.pdf')
#         # plt.show()
#
#
# if __name__ == '__main__':
#     main()
