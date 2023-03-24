import pathlib
from os.path import join
from pprint import pprint

import numpy as np
from keras.callbacks import ReduceLROnPlateau
from matplotlib import pyplot as plt
from pretty_confusion_matrix import pp_matrix_from_data
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.backend import categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Input

from model import PlotProgress, ModelFactory
from preprocessing import PermutationGenerator


class ModelTraining:
    info_dir = "training_info"
    debug_dir = 'debug'
    arch_dir = 'architecture'

    batch_size = 64
    epochs = 1000
    EARLY_STOPPING_PATIENCE = 20
    LR = 1e-3

    architecture_id = 0

    augmented = True
    debug = True

    def __init__(self, model_name, grid_shape, n_classes, architecture_id, permutations, model_type, classes_names):
        self.grid_shape = grid_shape
        self.architecture_id = architecture_id
        self.model_type = model_type
        self.classes_names = classes_names
        self.model_name = model_name

        self.n_classes = n_classes
        self.permutations = permutations

        self.model = None
        self.history = None
        self.training_info_dir = None
        self.debug_info_dir = None
        self.arch_info_dir = None
        self.model_factory = None
        self.subinput_shape = None

        print(f"Model name: {self.model_name}")

    def callbacks(self):
        return [
            EarlyStopping(monitor="val_loss", verbose=1, patience=self.EARLY_STOPPING_PATIENCE,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10, verbose=1, min_lr=1e-6),
            PlotProgress(self.training_info_dir, plot_lr=True),
            TensorBoard(log_dir=f'./{self.model_name}/graph', histogram_freq=1, write_graph=True,
                        write_images=True)
        ]

    def compile_opts(self):
        opts = {
            "loss": categorical_crossentropy,
            "optimizer": Adam(learning_rate=self.LR),
            "metrics": ['accuracy'],
        }
        return opts

    def get_perm_images_generator(self, x, y, augmented, debug=False, ind=None):
        aug = ImageDataGenerator(
            rescale=1 / 255,
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        ) if augmented else ImageDataGenerator(rescale=1. / 255)
        gen = PermutationGenerator(x, y, aug, self.subinput_shape, permutations=self.permutations,
                                   batch_size=self.batch_size, debug_path=self.debug_info_dir)
        if debug:
            gen.run_debug(ind)
        return gen

    def set_up(self, input_shape):
        self.training_info_dir = join(self.model_name, self.info_dir)
        self.debug_info_dir = join(self.model_name, self.debug_dir)
        self.arch_info_dir = join(self.model_name, self.arch_dir)
        pathlib.Path(self.training_info_dir).mkdir(exist_ok=True, parents=True)
        pathlib.Path(self.debug_info_dir).mkdir(exist_ok=True, parents=True)
        pathlib.Path(self.arch_info_dir).mkdir(exist_ok=True, parents=True)
        self.subinput_shape = (
            input_shape[0] // self.grid_shape[0], input_shape[1] // self.grid_shape[1], input_shape[2])
        self.model_factory = ModelFactory(self.arch_info_dir, self.n_classes, len(self.permutations),
                                          self.subinput_shape,
                                          self.architecture_id)

    def fit(self, x_train, y_train, x_val, y_val):
        self.set_up(x_train.shape[1:])
        print(self.subinput_shape)
        with open(join(self.training_info_dir, "permutations.txt"), 'w') as f:
            f.write(str(self.permutations).replace('array', 'np.array'))

        train_ds = self.get_perm_images_generator(x_train, y_train, augmented=self.augmented, debug=True,
                                                  ind=np.random.randint(len(x_train)))
        valid_ds = self.get_perm_images_generator(x_val, y_val, augmented=False, debug=True,
                                                  ind=np.random.randint(len(x_val)))

        self.model = self.model_factory.get_model(self.model_type, self.n_classes, name='federated-parallel')
        self.model.compile(**self.compile_opts())

        try:
            self.model.fit(train_ds, epochs=self.epochs, verbose=1, validation_data=valid_ds,
                           steps_per_epoch=len(x_train) // self.batch_size,
                           validation_steps=len(x_val) // self.batch_size,
                           callbacks=self.callbacks())
        except KeyboardInterrupt:
            print("\nInterrupted!")
        print(f"Saving model {self.model_name}")
        self.model.save(self.model_name)
        self.save_training_info(show=False)
        return self.model

    def predict(self, x_test, y_test):
        self.model = load_model(self.model_name)
        dic = ''
        with open(join(self.training_info_dir, "permutations.txt"), 'r') as f:
            for i in f.readlines():
                dic += i
        self.permutations = eval(dic)

        testing_path = join(self.model_name, "testing")
        pathlib.Path(testing_path).mkdir(exist_ok=True, parents=True)

        test_gen = self.get_perm_images_generator(x_test, y_test, augmented=False, debug=False)
        prediction = self.model.predict(test_gen, steps=len(x_test) // self.batch_size)
        actual_classes = np.argmax(y_test, axis=1)
        predicted_classes = np.argmax(prediction, axis=1)

        pp_matrix_from_data(actual_classes, predicted_classes, columns=self.classes_names)
        cr = classification_report(actual_classes, predicted_classes)
        with open(join(testing_path, "classification_scores"), 'w') as f:
            print(cr, file=f)
        return accuracy_score(actual_classes, predicted_classes)

    def save_training_info(self, show=False):
        history = self.model.history
        plt.clf()
        metrics = set([m.replace('val_', '') for m in history.history.keys()])
        for met in metrics:
            plt.plot(history.history[met])
            if f"val_{met}" in history.history:
                plt.plot(history.history[f"val_{met}"])
            plt.title(f"{met}")
            plt.ylabel(met)
            plt.xlabel('epoch')
            plt.grid()
            if f"val_{met}" in history.history:
                plt.legend(['train', 'validate'], loc='right')
            else:
                plt.legend(['train'], loc='right')
            plt.savefig(join(self.training_info_dir, met))
            if show:
                plt.show()
            plt.clf()
        with open(join(self.training_info_dir, "model_config"), 'w') as f:
            pprint(self.model.get_config(), f)


class TrainingSequential(ModelTraining):
    def __init__(self, model_name, subinput_shape, n_classes, architecture, permutations, classes_names):
        super().__init__(model_name, subinput_shape, n_classes, architecture, permutations, 'sequential', classes_names)
        self.sub_models = []

    def fit(self, x_train, y_train, x_val, y_val):
        self.set_up(x_train.shape[1:])
        for i, (coords, perm) in enumerate(self.permutations.items()):
            sub_model = join(self.model_name, "models", str(i))
            self.sub_models.append(sub_model)
            m = ModelTraining(sub_model, self.subinput_shape, self.n_classes, self.architecture_id, {coords: perm},
                              'single',
                              self.classes_names)
            m.fit(x_train, y_train, x_val, y_val)

        train_ds = self.get_perm_images_generator(x_train, y_train, augmented=self.augmented, debug=self.debug)
        valid_ds = self.get_perm_images_generator(x_val, y_val, augmented=False, debug=False)
        models = [load_model(m_p) for m_p in self.sub_models]
        _ins = [Input(shape=self.subinput_shape) for _ in models]
        for m in models:
            m.trainable = False
        self.model = self.model_factory.get_aggregating_model(_ins, models, name='sequential')
        self.model.compile(**self.compile_opts())

        try:
            self.model.fit(train_ds, epochs=self.epochs, verbose=1, validation_data=valid_ds,
                           steps_per_epoch=len(x_train) // self.batch_size,
                           validation_steps=len(x_val) // self.batch_size,
                           callbacks=self.callbacks())
        except KeyboardInterrupt:
            print("\nInterrupted!")
        print(f"Saving model {self.model_name}")
        self.model.save(self.model_name)
        self.save_training_info(show=False)
        return self.model

#
# if __name__ == '__main__':
#     seed = 42
#     sequentially = False
#     grid_shape = (2, 2)
#     shape = 'overlap_cross'
#
#     # DATASET = "mnist"
#     # DATASET = "fashion"
#     DATASET = "cifar"
#     v = 10
#
#     input_shape = (32, 32, 3) if DATASET == 'cifar' else (28, 28, 1)
#     subinput_shape = (input_shape[0] // grid_shape[0], input_shape[1] // grid_shape[1], input_shape[2])
#
#     base_name = f"models/{DATASET}-{grid_shape[0]}x{grid_shape[1]}"
#     base_name += f"{'-perm' if seed is not None else '-id'}"
#     base_name += f"{'-seq' if sequentially else '-par'}"
#     base_name += f"-{shape}"
#     version = 1
#     model_name = f"{base_name}-v{version}"
#     while exists(model_name):
#         version += 1
#         model_name = f"{base_name}-v{version}"
#
#     fashion_names = """
#     T-shirt/top
#     Trouser
#     Pullover
#     Dress
#     Coat
#     Sandal
#     Shirt
#     Sneaker
#     Bag
#     Ankle boot
#     """
#
#     cifar_names = """airplane
#     automobile
#     bird
#     cat
#     deer
#     dog
#     frog
#     horse
#     ship
#     truck"""
#     if DATASET == 'mnist':
#         dataset = mnist
#         classes = [i for i in range(10)]
#     elif DATASET == 'fashion':
#         dataset = fashion_mnist
#         classes = [c for c in fashion_names.split("\n") if c]
#     else:
#         dataset = cifar10
#         classes = [c for c in cifar_names.split("\n") if c]
#     (x_train, y_train), (x_test, y_test), n_classes = load_data(dataset, input_shape)
#     split_seed = 42
#     x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True,
#                                                       random_state=split_seed)
#     permutations = generate_permutations(seed, grid_shape, subinput_shape, shape)
#     if sequentially:
#
#         model = TrainingSequential(model_name, subinput_shape, n_classes, 0,
#                                    permutations,
#                                    classes)
#         model.fit(x_train, y_train, x_val, y_val).predict(x_test, y_test)
#
#     else:
#         model = ModelTraining(model_name, subinput_shape, n_classes, 0, permutations, 'parallel', classes)
#         model.fit(x_train, y_train, x_val, y_val).predict(x_test, y_test)
