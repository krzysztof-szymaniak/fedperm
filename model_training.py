import pathlib
import sys
from os.path import join, exists
from pprint import pprint

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from pretty_confusion_matrix import pp_matrix_from_data
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.backend import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Input
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import ModelFactory
from preprocessing import PermutationGenerator


def scheduler(epoch, lr):
    if epoch < 20:
        return 0.1
    if epoch < 40:
        return 0.01
    if epoch < 60:
        return 0.001
    else:
        return max(lr * tf.math.exp(-0.01), 1e-6)


class ModelTraining:
    info_dir = "training_info"
    debug_dir = 'debug'
    arch_dir = 'architecture'

    batch_size = 64
    epochs = 1000
    EARLY_STOPPING_PATIENCE = 20
    lr_patience = 8
    lr = 1e-3

    augmented = True
    debug = True

    def __init__(self, model_name, subinput_shape, n_classes, architecture_id, permutations, model_type, classes_names):
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
        self.subinput_shape = subinput_shape

        print(f"Model name: {self.model_name}")

    def callbacks(self):
        return [
            EarlyStopping(monitor="val_loss", verbose=1, patience=self.EARLY_STOPPING_PATIENCE,
                          restore_best_weights=True),
            # LearningRateScheduler(scheduler),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=self.lr_patience, verbose=1, min_lr=1e-7),
            PlotProgress(self.training_info_dir, plot_lr=True),
            TensorBoard(log_dir=f'./{self.model_name}/graph', histogram_freq=1, write_graph=True,
                        write_images=True)
        ]

    def compile_opts(self):
        opts = {
            "loss": categorical_crossentropy,
            # 'loss': binary_crossentropy,
            "optimizer": Adam(learning_rate=self.lr),
            # "optimizer": SGD(learning_rate=self.lr, momentum=0.9, nesterov=True),
            # "metrics": ['accuracy', Precision(), Recall()],
            "metrics": ['accuracy'],
        }
        return opts

    def get_perm_images_generator(self, x, y, augmented, debug=False, ind=0, batch_size=None, shuffle=True):
        aug = ImageDataGenerator(
            rescale=1 / 255,
            rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.2,  # Randomly zoom image
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            shear_range=20,
            horizontal_flip=True,
        ) if augmented else ImageDataGenerator(rescale=1. / 255)
        gen = PermutationGenerator(x, y, aug, self.subinput_shape, permutations=self.permutations,
                                   batch_size=batch_size, debug_path=self.debug_info_dir, shuffle_dataset=shuffle)
        if debug:
            gen.run_debug(ind)
        return gen

    def set_up(self):
        self.training_info_dir = join(self.model_name, self.info_dir)
        self.debug_info_dir = join(self.model_name, self.debug_dir)
        self.arch_info_dir = join(self.model_name, self.arch_dir)
        pathlib.Path(self.training_info_dir).mkdir(exist_ok=True, parents=True)
        pathlib.Path(self.debug_info_dir).mkdir(exist_ok=True, parents=True)
        pathlib.Path(self.arch_info_dir).mkdir(exist_ok=True, parents=True)
        self.model_factory = ModelFactory(self.arch_info_dir, self.n_classes, len(self.permutations),
                                          self.subinput_shape,
                                          self.architecture_id)
        with open(join(self.training_info_dir, "permutations.txt"), 'w') as f:
            f.write(str(self.permutations).replace('array', 'np.array'))

    def fit(self, x_train, y_train, x_val, y_val):
        if exists(join(self.model_name, 'saved_model.pb')):
            print("Model already trained, skipping")
            return
        self.set_up()
        train_ds = self.get_perm_images_generator(x_train, y_train, batch_size=self.batch_size,
                                                  augmented=self.augmented, debug=True,
                                                  ind=np.random.randint(len(x_train)))
        valid_ds = self.get_perm_images_generator(x_val, y_val, batch_size=self.batch_size,
                                                  augmented=False, debug=True,
                                                  ind=np.random.randint(len(x_val)))

        self.model = self.model_factory.get_model(self.model_type, self.n_classes)
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

        test_gen = self.get_perm_images_generator(x_test, y_test, augmented=False, debug=False, batch_size=len(x_test),
                                                  shuffle=False)
        x_data, _ = test_gen.next()
        prediction = self.model.predict(x_data)

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
        self.set_up()
        for i, (coords, perm) in enumerate(self.permutations.items()):
            sub_model = join(self.model_name, "subs", str(i))
            self.sub_models.append(sub_model)
            m = ModelTraining(sub_model, self.subinput_shape, self.n_classes, self.architecture_id, {coords: perm},
                              'single',
                              self.classes_names)
            m.fit(x_train, y_train, x_val, y_val)

        train_ds = self.get_perm_images_generator(x_train, y_train, batch_size=self.batch_size,
                                                  augmented=self.augmented, debug=self.debug,
                                                  ind=np.random.randint(len(x_train)))
        valid_ds = self.get_perm_images_generator(x_val, y_val, batch_size=self.batch_size,
                                                  augmented=False, debug=True,
                                                  ind=np.random.randint(len(x_val)))
        for m_p in self.sub_models:
            model = load_model(m_p)

        models = [load_model(m_p) for m_p in self.sub_models]
        for m in models:
            m.trainable = False
        _ins = [Input(shape=self.subinput_shape) for _ in models]
        outs = [m(_in) for m, _in in zip(models, _ins)]
        self.model = self.model_factory.get_aggregating_model(_ins, outs, name='sequential')
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


class PlotProgress(Callback):
    max_acc = 0
    max_val_acc = 0
    min_loss = sys.maxsize
    min_val_loss = sys.maxsize

    acc_ep = 0
    val_acc_ep = 0
    loss_ep = 0
    val_loss_ep = 0

    def __init__(self, i_dir, plot_lr=True, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.axs = None
        self.f = None
        self.metrics = None
        self.i_dir = join(i_dir, 'progress')
        pathlib.Path(self.i_dir).mkdir(exist_ok=True, parents=True)
        self.first_epoch = True
        self.plot_lr = plot_lr

    def on_train_begin(self, logs=None):
        plt.ion()
        self.metrics = {}

    def on_train_end(self, logs=None):
        self.f.savefig(f"{self.i_dir}/metrics")
        plt.close(self.f)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        n_met = len(logs)
        if self.f is None:
            # if n_met > 4:
            #     self.f, self.axs = plt.subplots(2, 3, figsize=(40, 20))
            # else:
            self.f, self.axs = plt.subplots(1, 3, figsize=(15, 5))
        acc = max(self.max_acc, round(logs.get("accuracy"), 4))
        val_acc = max(self.max_val_acc, round(logs.get("val_accuracy"), 4))
        loss = min(self.min_loss, round(logs.get("loss"), 4))
        val_loss = min(self.min_val_loss, round(logs.get("val_loss"), 4))

        if acc == self.max_acc:
            self.acc_ep += 1
        else:
            self.acc_ep = 0
        if val_acc == self.max_val_acc:
            self.val_acc_ep += 1
        else:
            self.val_acc_ep = 0

        if loss == self.min_loss:
            self.loss_ep += 1
        else:
            self.loss_ep = 0

        if val_loss == self.min_val_loss:
            self.val_loss_ep += 1
        else:
            self.val_loss_ep = 0

        self.max_acc = acc
        self.max_val_acc = val_acc
        self.min_loss = loss
        self.min_val_loss = val_loss
        acc_msg = f"{'Max accuracy': <16}: {self.max_acc:.4f}, not impr. in {self.acc_ep} epochs\n{'Max val_accuracy': <16}: {self.max_val_acc:.4f}, not impr. in {self.val_acc_ep} epochs"
        loss_msg = f"{'Min loss': <16}: {self.min_loss:.4f}, not impr. in {self.loss_ep} epochs\n{'Min val_loss': <16}: {self.min_val_loss:.4f}, not impr. in {self.val_loss_ep} epochs"
        metrics = [x for x in logs if 'val' not in x]
        for ax, metric in zip(self.axs.flatten(), metrics):
            ax.plot(range(1, epoch + 2), self.metrics[metric], color='blue', label=metric)
            if 'val_' + metric in logs:
                ax.plot(range(1, epoch + 2), self.metrics['val_' + metric], label='val_' + metric,
                        color='orange', )
                if metric == 'accuracy':
                    ax.set_title(acc_msg)
                elif metric == 'loss':
                    ax.set_title(loss_msg)
            if self.first_epoch:
                ax.legend()
                ax.grid()
        self.first_epoch = False
        plt.tight_layout()
        self.f.canvas.draw()
        self.f.canvas.flush_events()
        self.f.savefig(f"{self.i_dir}/progress_{epoch + 1}.png")
        # if self.verbose:
        #     print(acc_msg)
        #     print(loss_msg)
