import pathlib
import shutil
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
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler, ModelCheckpoint, \
    ReduceLROnPlateau
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier

from enums import Aggregation
from model import ModelFactory
from permutations import PermutationGenerator

info_dir_name = "train"
debug_dir_name = 'data_examples'
arch_dir_name = 'architecture'
testing_dir_name = 'test'

batch_size = 32
epochs = 1000
es_patience = 10
lr_patience = 5
lr = 3e-4


# lr = 1e-2


def scheduler(epoch, rate):
    if epoch < 10:
        return rate
    else:
        return rate * tf.math.exp(-0.05)


def compile_opts():
    opts = {
        "loss": categorical_crossentropy,
        # 'loss': binary_crossentropy,
        "optimizer": Adam(learning_rate=lr),
        # "optimizer": SGD(learning_rate=lr, momentum=0.9, nesterov=True),
        # "metrics": ['accuracy', Precision(), Recall()],
        "metrics": ['accuracy'],
    }
    return opts


def apply_flip(model_name):
    return True if '-mnist-' not in model_name and 'letters' not in model_name else False


def get_aug(model_name):
    if 'cifar' in model_name:
        return {
            # "rescale": 1 / 255,
            "width_shift_range": 0.05,  # randomly shift images horizontally (fraction of total width)
            "height_shift_range": 0.05,  # randomly shift images vertically (fraction of total height)
            # "rotation_range": 5,
            # "sheer_range": 3,
            "horizontal_flip": apply_flip(model_name)
        }
    return {
        # "rescale": 1 / 255,
        "width_shift_range": 0.02,  # randomly shift images horizontally (fraction of total width)
        "height_shift_range": 0.02,  # randomly shift images vertically (fraction of total height)
        # "rotation_range": 5,
        "horizontal_flip": apply_flip(model_name)
    }


class ModelTraining:
    augmented = True
    debug = True

    def __init__(self, model_name, subinput_shape, n_classes, permutations, model_type, classes_names, aggr,
                 m_id=None):
        self.aggr = aggr
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
        self.checkpoints_dir = None
        self.model_factory = None
        self.subinput_shape = subinput_shape
        self.m_id = m_id

        print(f"Model name: {self.model_name}")

    def callbacks(self):
        return [
            ModelCheckpoint(filepath=join(self.checkpoints_dir, 'weights.h5'),
                            save_freq='epoch', verbose=0, monitor='val_accuracy',
                            save_weights_only=True, save_best_only=True),
            EarlyStopping(monitor="val_loss", verbose=1, patience=es_patience, restore_best_weights=True),
            # LearningRateScheduler(scheduler),
            ReduceLROnPlateau(monitor="val_loss", factor=np.sqrt(0.1), patience=lr_patience, verbose=1, min_lr=0.5e-6),
            PlotProgress(self.training_info_dir),
            # TensorBoard(log_dir=f'./{self.model_name}/graph', histogram_freq=1, write_graph=True,
            #             write_images=True)
        ]

    def get_perm_images_generator(self, x, y, augmented, debug=False, bs=None, shuffle=True):
        aug = ImageDataGenerator(
            **get_aug(self.model_name)
        ) if augmented else ImageDataGenerator()
        gen = PermutationGenerator(x, y, aug, self.subinput_shape, permutations=self.permutations,
                                   batch_size=bs, debug_path=self.debug_info_dir, shuffle_dataset=shuffle)
        if debug:
            gen.run_debug()
        return gen

    def set_up(self):
        self.training_info_dir = join(self.model_name, info_dir_name)
        self.debug_info_dir = join(self.model_name, debug_dir_name)
        self.arch_info_dir = join(self.model_name, arch_dir_name)
        self.checkpoints_dir = join(self.model_name, 'checkpoint')

        pathlib.Path(self.debug_info_dir).mkdir(exist_ok=True, parents=True)
        if exists(self.arch_info_dir):
            shutil.rmtree(self.arch_info_dir)
        pathlib.Path(self.arch_info_dir).mkdir(exist_ok=True, parents=True)
        pathlib.Path(self.checkpoints_dir).mkdir(exist_ok=True, parents=True)
        self.model_factory = ModelFactory(self.arch_info_dir, self.n_classes, len(self.permutations),
                                          self.subinput_shape, self.aggr)

    def fit(self, x_train, y_train, x_val, y_val):
        self.set_up()
        if exists(join(self.model_name, 'saved_model.pb')):
            print("Model already trained, skipping")
            return

        with open(join(self.model_name, "permutations.txt"), 'w') as f:
            f.write(str(self.permutations).replace('array', 'np.array'))

        train_ds = self.get_perm_images_generator(x_train, y_train, bs=batch_size,
                                                  augmented=self.augmented, debug=True, )
        valid_ds = self.get_perm_images_generator(x_val, y_val, bs=batch_size,
                                                  augmented=False, debug=False, )

        self.model = self.model_factory.get_model(self.model_type, self.n_classes, m_id=self.m_id)
        self.model.compile(**compile_opts())

        try:
            self.model.fit(train_ds, epochs=epochs, verbose=1, validation_data=valid_ds,
                           steps_per_epoch=len(x_train) // batch_size,
                           validation_steps=len(x_val) // batch_size,
                           callbacks=self.callbacks())
        except KeyboardInterrupt:
            print("\nInterrupted!")
            self.model.load_weights(join(self.checkpoints_dir, 'weights.h5'))
        print(f"Saving model {self.model_name}")
        self.model.save(self.model_name)
        self.save_training_info(show=False)
        return self.model

    def predict(self, x_test, y_test):
        print(f'Predicting {self.model_name}')
        dic = ''
        with open(join(self.model_name, "permutations.txt"), 'r') as f:
            for i in f.readlines():
                dic += i
        self.permutations = eval(dic)

        if 'sequential' in self.model_name and 'subs' not in self.model_name:
            for i, (coords, perm) in enumerate(self.permutations.items()):
                sub_model = join(self.model_name, "subs", str(i))
                m = ModelTraining(sub_model, self.subinput_shape, self.n_classes, {coords: perm},
                                  'single',
                                  self.classes_names, aggr=None, m_id=i)
                m.predict(x_test, y_test)
        self.model = load_model(self.model_name)
        # history = np.load(join(self.training_info_dir, "history.npy"), allow_pickle=True).item()
        testing_path = join(self.model_name, testing_dir_name)
        pathlib.Path(testing_path).mkdir(exist_ok=True, parents=True)

        test_gen = self.get_perm_images_generator(x_test, y_test, augmented=False, debug=False, bs=len(x_test),
                                                  shuffle=False)
        x_test, y_test = test_gen.next()
        prediction = self.model.predict(x_test)

        actual_classes = np.argmax(y_test, axis=1)
        predicted_classes = np.argmax(prediction, axis=1)
        plt.ion()
        pp_matrix_from_data(actual_classes, predicted_classes, columns=self.classes_names)
        plt.savefig(join(testing_path, 'conf_matrix.png'))
        plt.close('all')
        plt.ioff()
        cr = classification_report(actual_classes, predicted_classes, target_names=self.classes_names)
        with open(join(testing_path, "classification_scores"), 'w') as f:
            print(cr, file=f)
        return accuracy_score(actual_classes, predicted_classes)

    def save_training_info(self, show=False):
        plt.close('all')
        plt.ioff()
        history = self.model.history
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
        np.save(join(self.training_info_dir, "history.npy"), history.history)
        with open(join(self.training_info_dir, "model_config"), 'w') as f:
            pprint(self.model.get_config(), f)


########################################################################################################################
########################################################################################################################
########################################################################################################################

class TrainingSequential(ModelTraining):
    def __init__(self, model_name, subinput_shape, n_classes, permutations, classes_names, aggr):
        super().__init__(model_name, subinput_shape, n_classes, permutations, 'sequential', classes_names, aggr)
        self.sub_models = []

    def fit(self, x_train, y_train, x_val, y_val):
        self.set_up()
        train_ds = self.get_perm_images_generator(x_train, y_train, bs=batch_size,
                                                  augmented=self.augmented, debug=True, )
        valid_ds = self.get_perm_images_generator(x_val, y_val, bs=batch_size,
                                                  augmented=False, debug=False, )
        with open(join(self.model_name, "permutations.txt"), 'w') as f:
            f.write(str(self.permutations).replace('array', 'np.array'))

        perms = enumerate(self.permutations.items())
        # perms = reversed(list(perms))  #
        for i, (coords, perm) in perms:
            sub_model = join(self.model_name, "subs", str(i))
            self.sub_models.append(sub_model)
            m = ModelTraining(sub_model, self.subinput_shape, self.n_classes, {coords: perm},
                              'single',
                              self.classes_names, aggr=None, m_id=i)
            m.fit(x_train, y_train, x_val, y_val)
        # self.sub_models = reversed(self.sub_models)  #

        run_dense = True
        if run_dense:
            models = []
            for m_p in self.sub_models:
                m = load_model(m_p)
                # model = Model(inputs=m.input, outputs=m.layers[-2].output, name=m.name)
                if self.aggr == Aggregation.CONCAT_STRIP.value:
                    # model = m
                    model = Model(inputs=m.input, outputs=m.layers[-2].output, name=m.name)
                else:
                    model = m
                model.trainable = False
                models.append(model)
            _ins = [Input(shape=self.subinput_shape) for _ in models]
            outs = [m(_in) for m, _in in zip(models, _ins)]
            self.model = self.model_factory.get_aggregating_model(_ins, outs, model_name='sequential')
            self.model.compile(**compile_opts())

            try:
                self.model.fit(train_ds, epochs=epochs, verbose=1, validation_data=valid_ds,
                               steps_per_epoch=len(x_train) // batch_size,
                               validation_steps=len(x_val) // batch_size,
                               callbacks=self.callbacks())
            except KeyboardInterrupt:
                print("\nInterrupted!")
                self.model.load_weights(join(self.checkpoints_dir, 'weights.h5'))
            print(f"Saving model {self.model_name}")
            self.model.save(self.model_name)
            self.save_training_info()
            print('Model saved')
            return self.model
        else:
            ensemble_train = []
            ensemble_val = []
            models = []
            train_gens = []
            valid_gens = []
            for i, (coords, perm) in perms:
                sub_model = join(self.model_name, "subs", str(i))
                m = ModelTraining(sub_model, self.subinput_shape, self.n_classes, {coords: perm},
                                  'single',
                                  self.classes_names, aggr=None, m_id=i)
                train_gen = m.get_perm_images_generator(x_train, y_train, shuffle=False, bs=len(x_train),
                                                        augmented=True)
                val_gen = m.get_perm_images_generator(x_val, y_val, shuffle=False, bs=len(x_val), augmented=False)
                model = load_model(sub_model)
                models.append(model)
                train_gens.append(train_gen)
                valid_gens.append(val_gen)
            ensemble_train = np.zeros((len(x_train), len(models), model.layers[-2].output.shape[-1]))
            ensemble_valid = np.zeros((len(x_val), len(models), model.layers[-2].output.shape[-1]))

            for i, (m, tg, vg) in enumerate(zip(models, train_gens, valid_gens)):
                ensemble_train[:, i, :]

            clf = RandomForestClassifier()
            clf.fit(ensemble_train, y_train)
            prediction = clf.predict(ensemble_val)
            actual_classes = np.argmax(y_val, axis=1)
            predicted_classes = np.argmax(prediction, axis=1)
            testing_path = join(self.model_name, testing_dir_name)
            pathlib.Path(testing_path).mkdir(exist_ok=True, parents=True)
            plt.ion()
            pp_matrix_from_data(actual_classes, predicted_classes, columns=self.classes_names)
            plt.savefig(join(testing_path, 'conf_matrix_val.png'))
            plt.close('all')
            plt.ioff()
            cr = classification_report(actual_classes, predicted_classes, target_names=self.classes_names)
            with open(join(testing_path, "classification_scores"), 'w') as f:
                print(cr, file=f)
            return accuracy_score(actual_classes, predicted_classes)


class PlotProgress(Callback):
    max_acc = 0
    max_val_acc = 0
    min_loss = sys.maxsize
    min_val_loss = sys.maxsize

    acc_ep = 0
    val_acc_ep = 0
    loss_ep = 0
    val_loss_ep = 0

    def __init__(self, i_dir, verbose=True):
        pathlib.Path(i_dir).mkdir(exist_ok=True, parents=True)
        super().__init__()
        self.verbose = verbose
        self.axs = None
        self.f = None
        self.metrics = None
        self.i_dir = i_dir
        pathlib.Path(self.i_dir).mkdir(exist_ok=True, parents=True)
        self.first_epoch = True
        self.metrics = {}
        plt.ion()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        n_met = len([x for x in logs if 'val' not in x])
        if self.f is None:
            if n_met > 3:
                self.f, self.axs = plt.subplots(2, 3, figsize=(12, 8))
            else:
                self.f, self.axs = plt.subplots(1, 3, figsize=(12, 4))

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
                    ax.set_ylim([0.0, 1.0])
                elif metric == 'loss':
                    ax.set_title(loss_msg)
                    ax.set_ylim([0.0, 3.0])
            if self.first_epoch:
                ax.legend()
                ax.grid()
        self.first_epoch = False
        plt.tight_layout()
        self.f.canvas.draw()
        self.f.canvas.flush_events()
        self.f.savefig(f"{self.i_dir}/progress.png")
