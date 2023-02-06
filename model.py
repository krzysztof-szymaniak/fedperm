import os.path
import sys
from contextlib import redirect_stdout

from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import utils
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv2D, concatenate, GlobalAveragePooling2D, BatchNormalization, Activation, \
    MaxPool2D, Dropout, Flatten
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2

from layers import SqueezeExcite, Inception, ReduceChannels

weight_decay = 1e-5


def cifar_submodel(_in):
    d_r = 0.25
    x = _in
    x = Conv2D(64, (1, 1), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(d_r)(x)

    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(d_r)(x)

    x = Conv2D(64, (5, 5), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(d_r)(x)

    x = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    x = SqueezeExcite(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D((2, 2))(x)

    x = Dropout(d_r)(x)
    x = Conv2D(128, (1, 1), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(d_r)(x)

    x = Conv2D(128, (3, 3), kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    x = SqueezeExcite(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(d_r)(x)

    x = Conv2D(128, (5, 5), kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    x = SqueezeExcite(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(d_r)(x)
    return x


def inception_model(_in, n_outputs):
    d_r = 0.15
    x = _in
    x = Inception(x, filters=[64, 64, 64])
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)

    x = Dropout(d_r)(x)
    x = Inception(x, filters=[128, 128, 128])
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)

    x = Dropout(d_r)(x)
    x = Inception(x, filters=[256, 256, 256])
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)

    x = Dropout(d_r)(x)
    x = Inception(x, filters=[512, 512, 512])
    x = BatchNormalization()(x)
    x = SqueezeExcite(x)
    x = ReduceChannels(x, channels=512)
    x = BatchNormalization()(x)
    x = SqueezeExcite(x)
    x = ReduceChannels(x, channels=256)
    x = BatchNormalization()(x)
    x = SqueezeExcite(x)
    x = ReduceChannels(x, channels=n_outputs)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    return x


def SubModel(_in, name=None, n_outputs=None):
    x = _in
    x = inception_model(x, n_outputs)
    _out = x
    model = Model(inputs=_in, outputs=_out, name=name)
    model.summary()
    return model


def get_aggregate_model(models_path, input_shape, n_classes, i_dir):
    models = [load_model(m_p) for m_p in [os.path.join(models_path, m) for m in os.listdir(models_path)]]
    _ins = [Input(shape=input_shape) for _ in models]
    for m in models:
        m.trainable = False
    x = concatenate([m(_in) for m, _in in zip(models, _ins)])
    x = Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.3)(x)
    x = Dense(n_classes, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    _out = x
    model = Model(inputs=_ins, outputs=_out, name="aggregated-sequential")
    model.summary()
    utils.plot_model(model, show_layer_names=False, show_shapes=True, to_file=f'{i_dir}/submodel.png')
    with open(f'{i_dir}/sub_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    return model


def get_single_model(input_shape, n_classes, i, i_dir):
    _in = Input(shape=input_shape)
    x = inception_model(_in, 128)
    x = Dense(n_classes, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    _out = x
    model = Model(inputs=_in, outputs=_out, name=f"single-{i + 1}")
    utils.plot_model(model, show_layer_names=False, show_shapes=True, to_file=f'{i_dir}/submodel.png')
    with open(f'{i_dir}/sub_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    return model


def get_composite_model(input_shape, n_classes, n_frames, i_dir):
    _ins = [Input(shape=input_shape) for _ in range(n_frames)]
    model_outputs = []
    for i, _in in enumerate(_ins):
        submodel = SubModel(_in, name=f"federated-{i}", n_outputs=128)
        if i == 0:
            utils.plot_model(submodel, show_layer_names=False, show_shapes=True, to_file=f'{i_dir}/submodel.png')
            with open(f'{i_dir}/sub_summary.txt', 'w') as f:
                with redirect_stdout(f):
                    submodel.summary()
        model_outputs.append(submodel(_in))
    aggr = concatenate(model_outputs, axis=-1)
    aggr = Dropout(0.3)(aggr)
    aggr = Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(aggr)
    aggr = Dropout(0.3)(aggr)
    _out = Dense(n_classes, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(
        aggr)
    ensemble_model = Model(inputs=_ins, outputs=_out, name="aggregated_model")
    ensemble_model.summary()
    utils.plot_model(ensemble_model, show_layer_names=False, show_shapes=True, to_file=f'{i_dir}/model.png')
    with open(f'{i_dir}/summary.txt', 'w') as f:
        with redirect_stdout(f):
            ensemble_model.summary()
    return ensemble_model


class PlotProgress(Callback):
    max_acc = 0
    max_val_acc = 0
    min_loss = sys.maxsize
    min_val_loss = sys.maxsize

    acc_ep = 0
    val_acc_ep = 0
    loss_ep = 0
    val_loss_ep = 0

    def __init__(self, i_dir):
        super().__init__()
        self.axs = None
        self.f = None
        self.metrics = None
        self.i_dir = i_dir
        self.first_epoch = True

    def on_train_begin(self, logs=None):
        plt.ion()
        if logs is None:
            logs = {}
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
        self.f, self.axs = plt.subplots(1, 3, figsize=(13, 4))

    def on_train_end(self, logs=None):
        self.f.savefig(f"{self.i_dir}/metrics")

    def on_epoch_end(self, epoch, logs=None):
        # Storing metrics
        if logs is None:
            logs = {}
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

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

        metrics = [x for x in logs if 'val' not in x]
        for i, metric in enumerate(metrics):
            self.axs[i].plot(range(1, epoch + 2), self.metrics[metric], color='blue', label=metric)
            if 'val_' + metric in logs:
                self.axs[i].plot(range(1, epoch + 2), self.metrics['val_' + metric], label='val_' + metric,
                                 color='orange', )
                if metric == 'accuracy':
                    self.axs[i].set_title(
                        f"{'Max accuracy': <16}: {self.max_acc:.4f}, not improved in {self.acc_ep} epochs\n{'Max val_accuracy': <16}: {self.max_val_acc:.4f}, not improved in {self.val_acc_ep} epochs")
                elif metric == 'loss':
                    self.axs[i].set_title(
                        f"{'Min loss': <16}: {self.min_loss:.4f}, not improved in {self.loss_ep} epochs\n{'Min val_loss': <16}: {self.min_val_loss:.4f}, not improved in {self.val_loss_ep} epochs")
            if self.first_epoch:
                self.axs[i].legend()
                self.axs[i].grid()
        self.first_epoch = False
        plt.tight_layout()
        self.f.canvas.draw()
        self.f.canvas.flush_events()
