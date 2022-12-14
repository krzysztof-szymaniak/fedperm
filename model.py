import sys
from contextlib import redirect_stdout
from os.path import join

from keras import Model
from keras import utils
from keras.callbacks import Callback
from keras.layers import Conv2D, MaxPooling2D, concatenate, GlobalAveragePooling2D, BatchNormalization, multiply, Add, \
    Activation, MaxPool2D, Flatten, Dropout
from keras.layers import Dense, Input
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from matplotlib import pyplot as plt

from config import info_dir

weight_decay = 1e-3


def SqueezeExcite(_in, ratio=8):
    """Squeeze-and-Excitation layers are considered to improve CNN performance.
    `Find out more <https://doi.org/10.48550/arXiv.1709.01507>`
    """
    filters = _in.shape[-1]
    x = GlobalAveragePooling2D()(_in)
    x = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    x = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    return multiply([_in, x])


def Inception(_in, filters=None):
    if filters is None:
        filters = [10, 10, 10]
    col_1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(_in)
    col_1 = Conv2D(filters[0], (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(col_1)

    col_2 = Conv2D(filters[1], (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(_in)
    col_2 = Conv2D(filters[1], (5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(col_2)

    col_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(_in)
    col_3 = Conv2D(filters[2], (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(col_3)

    out = concatenate([col_1, col_2, col_3])  # output size W x H x (f0 + f1 + f2)
    return out


def best_so_far(_in):
    x = _in
    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', kernel_initializer='he_normal')(x)
    x = SqueezeExcite(x)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), )(x)
    x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', kernel_initializer='he_normal')(x)
    x = SqueezeExcite(x)
    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', kernel_initializer='he_normal')(x)
    x = SqueezeExcite(x)
    x = Inception(x, filters=[10, 10, 10])
    x = Flatten()(x)
    return x


def cifar_submodel(_in):
    x = _in
    # x = Conv2D(64, (7, 7), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(_in)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)
    # x = SqueezeExcite(x)
    x = Conv2D(64, (1, 1), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SqueezeExcite(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SqueezeExcite(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (5, 5), padding='same', kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.25)(x)
    x = MaxPool2D((2, 2))(x)

    x = Dropout(0.25)(x)
    x = Conv2D(128, (1, 1), kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SqueezeExcite(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, (3, 3), kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SqueezeExcite(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, (5, 5), kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = GlobalAveragePooling2D()(x)
    return x


def SubModel(_in, name=None):
    x = _in
    x = cifar_submodel(x)
    _out = x
    model = Model(inputs=_in, outputs=_out, name=name)
    model.summary()
    return model


def get_composite_model(input_shape, n_classes, n_frames, i_dir):
    _ins = [Input(shape=input_shape) for _ in range(n_frames)]
    model_outputs = []
    for i, _in in enumerate(_ins):
        submodel = SubModel(_in, name=f"federated-{i}")
        if i == 0:
            utils.plot_model(submodel, show_layer_names=False, show_shapes=True, to_file=f'{i_dir}/submodel.png')
            with open(f'{i_dir}/sub_summary.txt', 'w') as f:
                with redirect_stdout(f):
                    submodel.summary()
        model_outputs.append(submodel(_in))

    # aggregated = Add()(model_outputs)  #  Add vs concatenate
    aggregated = concatenate(model_outputs, axis=-1)
    aggregated = Dropout(0.5)(aggregated)
    aggregated = Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(
        aggregated)
    aggregated = Dropout(0.5)(aggregated)
    _out = Dense(n_classes, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(
        aggregated)
    ensemble_model = Model(inputs=_ins, outputs=_out, name="aggregated_model")
    ensemble_model.compile(loss=categorical_crossentropy,
                           # optimizer=SGD(learning_rate=1e-4, momentum=0.9),
                           optimizer=Adam(learning_rate=1e-3, decay=0.001),
                           metrics=['accuracy'])
    ensemble_model.summary()
    utils.plot_model(ensemble_model, show_layer_names=False, show_shapes=True, to_file=f'{i_dir}/model.png')
    with open(f'{i_dir}/summary.txt', 'w') as f:
        with redirect_stdout(f):
            ensemble_model.summary()
    return ensemble_model


def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        )(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters

    X_shortcut = X
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               )(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = Add()([X, X_shortcut])  # SKIP Connection
    X = Activation('relu')(X)

    return X


def get_seed(model_name):
    with open(join(model_name, info_dir, "seed"), 'r') as f:
        seed = int(f.readline())
        return seed


class PlotProgress(Callback):
    max_acc = 0
    max_val_acc = 0
    min_loss = sys.maxsize
    min_val_loss = sys.maxsize

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

        self.f, self.axs = plt.subplots(1, 3, figsize=(15, 5))

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

        self.max_acc = max(self.max_acc, logs.get("accuracy"))
        self.max_val_acc = max(self.max_val_acc, logs.get("val_accuracy"))
        self.min_loss = min(self.min_loss, logs.get("loss"))
        self.min_val_loss = min(self.min_val_loss, logs.get("val_loss"))

        metrics = [x for x in logs if 'val' not in x]
        for i, metric in enumerate(metrics):
            self.axs[i].plot(range(1, epoch + 2), self.metrics[metric], color='blue', label=metric)
            if 'val_' + metric in logs:
                self.axs[i].plot(range(1, epoch + 2), self.metrics['val_' + metric], label='val_' + metric,
                                 color='orange', )
                if metric == 'accuracy':
                    self.axs[i].set_title(
                        f"{'Max accuracy': <25}: {self.max_acc:.4f}\n{'Max val_accuracy': <25}: {self.max_val_acc:.4f}")
                elif metric == 'loss':
                    self.axs[i].set_title(
                        f"{'Min loss': <25}: {self.min_loss:.4f}\n{'Min val_loss': <25}: {self.min_val_loss:.4f}")
            if self.first_epoch:
                self.axs[i].legend()
                self.axs[i].grid()
        self.first_epoch = False
        plt.tight_layout()
        self.f.canvas.draw()
        self.f.canvas.flush_events()
