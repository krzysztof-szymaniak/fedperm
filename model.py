import sys
from os.path import join

from keras import Model
from keras import utils
from keras.callbacks import Callback
from keras.layers import Conv2D, MaxPooling2D, concatenate, GlobalAveragePooling2D, BatchNormalization, multiply, Add, \
    Activation, MaxPool2D, Flatten, Dropout
from keras.layers import Dense, Input
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from matplotlib import pyplot as plt

from config import info_dir


def SqueezeExcite(_in, ratio=8):
    """Squeeze-and-Excitation layers are considered to improve CNN performance.
    `Find out more <https://doi.org/10.48550/arXiv.1709.01507>`
    """
    filters = _in.shape[-1]
    x = GlobalAveragePooling2D()(_in)
    x = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(x)
    x = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)
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
    # x = Conv2D(32, (7, 7), padding='same', strides=(1, 1), activation='relu', kernel_initializer='he_normal')(x)
    # x = SqueezeExcite(x)
    # x = Inception(x, filters=[32, 32, 32])
    # x = MaxPool2D()(x)
    x = Conv2D(128, (5, 5), padding='same', strides=(2, 2), activation='relu', kernel_initializer='he_normal')(x)
    x = SqueezeExcite(x)
    x = Inception(x, filters=[32, 32, 32])
    x = SqueezeExcite(x)
    x = MaxPool2D()(x)
    x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', kernel_initializer='he_normal')(x)
    # x = Inception(x, filters=[32, 32, 32])
    # x = SqueezeExcite(x)
    # x = Inception(x, filters=[32, 32, 32])
    # x = SqueezeExcite(x)
    # x = Conv2D(32, (1, 1), strides=(1, 1), activation='relu', kernel_initializer='he_normal')(x)
    # x = Flatten()(x) if not global_pooling else GlobalAveragePooling2D()(x)
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
        model_outputs.append(submodel(_in))

    # aggregated = Add()(model_outputs)  # TODO Add vs concatenate
    aggregated = concatenate(model_outputs, axis=-1)
    aggregated = Conv2D(128, (2, 2), padding='same', strides=(2, 2), activation='relu', kernel_initializer='he_normal')(aggregated)
    aggregated = GlobalAveragePooling2D()(aggregated)
    _out = Dense(n_classes, activation='softmax', kernel_initializer='he_normal')(aggregated)
    ensemble_model = Model(inputs=_ins, outputs=_out, name="aggregated_model")
    ensemble_model.compile(loss=categorical_crossentropy,
                           # optimizer=SGD(learning_rate=1e-4, momentum=0.9),
                           optimizer=Adam(learning_rate=1e-2, decay=0.01),
                           metrics=['accuracy'])
    ensemble_model.summary()
    utils.plot_model(ensemble_model, show_layer_names=False, show_shapes=True, to_file=f'{i_dir}/model.png')
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

    def on_train_begin(self, logs=None):
        plt.ion()
        self.first_epoch = True
        if logs is None:
            logs = {}
        self.metrics = {}
        self.f, self.axs = plt.subplots(1, 3, figsize=(15, 5))

        for metric in logs:
            self.metrics[metric] = []

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
