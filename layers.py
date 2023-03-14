from keras.layers import Conv2D, MaxPooling2D, concatenate, GlobalAveragePooling2D, BatchNormalization, multiply, \
    Activation, Dropout, MaxPool2D
from keras.layers import Dense
from keras.regularizers import l2

weight_decay = 1e-4


def SqueezeExcite(_in, ratio=8):
    filters = _in.shape[-1]
    x = GlobalAveragePooling2D()(_in)
    x = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False,
              kernel_regularizer=l2(weight_decay))(x)
    x = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False,
              kernel_regularizer=l2(weight_decay))(x)
    return multiply([_in, x])


def ConvBlock(_in, filters, kernel_size, d_r=0.0, strides=(1, 1)):
    x = _in
    x = Conv2D(filters, kernel_size, padding='same', strides=strides, activation='relu', kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = SqueezeExcite(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(d_r)(x)
    return x


def ReduceChannels(_in, channels=0):
    return Conv2D(channels, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal',
                  kernel_regularizer=l2(weight_decay))(_in)


def Inception(_in, filters=None):
    if filters is None:
        filters = [10, 10, 10]
    col_1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu', kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(_in)
    col_1 = Conv2D(filters[0], (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(col_1)

    col_2 = Conv2D(filters[1], (1, 1), padding='same', activation='relu', kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(_in)
    col_2 = Conv2D(filters[1], (5, 5), padding='same', activation='relu', kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(col_2)

    col_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(_in)
    col_3 = Conv2D(filters[2], (1, 1), padding='same', activation='relu', kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(col_3)

    out = concatenate([col_1, col_2, col_3])  # output size W x H x (f0 + f1 + f2)
    return out


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
