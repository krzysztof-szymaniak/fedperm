from keras.layers import Conv2D, MaxPooling2D, concatenate, GlobalAveragePooling2D, BatchNormalization, multiply, Add, \
    Activation, Dropout
from keras.layers import Dense
from keras.regularizers import l2

weight_decay = 1e-4


def SqueezeExcite(_in, ratio=8):
    """Squeeze-and-Excitation layers are considered to improve CNN performance.
    `Find out more <https://doi.org/10.48550/arXiv.1709.01507>`
    """
    filters = _in.shape[-1]
    x = GlobalAveragePooling2D()(_in)
    x = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False,
              kernel_regularizer=l2(weight_decay))(x)
    x = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False,
              kernel_regularizer=l2(weight_decay))(x)
    return multiply([_in, x])


def ConvBlock(_in, filters, kernel_size, d_r=0.0, strides=(1, 1)):
    x = _in
    x = Conv2D(filters, kernel_size, padding='same', strides=strides, activation='relu', kernel_initializer='he_normal')(x)
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
