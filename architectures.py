from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, GlobalAveragePooling2D, BatchNormalization, \
    multiply, \
    Dropout, ReLU
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.layers import Add

weight_decay = 1e-7
alpha = 0.2


def Conv_block(x, F, k, s, name, padding='same', act=True):
    x = Conv2D(F, k, strides=s, name=name, kernel_initializer='he_normal', padding=padding,
               kernel_regularizer=l2(weight_decay))(x)
    x = SqueezeExcite(x)
    x = BatchNormalization()(x)
    if act:
        x = ReLU()(x)
    return x


def SqueezeExcite(_in, i=None):
    ratio = 16
    filters = _in.shape[-1]
    x = GlobalAveragePooling2D()(_in)
    x = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False,
              kernel_regularizer=l2(weight_decay))(x)
    x = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False,
              kernel_regularizer=l2(weight_decay))(x)
    _out = multiply([_in, x])
    # se = Model(inputs=_in, outputs=_out, name=f"squeeze_excite_{i}")
    # self.save_model_info(se, f"squeeze_excite_{i}")
    # return se(_in)
    return _out


def Inception_resnet(_in, filters, stage, block):
    conv_name_base = f'inc-res_stage_{stage}_block_{block}_branch'
    F1, F2, F3, F4 = filters
    b1 = Conv_block(_in, F1, (1, 1), s=1, name=conv_name_base + '1a')
    b1 = Conv_block(b1, F1, (3, 3), s=1, name=conv_name_base + '1b')

    b2 = Conv_block(_in, F2, (1, 1), s=1, name=conv_name_base + '2a')
    b2 = Conv_block(b2, F2, (3, 3), s=1, name=conv_name_base + '2b')
    b2 = Conv_block(b2, F2, (3, 3), s=1, name=conv_name_base + '2c')

    b3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(_in)
    b3 = Conv_block(b3, F3, (1, 1), s=1, name=conv_name_base + '31')

    merged = concatenate([b1, b2, b3])  # output size W x H x f3
    merged = Conv_block(merged, F4, (1, 1), s=1, name=conv_name_base + '-merged', act=False)
    _out = Add()([merged, _in])
    _out = ReLU()(_out)

    # inception = Model(inputs=_in, outputs=_out, name=f"inception_{i}")
    # self.save_model_info(inception, f"inception_{i}")
    # return inception(_in)
    return _out


# def resnet_inception(_in):
#     d_r = 0.15
#     x = _in
#     x = Conv2D(64, (7, 7), strides=(1, 1), name='conv1', kernel_initializer='he_normal', padding='same',
#                kernel_regularizer=l2(weight_decay))(x)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     x = SqueezeExcite(x)
#     x = Dropout(d_r)(x)
#     stage = 1
#     filters = [64, 64, 128]
#     i_filters = [64, 64, 64, 128]
#     kernel = 3
#     x = convolutional_block(x, f=kernel, filters=filters, stage=stage, block='a', s=1)
#     x = Dropout(d_r)(x)
#     for _ in range(2):
#         x = Inception_resnet(x, filters=i_filters)
#         x = Dropout(d_r)(x)
#
#     stage = stage + 1
#     filters = [128, 128, 256]
#     i_filters = [128, 128, 128, 256]
#     kernel = 3
#     x = convolutional_block(x, f=kernel, filters=filters, stage=stage, block='a', s=2)
#     x = Dropout(d_r)(x)
#     for _ in range(3):
#         x = Inception_resnet(x, filters=i_filters)
#         x = Dropout(d_r)(x)
#
#     stage = stage + 1
#     filters = [256, 256, 512]
#     i_filters = [256, 256, 256, 512]
#     kernel = 3
#     x = convolutional_block(x, f=kernel, filters=filters, stage=stage, block='a', s=2)
#     x = Dropout(d_r)(x)
#     for _ in range(2):
#         x = Inception_resnet(x, filters=i_filters)
#         x = Dropout(d_r)(x)
#     x = GlobalAveragePooling2D()(x)
#     x = Dropout(0.4)(x)
#     return x


def get_resnet(_in):
    stages = [
        {
            'n_blocks': 3,
            'filters': [64, 64, 128],
            'kernel': 5,
            'stride': 1
        },
        {
            'n_blocks': 5,
            'filters': [128, 128, 256],
            'kernel': 5,
            'stride': 2
        },
        # {
        #     'n_blocks': 3,
        #     'filters': [256, 256, 512],
        #     'kernel': 3,
        # }
    ]
    return resnet_builder(_in, stages)


def resnet_builder(_in, stages):
    d_r = 0.1
    x = _in
    x = Conv_block(x, 64, (11, 11), s=2, name='conv1')
    x = Dropout(d_r)(x)
    for st, params in enumerate(stages):
        n_blocks, filters, kernel, stride = params.values()
        x = convolutional_block(x, f=kernel, filters=filters, stage=st, block=0, s=stride)
        x = Dropout(d_r)(x)
        for i in range(1, n_blocks + 1):
            x = identity_block(x, kernel, filters, stage=st, block=i)
            x = Dropout(d_r)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    return x


def resnet_fashion(_in):
    d_r = 0.1
    x = _in
    x = Conv2D(128, (7, 7), strides=(2, 2), name='conv1', kernel_initializer='he_normal', padding='same',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SqueezeExcite(x)
    x = Dropout(d_r)(x)

    stage = 1
    filters = [128, 128, 256]
    kernel = 3
    x = convolutional_block(x, f=kernel, filters=filters, stage=stage, block='a', s=1)
    x = Dropout(d_r)(x)
    n_blocks = 6
    for i in range(n_blocks):
        x = identity_block(x, kernel, filters, stage=stage, block=i)
        x = Dropout(d_r)(x)

    stage = stage + 1
    filters = [128, 128, 256]
    kernel = 3
    x = convolutional_block(x, f=kernel, filters=filters, stage=stage, block='a', s=2)
    x = Dropout(d_r)(x)
    n_blocks = 4
    for i in range(n_blocks):
        x = identity_block(x, kernel, filters, stage=stage, block=i)
        x = Dropout(d_r)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    return x


def convolutional_block(x, f, filters, stage, block, s=2):
    conv_name_base = f'conv_stage_{stage}_block_{block}_branch'

    F1, F2, F3 = filters

    x_skip = x
    x = Conv_block(x, F1, (1, 1), s=s, name=conv_name_base + '2a', padding='valid')
    x = Conv_block(x, F2, (f, f), s=1, name=conv_name_base + '2b')
    x = Conv_block(x, F3, (1, 1), s=1, name=conv_name_base + '2c', padding='valid')

    x_skip = Conv_block(x_skip, F3, (1, 1), s=1, name=conv_name_base + '1', padding='valid', act=False)

    x = Add()([x, x_skip])
    x = ReLU()(x)

    return x


def identity_block(x, f, filters, stage, block):
    conv_name_base = f'res_stage_{stage}_block_{block}_branch'
    F1, F2, F3 = filters

    x_skip = x
    x = Conv_block(x, F1, (1, 1), s=1, name=conv_name_base + '2a', padding='valid')
    x = Conv_block(x, F2, (f, f), s=1, name=conv_name_base + '2b')
    x = Conv_block(x, F3, (1, 1), s=1, name=conv_name_base + '2c', padding='valid')

    x = Add()([x, x_skip])
    x = ReLU()(x)

    return x
