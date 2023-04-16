import pathlib
from pprint import pprint

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, BatchNormalization, \
    Dropout, ReLU, SpatialDropout2D, LeakyReLU, Multiply, Add, Dense, Flatten, Lambda, Average
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from contextlib import redirect_stdout
from tensorflow.keras import utils

from resnets_configs import fashion_config, cifar_config, cats_dogs_config

VISUALIZE = True

weight_decay = 1e-7
alpha = 0.05
d_r = 0.1


def get_resnet(_in, m_id, i_dir):
    if 'cifar' in i_dir:
        return ResnetBuilder(m_id, i_dir).resnet(_in, cifar_config)
    elif 'fashion' in i_dir:
        return ResnetBuilder(m_id, i_dir).resnet(_in, fashion_config)
    elif 'cats' in i_dir:
        return ResnetBuilder(m_id, i_dir).resnet(_in, cats_dogs_config)


def aggregate(models, n_classes):
    x = Concatenate()(models)
    x = Dense(512, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = LeakyReLU(alpha)(x)
    x = Dropout(0.2)(x)
    x = Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = LeakyReLU(alpha)(x)
    if n_classes != 2:
        x = Dense(n_classes, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(
            x)
    else:
        x = Dense(1, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    return x


class ResnetBuilder:
    def __init__(self, m_id, i_dir):
        self.i_dir = i_dir
        self.m_id = m_id

    def store_config(self, conf):
        with open(f'{self.i_dir}/resnet.config', 'w') as f:
            pprint(conf, f)

    def resnet(self, _in, config):
        self.store_config(config)
        stages = config['stages']
        f, k, s = config['stem_layer'].values()
        x = _in
        x = self.Conv(x, f, (k, k), s=s, name=f'Conv{f}x{f}-stem_m{self.m_id}')
        for st, params in enumerate(stages):
            if len(params) == 5:
                n_blocks, filters, kernel, stride, i_filters = params.values()
            else:
                i_filters = None
                n_blocks, filters, kernel, stride = params.values()
            # x = SpatialDropout2D(d_r)(x)
            x = self.convolutional_block(x, f=kernel, filters=filters, stage=st, block=0, s=stride)
            for i in range(1, n_blocks):
                if i_filters:
                    # x = SpatialDropout2D(d_r)(x)
                    x = self.Inception_resnet(x, i_filters, stage=st, block=i)
                else:
                    # x = SpatialDropout2D(d_r)(x)
                    x = self.identity_block(x, kernel, filters, stage=st, block=i)
        for layer in config['output_layers']:
            x = eval(layer)
        return x

    def convolutional_block(self, prev, f, filters, stage, block, s):
        _in = Input(prev.shape[1:])
        conv_name_base = f'Conv[f]x[f]-m{self.m_id}_st{stage}_bl{block}'
        x = _in
        F1, F2, F3 = filters

        x_skip = x
        x = self.Conv(x, F1, (1, 1), s=s, name=conv_name_base.replace('[f]', '1') + '_br2a', padding='valid')
        x = self.Conv(x, F2, (f, f), s=1, name=conv_name_base.replace('[f]', str(f)) + '_br2b')
        x = self.Conv(x, F3, (1, 1), s=1, name=conv_name_base.replace('[f]', '1') + '_br2c', padding='valid',
                      act=False)

        x_skip = self.Conv(x_skip, F3, (1, 1), s=s, name=conv_name_base.replace('[f]', '1') + 'skip',
                           padding='valid', act=False)

        x = Add()([x, x_skip])
        x = LeakyReLU(alpha)(x)

        m_name = f'ConvBlock{f}x{f}-m{self.m_id}_st{stage}_bl{block}'
        m = Model(inputs=_in, outputs=x, name=m_name)
        save_model_info(f'{self.i_dir}/resnet-blocks', m, m_name)
        return m(prev)

    def identity_block(self, prev, f, filters, stage, block, ):
        _in = Input(prev.shape[1:])
        conv_name_base = f'Conv[f]x[f]-m{self.m_id}_st{stage}_bl{block}'
        F1, F2, F3 = filters
        x = _in
        x_skip = x
        x = self.Conv(x, F1, (1, 1), s=1, name=conv_name_base.replace('[f]', '1') + '_br2a', padding='valid')
        x = self.Conv(x, F2, (f, f), s=1, name=conv_name_base.replace('[f]', str(f)) + '_br2b')
        x = self.Conv(x, F3, (1, 1), s=1, name=conv_name_base.replace('[f]', '1') + '_br2c', padding='valid', act=False)

        x = Add()([x, x_skip])
        x = LeakyReLU(alpha)(x)

        m_name = f'IdentityBlock{f}x{f}-m{self.m_id}_st{stage}_bl{block}'
        m = Model(inputs=_in, outputs=x, name=m_name)
        save_model_info(f'{self.i_dir}/resnet-blocks', m, m_name)
        return m(prev)

    def Conv(self, prev, F, k, s, name, padding='same', act=True):
        _in = Input(prev.shape[1:])
        x = Conv2D(F, k, strides=s, name=name, kernel_initializer='he_normal', padding=padding,
                   kernel_regularizer=l2(weight_decay))(_in)
        x = self.SqueezeExcite(x, 'SqueezeExcite' + name)
        x = BatchNormalization()(x)
        if act:
            x = LeakyReLU(alpha)(x)
        m = Model(inputs=_in, outputs=x, name=name)
        save_model_info(f'{self.i_dir}/conv', m, name)
        return m(prev)

    def SqueezeExcite(self, prev, name):
        _in = Input(prev.shape[1:])
        ratio = 16
        filters = _in.shape[-1]
        x = GlobalAveragePooling2D()(_in)
        x = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False,
                  kernel_regularizer=l2(weight_decay))(x)
        x = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False,
                  kernel_regularizer=l2(weight_decay))(x)
        _out = Multiply()([_in, x])
        m = Model(inputs=_in, outputs=_out, name=name)
        save_model_info(f'{self.i_dir}/se', m, name)
        return m(prev)

    def Inception_resnet(self, prev, filters, stage, block):
        _in = Input(prev.shape[1:])
        conv_name_base = f'Conv[f]x[f]-m{self.m_id}_st{stage}_bl{block}'
        F1, F2, F3, F4 = filters
        b1 = self.Conv(_in, F1, (1, 1), s=1, name=conv_name_base.replace('[f]', '1') + '_br1a')
        b1 = self.Conv(b1, F1, (3, 3), s=1, name=conv_name_base.replace('[f]', '3') + '_br1b')

        b2 = self.Conv(_in, F2, (1, 1), s=1, name=conv_name_base.replace('[f]', '1') + '_br2a')
        b2 = self.Conv(b2, F2, (3, 3), s=1, name=conv_name_base.replace('[f]', '3') + '_br2b')
        b2 = self.Conv(b2, F2, (3, 3), s=1, name=conv_name_base.replace('[f]', '3') + '_br2c')

        b3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(_in)
        b3 = self.Conv(b3, F3, (1, 1), s=1, name=conv_name_base.replace('[f]', '1') + '_br31')

        merged = Concatenate()([b1, b2, b3])  # output size W x H x f4
        merged = self.Conv(merged, F4, (1, 1), s=1, name=conv_name_base.replace('[f]', '1') + '-merged', act=False)
        x = Add()([merged, _in])
        x = LeakyReLU(alpha)(x)

        m_name = f'InceptionBlock-m{self.m_id}_st{stage}_bl{block}'
        m = Model(inputs=_in, outputs=x, name=m_name)
        save_model_info(f'{self.i_dir}/resnet-inception-blocks', m, m_name)
        return m(prev)


def save_model_info(i_dir, model, filename):
    pathlib.Path(i_dir).mkdir(exist_ok=True, parents=True)
    utils.plot_model(model, show_layer_names=True, show_shapes=True, to_file=f'{i_dir}/{filename}.png')
    with open(f'{i_dir}/{filename}.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
