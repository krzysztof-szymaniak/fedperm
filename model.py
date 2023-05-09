import pathlib
from pprint import pprint

import numpy as np
from keras import Sequential
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, BatchNormalization, ReLU, SpatialDropout2D,
    Multiply, Add, Dense, Average, SeparableConv2D, Lambda, Dropout, Flatten, DepthwiseConv2D, Reshape)
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from contextlib import redirect_stdout
from tensorflow.keras import utils

import tensorflow.keras.backend as K

# from PermutationMatrix import PermutationMatrix, matrix_penalty
from PixelShuffle import PixelShuffler
from enums import Aggregation
from prototyping import cifar_config, bigger_mnist_config  # , smaller_mnist_config,

name = 'model'


# l2_reg = 1e-4


def matrix_penalty(gamma):
    def l1_2(weights):
        N = weights.shape[0]
        s = tf.reduce_sum(tf.reduce_sum(tf.abs(weights), axis=0) - tf.sqrt(tf.reduce_sum(tf.pow(weights, 2), axis=0)))
        s = s + tf.reduce_sum(
            tf.reduce_sum(tf.abs(weights), axis=1) - tf.sqrt(tf.reduce_sum(tf.pow(weights, 2), axis=1)))
        return gamma / (N * N) * s

    return l1_2


def get_resnet(_in, m_id, i_dir, ):
    if 'cifar' in i_dir:
        return ResnetBuilder(m_id, i_dir).run(_in, cifar_config)
    elif 'fashion' in i_dir or 'letters' in i_dir:
        return ResnetBuilder(m_id, i_dir).run(_in, bigger_mnist_config)
    # elif 'mnist' in i_dir:
    #     return ResnetBuilder(m_id, i_dir).resnet(_in, smaller_mnist_config)
    else:
        raise Exception("No matching config")
    # elif 'cats' in i_dir:
    #     return ResnetBuilder(m_id, i_dir).resnet(_in, cats_dogs_config)


def aggregate(models, n_classes, aggr):
    n_out = models[0].shape[-1]
    models = [BatchNormalization()(z) for z in models]
    x = {
        Aggregation.CONCAT_STRIP.value: Concatenate,
        Aggregation.CONCATENATE.value: Concatenate,
        Aggregation.ADD.value: Add,
        Aggregation.AVERAGE.value: Average,
    }[aggr]()(models)
    # x = BatchNormalization()(x)
    # x = Dropout(0.4)(x)
    # x = Dense(n_out * len(models), activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.2)(x)
    x = Dense(n_out, activation='relu', kernel_regularizer=l2(1e-2))(x)
    # x = Dropout(0.4)(x)

    if n_classes != 2:
        x = Dense(n_classes, activation='softmax')(x)
    else:
        x = Dense(1, activation='sigmoid')(x)
    return x


class ResnetBuilder:
    def __init__(self, m_id, i_dir, ):
        self.i_dir = i_dir
        self.m_id = m_id

    def build_vgg_block(self, x, params, st):
        f = params.get('filters')
        k = params.get('kernel')
        s = params.get('stride')
        dr = params.get('dropout')
        l2_reg = params.get('l2')
        n_blocks = params.get('n_blocks')
        x = self.Conv(x, f, (k, k), s=s, block_name=f'Conv{k}x{k}-st{st}-0_m{self.m_id}', dr=dr, se=True, l2_reg=l2_reg)
        for i in range(n_blocks - 1):
            x = self.Conv(x, f, (k, k), s=1, block_name=f'Conv{k}x{k}-st{st}-{i + 1}_m{self.m_id}', dr=dr, se=True,
                          l2_reg=l2_reg)
        if params.get('maxpool'):
            x = MaxPooling2D((3, 3), strides=2)(x)
        return x

    def build_resnet_stage(self, x, params, st, v):
        f = params.get('filters')
        k = params.get('kernel')
        s = params.get('stride')
        dr = params.get('dropout')
        l2_reg = params.get('l2')
        n_blocks = params.get('n_blocks')
        for i in range(n_blocks):
            if i == 0:
                x = self.convolutional_block(x, k=k, f=f, stage=st, block=0, s=s, v=v, dr=dr, l2_reg=l2_reg)
            else:
                x = self.identity_block(x, k, f, stage=st, block=i, v=v, dr=dr, l2_reg=l2_reg)
        return x

    def store_config(self, conf):
        with open(f'{self.i_dir}/model_layers.txt', 'w') as f:
            pprint(conf, f)

    def blockwise_network(self, _in, config):
        blockwise = 2
        stages = config['stages']
        x = _in
        sr, sc = x.shape[1] // blockwise, x.shape[2] // blockwise
        blocks = []
        for row in range(blockwise):
            for col in range(blockwise):
                r_s = slice(int(row * sr), int((row + 1) * sr))
                c_s = slice(int(col * sc), int((col + 1) * sc))
                z = Lambda(lambda y: y[:, r_s, c_s, :])(x)
                z = Conv2D(256, (sr, sc), activation='relu')(z)
                # z = Reshape((1, 128))(z)
                z = Flatten()(z)
                blocks.append(z)

        x = Concatenate()(blocks)

        # x = PermutationMatrix()(x)
        n_out = blocks[0].shape[-1]
        # x = Dense(n_out * len(blocks), use_bias=False, kernel_regularizer=matrix_penalty(0.001))(x)
        x = Reshape((blockwise, blockwise, n_out))(x)
        x = PixelShuffler(size=(8, 8))(x)
        x = self.network(x, config)
        return x

    def run(self, _in, config):
        self.store_config(config)
        if config.get('blockwise'):
            return self.blockwise_network(_in, config)
        return self.network(_in, config)

    def pixel_shuffler_model(self, x, params):
        f = params.get('filters')
        dr = params.get('dropout')
        l2_reg = params.get('l2')
        k = x.shape[1]
        x = self.Conv(x, f, (k, k), s=1, block_name=f'Conv{k}x{k}-stem_m{self.m_id}', dr=dr, padding='valid',
                      l2_reg=l2_reg, se=True)
        x = PixelShuffler(size=params.get('pixel_shuffler'))(x)
        return x

    def network(self, _in, config):
        x = _in
        if config.get('stem_layer'):
            params = config['stem_layer']
            if params.get('pixel_shuffler'):
                x = self.pixel_shuffler_model(x, params)
            else:
                f = params.get('filters')
                k = params.get('kernel')
                s = params.get('stride')
                dr = params.get('dropout')
                l2_reg = params.get('l2')
                x = self.Conv(x, f, (k, k), s=s, block_name=f'Conv{k}x{k}-stem_m{self.m_id}', dr=dr, l2_reg=l2_reg)
        stages = config['stages']
        version = config['v']
        for st, params in enumerate(stages):
            if version == 0:
                x = self.build_vgg_block(x, params, st)
            else:
                x = self.build_resnet_stage(x, params, st, version)
        for layer in config['outro_layers']:
            x = eval(layer)
        return x

    def Conv(self, prev_layer, F, k, s, block_name, padding='same', act=True, bn=True, dr=None, se=False, l2_reg=None):
        if s is None:
            s = 1
        _in = Input(prev_layer.shape[1:])
        x = _in
        x = Conv2D(F, k, strides=s, name=block_name, padding=padding,
                   kernel_regularizer=l2(l2_reg) if l2_reg else None)(x)
        if se:
            x = self.SqueezeExcite(x, f'SE-{block_name}')
        if bn:
            x = BatchNormalization()(x)
        if act:
            x = ReLU()(x)
        if dr:
            x = SpatialDropout2D(dr)(x)
        m = Model(inputs=_in, outputs=x, name=block_name)
        save_model_info(f'{self.i_dir}/conv', m, block_name)
        return m(prev_layer)

    def convolutional_block(self, prev_layer, k, f, stage, block, s, v, dr, l2_reg):
        _in = Input(prev_layer.shape[1:])
        conv_name_base = f'Conv[k]x[k]-st{stage}-bl{block}-m{self.m_id}'
        x = _in

        x_skip = x
        if v == 2:
            x = self.Conv(x, f, (1, 1), s=s, block_name=conv_name_base.replace('[k]', '1') + '_br2a', padding='valid',
                          dr=dr, l2_reg=l2_reg)
            x = self.Conv(x, f, (k, k), s=1, block_name=conv_name_base.replace('[k]', str(k)) + '_br2b', dr=dr,
                          l2_reg=l2_reg)
            x = self.Conv(x, 4 * f, (1, 1), s=1, block_name=conv_name_base.replace('[k]', '1') + '_br2c',
                          padding='valid', act=False, dr=dr, l2_reg=l2_reg)
            x_skip = self.Conv(x_skip, 4 * f, (1, 1), s=s, block_name=conv_name_base.replace('[k]', '1') + '_skip',
                               padding='valid', act=False, bn=False, dr=dr, l2_reg=l2_reg)
        if v == 1:
            x = self.Conv(x, f, (k, k), s=s, block_name=conv_name_base.replace('[k]', str(k)) + '_br2a', dr=dr,
                          l2_reg=l2_reg)
            x = self.Conv(x, f, (k, k), s=1, block_name=conv_name_base.replace('[k]', str(k)) + '_br2b', act=False,
                          dr=dr, l2_reg=l2_reg)

            x_skip = self.Conv(x_skip, f, (1, 1), s=s, block_name=conv_name_base.replace('[k]', '1') + '_skip',
                               padding='valid', act=False, bn=False, dr=dr, l2_reg=l2_reg)

        x = self.SqueezeExcite(x, f'SqueezeExcite-st{stage}-bl{block}-m{self.m_id}')
        x = Add()([x, x_skip])
        # x = self.SqueezeExcite(x, f'SqueezeExcite-st{stage}-bl{block}-m{self.m_id}')
        x = BatchNormalization()(x)
        x = ReLU()(x)

        m_name = f'ConvBlock{k}x{k}-st{stage}-bl{block}-m{self.m_id}'
        m = Model(inputs=_in, outputs=x, name=m_name)
        save_model_info(f'{self.i_dir}/resnet-blocks', m, m_name)
        return m(prev_layer)

    def identity_block(self, prev_layer, k, f, stage, block, v, dr, l2_reg):
        _in = Input(prev_layer.shape[1:])
        conv_name_base = f'Conv[k]x[k]-st{stage}-bl{block}-m{self.m_id}'
        x = _in
        x_skip = x
        if v == 2:
            x = self.Conv(x, f, (1, 1), s=1, block_name=conv_name_base.replace('[k]', '1') + '_br2a', padding='valid',
                          dr=dr, l2_reg=l2_reg)
            x = self.Conv(x, f, (k, k), s=1, block_name=conv_name_base.replace('[k]', str(k)) + '_br2b', dr=dr,
                          l2_reg=l2_reg
                          )
            x = self.Conv(x, 4 * f, (1, 1), s=1, block_name=conv_name_base.replace('[k]', '1') + '_br2c',
                          padding='valid', dr=dr, act=False)
        if v == 1:
            x = self.Conv(x, f, (k, k), s=1, block_name=conv_name_base.replace('[k]', str(k)) + '_br2a', dr=dr,
                          l2_reg=l2_reg)
            x = self.Conv(x, f, (k, k), s=1, block_name=conv_name_base.replace('[k]', str(k)) + '_br2b', act=False,
                          dr=dr, l2_reg=l2_reg)

        x = self.SqueezeExcite(x, f'SqueezeExcite-_st{stage}_bl{block}-m{self.m_id}')
        x = Add()([x, x_skip])
        x = BatchNormalization()(x)
        x = ReLU()(x)

        m_name = f'IdentityBlock{k}x{k}_st{stage}_bl{block}_m{self.m_id}'
        m = Model(inputs=_in, outputs=x, name=m_name)
        save_model_info(f'{self.i_dir}/resnet-blocks', m, m_name)
        return m(prev_layer)

    def SqueezeExcite(self, prev_layer, block_name):
        _in = Input(prev_layer.shape[1:])
        ratio = 16
        filters = _in.shape[-1]
        x = GlobalAveragePooling2D()(_in)
        x = Dense(filters // ratio, activation='relu', use_bias=False, )(x)
        x = Dense(filters, activation='sigmoid', use_bias=False, )(x)
        _out = Multiply()([_in, x])
        m = Model(inputs=_in, outputs=_out, name=block_name)
        save_model_info(f'{self.i_dir}/se', m, block_name)
        return m(prev_layer)

    def Inception_resnet(self, prev_layer, filters, stage, block):
        _in = Input(prev_layer.shape[1:])
        conv_name_base = f'Conv[k]x[k]-st{stage}-bl{block}-m{self.m_id}'
        F1, F2, F3, F4 = filters
        b1 = self.Conv(_in, F1, (1, 1), s=1, block_name=conv_name_base.replace('[k]', '1') + '_br1a')
        b1 = self.Conv(b1, F1, (3, 3), s=1, block_name=conv_name_base.replace('[k]', '3') + '_br1b', )

        b2 = self.Conv(_in, F2, (1, 1), s=1, block_name=conv_name_base.replace('[k]', '1') + '_br2a')
        b2 = self.Conv(b2, F2, (5, 5), s=1, block_name=conv_name_base.replace('[k]', '5') + '_br2b', )

        b3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(_in)
        b3 = self.Conv(b3, F3, (1, 1), s=1, block_name=conv_name_base.replace('[k]', '1') + '_br31')

        merged = Concatenate()([b1, b2, b3])
        merged = self.Conv(merged, F4, (1, 1), s=1, block_name=conv_name_base.replace('[k]', '1') + '-merged',
                           act=False)
        merged = self.SqueezeExcite(merged, f'SqueezeExcite-_st{stage}_bl{block}-m{self.m_id}')
        x = Add()([merged, _in])
        x = ReLU()(x)

        m_name = f'InceptionBlock-st{stage}-bl{block}-m{self.m_id}'
        m = Model(inputs=_in, outputs=x, name=m_name)
        save_model_info(f'{self.i_dir}/resnet-inception-blocks', m, m_name)
        return m(prev_layer)


def save_model_info(i_dir, model, filename):
    pathlib.Path(i_dir).mkdir(exist_ok=True, parents=True)
    utils.plot_model(model, show_layer_names=True, show_shapes=True, to_file=f'{i_dir}/{filename}.png')
    with open(f'{i_dir}/{filename}.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()


class ModelFactory:
    def __init__(self, i_dir, n_classes, n_frames, input_shape, aggr):
        self.n_frames = n_frames
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.i_dir = i_dir
        self.model_type = None
        self.m_id = None
        self.aggr = aggr

    def get_model(self, model_type, n_outputs, m_id=None):
        self.model_type = model_type
        if m_id is not None:
            self.m_id = m_id
        return {
            'single': self.get_single,
            'parallel': self.get_composite_model,
        }[model_type](n_outputs)

    def get_single(self, n_outputs):
        _in = Input(shape=self.input_shape)
        assert self.m_id is not None
        _out = self.sequential_submodel(_in, n_outputs, self.m_id)
        m_name = f'{name}_{self.m_id}'
        model = Model(inputs=_in, outputs=_out, name=m_name)
        save_model_info(self.i_dir, model, name)
        model.summary()
        return model

    def get_composite_model(self, n_outputs):
        _ins = [Input(shape=self.input_shape) for _ in range(self.n_frames)]
        model_outputs = []
        for i, _in in enumerate(_ins):
            _out = self.parallel_submodel(_in, n_outputs, i)
            m_name = f'{name}_{i}'
            submodel = Model(inputs=_in, outputs=_out, name=m_name)
            if i == 0:
                save_model_info(self.i_dir, submodel, name)
            submodel.summary()
            model_outputs.append(submodel)
        outs = [m(_in) for m, _in in zip(model_outputs, _ins)]
        model = self.get_aggregating_model(_ins, outs, self.model_type)
        return model

    def get_aggregating_model(self, inputs, outs, model_name):
        _out = aggregate(outs, self.n_classes, self.aggr)
        model = Model(inputs=inputs, outputs=_out, name=model_name)
        model.summary()
        save_model_info(self.i_dir, model, model_name)
        return model

    def sequential_submodel(self, _in, n_outputs, m_id):
        x = get_resnet(_in, m_id, self.i_dir)
        if n_outputs != 2:
            x = Dense(n_outputs, activation='softmax', )(x)
        else:
            x = Dense(1, activation='sigmoid', )(x)
        return x

    def parallel_submodel(self, _in, _, m_id):
        x = get_resnet(_in, m_id, self.i_dir)
        return x
