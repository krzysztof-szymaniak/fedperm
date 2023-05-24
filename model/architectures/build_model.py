import os
import pathlib
import shutil
from pprint import pprint

from keras.regularizers import l1_l2
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    MaxPooling2D, Concatenate, BatchNormalization, Add, Dense, Average, Dropout, GlobalAveragePooling2D)

from enums import Aggregation
from model.architectures.blocks.PixelShuffle import PixelShuffler
from model.architectures.blocks.basic import Conv
from model.architectures.blocks.conv_mixer import conv_mixer_block
from model.architectures.blocks.inception import Inception
from model.architectures.blocks.resnet import convolutional_block, identity_block
from model.architectures.get_configuration import get_config
from model.visualisation import plot_model

# name = 'model'
_ = GlobalAveragePooling2D


def get_model(model_type, arch_dir, sub_input_shape, n_classes, m_id, ds_name):
    _in = Input(shape=sub_input_shape)
    x = network(_in, model_type, ds_name, m_id, arch_dir)
    _out = Dense(n_classes, activation='softmax')(x) if n_classes != 2 else Dense(1, activation='sigmoid', )(x)
    name = model_type
    m_name = f'{name}_{m_id}'
    model = Model(inputs=_in, outputs=_out, name=m_name)
    plot_model(arch_dir, model, name)
    model.summary()
    return model


def network(_in, model_type, ds_name, m_id, i_dir):
    if os.path.exists(i_dir):
        shutil.rmtree(i_dir)
        pathlib.Path(i_dir).mkdir()

    config = get_config(model_type)
    return builder(_in, config, m_id, i_dir)


def aggregate(models, n_classes, aggr):
    n_out = models[0].shape[-1]
    models = [BatchNormalization()(z) for z in models]
    x = {
        Aggregation.STRIP_CONCAT: Concatenate,
        Aggregation.ADD: Add,
        Aggregation.AVERAGE: Average,
    }[aggr]()(models)
    # x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(n_out, activation='relu', kernel_regularizer=l1_l2(1e-4, 1e-4))(x)
    x = Dropout(0.4)(x)
    x = Dense(n_classes, activation='softmax')(x) if n_classes > 2 else Dense(1, activation='sigmoid')(x)
    return x


def build_vgg_block(x, params, st, m_id, i_dir):
    f = params.get('filters')
    k = params.get('kernel')
    s = params.get('stride')
    dr = params.get('dropout')
    n_blocks = params.get('n_blocks')
    x = Conv(x, f, (k, k), s=s, block_name=f'Conv{k}x{k}-st{st}-0_m{m_id}', dr=dr, se=True, i_dir=i_dir)
    for i in range(n_blocks - 1):
        x = Conv(x, f, (k, k), s=1, block_name=f'Conv{k}x{k}-st{st}-{i + 1}_m{m_id}', dr=dr, se=True, i_dir=i_dir)
    if params.get('maxpool'):
        x = MaxPooling2D((3, 3), strides=2)(x)
    return x


def build_resnet_stage(x, params, st, v, m_id, i_dir):
    f = params.get('filters')
    k = params.get('kernel')
    s = params.get('stride')
    dr = params.get('dropout')
    n_blocks = params.get('n_blocks')
    for i in range(n_blocks):
        if i == 0:
            x = convolutional_block(x, kernel=k, f=f, stage=st, block=0, s=s, v=v, dr=dr, m_id=m_id,
                                    i_dir=i_dir)
        else:
            x = identity_block(x, k, f, stage=st, block=i, v=v, dr=dr, m_id=m_id, i_dir=i_dir)
    return x


def build_inception_stage(x, params, st, m_id, i_dir):
    f = params.get('filters')
    if type(f) != list:
        return build_vgg_block(x, params, st, m_id, i_dir)
    dr = params.get('dropout')
    n_blocks = params.get('n_blocks')
    for i in range(n_blocks):
        x = Inception(x, f, st, i, dr, i_dir=i_dir, m_id=m_id)
    if params.get('maxpool'):
        x = MaxPooling2D((3, 3), strides=2)(x)
    return x


def store_config(i_dir, conf):
    with open(f'{i_dir}/model_layers.txt', 'w') as f:
        pprint(conf, f)


def conv_and_upscale(x, params, m_id, i_dir):
    f = params.get('filters')
    k = params.get('kernel', x.shape[1])
    s = params.get('stride', 1)
    dr = params.get('dropout')
    padding = params.get('padding', 'valid')

    x = Conv(x, f, (k, k), s=s, se=True,
             block_name=f'Conv{k}x{k}-adaptation_m{m_id}', dr=dr, padding=padding, i_dir=i_dir)
    x = PixelShuffler(size=params.get('upscale_factor'))(x)
    return x


def build_conv_mixer_block(x, params, st, m_id, i_dir):
    f = params.get('filters')
    k = params.get('kernel')
    dr = params.get('dropout')
    n_blocks = params.get('n_blocks')
    for i in range(n_blocks):
        x = conv_mixer_block(x, f, k, i + 1, m_id, i_dir, dr=dr)
    return x


def builder(_in, config, m_id, i_dir):
    x = _in
    if config.get('stem_layer'):
        params = config['stem_layer']
        if params.get('upscale_factor'):
            x = conv_and_upscale(x, params, m_id, i_dir)
        else:
            f = params.get('filters')
            k = params.get('kernel')
            s = params.get('stride')
            dr = params.get('dropout')
            padding = params.get('padding', 'same')
            x = Conv(x, f, (k, k), s=s, block_name=f'Conv{k}x{k}-adaptation_m{m_id}',
                     dr=dr, i_dir=i_dir, padding=padding, se=True)
    stages = config['stages']
    version = config['v']
    for st, params in enumerate(stages):
        if version == 'vgg':
            x = build_vgg_block(x, params, st, m_id, i_dir)
        elif version in [1, 2]:
            x = build_resnet_stage(x, params, st, version, m_id, i_dir)
        elif version == 'inception':
            x = build_inception_stage(x, params, st, m_id, i_dir)
        elif version == 'conv-mixer':
            x = build_conv_mixer_block(x, params, st, m_id, i_dir)
    for layer in config['outro_layers']:
        x = eval(layer)
    return x
