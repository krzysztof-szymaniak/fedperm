import os
import pathlib
import shutil
from pprint import pprint

from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Concatenate, Add, Dense, Average, Dropout, GlobalAveragePooling2D, BatchNormalization
)

from enums import Aggregation
from model.architectures.blocks.basic import ConvBlock
from model.architectures.blocks.conv_mixer import ConvMixerBlock
from model.architectures.model_configs import get_config
from model.train_configs import BATCH_SIZE
from model.visualisation import plot_model


def get_model(model_type, arch_dir, sub_input_shape, n_classes, m_id):
    name = model_type.name.lower()
    _in = Input(shape=sub_input_shape)
    x = network(_in, model_type, m_id, arch_dir)
    _out = Dense(n_classes, activation='softmax')(x) if n_classes != 2 else Dense(1, activation='sigmoid')(x)
    model = Model(inputs=_in, outputs=_out, name=f'{name}_{m_id}')
    plot_model(arch_dir, model, name)
    return model


def network(_in, model_type, m_id, i_dir):
    if os.path.exists(i_dir):
        shutil.rmtree(i_dir)
        pathlib.Path(i_dir).mkdir()

    config = get_config(model_type)
    store_config(i_dir, config)
    return builder(_in, config, m_id, i_dir)


def aggregate(models, n_classes, aggr):
    # n_out = models[0].shape[-1] // 2
    models = [BatchNormalization()(z) for z in models]
    x = {
        Aggregation.STRIP_CONCAT: Concatenate,
        Aggregation.CONCAT: Concatenate,
        Aggregation.ADD: Add,
        Aggregation.AVERAGE: Average,
    }[aggr]()(models)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x) if n_classes > 2 else Dense(1, activation='sigmoid')(x)
    return x


def store_config(i_dir, conf):
    with open(f'{i_dir}/model_layers.txt', 'w') as f:
        print(f'{BATCH_SIZE=}', file=f)
        pprint(conf, f)


def build_conv_mixer_block(x, params, st, m_id, i_dir):
    f = params.get('filters')
    k = params.get('kernel')
    dr = params.get('dropout')
    n_blocks = params.get('n_blocks')
    for i in range(n_blocks):
        conv_mixer = ConvMixerBlock(f, k, x.shape, block_name=f"ConvMixer{k}x{k}-st{st}-m{m_id}", dr=dr)
        if i == 0:
            conv_mixer.plot_layer(x.shape, i_dir)
        x = conv_mixer(x)
    return x


def builder(_in, config, m_id, i_dir):
    x = _in
    if config.get('stem_layer'):
        params = config['stem_layer']
        f = params.get('filters')
        k = params.get('kernel')
        s = params.get('stride')
        dr = params.get('dropout')
        padding = params.get('padding', 'same')
        conv = ConvBlock(x.shape, f, k, s, block_name=f'Conv{k}x{k}-adaptation_m{m_id}', dr=dr, padding=padding)
        conv.plot_layer(x.shape, i_dir)
        x = conv(x)
    stages = config['stages']
    version = config['v']
    for st, params in enumerate(stages):
        # if version == 'vgg':
        #     x = build_vgg_block(x, params, st, m_id, i_dir)
        # elif version in [1, 2]:
        #     x = build_resnet_stage(x, params, st, version, m_id, i_dir)
        # elif version == 'inception':
        #     x = build_inception_stage(x, params, st, m_id, i_dir)
        if version == 'conv-mixer':
            x = build_conv_mixer_block(x, params, st, m_id, i_dir)
    for layer in config['outro_layers']:
        x = eval(layer)
    return x


_ = GlobalAveragePooling2D  # import to use in eval()
