from pprint import pprint

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, BatchNormalization, ReLU, SpatialDropout2D,
    Multiply, Add, Dense, Average, Dropout, Flatten, UpSampling2D)
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2, l1_l2
from ConvMixer import ConvMixer
from PixelShuffle import PixelShuffler
from enums import Aggregation, ModelType
from layers import Conv, convolutional_block, identity_block, Inception
from prototyping import cifar_model, mnist_model
from transformer import Transformer
from utils import save_model_info

name = 'model'


def get_resnet(_in, m_id, i_dir, model_type):
    patch_size = int(i_dir.split('perm-bs_')[1].split('/')[0].split("_")[0])
    if model_type == ModelType.MLP.value:
        return ResnetBuilder(m_id, i_dir).build_mlp(_in)
    if model_type == ModelType.CONV_MIXER.value:
        if 'cifar' in i_dir or 'eurosat' in i_dir:
            filters = 196  # 128
            depth = 6
        elif 'mnist' in i_dir:
            filters = 64
            depth = 5
        else:
            filters = 256
            depth = 10
        return ConvMixer(_in, filters=filters, depth=depth, patch_size=patch_size, kernel_size=3, upscale=(2, 2), m_id=m_id, i_dir=i_dir)

    if model_type == ModelType.VISION_TRANSFORMER.value:
        if 'cifar' in i_dir or 'eurosat' in i_dir:
            dim = 32
            depth = 6
            n_heads = 6
            mlp_head = [256]

        else:
            dim = 16
            depth = 4
            n_heads = 4
            mlp_head = [128]
        return Transformer(_in,
                           n_dims=dim, n_layers=depth, mlp_head=mlp_head,
                           m_id=m_id, i_dir=i_dir, patch_size=patch_size, n_heads=n_heads)

    if 'cifar' in i_dir or 'eurosat' in i_dir:
        cifar_config = cifar_model[model_type]
        return ResnetBuilder(m_id, i_dir).run(_in, cifar_config)
    elif 'fashion' in i_dir or 'letters' in i_dir or 'mnist' in i_dir:
        mnist_config = mnist_model[model_type]
        return ResnetBuilder(m_id, i_dir).run(_in, mnist_config)
    else:
        raise Exception("No matching config")


def aggregate(models, n_classes, aggr):
    n_out = models[0].shape[-1]
    models = [BatchNormalization()(z) for z in models]
    x = {
        Aggregation.CONCATENATE.value: Concatenate,
        Aggregation.ADD.value: Add,
        Aggregation.AVERAGE.value: Average,
    }[aggr]()(models)
    # x = BatchNormalization()(x)

    x = Dropout(0.4)(x)
    x = Dense(n_out, activation='relu', kernel_regularizer=l1_l2(1e-4, 1e-4))(x)
    x = Dropout(0.4)(x)

    if n_classes != 2:
        x = Dense(n_classes, activation='softmax')(x)
    else:
        x = Dense(1, activation='sigmoid')(x)
    return x


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu, kernel_regularizer=l1_l2(1e-4, 1e-4))(x)
        x = Dropout(dropout_rate)(x)
    return x


class ResnetBuilder:
    def __init__(self, m_id, i_dir, ):
        self.i_dir = i_dir
        self.m_id = m_id

    def build_mlp(self, x):
        k = x.shape[1]
        x = Conv(x, 256, (k, k), s=1, block_name=f'Conv{k}x{k}-stem_m{self.m_id}', padding='valid', l2_reg=1e-4,
                 se=True, i_dir=self.i_dir)
        x = Flatten()(x)
        x = mlp(x, [512, 256], dropout_rate=0.4)
        return x

    def build_vgg_block(self, x, params, st):
        f = params.get('filters')
        k = params.get('kernel')
        s = params.get('stride')
        dr = params.get('dropout')
        l2_reg = params.get('l2')
        n_blocks = params.get('n_blocks')
        x = Conv(x, f, (k, k), s=s, block_name=f'Conv{k}x{k}-st{st}-0_m{self.m_id}', dr=dr, se=True, l2_reg=l2_reg,
                 i_dir=self.i_dir)
        for i in range(n_blocks - 1):
            x = Conv(x, f, (k, k), s=1, block_name=f'Conv{k}x{k}-st{st}-{i + 1}_m{self.m_id}', dr=dr, se=True,
                     l2_reg=l2_reg, i_dir=self.i_dir)
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
                x = convolutional_block(x, k=k, f=f, stage=st, block=0, s=s, v=v, dr=dr, l2_reg=l2_reg, m_id=self.m_id,
                                        i_dir=self.i_dir)
            else:
                x = identity_block(x, k, f, stage=st, block=i, v=v, dr=dr, l2_reg=l2_reg, m_id=self.m_id,
                                   i_dir=self.i_dir)
        return x

    def build_inception_stage(self, x, params, st):
        f = params.get('filters')
        if type(f) != list:
            return self.build_vgg_block(x, params, st)
        dr = params.get('dropout')
        l2_reg = params.get('l2')
        n_blocks = params.get('n_blocks')
        for i in range(n_blocks):
            x = Inception(x, f, st, i, dr, l2_reg, i_dir=self.i_dir, m_id=self.m_id)
        if params.get('maxpool'):
            x = MaxPooling2D((3, 3), strides=2)(x)
        return x

    def store_config(self, conf):
        with open(f'{self.i_dir}/model_layers.txt', 'w') as f:
            pprint(conf, f)

    def run(self, _in, config):
        self.store_config(config)
        return self.network(_in, config)

    def pixel_shuffler_model(self, x, params, ps=True):
        f = params.get('filters')
        l2_reg = params.get('l2', 1e-4)
        k = params.get('kernel', x.shape[1])
        s = params.get('stride', 1)
        dr = params.get('dropout')
        padding = params.get('padding', 'valid')

        x = Conv(x, f, (k, k), s=s, block_name=f'Conv{k}x{k}-adaptation_m{self.m_id}', dr=dr, padding=padding,
                 l2_reg=l2_reg, se=True, i_dir=self.i_dir)
        if ps:
            x = PixelShuffler(size=params.get('upscale_factor'))(x)
        else:
            x = UpSampling2D(size=params.get('upscale_factor'))(x)
        return x

    def network(self, _in, config):
        x = _in
        if config.get('stem_layer'):
            params = config['stem_layer']
            if params.get('upscale_factor'):
                x = self.pixel_shuffler_model(x, params)
            else:
                f = params.get('filters')
                k = params.get('kernel')
                s = params.get('stride')
                dr = params.get('dropout')
                l2_reg = params.get('l2')
                padding = params.get('padding', 'same')
                x = Conv(x, f, (k, k), s=s, block_name=f'Conv{k}x{k}-adaptation_m{self.m_id}', dr=dr, l2_reg=l2_reg,
                         i_dir=self.i_dir, padding=padding, se=True)
        stages = config['stages']
        version = config['v']
        for st, params in enumerate(stages):
            if version == 'vgg':
                x = self.build_vgg_block(x, params, st)
            elif version in [1, 2]:
                x = self.build_resnet_stage(x, params, st, version)
            elif version == 'inception':
                x = self.build_inception_stage(x, params, st)
        for layer in config['outro_layers']:
            x = eval(layer)
        return x


class ModelFactory:
    def __init__(self, i_dir, n_classes, n_frames, input_shape, aggr):
        self.n_frames = n_frames
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.i_dir = i_dir
        self.model_type = None
        self.m_id = None
        self.aggr = aggr

    def get_model(self, model_type, n_outputs, model_arch, m_id=None):
        self.model_type = model_type
        if m_id is not None:
            self.m_id = m_id
        return {
            'single': self.get_single,
            'parallel': self.get_composite_model,
        }[model_type](n_outputs, model_arch)

    def get_single(self, n_outputs, model_arch):
        _in = Input(shape=self.input_shape)
        assert self.m_id is not None
        _out = self.sequential_submodel(_in, n_outputs, self.m_id, model_arch)
        m_name = f'{name}_{self.m_id}'
        model = Model(inputs=_in, outputs=_out, name=m_name)
        save_model_info(self.i_dir, model, name)
        model.summary()
        return model

    def get_composite_model(self, n_outputs, model_arch):
        _ins = [Input(shape=self.input_shape) for _ in range(self.n_frames)]
        model_outputs = []
        for i, _in in enumerate(_ins):
            _out = self.parallel_submodel(_in, n_outputs, i, model_arch)
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

    def sequential_submodel(self, _in, n_outputs, m_id, model_arch):
        x = get_resnet(_in, m_id, self.i_dir, model_arch)
        if n_outputs != 2:
            x = Dense(n_outputs, activation='softmax', )(x)
        else:
            x = Dense(1, activation='sigmoid', )(x)
        return x

    def parallel_submodel(self, _in, _, m_id, model_arch):
        x = get_resnet(_in, m_id, self.i_dir, model_arch)
        return x
