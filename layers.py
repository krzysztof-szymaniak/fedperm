from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, BatchNormalization, ReLU, SpatialDropout2D,
    Multiply, Add, Dense, Average, Dropout, Flatten)
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2, l1_l2
from utils import save_model_info


def Conv(prev_layer, F, k, s, block_name, padding='same', act=True, bn=True, dr=None, se=False, l2_reg=None,
         i_dir=None):
    if s is None:
        s = 1
    _in = Input(prev_layer.shape[1:])
    x = _in
    x = Conv2D(F, k, strides=s, name=block_name, padding=padding, use_bias=False,
               kernel_regularizer=l2(l2_reg) if l2_reg else None)(x)
    if se:
        x = SqueezeExcite(x, f'SE-{block_name}', i_dir)
    if bn:
        x = BatchNormalization()(x)
    if act:
        x = ReLU()(x)
    if dr:
        x = SpatialDropout2D(dr)(x)
    m = Model(inputs=_in, outputs=x, name=block_name)
    save_model_info(f'{i_dir}/conv', m, block_name)
    return m(prev_layer)


def SqueezeExcite(prev_layer, block_name, i_dir):
    _in = Input(prev_layer.shape[1:])
    ratio = 16
    filters = _in.shape[-1]
    x = GlobalAveragePooling2D()(_in)
    x = Dense(filters // ratio, activation='relu', use_bias=False, )(x)
    x = Dense(filters, activation='sigmoid', use_bias=False, )(x)
    _out = Multiply()([_in, x])
    m = Model(inputs=_in, outputs=_out, name=block_name)
    save_model_info(f'{i_dir}/se', m, block_name)
    return m(prev_layer)


def convolutional_block(prev_layer, k, f, stage, block, s, v, dr, l2_reg, m_id, i_dir):
    _in = Input(prev_layer.shape[1:])
    conv_name_base = f'Conv[k]x[k]-st{stage}-bl{block}-m{m_id}'
    x = _in

    x_skip = x
    if v == 2:
        x = Conv(x, f, (1, 1), s=s, block_name=conv_name_base.replace('[k]', '1') + '_br2a', padding='valid',
                 dr=dr, l2_reg=l2_reg, i_dir=i_dir)
        x = Conv(x, f, (k, k), s=1, block_name=conv_name_base.replace('[k]', str(k)) + '_br2b', dr=dr,
                 l2_reg=l2_reg, i_dir=i_dir)
        x = Conv(x, 4 * f, (1, 1), s=1, block_name=conv_name_base.replace('[k]', '1') + '_br2c',
                 padding='valid', act=False, dr=dr, l2_reg=l2_reg, i_dir=i_dir)
        x_skip = Conv(x_skip, 4 * f, (1, 1), s=s, block_name=conv_name_base.replace('[k]', '1') + '_skip',
                      padding='valid', act=False, bn=False, dr=dr, l2_reg=l2_reg, i_dir=i_dir)
    if v == 1:
        x = Conv(x, f, (k, k), s=s, block_name=conv_name_base.replace('[k]', str(k)) + '_br2a', dr=dr,
                 l2_reg=l2_reg, i_dir=i_dir)
        x = Conv(x, f, (k, k), s=1, block_name=conv_name_base.replace('[k]', str(k)) + '_br2b', act=False,
                 dr=dr, l2_reg=l2_reg, i_dir=i_dir)

        x_skip = Conv(x_skip, f, (1, 1), s=s, block_name=conv_name_base.replace('[k]', '1') + '_skip',
                      padding='valid', act=False, bn=False, dr=dr, l2_reg=l2_reg, i_dir=i_dir, )

    x = SqueezeExcite(x, f'SqueezeExcite-st{stage}-bl{block}-m{m_id}', i_dir)
    x = Add()([x, x_skip])
    x = BatchNormalization()(x)
    x = ReLU()(x)

    m_name = f'ConvBlock{k}x{k}-st{stage}-bl{block}-m{m_id}'
    m = Model(inputs=_in, outputs=x, name=m_name)
    save_model_info(f'{i_dir}/resnet-blocks', m, m_name)
    return m(prev_layer)


def identity_block(prev_layer, k, f, stage, block, v, dr, l2_reg, m_id, i_dir):
    _in = Input(prev_layer.shape[1:])
    conv_name_base = f'Conv[k]x[k]-st{stage}-bl{block}-m{m_id}'
    x = _in
    x_skip = x
    if v == 2:
        x = Conv(x, f, (1, 1), s=1, block_name=conv_name_base.replace('[k]', '1') + '_br2a', padding='valid',
                 dr=dr, l2_reg=l2_reg, i_dir=i_dir)
        x = Conv(x, f, (k, k), s=1, block_name=conv_name_base.replace('[k]', str(k)) + '_br2b', dr=dr,
                 l2_reg=l2_reg, i_dir=i_dir)
        x = Conv(x, 4 * f, (1, 1), s=1, block_name=conv_name_base.replace('[k]', '1') + '_br2c',
                 padding='valid', dr=dr, act=False, i_dir=i_dir)
    if v == 1:
        x = Conv(x, f, (k, k), s=1, block_name=conv_name_base.replace('[k]', str(k)) + '_br2a', dr=dr,
                 l2_reg=l2_reg, i_dir=i_dir)
        x = Conv(x, f, (k, k), s=1, block_name=conv_name_base.replace('[k]', str(k)) + '_br2b', act=False,
                 dr=dr, l2_reg=l2_reg, i_dir=i_dir)

    x = SqueezeExcite(x, f'SqueezeExcite-_st{stage}_bl{block}-m{m_id}', i_dir)
    x = Add()([x, x_skip])
    x = BatchNormalization()(x)
    x = ReLU()(x)

    m_name = f'IdentityBlock{k}x{k}_st{stage}_bl{block}_m{m_id}'
    m = Model(inputs=_in, outputs=x, name=m_name)
    save_model_info(f'{i_dir}/resnet-blocks', m, m_name)
    return m(prev_layer)


def Inception(prev_layer, filters, stage, block, dr, l2_reg, m_id, i_dir):
    _in = Input(prev_layer.shape[1:])
    conv_name_base = f'Conv[k]x[k]-st{stage}-bl{block}-m{m_id}'
    F1, F2, F3, F4 = filters
    # F4 = sum(filters)
    b1 = Conv(_in, F1, (1, 1), s=1, block_name=conv_name_base.replace('[k]', '1') + '_br1a', dr=dr,
              l2_reg=l2_reg, i_dir=i_dir)
    b1 = Conv(b1, F1, (3, 3), s=1, block_name=conv_name_base.replace('[k]', '3') + '_br1b', dr=dr,
              l2_reg=l2_reg, i_dir=i_dir)

    b2 = Conv(_in, F2, (1, 1), s=1, block_name=conv_name_base.replace('[k]', '1') + '_br2a', dr=dr,
              l2_reg=l2_reg, i_dir=i_dir)
    b2 = Conv(b2, F2, (5, 5), s=1, block_name=conv_name_base.replace('[k]', '5') + '_br2b', dr=dr,
              l2_reg=l2_reg, i_dir=i_dir)

    b3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(_in)
    b3 = Conv(b3, F3, (1, 1), s=1, block_name=conv_name_base.replace('[k]', '1') + '_br31', dr=dr,
              l2_reg=l2_reg, i_dir=i_dir)

    b4 = Conv(_in, F4, (1, 1), s=1, block_name=conv_name_base.replace('[k]', '1') + '_br4', dr=dr,
              l2_reg=l2_reg, i_dir=i_dir)

    x = Concatenate()([b1, b2, b3, b4])
    x = SqueezeExcite(x, f'SqueezeExcite-_st{stage}_bl{block}-m{m_id}', i_dir)

    m_name = f'InceptionBlock-st{stage}-bl{block}-m{m_id}'
    m = Model(inputs=_in, outputs=x, name=m_name)
    save_model_info(f'{i_dir}/resnet-inception-blocks', m, m_name)
    return m(prev_layer)
