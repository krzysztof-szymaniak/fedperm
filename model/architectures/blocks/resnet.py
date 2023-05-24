from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, ReLU, Add

from model.architectures.blocks.basic import Conv, SqueezeExcite
from model.visualisation import plot_model, VISUALIZE_IN_SEGMENTS


def convolutional_block(prev_layer, kernel, f, stage, block, s, v, m_id, i_dir, dr):
    _in = Input(prev_layer.shape[1:]) if VISUALIZE_IN_SEGMENTS else prev_layer
    conv_name_base = f'Conv[k]x[k]-st{stage}-bl{block}-m{m_id}'
    x = _in

    x_skip = x
    if v == 2:
        x = Conv(x, f, (1, 1), s=s, padding='valid',
                 block_name=conv_name_base.replace('[k]', '1') + '_br2a', i_dir=i_dir, dr=dr)
        x = Conv(x, f, (kernel, kernel), s=1,
                 block_name=conv_name_base.replace('[k]', str(kernel)) + '_br2b', i_dir=i_dir, dr=dr)
        x = Conv(x, 4 * f, (1, 1), s=1, padding='valid', act=False,
                 block_name=conv_name_base.replace('[k]', '1') + '_br2c', i_dir=i_dir, dr=dr)
        x_skip = Conv(x_skip, 4 * f, (1, 1), s=s, padding='valid', act=False, bn=False,
                      block_name=conv_name_base.replace('[k]', '1') + '_skip', i_dir=i_dir, dr=dr)
    if v == 1:
        x = Conv(x, f, (kernel, kernel), s=s,
                 block_name=conv_name_base.replace('[k]', str(kernel)) + '_br2a', i_dir=i_dir, dr=dr)
        x = Conv(x, f, (kernel, kernel), s=1, act=False,
                 block_name=conv_name_base.replace('[k]', str(kernel)) + '_br2b', i_dir=i_dir, dr=dr)

        x_skip = Conv(x_skip, f, (1, 1), s=s, padding='valid', act=False, bn=False,
                      block_name=conv_name_base.replace('[k]', '1') + '_skip', i_dir=i_dir, dr=dr)
    m_name = f'ConvBlock{kernel}x{kernel}-st{stage}-bl{block}-m{m_id}'
    x = SqueezeExcite(x.shape[-1], i_dir, f'SE-{m_name}')(x)
    x = Add()([x, x_skip])
    x = BatchNormalization()(x)
    x = ReLU()(x)
    if VISUALIZE_IN_SEGMENTS:
        m = Model(inputs=_in, outputs=x, name=m_name)
        plot_model(f'{i_dir}/resnet-blocks', m, m_name)
        return m(prev_layer)
    return x


def identity_block(prev_layer, kernel, filters, stage, block, v, m_id, i_dir, dr):
    _in = Input(prev_layer.shape[1:]) if VISUALIZE_IN_SEGMENTS else prev_layer
    conv_name_base = f'Conv[k]x[k]-st{stage}-bl{block}-m{m_id}'
    x = _in
    x_skip = x
    if v == 2:
        x = Conv(x, filters, (1, 1), s=1, padding='valid',
                 block_name=conv_name_base.replace('[k]', '1') + '_br2a', i_dir=i_dir, dr=dr)
        x = Conv(x, filters, (kernel, kernel), s=1,
                 block_name=conv_name_base.replace('[k]', str(kernel)) + '_br2b', i_dir=i_dir, dr=dr)
        x = Conv(x, 4 * filters, (1, 1), s=1, padding='valid', act=False,
                 block_name=conv_name_base.replace('[k]', '1') + '_br2c', i_dir=i_dir, dr=dr)
    if v == 1:
        x = Conv(x, filters, (kernel, kernel), s=1,
                 block_name=conv_name_base.replace('[k]', str(kernel)) + '_br2a', i_dir=i_dir, dr=dr)
        x = Conv(x, filters, (kernel, kernel), s=1, act=False,
                 block_name=conv_name_base.replace('[k]', str(kernel)) + '_br2b', i_dir=i_dir, dr=dr)

    m_name = f'IdentityBlock{kernel}x{kernel}-st{stage}-bl{block}-m{m_id}'
    x = SqueezeExcite(x.shape[-1], i_dir, f'SE-{m_name}')(x)
    x = Add()([x, x_skip])
    x = BatchNormalization()(x)
    x = ReLU()(x)
    if VISUALIZE_IN_SEGMENTS:
        m = Model(inputs=_in, outputs=x, name=m_name)
        plot_model(f'{i_dir}/resnet-blocks', m, m_name)
        return m(prev_layer)
    return x
