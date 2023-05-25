from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    MaxPooling2D, Concatenate)

from model.architectures.blocks.basic import Conv, SqueezeExcite
from model.visualisation import plot_model, VISUALIZE_IN_SEGMENTS


def Inception(prev_layer, filters, stage, block, dr, m_id, i_dir):
    _in = Input(prev_layer.shape[1:]) if VISUALIZE_IN_SEGMENTS else prev_layer
    conv_name_base = f'Conv[k]x[k]-st{stage}-bl{block}-m{m_id}'
    F1, F2, F3, F4 = filters
    b1 = Conv(_in, F1, (1, 1), s=1,
              block_name=conv_name_base.replace('[k]', '1') + '_br1a', dr=dr, i_dir=i_dir)
    b1 = Conv(b1, F1, (3, 3), s=1, block_name=conv_name_base.replace('[k]', '3') + '_br1b', dr=dr, i_dir=i_dir)

    b2 = Conv(_in, F2, (1, 1), s=1,
              block_name=conv_name_base.replace('[k]', '1') + '_br2a', dr=dr, i_dir=i_dir)
    b2 = Conv(b2, F2, (5, 5), s=1,
              block_name=conv_name_base.replace('[k]', '5') + '_br2b', dr=dr, i_dir=i_dir)

    b3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(_in)
    b3 = Conv(b3, F3, (1, 1), s=1,
              block_name=conv_name_base.replace('[k]', '1') + '_br31', dr=dr, i_dir=i_dir)

    b4 = Conv(_in, F4, (1, 1), s=1,
              block_name=conv_name_base.replace('[k]', '1') + '_br4', dr=dr, i_dir=i_dir)

    x = Concatenate()([b1, b2, b3, b4])
    m_name = f'InceptionBlock-st{stage}-bl{block}-m{m_id}'
    se = SqueezeExcite(x.shape[-1])
    se.plot_model(f'SE-{m_name}', i_dir, x.shape[-1])
    x = se(x)
    if VISUALIZE_IN_SEGMENTS:
        m = Model(inputs=_in, outputs=x, name=m_name)
        plot_model(f'{i_dir}/resnet-inception-blocks', m, m_name)
        return m(prev_layer)
    return x
