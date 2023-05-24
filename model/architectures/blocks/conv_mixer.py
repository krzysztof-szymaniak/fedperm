from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Conv2D, Activation, DepthwiseConv2D, GlobalAveragePooling2D, BatchNormalization, ReLU, SpatialDropout2D,
    Multiply, Add, Dense, Dropout, Flatten, Layer)
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2, l1_l2

from model.architectures.blocks.basic import SqueezeExcite
from model.visualisation import plot_model, VISUALIZE_IN_SEGMENTS


def conv_mixer_block(prev_layer, filters, kernel_size, st, m_id, i_dir, dr):
    block_name = f"ConvMixer{kernel_size}x{kernel_size}-st{st}-m{m_id}"
    _in = Input(prev_layer.shape[1:]) if VISUALIZE_IN_SEGMENTS else prev_layer
    x_skip = _in
    x = DepthwiseConv2D(kernel_size=kernel_size, padding="same", depthwise_regularizer=l2(1e-4))(_in)
    x = SqueezeExcite(x.shape[-1], i_dir, f'SE-dep-{block_name}')(x)
    x = Activation("gelu")(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_skip])  # Residual.
    if dr:
        x = SpatialDropout2D(dr)(x)
    x = Conv2D(filters, kernel_size=1, kernel_regularizer=l2(1e-4))(x)
    x = SqueezeExcite(x.shape[-1], i_dir, block_name=f'SE-pix-{block_name}')(x)
    x = Activation("gelu")(x)
    x = BatchNormalization()(x)
    if VISUALIZE_IN_SEGMENTS:
        m = Model(inputs=_in, outputs=x, name=block_name)
        plot_model(f'{i_dir}/conv-mixer', m, block_name)
        return m(prev_layer)
    return x
