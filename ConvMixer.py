from tensorflow.keras.regularizers import l2

from PixelShuffle import PixelShuffler
from layers import Conv, SqueezeExcite
from transformer import Patches, PatchEncoder
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, ReLU, Conv2D, SpatialDropout2D, UpSampling2D

from utils import save_model_info


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_mixer_block(prev_layer, filters, kernel_size, st, m_id, i_dir):
    _in = Input(prev_layer.shape[1:])
    x = _in
    x0 = x
    x = layers.SpatialDropout2D(0.15)(x)
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same", depthwise_regularizer=l2(1e-4))(x)
    x = activation_block(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    x = layers.Conv2D(filters, kernel_size=1, kernel_regularizer=l2(1e-4))(x)
    x = activation_block(x)
    x = SqueezeExcite(x, f'SE-{st}', i_dir)
    block_name = f"ConvMixer{kernel_size}x{kernel_size}-st{st}-m{m_id}"
    m = Model(inputs=_in, outputs=x, name=block_name)
    save_model_info(f'{i_dir}/conv-mixer', m, block_name)
    return m(prev_layer)


def ConvMixer(_in, filters=None, depth=None, kernel_size=None, patch_size=None, upscale=False, m_id=None,
              i_dir=None):
    """ConvMixer: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    """
    x = Conv(_in, filters, k=patch_size, s=patch_size, se=True, l2_reg=1e-4, padding='valid', i_dir=i_dir, dr=None,
             block_name=f'Conv{patch_size}x{patch_size}-adaptation')
    if upscale:
        x = PixelShuffler(size=upscale)(x)
        # x = UpSampling2D(size=upscale)(x)
    for i in range(depth):
        x = conv_mixer_block(x, filters, kernel_size, st=i + 1, m_id=m_id, i_dir=i_dir)

    x = layers.GlobalAvgPool2D()(x)
    return x
