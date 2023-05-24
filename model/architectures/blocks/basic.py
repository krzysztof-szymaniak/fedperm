from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D, GlobalAveragePooling2D, BatchNormalization, ReLU, SpatialDropout2D,
    Multiply, Dense, Layer)
from tensorflow.keras.regularizers import l2

from model.visualisation import plot_model, VISUALIZE_IN_SEGMENTS

l2_reg = 1e-4


def Conv(prev_layer, F, k, s, block_name, padding='same', act=True, bn=True, se=False, i_dir=None, dr=None):
    if s is None:
        s = 1
    _in = Input(prev_layer.shape[1:]) if VISUALIZE_IN_SEGMENTS else prev_layer
    x = _in
    x = Conv2D(F, k, strides=s, name=block_name, padding=padding, kernel_regularizer=l2(l2_reg) if l2_reg else None)(x)
    if se:
        x = SqueezeExcite(x.shape[-1], i_dir, f'SE-{block_name}')(x)
    if bn:
        x = BatchNormalization()(x)
    if act:
        x = ReLU()(x)
    if dr:
        x = SpatialDropout2D(dr)(x)
    if VISUALIZE_IN_SEGMENTS:
        m = Model(inputs=_in, outputs=x, name=block_name)
        plot_model(f'{i_dir}/conv', m, block_name)
        return m(prev_layer)
    return x


class SqueezeExcite(Layer):
    def __init__(self, filters, i_dir, block_name, **kwargs):
        super().__init__(**kwargs)
        ratio = 16
        self.squeeze = Dense(filters // ratio, activation='relu', use_bias=False)
        self.excite = Dense(filters, activation='sigmoid', use_bias=False)
        self.plot_model(block_name, i_dir, filters)

    def plot_model(self, block_name, i_dir, filters):
        inputs = Input(shape=(16, 16, filters))
        x = GlobalAveragePooling2D()(inputs)
        x = self.squeeze(x)
        x = self.excite(x)
        _out = Multiply()([inputs, x])
        m = Model(inputs=inputs, outputs=_out, name=block_name)
        plot_model(f'{i_dir}/conv', m, block_name)

    def call(self, inputs):
        x = GlobalAveragePooling2D()(inputs)
        x = self.squeeze(x)
        x = self.excite(x)
        _out = Multiply()([inputs, x])
        return _out

    def get_config(self):
        config = {'filters': self.filters}
        base_config = super(SqueezeExcite, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
