from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D, GlobalAveragePooling2D, BatchNormalization, Activation, SpatialDropout2D,
    Multiply, Dense, Layer)
from tensorflow.keras.regularizers import l2

from model.visualisation import plot_model

l2_reg = 1e-4


class ConvBlock(Layer):
    def __init__(self, in_shape, filters, kernel_size, stride, block_name, padding='same', dr=None, **kwargs):
        super().__init__(**kwargs)
        self.dr = dr
        self.padding = padding
        self.block_name = block_name
        self.stride = stride if stride is not None else 1
        self.kernel_size = kernel_size
        self.filters = filters
        self.in_shape = in_shape
        self.conv = Conv2D(
            filters, kernel_size, strides=stride, name=block_name, padding=padding, kernel_regularizer=l2(l2_reg)
        )
        self.se = SqueezeExcite(filters)
        self.act = Activation("gelu")
        self.bn = BatchNormalization()
        self.dropout = SpatialDropout2D(dr) if dr else None

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.se(x)
        x = self.act(x)
        x = self.bn(x)
        if self.dropout:
            x = self.dropout(x)
        return x

    def plot_layer(self, in_shape, info_dir):
        self.se.plot_model(f'SE-{self.block_name}', info_dir)
        inputs = Input(in_shape[1:])
        _out = self.call(inputs)
        m = Model(inputs=inputs, outputs=_out, name=self.block_name)
        plot_model(f'{info_dir}/conv', m, self.block_name)

    def get_config(self):
        layer_config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'block_name': self.block_name,
            'dr': self.dr,
            'in_shape': self.in_shape,
            'stride': self.stride,
        }
        base_config = super(ConvBlock, self).get_config()

        return dict(list(base_config.items()) + list(layer_config.items()))

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)


class SqueezeExcite(Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        ratio = 16
        self.filters = filters
        self.squeeze = Dense(filters // ratio, activation='relu', use_bias=False)
        self.excite = Dense(filters, activation='sigmoid', use_bias=False)

    def plot_model(self, block_name, info_dir):
        inputs = Input((16, 16, self.filters))
        _out = self.call(inputs)
        m = Model(inputs=inputs, outputs=_out, name=block_name)
        plot_model(f'{info_dir}/conv', m, block_name)

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
