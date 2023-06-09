from tensorflow.keras import Input
from tensorflow.keras.layers import (Conv2D, Activation, DepthwiseConv2D, BatchNormalization, SpatialDropout2D, Add,
                                     Layer)
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2

from model.architectures.blocks.basic import SqueezeExcite
from model.visualisation import plot_model


class ConvMixerBlock(Layer):
    def __init__(self, filters, kernel_size, in_shape, block_name, dr, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.in_shape = in_shape
        self.dr = dr
        self.block_name = block_name
        self.depthwise = DepthwiseConv2D(
            kernel_size=kernel_size, padding="same", depthwise_regularizer=l2(1e-4), kernel_initializer='he_normal'
        )
        self.act1 = Activation("gelu")
        self.bn1 = BatchNormalization()
        self.add = Add()
        self.pointwise = Conv2D(
            filters, kernel_size=1, kernel_regularizer=l2(1e-4), kernel_initializer='he_normal'
        )
        self.se = SqueezeExcite(in_shape[-1])
        self.act2 = Activation("gelu")
        self.bn2 = BatchNormalization()
        self.dropout = SpatialDropout2D(dr) if dr else None

    def call(self, inputs):
        x_skip = inputs
        x = self.depthwise(inputs)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.add([x, x_skip])
        x = self.pointwise(x)
        x = self.se(x)
        x = self.act2(x)
        x = self.bn2(x)
        if self.dropout:
            x = self.dropout(x)
        return x

    def plot_layer(self, in_shape, i_dir):
        self.se.plot_model(f'SE-{self.block_name}', i_dir)
        inputs = Input(in_shape[1:])
        _out = self.call(inputs)
        model = Model(inputs=inputs, outputs=_out, name=self.block_name)
        plot_model(f'{i_dir}/conv-mixer', model, self.block_name)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'block_name': self.block_name,
            'dr': self.dr,
            'in_shape': self.in_shape,
        }
        base_config = super(ConvMixerBlock, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
