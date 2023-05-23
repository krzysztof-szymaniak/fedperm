import pathlib

from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout, Embedding, Add, LayerNormalization, MultiHeadAttention, Flatten, \
    Layer, GlobalAveragePooling1D
from tensorflow.keras import Model
import tensorflow as tf

from utils import save_model_info


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x


class Patches(Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        # print(f'{patch_dims=}')
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def transformer_block(prev_layer, num_heads, n_dim, units, bl, m_id, i_dir):
    _in = Input(prev_layer.shape[1:])
    x1 = LayerNormalization(epsilon=1e-6)(_in)
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=n_dim, dropout=0.1
    )(x1, x1)
    x2 = Add()([attention_output, _in])
    x3 = LayerNormalization(epsilon=1e-6)(x2)
    x3 = mlp(x3, hidden_units=units, dropout_rate=0.1)
    x = Add()([x3, x2])
    m = Model(inputs=_in, outputs=x, name=f'Transformer-Encoder-bl{bl}-m{m_id}')
    save_model_info(f'{i_dir}/transformer', m, m.name)
    return m(prev_layer)


def Transformer(_in, patch_size=None, n_dims=None, n_layers=None, n_heads=None, mlp_head=None, m_id=None, i_dir=None):
    num_patches = (_in.shape[-2] // patch_size) ** 2
    transformer_units = [
        n_dims * 2,
        n_dims,
    ]
    patches = Patches(patch_size)(_in)
    x = PatchEncoder(num_patches, n_dims)(patches)
    for i in range(n_layers):
        x = transformer_block(x, num_heads=n_heads, n_dim=n_dims,
                              units=transformer_units,
                              bl=i + 1, m_id=m_id, i_dir=i_dir)

    representation = LayerNormalization(epsilon=1e-6)(x)
    representation = Flatten()(representation)
    representation = Dropout(0.4)(representation)
    representation = mlp(representation, hidden_units=mlp_head, dropout_rate=0.4)
    return representation
