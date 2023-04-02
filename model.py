from contextlib import redirect_stdout

from tensorflow.keras import Model
from tensorflow.keras import utils
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, GlobalAveragePooling2D, BatchNormalization, \
    multiply, \
    Dropout, LeakyReLU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.layers import Add

weight_decay = 1e-7


def aggregate(models, n_classes):
    x = concatenate(models)
    # x = Dense(512, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.4)(x)
    # x = Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    # x = Dropout(0.2)(x)
    if n_classes != 2:
        x = Dense(n_classes, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(
            x)
    else:
        x = Dense(1, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
    return x


class ModelFactory:
    def __init__(self, i_dir, n_classes, n_frames, input_shape, architecture_id):
        self.arch_id = architecture_id
        self.n_frames = n_frames
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.i_dir = i_dir
        self.model_type = None

    def get_model(self, model_type, n_outputs):
        self.model_type = model_type
        return {
            'single': self.get_single,
            'parallel': self.get_composite_model,
        }[model_type](n_outputs)

    def save_model_info(self, model, filename):
        utils.plot_model(model, show_layer_names=True, show_shapes=True, to_file=f'{self.i_dir}/{filename}.png')
        with open(f'{self.i_dir}/{filename}.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()

    def Submodel(self, _in, n_outputs, m_id):
        _out, name = {
            0: self.sequential_submodel,
            # 1: self.inception_model,
            2: self.parallel_submodel,
            3: self.sequential_cats_dogs,
            4: self.parallel_cats_dogs,

        }[self.arch_id](_in, n_outputs)
        m_name = f'{name}_{m_id}'
        model = Model(inputs=_in, outputs=_out, name=m_name)
        if m_id == 0:
            self.save_model_info(model, m_name)
        model.summary()
        return model

    def get_single(self, n_outputs):
        _in = Input(shape=self.input_shape)
        model = self.Submodel(_in, n_outputs=n_outputs, m_id=0)
        return model

    def get_composite_model(self, n_outputs):
        _ins = [Input(shape=self.input_shape) for _ in range(self.n_frames)]
        model_outputs = []
        for i, _in in enumerate(_ins):
            submodel = self.Submodel(_in, n_outputs, m_id=i)
            model_outputs.append(submodel)
        outs = [m(_in) for m, _in in zip(model_outputs, _ins)]
        model = self.get_aggregating_model(_ins, outs, self.model_type)
        return model

    def get_aggregating_model(self, inputs, outs, name):
        _out = aggregate(outs, self.n_classes)
        model = Model(inputs=inputs, outputs=_out, name=name)
        model.summary()
        self.save_model_info(model, name)
        return model

    ####################################################################################################################
    ####################################################################################################################
    def SqueezeExcite(self, _in, i=None):
        ratio = 16
        filters = _in.shape[-1]
        x = GlobalAveragePooling2D()(_in)
        x = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False,
                  kernel_regularizer=l2(weight_decay))(x)
        x = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False,
                  kernel_regularizer=l2(weight_decay))(x)
        _out = multiply([_in, x])
        # se = Model(inputs=_in, outputs=_out, name=f"squeeze_excite_{i}")
        # self.save_model_info(se, f"squeeze_excite_{i}")
        # return se(_in)
        return _out

    ####################################################################################################################
    def Inception(self, _in, filters=None, i=None):
        if filters is None:
            filters = [16, 16, 16]
        b1 = Conv2D(filters[0], (1, 1), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay))(_in)
        b1 = BatchNormalization()(b1)
        b1 = LeakyReLU()(b1)

        b1 = Conv2D(filters[0], (3, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay))(b1)
        b1 = BatchNormalization()(b1)
        b1 = LeakyReLU()(b1)

        b2 = Conv2D(filters[1], (1, 1), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay))(_in)
        b2 = BatchNormalization()(b2)
        b2 = LeakyReLU()(b2)
        b2 = Conv2D(filters[1], (5, 5), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay))(b2)
        b2 = BatchNormalization()(b2)
        b2 = LeakyReLU()(b2)

        b3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(_in)
        b3 = Conv2D(filters[2], (1, 1), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay))(b3)
        b3 = BatchNormalization()(b3)
        b3 = LeakyReLU()(b3)

        _out = concatenate([b1, b2, b3])  # output size W x H x (f0 + f1 + f2)
        # inception = Model(inputs=_in, outputs=_out, name=f"inception_{i}")
        # self.save_model_info(inception, f"inception_{i}")
        # return inception(_in)
        return _out

    ####################################################################################################################
    def sequential_submodel(self, _in, n_outputs):
        x = self.resnet_fashion(_in)
        x = Dense(n_outputs, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(
            x)
        return x, "resnet"

    def parallel_submodel(self, _in, n_outputs):
        x = self.resnet_fashion(_in)
        x = Dense(32, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        return x, "resnet"

    ####################################################################################################################
    def sequential_cats_dogs(self, _in, n_outputs):
        x = self.resnet_cats_dogs(_in)
        x = Dense(1, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(
            x)
        return x, "resnet"

    def parallel_cats_dogs(self, _in, n_outputs):
        x = self.resnet_cats_dogs(_in)
        x = Dense(32, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        return x, "resnet"

    ####################################################################################################################

    def resnet_fashion(self, _in):
        d_r = 0.1
        x = _in
        x = Conv2D(32, (7, 7), strides=(1, 1), name='conv1', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)

        stage = 2
        filters = [32, 32, 64]
        i_filters = [20, 24, 20]
        kernel = 3
        x = self.convolutional_block(x, f=kernel, filters=filters, stage=stage, block='a', s=1)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, kernel, filters, stage=stage, block='b')
        x = Dropout(d_r)(x)
        x = self.Inception(x, filters=i_filters)
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)

        stage = stage + 1
        filters = [64, 64, 128]
        i_filters = [40, 48, 40]
        kernel = 3
        x = self.convolutional_block(x, f=kernel, filters=filters, stage=stage, block='a', s=2)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, kernel, filters, stage=stage, block='b')
        x = Dropout(d_r)(x)
        x = self.identity_block(x, kernel, filters, stage=stage, block='c')
        x = Dropout(d_r)(x)
        x = self.Inception(x, filters=i_filters)
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)

        stage = stage + 1
        filters = [128, 128, 256]
        i_filters = [80, 96, 80]
        kernel = 3
        x = self.convolutional_block(x, f=kernel, filters=filters, stage=stage, block='a', s=2)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, kernel, filters, stage=stage, block='b')
        x = Dropout(d_r)(x)
        x = self.identity_block(x, kernel, filters, stage=stage, block='c')
        x = Dropout(d_r)(x)
        x = self.Inception(x, filters=i_filters)
        x = self.SqueezeExcite(x)
        x = GlobalAveragePooling2D()(x)
        return x

    def resnet_cats_dogs(self, _in):
        d_r = 0.1
        x = _in
        x = Conv2D(64, (11, 11), strides=(2, 2), name='conv1', kernel_initializer='he_normal', padding='valid',
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)

        stage = 2
        filters = [64, 64, 128]
        x = self.convolutional_block(x, f=11, filters=filters, stage=stage, block='a', s=1)
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 11, filters, stage=stage, block='b')
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 11, filters, stage=stage, block='c')
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)

        stage = stage + 1
        filters = [128, 128, 256]
        x = self.convolutional_block(x, f=7, filters=filters, stage=stage, block='a', s=2)
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 7, filters, stage=stage, block='b')
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 7, filters, stage=stage, block='c')
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 7, filters, stage=stage, block='d')
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)

        stage = stage + 1
        filters = [256, 256, 512]
        x = self.convolutional_block(x, f=5, filters=filters, stage=stage, block='a', s=2)
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 5, filters, stage=stage, block='b')
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 5, filters, stage=stage, block='c')
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        # x = self.identity_block(x, 3, filters, stage=stage, block='d')
        # x = self.SqueezeExcite(x)
        # x = Dropout(d_r)(x)
        # x = self.identity_block(x, 3, filters, stage=stage, block='e')
        # x = self.SqueezeExcite(x)
        # x = Dropout(d_r)(x)
        # x = self.identity_block(x, 3, filters, stage=stage, block='f')
        # x = self.SqueezeExcite(x)
        # x = Dropout(d_r)(x)

        stage = stage + 1
        filters = [512, 512, 1024]
        x = self.convolutional_block(x, f=3, filters=filters, stage=stage, block='a', s=2)
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 3, filters, stage=stage, block='b')
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 3, filters, stage=stage, block='c')
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)

        stage = stage + 1
        filters = [1024, 1024, 2048]
        x = self.convolutional_block(x, f=3, filters=filters, stage=stage, block='a', s=2)
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 3, filters, stage=stage, block='b')
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 3, filters, stage=stage, block='c')
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = GlobalAveragePooling2D()(x)
        return x

    def resnet(self, _in):
        d_r = 0.1
        x = _in
        x = Conv2D(128, (5, 5), strides=(1, 1), name='conv1', kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=3, name='bn_conv1')(x)
        x = LeakyReLU()(x)
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)

        stage = 2
        filters = [256, 256, 512]
        x = self.convolutional_block(x, f=3, filters=filters, stage=stage, block='a', s=1)  # 16
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 3, filters, stage=stage, block='b')
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 3, filters, stage=stage, block='c')
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)

        stage = stage + 1
        filters = [256, 256, 512]
        x = self.convolutional_block(x, f=3, filters=filters, stage=stage, block='a', s=1)  # 16
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 3, filters, stage=stage, block='b')
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 3, filters, stage=stage, block='c')
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)

        stage = stage + 1
        filters = [512, 512, 1024]
        x = self.convolutional_block(x, f=3, filters=filters, stage=stage, block='a', s=2)  # 8
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 3, filters, stage=stage, block='b')
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 3, filters, stage=stage, block='c')
        x = Dropout(d_r)(x)
        x = self.SqueezeExcite(x)

        stage = stage + 1
        filters = [512, 512, 1024]
        x = self.convolutional_block(x, f=3, filters=filters, stage=stage, block='a', s=1)  # 8
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 3, filters, stage=stage, block='b')
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 3, filters, stage=stage, block='c')
        x = Dropout(d_r)(x)
        x = self.SqueezeExcite(x)

        # stage = stage + 1
        # filters = [256, 256, 512]
        # x = self.convolutional_block(x, f=3, filters=filters, stage=stage, block='a', s=1)  # 8
        # x = self.SqueezeExcite(x)
        # x = Dropout(d_r)(x)
        # x = self.identity_block(x, 3, filters, stage=stage, block='b')
        # x = self.SqueezeExcite(x)
        # x = Dropout(d_r)(x)
        # x = self.identity_block(x, 3, filters, stage=stage, block='c')
        # x = Dropout(d_r)(x)
        # x = self.SqueezeExcite(x)

        stage = stage + 1
        filters = [1024, 1024, 2048]
        x = self.convolutional_block(x, f=3, filters=filters, stage=stage, block='a', s=2)
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = self.identity_block(x, 3, filters, stage=stage, block='b')
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        # x = self.identity_block(x, 3, filters, stage=stage, block='c')
        # x = self.SqueezeExcite(x)
        # x = Dropout(d_r)(x)

        # stage = stage + 1
        # filters = [512, 512, 1024]
        # x = self.convolutional_block(x, f=3, filters=filters, stage=stage, block='a', s=1)
        # x = self.SqueezeExcite(x)
        # x = Dropout(d_r)(x)
        # x = self.identity_block(x, 3, filters, stage=stage, block='b')
        # x = self.SqueezeExcite(x)
        # x = Dropout(d_r)(x)
        # x = self.identity_block(x, 3, filters, stage=stage, block='c')
        # x = self.SqueezeExcite(x)
        # x = Dropout(d_r)(x)

        x = Conv2D(1024, (1, 1), name='conv_reduce', kernel_initializer='he_normal', padding='valid',
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU()(x)
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)
        x = Conv2D(1024, (3, 3), strides=(1, 1), name='conv_out', kernel_initializer='he_normal', padding='valid',
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU()(x)
        x = self.SqueezeExcite(x)
        x = Dropout(d_r)(x)

        x = GlobalAveragePooling2D()(x)
        return x

    def convolutional_block(self, x, f, filters, stage, block, s=2):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        F1, F2, F3 = filters

        x_skip = x

        x = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(name=bn_name_base + '2a')(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(name=bn_name_base + '2b')(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(name=bn_name_base + '2c')(x)
        x = self.SqueezeExcite(x)

        x_skip = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x_skip)
        x_skip = BatchNormalization(name=bn_name_base + '1')(x_skip)

        x = Add()([x, x_skip])
        x = LeakyReLU()(x)

        return x

    def identity_block(self, x, f, filters, stage, block):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        F1, F2, F3 = filters

        x_skip = x

        x = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

        x = BatchNormalization(name=bn_name_base + '2a')(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(name=bn_name_base + '2b')(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(name=bn_name_base + '2c')(x)
        x = self.SqueezeExcite(x)

        x = Add()([x, x_skip])  # SKIP Connection
        x = LeakyReLU()(x)

        return x
