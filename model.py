from contextlib import redirect_stdout

from tensorflow.keras import Model
from tensorflow.keras import utils
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate, Dropout
from tensorflow.keras.regularizers import l2

from architectures import resnet_inception, get_resnet

weight_decay = 1e-7
alpha = 0.2


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

    def sequential_submodel(self, _in, n_outputs):
        x = get_resnet(_in)
        x = Dense(n_outputs, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(
            x)
        return x, "resnet"

    def parallel_submodel(self, _in, n_outputs):
        x = get_resnet(_in)
        x = Dense(32, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        return x, "resnet"
