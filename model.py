from tensorflow.keras import Model

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate, Dropout
from tensorflow.keras.regularizers import l2

from architectures import get_resnet, save_model_info, aggregate

weight_decay = 1e-7
alpha = 0.2


class ModelFactory:
    def __init__(self, i_dir, n_classes, n_frames, input_shape):
        self.n_frames = n_frames
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.i_dir = i_dir
        self.model_type = None
        self.m_id = None

    def get_model(self, model_type, n_outputs, m_id=None):
        self.model_type = model_type
        if m_id is not None:
            self.m_id = m_id
        return {
            'single': self.get_single,
            'parallel': self.get_composite_model,
        }[model_type](n_outputs)

    def get_single(self, n_outputs):
        _in = Input(shape=self.input_shape)
        assert self.m_id is not None
        _out, name = self.sequential_submodel(_in, n_outputs, self.m_id)
        m_name = f'{name}_{self.m_id}'
        model = Model(inputs=_in, outputs=_out, name=m_name)
        save_model_info(self.i_dir, model, name)
        model.summary()
        return model

    def get_composite_model(self, n_outputs):
        _ins = [Input(shape=self.input_shape) for _ in range(self.n_frames)]
        model_outputs = []
        for i, _in in enumerate(_ins):
            _out, name = self.parallel_submodel(_in, n_outputs, i)
            m_name = f'{name}_{i}'
            submodel = Model(inputs=_in, outputs=_out, name=m_name)
            save_model_info(self.i_dir, submodel, name)
            submodel.summary()
            model_outputs.append(submodel)
        outs = [m(_in) for m, _in in zip(model_outputs, _ins)]
        model = self.get_aggregating_model(_ins, outs, self.model_type)
        return model

    def get_aggregating_model(self, inputs, outs, name):
        _out = aggregate(outs, self.n_classes)
        model = Model(inputs=inputs, outputs=_out, name=name)
        model.summary()
        save_model_info(self.i_dir, model, name)
        return model

    def sequential_submodel(self, _in, n_outputs, m_id):
        x = get_resnet(_in, m_id, self.i_dir)
        if n_outputs != 2:
            x = Dense(n_outputs, activation='softmax', kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay))(
                x)
        else:
            x = Dense(1, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        return x, "resnet"

    def parallel_submodel(self, _in, _, m_id):
        x = get_resnet(_in, m_id, self.i_dir)
        return x, "resnet"
