import sys
from contextlib import redirect_stdout

from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import utils
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input

from layers import get_architecture, aggregate


class ModelFactory:
    def __init__(self, i_dir, n_classes, n_frames, input_shape, architecture_id):
        self.arch_id = architecture_id
        self.n_frames = n_frames
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.i_dir = i_dir

    def get_model(self, model_type, n_outputs, name):
        return {
            'single': self.get_submodel,
            'parallel': self.get_composite_model,
        }[model_type](n_outputs, name)

    def save_model_info(self, model, filename):
        utils.plot_model(model, show_layer_names=False, show_shapes=True, to_file=f'{self.i_dir}/{filename}.png')
        with open(f'{self.i_dir}/{filename}.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()

    def submodel(self, _in, n_outputs, name=None):
        model = Model(inputs=_in, outputs=get_architecture(self.arch_id)(_in, n_outputs), name=name)
        model.summary()
        self.save_model_info(model, name)
        return model

    def get_submodel(self, n_outputs, name):
        _in = Input(shape=self.input_shape)
        return self.submodel(_in, name=name, n_outputs=n_outputs)

    def get_composite_model(self, n_outputs, name):
        _ins = [Input(shape=self.input_shape) for _ in range(self.n_frames)]
        model_outputs = []
        for i, _in in enumerate(_ins):
            subname = f'sub-{i}'
            submodel = self.submodel(_in, n_outputs, name=subname)
            if i == 0:
                self.save_model_info(submodel, subname)
            model_outputs.append(submodel)
        model = self.get_aggregating_model(_ins, model_outputs, name)
        return model

    def get_aggregating_model(self, inputs, outputs, name):
        _out = aggregate([m(_in) for m, _in in zip(outputs, inputs)], self.n_classes)
        model = Model(inputs=inputs, outputs=_out, name=name)
        self.save_model_info(model, name)
        return model


class PlotProgress(Callback):
    max_acc = 0
    max_val_acc = 0
    min_loss = sys.maxsize
    min_val_loss = sys.maxsize

    acc_ep = 0
    val_acc_ep = 0
    loss_ep = 0
    val_loss_ep = 0

    def __init__(self, i_dir, plot_lr=True):
        super().__init__()
        self.axs = None
        self.f = None
        self.metrics = None
        self.i_dir = i_dir
        self.first_epoch = True
        self.plot_lr = plot_lr

    def on_train_begin(self, logs=None):
        plt.ion()
        if logs is None:
            logs = {}
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
        self.f, self.axs = plt.subplots(1, 3, figsize=(13, 4)) if self.plot_lr else plt.subplots(1, 2, figsize=(9, 4))

    def on_train_end(self, logs=None):
        self.f.savefig(f"{self.i_dir}/metrics")
        plt.ioff()
        plt.close(self.f)

    def on_epoch_end(self, epoch, logs=None):
        # Storing metrics
        if logs is None:
            logs = {}
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        acc = max(self.max_acc, round(logs.get("accuracy"), 4))
        val_acc = max(self.max_val_acc, round(logs.get("val_accuracy"), 4))
        loss = min(self.min_loss, round(logs.get("loss"), 4))
        val_loss = min(self.min_val_loss, round(logs.get("val_loss"), 4))

        if acc == self.max_acc:
            self.acc_ep += 1
        else:
            self.acc_ep = 0
        if val_acc == self.max_val_acc:
            self.val_acc_ep += 1
        else:
            self.val_acc_ep = 0

        if loss == self.min_loss:
            self.loss_ep += 1
        else:
            self.loss_ep = 0

        if val_loss == self.min_val_loss:
            self.val_loss_ep += 1
        else:
            self.val_loss_ep = 0

        self.max_acc = acc
        self.max_val_acc = val_acc
        self.min_loss = loss
        self.min_val_loss = val_loss

        metrics = [x for x in logs if 'val' not in x]
        for i, metric in enumerate(metrics):
            self.axs[i].plot(range(1, epoch + 2), self.metrics[metric], color='blue', label=metric)
            if 'val_' + metric in logs:
                self.axs[i].plot(range(1, epoch + 2), self.metrics['val_' + metric], label='val_' + metric,
                                 color='orange', )
                if metric == 'accuracy':
                    self.axs[i].set_title(
                        f"{'Max accuracy': <16}: {self.max_acc:.4f}, not impr. in {self.acc_ep} epochs\n{'Max val_accuracy': <16}: {self.max_val_acc:.4f}, not impr. in {self.val_acc_ep} epochs")
                elif metric == 'loss':
                    self.axs[i].set_title(
                        f"{'Min loss': <16}: {self.min_loss:.4f}, not impr. in {self.loss_ep} epochs\n{'Min val_loss': <16}: {self.min_val_loss:.4f}, not impr. in {self.val_loss_ep} epochs")
            if self.first_epoch:
                self.axs[i].legend()
                self.axs[i].grid()
        self.first_epoch = False
        plt.tight_layout()
        self.f.canvas.draw()
        self.f.canvas.flush_events()
