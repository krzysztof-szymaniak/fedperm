import pathlib
import sys
from os.path import join
from pprint import pprint

import numpy as np
from keras.callbacks import Callback
from matplotlib import pyplot as plt

info_dir_name = "train"
examples_dir_name = 'data_examples'
arch_dir_name = 'architecture'
testing_dir_name = 'test'


def save_training_info(model, training_info_dir):
    plt.close('all')
    plt.ioff()
    history = model.history
    metrics = set([m.replace('val_', '') for m in history.history.keys()])
    for met in metrics:
        plt.plot(history.history[met])
        if f"val_{met}" in history.history:
            plt.plot(history.history[f"val_{met}"])
        plt.title(f"{met}")
        plt.ylabel(met)
        plt.xlabel('epoch')
        plt.grid()
        if f"val_{met}" in history.history:
            plt.legend(['train', 'validate'], loc='right')
        else:
            plt.legend(['train'], loc='right')
        plt.savefig(join(training_info_dir, met))
        plt.clf()
    np.save(join(training_info_dir, "history.npy"), history.history)
    # history = np.load(join(self.training_info_dir, "history.npy"), allow_pickle=True).item()
    with open(join(training_info_dir, "model_config"), 'w') as f:
        pprint(model.get_config(), f)


def create_dir(name):
    pathlib.Path(name).mkdir(exist_ok=True, parents=True)


def set_up_dirs(model_name):
    training_info_dir = join(model_name, info_dir_name)
    examples_info_dir = join(model_name, examples_dir_name)
    arch_info_dir = join(model_name, arch_dir_name)
    checkpoints_dir = join(model_name, 'checkpoint')

    create_dir(training_info_dir)
    create_dir(examples_info_dir)
    create_dir(arch_info_dir)
    create_dir(checkpoints_dir)

    return training_info_dir, examples_info_dir, arch_info_dir, checkpoints_dir


class PlotProgress(Callback):
    max_acc = 0
    max_val_acc = 0
    min_loss = sys.maxsize
    min_val_loss = sys.maxsize

    acc_ep = 0
    val_acc_ep = 0
    loss_ep = 0
    val_loss_ep = 0

    def __init__(self, i_dir, verbose=True):
        pathlib.Path(i_dir).mkdir(exist_ok=True, parents=True)
        super().__init__()
        self.verbose = verbose
        self.axs = None
        self.f = None
        self.metrics = None
        self.i_dir = i_dir
        pathlib.Path(self.i_dir).mkdir(exist_ok=True, parents=True)
        self.first_epoch = True
        self.metrics = {}
        plt.ion()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        n_met = len([x for x in logs if 'val' not in x])
        if self.f is None:
            if n_met > 3:
                self.f, self.axs = plt.subplots(2, 3, figsize=(12, 8))
            else:
                self.f, self.axs = plt.subplots(1, 3, figsize=(12, 4))

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
        acc_msg = f"{'Max accuracy': <16}: {self.max_acc:.4f}, not impr. in {self.acc_ep} epochs\n{'Max val_accuracy': <16}: {self.max_val_acc:.4f}, not impr. in {self.val_acc_ep} epochs"
        loss_msg = f"{'Min loss': <16}: {self.min_loss:.4f}, not impr. in {self.loss_ep} epochs\n{'Min val_loss': <16}: {self.min_val_loss:.4f}, not impr. in {self.val_loss_ep} epochs"
        metrics = [x for x in logs if 'val' not in x]
        for ax, metric in zip(self.axs.flatten(), metrics):
            ax.plot(range(1, epoch + 2), self.metrics[metric], color='blue', label=metric)
            if 'val_' + metric in logs:
                ax.plot(range(1, epoch + 2), self.metrics['val_' + metric], label='val_' + metric,
                        color='orange', )
                if metric == 'accuracy':
                    ax.set_title(acc_msg)
                    ax.set_ylim([0.0, 1.0])
                elif metric == 'loss':
                    ax.set_title(loss_msg)
                    ax.set_ylim([0.0, 3.0])
            if metric == 'lr':
                ax.set_title(f'Learning rate: {self.metrics[metric][-1]:.4e}')
            if self.first_epoch:
                ax.legend()
                ax.grid()
        self.first_epoch = False
        plt.tight_layout()
        self.f.canvas.draw()
        self.f.canvas.flush_events()
        self.f.savefig(f"{self.i_dir}/progress.png")
