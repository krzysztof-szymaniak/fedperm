from os.path import join

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, \
    TensorBoard

from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import SGD, Adam

from model.utils import PlotProgress

BATCH_SIZE = 32
MAX_EPOCHS = 200


def scheduler(start_ep=15, decay_rate=-0.04, min_rate=5e-7):
    def schedule(epoch, lr):
        if epoch < start_ep:
            return lr
        else:
            return max(lr * tf.math.exp(decay_rate), min_rate)

    return schedule


def compile_options(n_classes, opt='adam'):
    opts = {
        "loss": categorical_crossentropy if n_classes > 2 else binary_crossentropy,
        "optimizer":
            SGD(learning_rate=1e-2, momentum=0.9, nesterov=True) if opt == 'sgd'
            else Adam(learning_rate=1e-3),
        "metrics": ['accuracy'] if n_classes > 2 else ['accuracy', Precision(), Recall()],
    }
    return opts


def callbacks(checkpoints_dir, training_info_dir, name):
    stopping_patience = 8
    reduce_patience = 4
    monitor_metric = 'val_accuracy'

    return [
        ModelCheckpoint(
            filepath=join(checkpoints_dir, 'weights.h5'),
            save_freq='epoch',
            verbose=0,
            monitor=monitor_metric,
            save_weights_only=True,
            save_best_only=True
        ),
        EarlyStopping(
            monitor=monitor_metric,
            verbose=1,
            patience=stopping_patience,
            restore_best_weights=True
        ),
        LearningRateScheduler(scheduler()),
        ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=np.sqrt(0.1),
            patience=reduce_patience,
            verbose=1,
            min_lr=5e-7
        ),
        PlotProgress(training_info_dir, name),
        TensorBoard(log_dir=f'{training_info_dir}/graph', histogram_freq=1, write_graph=True, write_images=True)
    ]
