from os.path import join

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import SGD, Adam

from model.utils import PlotProgress

# reverse = False

BATCH_SIZE = 32
epochs = 1000
stopping_patience = 12
reduce_patience = 5


def scheduler(ep_threshold=15, decay_rate=-0.02, min_rate=5e-6):
    def schedule(epoch, lr):
        if epoch < ep_threshold:
            return lr
        else:
            return max(lr * tf.math.exp(decay_rate), min_rate)

    return schedule


def compile_opts(n_classes):
    opts = {
        "loss": categorical_crossentropy if n_classes > 2 else binary_crossentropy,
        "optimizer": Adam(learning_rate=3e-4),
        # "optimizer": SGD(learning_rate=1e-2, momentum=0.9, nesterov=True),
        "metrics": ['accuracy'] if n_classes > 2 else ['accuracy', Precision(), Recall()],
    }
    return opts


def callbacks(checkpoints_dir, training_info_dir):
    return [
        ModelCheckpoint(filepath=join(checkpoints_dir, 'weights.h5'),
                        save_freq='epoch',
                        verbose=0,
                        monitor='val_accuracy',
                        save_weights_only=True,
                        save_best_only=True),
        EarlyStopping(monitor="val_loss",
                      verbose=1,
                      patience=stopping_patience,
                      restore_best_weights=True),
        LearningRateScheduler(scheduler()),
        ReduceLROnPlateau(monitor="val_loss",
                          factor=np.sqrt(0.1),
                          patience=reduce_patience,
                          verbose=1,
                          min_lr=0.5e-6),
        PlotProgress(training_info_dir),
        # TensorBoard(log_dir=f'./{self.model_name}/graph', histogram_freq=1, write_graph=True,
        #             write_images=True)
    ]


def augmentation(apply_flip):
    return {
        "width_shift_range": 0.08,  # randomly shift images horizontally (fraction of total width)
        "height_shift_range": 0.08,  # randomly shift images vertically (fraction of total height)
        "rotation_range": 5,
        "horizontal_flip": apply_flip
    }
