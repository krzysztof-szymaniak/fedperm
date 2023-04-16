from os.path import exists, join

import numpy as np

from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.client import device_lib

from training import ModelTraining, TrainingSequential
from permutations import generate_permutations, Overlap
from preprocessing import load_data, get_classes_names_for_dataset

print(device_lib.list_local_devices())


def get_training_env(model_params, ds_name, f_id, n_classes, input_shape):
    model_type = model_params['type']
    grid_size = model_params['grid_size']
    seed = model_params['seed']
    overlap = model_params['overlap']

    subinput_shape = (input_shape[0] // grid_size[0], input_shape[1] // grid_size[1], input_shape[2])
    permutations = generate_permutations(seed, grid_size, subinput_shape, overlap)

    model_name = f"models/{model_type}-{ds_name}-fold_{f_id}-{grid_size[0]}x{grid_size[1]}-"
    model_name += f"-{overlap}"
    model_name += f"{'-permuted' if seed is not None else '-identity'}"

    classes = get_classes_names_for_dataset(ds_name)
    if model_type == 'parallel':
        return ModelTraining(model_name, subinput_shape, n_classes, permutations, 'parallel', classes)
    elif model_type == 'sequential':
        return TrainingSequential(model_name, subinput_shape, n_classes, permutations, classes)


class Experiment:
    n_splits = 5

    def __init__(self):
        self.models = None
        self.skf = StratifiedKFold(n_splits=self.n_splits)
        self.datasets = None
        self.scores = None

    def fit_models_and_save_scores(self, modes_params, datasets, scores_file):
        self.scores = np.zeros((len(datasets), len(modes_params), self.n_splits))
        for d_id, ds_name in enumerate(datasets):
            (x, y), (x_test, y_test), n_classes = load_data(ds_name)
            if n_classes != 2:
                y_s = np.argmax(y, axis=1)
            else:
                y_s = y
            for f_id, (train, valid) in enumerate(self.skf.split(x, y_s)):
                for m_id, m in enumerate(modes_params):
                    model = get_training_env(m, ds_name, f_id, n_classes, x.shape[1:])
                    if not exists(join(model.model_name, 'saved_model.pb')):
                        model.fit(x[train], y[train], x[valid], y[valid])
                    else:
                        print(f"Skipping training {model.model_name}")
                    self.scores[d_id, m_id, f_id] = model.predict(x_test, y_test)
        np.save(scores_file, self.scores)
        self.run_stats(scores_file)

    def run_stats(self, scores_file, alfa=0.05):
        self.scores = np.load(scores_file)
        n_models = self.scores.shape[1]
        for ds_scores in self.scores:
            t_statistic = np.zeros((n_models, n_models))
            p_value = np.zeros((n_models, n_models))
            for i in range(n_models):
                for j in range(n_models):
                    t_statistic[i, j], p_value[i, j] = ttest_rel(ds_scores[i], ds_scores[j])
            np.save('t_stat.npy', t_statistic)
            print(f'{t_statistic=}')
            print(f'{p_value=}')
            np.save('p_stat.npy', p_value)
            advantage = np.zeros((n_models, n_models))
            advantage[t_statistic > 0] = 1
            significance = np.zeros((n_models, n_models))
            significance[p_value <= alfa] = 1
            stat_better = significance * advantage
            print(stat_better)


if __name__ == '__main__':
    models = [
        {
            'type': 'sequential',
            'seed': 123,
            'grid_size': (2, 2),
            'overlap': Overlap.CENTER.value,
        },
        {
            'type': 'parallel',
            'seed': 123,
            'grid_size': (2, 2),
            'overlap': Overlap.CENTER.value,
        },
        {
            'type': 'sequential',
            'seed': None,
            'grid_size': (2, 2),
            'overlap': Overlap.CENTER.value,
        },
        {
            'type': 'parallel',
            'seed': None,
            'grid_size': (2, 2),
            'overlap': Overlap.CENTER.value,
        },
    ]
    datasets = [
        # 'mnist',
        'fashion_mnist',
        # 'cifar10',
        # 'kmnist',
        # 'emnist-letters',
        # 'eurosat',
        # 'cats_vs_dogs'
    ]
    Experiment().fit_models_and_save_scores(models, datasets, 'scores.npy')
