import numpy as np
from keras.datasets import fashion_mnist
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold

from model_training import ModelTraining, TrainingSequential
from permutations import generate_permutations
from preprocessing import load_data


def get_classes_names_for_dataset(ds_name):
    fashion_names = """
        T-shirt/top
        Trouser
        Pullover
        Dress
        Coat
        Sandal
        Shirt
        Sneaker
        Bag
        Ankle boot
        """
    cifar_names = """airplane
        automobile
        bird
        cat
        deer
        dog
        frog
        horse
        ship
        truck"""
    classes = None
    if ds_name == 'mnist':
        classes = [i for i in range(10)]
    elif ds_name == 'fashion':
        classes = [c for c in fashion_names.split("\n") if c]
    elif ds_name == 'cifar10':
        classes = [c for c in cifar_names.split("\n") if c]
    return classes


class Experiment:
    n_splits = 5
    seed = 420

    def __init__(self):
        self.models = None
        self.skf = StratifiedKFold(n_splits=self.n_splits)
        self.datasets = None
        self.scores = None

    def fit_models_and_save_scores(self, modes_params, datasets, scores_file):
        self.scores = np.zeros((len(datasets), len(modes_params), self.n_splits))
        for d_id, (data, ds_name) in enumerate(datasets):
            (x, y), (x_test, y_test), n_classes = load_data(data)
            for f_id, (train, valid) in enumerate(self.skf.split(x, np.argmax(y, axis=1))):
                for m_id, m in enumerate(modes_params):
                    model = self.get_training_env(m, ds_name, m_id, f_id, n_classes, x.shape[1:])
                    model.fit(x[train], y[train], x[valid], y[valid])
                    self.scores[d_id, m_id, f_id] = model.predict(x_test, y_test)
        np.save(scores_file, self.scores)
        self.run_stats(scores_file)

    def get_training_env(self, model_params, ds_name, m_id, f_id, n_classes, input_shape):
        model_type = model_params['type']
        grid_size = model_params['grid_size']
        seed = model_params['seed']
        shape = model_params['shape']
        arch_id = model_params['arch_id']

        subinput_shape = (input_shape[0] // grid_size[0], input_shape[1] // grid_size[1], input_shape[2])
        permutations = generate_permutations(seed, grid_size, subinput_shape, shape)

        model_name = f"models/{ds_name}-model_{m_id}-fold_{f_id}-{grid_size[0]}x{grid_size[1]}-"
        model_name += f"{'-permuted' if seed is not None else '-identity'}"
        model_name += f"-{shape}"
        model_name += f"-{model_type}"
        classes = get_classes_names_for_dataset(ds_name)
        if model_type == 'parallel':
            return ModelTraining(model_name, subinput_shape, n_classes, arch_id, permutations, 'parallel', classes)
        elif model_type == 'sequential':
            return TrainingSequential(model_name, subinput_shape, n_classes, arch_id, permutations, classes)

    def run_stats(self, scores_file, alfa=0.05):
        self.scores = np.load(scores_file)
        n_models = self.scores.shape[1]
        for ds_scores in self.scores:
            t_statistic = np.zeros((n_models, n_models))
            p_value = np.zeros((n_models, n_models))
            for i in range(n_models):
                for j in range(n_models):
                    t_statistic[i, j], p_value[i, j] = ttest_rel(ds_scores[i], ds_scores[j])
            print(f'{t_statistic=}')
            print(f'{p_value=}')
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
            'seed': 420,
            'grid_size': (2, 2),
            'shape': 'overlap_cross',
            'arch_id': 0
        },
        {
            'type': 'parallel',
            'seed': 420,
            'grid_size': (2, 2),
            'shape': 'overlap_cross',
            'arch_id': 0
        },
    ]
    datasets = [
        # (mnist, 'mnist'),
        (fashion_mnist, 'fashion_mnist'),
        # (cifar10, 'cifar10'),
    ]
    Experiment().fit_models_and_save_scores(models, datasets, 'scores.npy')
