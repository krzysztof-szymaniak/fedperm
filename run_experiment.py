import os
import pathlib
from pprint import pprint

import numpy as np
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.client import device_lib

from datasets import load_data, get_classes_names_for_dataset
from experimen_configs import get_experiment_config, datasets
from model.training import train_model, predict, skip_training
from permutation.permutations import generate_permutations

print(device_lib.list_local_devices())
exp_id = "1"

# if lab:


print(f'{exp_id=}')

lab = os.path.exists('lab')
lab = True

run_faulty_test = True
run_from_saved_models = False


def parse_config(model_params, ds_name, f_id, n_classes, input_shape):
    print(model_params)
    mode = model_params['type']
    grid_size = model_params['grid_size']
    seed = model_params['seed']
    overlap = model_params['overlap']
    aggr_scheme = model_params['aggregation']
    scheme = model_params.get('permutation_scheme')
    arch = model_params.get('model_architecture')

    sub_input_shape = (input_shape[0] // grid_size[0], input_shape[1] // grid_size[1], input_shape[2])

    permutations = generate_permutations(seed, grid_size, sub_input_shape, overlap, scheme)

    model_path = f"{'experiments' if not run_from_saved_models else 'saved'}/v{exp_id}/{ds_name}/{mode}/" \
                 f"{arch.value}/" \
                 f"{'perm-' if seed is not None else 'identity'}" \
                 f"{scheme.name.lower() if scheme and seed else ''}/" \
                 f"ov_{overlap.name.lower()}-agg_{aggr_scheme.name.lower()}-{grid_size[0]}x{grid_size[1]}/fold_{f_id}"

    classes = get_classes_names_for_dataset(ds_name)

    return (model_path, permutations, sub_input_shape, n_classes, ds_name, arch, mode, aggr_scheme), classes


class Experiment:
    n_splits = 5

    def __init__(self):
        self.models = None
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=69)
        self.datasets = None
        self.scores = None

    def fit_models_and_save_scores(self, datasets, scores_file):
        models_params = list(get_experiment_config())
        exp_dir = f'experiments/v{exp_id}'
        pathlib.Path(exp_dir).mkdir(exist_ok=True, parents=True)
        with open(f'{exp_dir}/experiment_config', 'w') as conf:
            pprint(models_params, conf)
        self.scores = np.zeros((len(datasets), len(models_params), self.n_splits))
        for d_id, ds_name in enumerate(datasets):
            (x, y), (x_test, y_test), n_classes = load_data(ds_name)
            if n_classes != 2:
                y_s = np.argmax(y, axis=1)
            else:
                y_s = y
            for f_id, (train, valid) in enumerate(self.skf.split(x, y_s)):
                for m_id, m in enumerate(models_params):
                    params, classes_names = parse_config(m, ds_name, f_id, n_classes, x.shape[1:])
                    model_path = params[0]
                    if not skip_training(model_path):
                        data = (x[train], y[train], x[valid], y[valid])
                        train_model(data, *params)
                    if run_faulty_test:
                        print("Running test with faulty data")
                        invalid_test_config = {
                            'seed': 11111,
                            'overlap': m['overlap'],
                            'grid': m['grid_size'],
                            'scheme': m['permutation_scheme']
                        }
                        predict(model_path, x_test, y_test, classes_names, invalid_test=invalid_test_config,
                                test_dir_name='test_wrong_seed')
                    self.scores[d_id, m_id, f_id] = predict(model_path, x_test, y_test, classes_names)
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
            np.save(f'models/v{exp_id}/t_stat{exp_id}.npy', t_statistic)
            print(f'{t_statistic=}')
            print(f'{p_value=}')
            np.save(f'models/v{exp_id}/p_stat{exp_id}.npy', p_value)
            advantage = np.zeros((n_models, n_models))
            advantage[t_statistic > 0] = 1
            significance = np.zeros((n_models, n_models))
            significance[p_value <= alfa] = 1
            stat_better = significance * advantage
            print(stat_better)


if __name__ == '__main__':
    Experiment().fit_models_and_save_scores(datasets, f'models/v{exp_id}/scores{exp_id}.npy')
