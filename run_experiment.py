import gc
import os
import shutil

from enums import Overlap

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress logging
import pathlib
import pickle
from contextlib import redirect_stdout
from copy import copy
from pprint import pprint

import numpy as np
from scipy.stats import ttest_rel
from sklearn.model_selection import RepeatedStratifiedKFold
from tensorflow.python.client import device_lib
from tabulate import tabulate

from datasets import load_data, get_classes_names_for_dataset
from experiment_configs import get_experiment
from model.training import train_model, predict, skip_training
from permutation.permutations import generate_permutations

print(device_lib.list_local_devices())

experiment_name = 'exp-3'

N_REPEATS = 5
N_SPLITS = 2
kfold = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)

ds = [
    'cifar10',
    # 'cifar100',
    # 'fashion_mnist',
    # 'emnist-letters',
    # 'cats_vs_dogs',
    # 'mnist',
    # 'eurosat',
]


def reuse_trained_models(model_path, overlap):
    source_path = model_path.replace('ov_' + overlap.name.lower(), 'ov_' + Overlap.FULL.name.lower())
    if os.path.exists(source_path) and source_path != model_path:
        if overlap == Overlap.CENTER:
            n_models = 5
        elif overlap == Overlap.NONE:
            n_models = 4
        else:
            return
        for i in range(n_models):
            sub_source_path = os.path.join(source_path, 'subs', str(i))
            target_path = os.path.join(model_path, 'subs', str(i))
            if os.path.exists(target_path):
                continue
            print(f"Reusing model {sub_source_path}")
            shutil.copytree(sub_source_path, target_path)


def get_path_from_config(model_params, ds_name, f_id):
    mode = model_params['type']
    grid_size = model_params['grid_size']
    seed = model_params['seed']
    overlap = model_params['overlap']
    aggr_scheme = model_params['aggregation']
    scheme = model_params.get('permutation_scheme')
    arch = model_params.get('model_architecture')
    model_path = f"experiments/{experiment_name}/{ds_name}/{mode}/" \
                 f"{arch.value}/" \
                 f"{'perm-' if seed is not None else 'identity'}" \
                 f"{scheme.name.lower() if scheme and seed else ''}/" \
                 f"ov_{overlap.name.lower()}-agg_{aggr_scheme.name.lower()}-{grid_size[0]}x{grid_size[1]}/fold_{f_id}"
    return model_path


def parse_config(model_params, ds_name, f_id, n_classes, input_shape):
    mode = model_params['type']
    grid_size = model_params['grid_size']
    seed = model_params['seed']
    overlap = model_params['overlap']
    aggr_scheme = model_params['aggregation']
    scheme = model_params.get('permutation_scheme')
    arch = model_params.get('model_architecture')

    print(f'{input_shape=} {grid_size=}')
    sub_input_shape = (input_shape[0] // grid_size[0], input_shape[1] // grid_size[1], input_shape[2])

    permutations = generate_permutations(seed, grid_size, sub_input_shape, overlap, scheme)

    model_path = get_path_from_config(model_params, ds_name, f_id)

    classes = get_classes_names_for_dataset(ds_name)
    reuse_trained_models(model_path, overlap)
    print(
        f"Running with ({mode}, {arch.name.lower()}, {scheme.name.lower()}, {aggr_scheme.name.lower()},"
        f" {overlap.name.lower()})")
    return (model_path, permutations, sub_input_shape, n_classes, ds_name, arch, mode, aggr_scheme), classes


def train_models(data, models):
    for d_id, ds_name in enumerate(data):
        (x, y), _, n_classes = load_data(ds_name)
        y_s = np.argmax(y, axis=1) if n_classes != 2 else y
        for m_id, m_config in enumerate(models):
            for f_id, (train, valid) in enumerate(kfold.split(x, y_s)):
                params, _ = parse_config(m_config, ds_name, f_id, n_classes, x.shape[1:])
                print(f'{m_id=} , {f_id=}')
                model_path = params[0]
                if not skip_training(model_path):
                    train_model(x[train], y[train], x[valid], y[valid], *params)


def evaluate_models(data, models, scores_path, run_faulty_test=True):
    configs = [[[None for _ in range(kfold.get_n_splits())] for _ in range(len(models))] for _ in range(len(data))]
    scores = np.zeros((len(data), len(models), kfold.get_n_splits()))
    for d_id, ds_name in enumerate(data):
        _, (x_test, y_test), n_classes = load_data(ds_name)
        for f_id in range(kfold.get_n_splits()):
            for m_id, m_config in enumerate(models):
                params, classes_names = parse_config(m_config, ds_name, f_id, n_classes, x_test.shape[1:])
                model_path = params[0]
                if run_faulty_test:
                    print("Running test with invalid key")
                    invalid_test_config = copy(m_config)
                    invalid_test_config['seed'] = 1111
                    acc = predict(
                        model_path, x_test, y_test, params[2], classes_names,
                        invalid_test=invalid_test_config,
                        test_dir_name='test_invalid_perm'
                    )
                    print("False Accuracy: ", acc)
                acc = predict(model_path, x_test, y_test, params[2], classes_names, mode=params[6])
                print("Accuracy: ", acc)
                scores[d_id, m_id, f_id] = acc
                configs[d_id][m_id][f_id] = m_config
    result = {
        'configs': configs,
        'scores': scores
    }
    with open(scores_path, 'wb') as file:
        pickle.dump(result, file)
    return result


def run_tests(data):
    exp_dir = f'experiments/{experiment_name}'
    scores_path = f'{exp_dir}/scores'
    pathlib.Path(exp_dir).mkdir(exist_ok=True, parents=True)
    models_params = get_experiment()
    with open(f'{exp_dir}/experiment_config', 'w') as conf:
        pprint(models_params, conf)

    train_models(data, models_params)

    if not os.path.exists(scores_path):
        results = evaluate_models(data, models_params, scores_path)
    else:
        with open(scores_path, 'rb') as file:
            results = pickle.load(file)
    run_stats(results['scores'], exp_dir, models_params)


def run_stats(scores, exp_dir, models_params, alfa=0.05):
    headers = []
    for m_config in models_params:
        overlap = m_config['overlap'].name.lower()
        scheme = m_config.get('permutation_scheme').name.lower()
        m_type = m_config['type'][:4]
        headers.append(f'CM-{m_type}-{overlap}-{scheme}')

    pathlib.Path(exp_dir).mkdir(exist_ok=True)
    n_models = scores.shape[1]
    for d_id, ds_scores in enumerate(scores):
        ds_name = ds[d_id]
        t_statistic = np.zeros((n_models, n_models))
        p_value = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    t_statistic[i, j], p_value[i, j] = ttest_rel(ds_scores[i], ds_scores[j])

        advantage = np.zeros((n_models, n_models))
        advantage[t_statistic > 0] = 1
        significance = np.zeros((n_models, n_models))
        significance[p_value <= alfa] = 1
        adv_table = significance * advantage
        print_pretty_table(t_statistic, p_value, adv_table, save_path=f'{exp_dir}/{ds_name}', headers=headers)
        print_some_more_stats(ds_scores, models_params)


def print_some_more_stats(ds_scores, models_params, ds_name='cifar10'):
    n_folds = ds_scores.shape[1]
    for m_id, m_config in enumerate(models_params):
        scores = []
        for f_id in range(n_folds):
            model_path = get_path_from_config(m_config, ds_name, f_id)
            scores_path = os.path.join(model_path, 'test', 'sub_preds.npy')
            if os.path.exists(scores_path):
                scores.append(np.load(scores_path))
        if scores:
            print(f"avg subscores: {np.round(np.average(scores, axis=0), 4)}")
            print(f"std subscores: {np.round(np.std(scores, axis=0), 4)}")
        print(f'avg & std total score {np.average(ds_scores[m_id]):.4f} & {np.std(ds_scores[m_id]):.4f}')
        print()
        print()


def print_pretty_table(t_statistic, p_value, advantage_table, save_path, headers):
    names_column = np.array([[n] for n in headers])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")

    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".8f")

    adv_table = np.concatenate((names_column, advantage_table), axis=1)
    adv_table = tabulate(adv_table, headers)

    results = f"t-statistic:\n {t_statistic_table}" \
              f"\n\np-value:\n{p_value_table}" \
              f"\n\nadvantage-table:\n{adv_table}"
    print(results)
    with open(f'{save_path}/summary.txt', 'w') as f:
        with redirect_stdout(f):
            print(results)


if __name__ == '__main__':
    run_tests(ds)
