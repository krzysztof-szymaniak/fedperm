import os
import pathlib
import pickle
from pprint import pprint

import numpy as np
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.client import device_lib

from datasets import load_data, get_classes_names_for_dataset
from experiment_configs import get_experiment_config, datasets
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
    print(
        f"Running with ({mode}, {arch.name.lower()}, {scheme.name.lower()}, {aggr_scheme.name.lower()}, {overlap.name.lower()})")
    return (model_path, permutations, sub_input_shape, n_classes, ds_name, arch, mode, aggr_scheme), classes


def train_models(data, models, skf):
    for d_id, ds_name in enumerate(data):
        (x, y), _, n_classes = load_data(ds_name)
        if n_classes != 2:
            y_s = np.argmax(y, axis=1)
        else:
            y_s = y
        for f_id, (train, valid) in enumerate(skf.split(x, y_s)):
            for m_id, m_config in enumerate(models):
                params, _ = parse_config(m_config, ds_name, f_id, n_classes, x.shape[1:])
                model_path = params[0]
                if not skip_training(model_path):
                    train_model(x[train], y[train], x[valid], y[valid], *params)


def evaluate_models(data, models, skf, scores_path):
    configs = [[[None for _ in range(skf.n_splits)] for _ in range(len(models))] for _ in range(len(data))]
    scores = np.zeros((len(data), len(models), skf.n_splits))
    for d_id, ds_name in enumerate(data):
        _, (x_test, y_test), n_classes = load_data(ds_name)
        for f_id in range(skf.n_splits):
            for m_id, m_config in enumerate(models):
                params, classes_names = parse_config(m_config, ds_name, f_id, n_classes, x_test.shape[1:])
                model_path = params[0]
                if run_faulty_test:
                    print("Running test with faulty data")
                    invalid_test_config = {
                        'seed': 11111,
                        'overlap': m_config['overlap'],
                        'grid': m_config['grid_size'],
                        'scheme': m_config['permutation_scheme']
                    }
                    predict(
                        model_path, x_test, y_test, params[2], classes_names,
                        invalid_test=invalid_test_config,
                        test_dir_name='test_invalid_perm'
                    )
                acc = predict(model_path, x_test, y_test, params[2], classes_names)
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


def run_tests(data, n_splits=5):
    models_params = list(get_experiment_config())
    exp_dir = f'experiments/v{exp_id}'
    scores_path = f'{exp_dir}/scores'
    pathlib.Path(exp_dir).mkdir(exist_ok=True, parents=True)
    with open(f'{exp_dir}/experiment_config', 'w') as conf:
        pprint(models_params, conf)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=69)
    train_models(data, models_params, skf)
    if not os.path.exists(scores_path):
        results = evaluate_models(data, models_params, skf, scores_path)
    else:
        with open(scores_path, 'rb') as file:
            results = pickle.load(file)
    stats_path = f'{exp_dir}/stats'
    run_stats(results['scores'], stats_path)


def run_stats(scores, stats_path, datasets, alfa=0.05):
    print(scores)
    pathlib.Path(stats_path).mkdir(exist_ok=True)
    n_models = scores.shape[1]
    for d_id, ds_scores in enumerate(scores):
        t_statistic = np.zeros((n_models, n_models))
        p_value = np.zeros((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                t_statistic[i, j], p_value[i, j] = ttest_rel(ds_scores[i], ds_scores[j])
        np.save(f'{stats_path}/{datasets[d_id]}-t_stat.npy', t_statistic)
        print(f'{t_statistic=}')
        print(f'{p_value=}')
        np.save(f'{stats_path}/{datasets[d_id]}-p_stat.npy', p_value)
        advantage = np.zeros((n_models, n_models))
        advantage[t_statistic > 0] = 1
        significance = np.zeros((n_models, n_models))
        significance[p_value <= alfa] = 1
        stat_better = significance * advantage
        print(stat_better)
        np.save(f'{stats_path}/{datasets[d_id]}-stat_better.npy', stat_better)


if __name__ == '__main__':
    run_tests(datasets)
