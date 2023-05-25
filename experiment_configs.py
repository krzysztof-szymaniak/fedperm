from enums import Overlap, PermSchemas, ModelType, Aggregation

datasets = [
    'cifar10',
    # 'cifar100',
    # 'fashion_mnist',
    # 'emnist-letters',
    # 'cats_vs_dogs',
    # 'mnist',
    # 'eurosat',
]


def get_configs(overlap, aggr, perm_scheme, model_arch):
    seed = 42
    return [
        {
            'type': 'composite',
            'seed': seed,
            'permutation_scheme': perm_scheme,
            'grid_size': (2, 2),
            'overlap': overlap,
            'aggregation': aggr,
            'model_architecture': model_arch,
        },
        {
            'type': 'composite',
            'seed': None,
            'permutation_scheme': PermSchemas.IDENTITY,
            'grid_size': (2, 2),
            'overlap': overlap,
            'aggregation': aggr,
            'model_architecture': model_arch,
        },
    ]


overlaps = [
    Overlap.NONE,
    # Overlap.CENTER,
    # Overlap.FULL,
]

aggregations = [
    Aggregation.STRIP_CONCAT,
    # Aggregation.CONCAT,
]

perm_schemas = [
    PermSchemas.BS_4_3,
    # PermSchemas.FULL,
]

model_types = [
    ModelType.CONV_MIXER,
    # ModelType.ADAPTATION_RESNET_V2,
    # ModelType.ADAPTATION_INCEPTION,
    # ModelType.ADAPTATION_VGG
]


def get_experiment_config():
    for ov in overlaps:
        for aggr in aggregations:
            for perm in perm_schemas:
                for model in model_types:
                    seedful, seedless = get_configs(
                        overlap=ov,
                        aggr=aggr,
                        perm_scheme=perm,
                        model_arch=model
                    )
                    yield seedful
                    yield seedless
