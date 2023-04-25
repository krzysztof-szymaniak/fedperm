bigger_mnist_config = {
    'v': 2,
    'separable': True,
    'stem_layer': {
        'filters': 16,
        'kernel': 3,
        'stride': 1,
        # 'dropout': 0.01,
    },
    'stages': [
        {
            'n_blocks': 2,
            'filters': 16,
            'kernel': 3,
            'stride': 1,
            # 'dropout': 0.05
        },
        {
            'n_blocks': 3,
            'filters': 32,
            'kernel': 3,
            'stride': 2,
            # 'dropout': 0.05,
        },
        {
            'n_blocks': 1,
            'filters': 64,
            'kernel': 3,
            'stride': 2,
            # 'dropout': 0.05,
        },
    ],
    'output_layers': [
        'GlobalAveragePooling2D()(x)',
    ]
}

cifar_config = {
    'v': 2,
    'separable': True,
    'stem_layer': {
        'filters': 32,
        'kernel': 3,
        'stride': 1,
        # 'dropout': 0.05
    },
    'stages': [
        {
            'n_blocks': 3,
            'filters': 32,
            'kernel': 3,
            'stride': 1,
            # 'dropout': 0.05,
        },
        {
            'n_blocks': 3,
            'filters': 64,
            'kernel': 3,
            'stride': 2,
            # 'dropout': 0.1
        },
        {
            'n_blocks': 1,
            'filters': 128,
            'kernel': 3,
            'stride': 2,
            # 'dropout': 0.05,
        },
    ],
    'output_layers': [
        'GlobalAveragePooling2D()(x)',
    ]
}

smaller_mnist_config = {
    'stem_layer': {
        'filters': 16,
        'kernel': 3,
        'stride': 1,
        # 'dropout': 0.01,
    },
    'v': 1,
    'stages': [
        {
            'n_blocks': 2,
            'filters': 16,
            'kernel': 3,
            'stride': 1,
            # 'dropout': 0.05
        },
        {
            'n_blocks': 3,
            'filters': 32,
            'kernel': 3,
            'stride': 2,
            # 'dropout': 0.05,
        },
        {
            'n_blocks': 1,
            'filters': 64,
            'kernel': 3,
            'stride': 2,
            # 'dropout': 0.05,
        },
    ],
    'output_layers': [
        'GlobalAveragePooling2D()(x)',
    ]
}
