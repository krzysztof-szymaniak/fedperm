n_filters = 32

bigger_mnist_config = {
    'v': 0,
    'stem_layer': {
        'pixel_shuffler': (8, 8),
        'filters': 256,
        # 'filters': 16,
        # 'kernel': 7,
        # 'stride': 1,
        'l2': 1e-4,
        'dropout': 0.1,
    },
    'stages': [
        {
            'n_blocks': 2,  # 3
            'filters': n_filters,
            'kernel': 5,
            # 'stride': 1,
            'maxpool': True,
            'l2': 1e-4,
            'dropout': 0.05,
        },
        {
            'n_blocks': 1,
            'filters': n_filters * 2,
            'kernel': 3,
            # 'stride': 2,
            'dropout': 0.05,
        },

    ],
    'outro_layers': [
        'GlobalAveragePooling2D()(x)',
        # 'Flatten()(x)',
        "Dropout(0.3)(x)",
        f"Dense({n_filters * 2}, activation='relu', kernel_regularizer=l2(1e-4))(x)"
    ]
}

n_filters = 64

cifar_config = {
    'v': 0,
    # 'blockwise': True,
    'stem_layer': {
        # 'filters': 32,
        # 'kernel': 7,
        'filters': 512,
        'pixel_shuffler': (8, 8),
        # 'stride': 1,
        'dropout': 0.3,
        'l2': 1e-4,
    },
    'stages': [
        {
            'n_blocks': 2,
            'filters': n_filters,
            'kernel': 5,
            # 'stride': 1,
            'maxpool': True,
            'dropout': 0.1,
            'l2': 1e-4,
        },
        {
            'n_blocks': 1,
            'filters': n_filters * 2,
            'kernel': 3,
            # 'stride': 2,
            # 'maxpool': True,
            'dropout': 0.1,
            'l2': 1e-4,
        },

    ],
    'outro_layers': [
        'GlobalAveragePooling2D()(x)',
        # 'Flatten()(x)',
        "Dropout(0.3)(x)",
        f"Dense({n_filters * 2}, activation='relu', kernel_regularizer=l2(1e-4))(x)"
    ]
}
