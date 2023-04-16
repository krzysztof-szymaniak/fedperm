fashion_stages = [
    {
        'n_blocks': 2,
        'filters': [64, 64, 64],
        'kernel': 5,
        'stride': 1,
        'inc_filters': [64, 64, 64, 64]
    },
    {
        'n_blocks': 2,
        'filters': [128, 128, 128],
        'kernel': 5,
        'stride': 2,
        'inc_filters': [128, 128, 128, 128]
    },
    {
        'n_blocks': 2,
        'filters': [256, 256, 256],
        'kernel': 3,
        'stride': 2,
        # 'inc_filters': [128, 128, 128, 512]
    },
]
cifar_inc_resnet = [
    {
        'n_blocks': 2,
        'filters': [64, 64, 64],
        'kernel': 5,
        'stride': 1,
        'inc_filters': [64, 64, 64, 64]
    },
    {
        'n_blocks': 2,
        'filters': [128, 128, 128],
        'kernel': 5,
        'stride': 2,
        'inc_filters': [128, 128, 128, 128]
    },
    {
        'n_blocks': 1,
        'filters': [256, 256, 256],
        'kernel': 3,
        'stride': 2,
        # 'inc_filters': [256, 256, 256, 256]
    },
]

cats_dogs_stages = [
    {  # 100
        'n_blocks': 3,
        'filters': [64, 64, 64],
        'kernel': 3,
        'stride': 1,
        # 'inc_filters': [64, 64, 64, 64]
    },
    {  # 50
        'n_blocks': 5,
        'filters': [128, 128, 128],
        'kernel': 3,
        'stride': 2,
        # 'inc_filters': [128, 128, 128, 128]
    },
    {  # 25
        'n_blocks': 5,
        'filters': [256, 256, 256],
        'kernel': 3,
        'stride': 2,
        # 'inc_filters': [256, 256, 256, 256]
    },
    {  # 12
        'n_blocks': 3,
        'filters': [512, 512, 512],
        'kernel': 3,
        'stride': 2,
        # 'inc_filters': [512, 512, 512, 512]
    },
    {  # 12
        'n_blocks': 1,
        'filters': [1024, 1024, 1024],
        'kernel': 3,
        'stride': 2,
        # 'inc_filters': [512, 512, 512, 512]
    },
]

stem_layer = {
    'filters': 32,
    'kernel': 5,
    'stride': 1,
}

fashion_config = {
    'stem_layer': stem_layer,
    'stages': fashion_stages,
    'output_layers': [
        'GlobalAveragePooling2D()(x)',
        # 'Flatten()(x)',
        "Dense(256, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(x)",
    ]
}

cifar_config = {
    'stem_layer': stem_layer,
    'stages': cifar_inc_resnet,
    'output_layers': [
        'GlobalAveragePooling2D()(x)',
        # 'Flatten()(x)',
        "Dense(512, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(x)",
    ]
}

cats_dogs_config = {
    'stem_layer': {
        'filters': 128,
        'kernel': 11,
        'stride': 2,
    },
    'stages': cats_dogs_stages,
    'output_layers': [
        'GlobalAveragePooling2D()(x)',
        # 'Flatten()(x)',
        "Dense(512, kernel_initializer='he_normal', activation='relu', kernel_regularizer=l2(weight_decay))(x)",
    ]
}
