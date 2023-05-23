from enums import ModelType

mnist_pixel_shuffle = {
    'v': 'vgg',
    'stem_layer': {
        'filters': 128,
        'upscale_factor': (8, 8),
        'l2': 1e-4,
    },
    'stages': [
        {
            'n_blocks': 3,
            'filters': 32,
            'kernel': 5,
            'maxpool': True,
            'l2': 1e-4,
        },
        {
            'n_blocks': 1,
            'filters': 64,
            'kernel': 3,
            'l2': 1e-4,
        },

    ],
    'outro_layers': [
        'GlobalAveragePooling2D()(x)',
        "Dropout(0.2)(x)",
    ]
}

mnist_pixel_shuffle_inception = {
    'v': 'inception',
    'stem_layer': {
        'filters': 256,
        'upscale_factor': (8, 8),
        'l2': 1e-4,
        'dropout': 0.1,
    },
    'stages': [
        {
            'n_blocks': 3,
            'filters': [16, 32, 16, 32],
            'maxpool': True,
            'dropout': 0.05,
            'l2': 1e-4,
        },

    ],
    'outro_layers': [
        'GlobalAveragePooling2D()(x)',
        "Dropout(0.3)(x)",
    ]

}

mnist_vgg = {
    'v': 'vgg',
    'stages': [
        {
            'n_blocks': 3,
            'filters': 16,
            'kernel': 3,
            'maxpool': True,

            'l2': 1e-4,
        },
        {
            'n_blocks': 3,
            'filters': 32,
            'kernel': 3,
            'maxpool': True,

            'l2': 1e-4,
        },
        {
            'n_blocks': 3,
            'filters': 64,
            'kernel': 3,

            'l2': 1e-4,
        },

    ],
    'outro_layers': [
        'GlobalAveragePooling2D()(x)',
        "Dropout(0.2)(x)",
    ],
}

mnist_resnet_v1 = {
    'v': 1,
    'stem_layer': {
        'filters': 8,
        'kernel': 7,
        'stride': 2,
        #
        'l2': 1e-4,
    },
    'stages': [
        {
            'n_blocks': 3,
            'filters': 16,
            'kernel': 3,

            'l2': 1e-4,
        },
        {
            'n_blocks': 3,
            'filters': 32,
            'kernel': 3,
            'stride': 2,

            'l2': 1e-4,
        },
    ],
    'outro_layers': [
        'GlobalAveragePooling2D()(x)',
        "Dropout(0.2)(x)",
    ],
}

mnist_resnet_v2 = {
    'v': 2,
    'stem_layer': {
        'filters': 8,
        'kernel': 7,
        'stride': 2,
        'l2': 1e-4,
    },
    'stages': [
        {
            'n_blocks': 3,
            'filters': 16,
            'kernel': 3,
            'l2': 1e-4,
        },
        {
            'n_blocks': 3,
            'filters': 32,
            'kernel': 3,
            'stride': 2,
            'l2': 1e-4,
        },
    ],
    'outro_layers': [
        'GlobalAveragePooling2D()(x)',
        "Dropout(0.2)(x)",
    ],
}

mnist_pixel_shuffle_resnet_v1 = {}

mnist_pixel_shuffle_resnet_v2 = {
    'v': 2,
    'stem_layer': {
        'filters': 64,
        'kernel': 7,
        'stride': 7,
        'l2': 1e-4,
        'upscale_factor': (4, 4),
        'dropout': 0.05,
    },
    'stages': [
        {
            'n_blocks': 3,
            'filters': 32,
            'kernel': 3,
            'l2': 1e-4,
            'dropout': 0.05,
        },
    ],
    'outro_layers': [
        'GlobalAveragePooling2D()(x)',
        "Dropout(0.1)(x)",
    ],
}

########################################################################################################################
cifar_pixel_shuffle = {
    'v': 'vgg',
    'stem_layer': {
        'filters': 256,
        # 'upscale_factor': (4, 4),
        'kernel': 4,
        'stride': 4,
        'dropout': 0.05,
        'l2': 1e-5,
    },
    'stages': [
        # {
        #     'n_blocks': 3,
        #     'filters': 32,
        #     'kernel': 5,
        #     # 'dropout': 0.05,
        #     'maxpool': True,
        #     'l2': 1e-5,
        # },
        {
            'n_blocks': 1,
            'filters': 128,
            'kernel': 3,
            'l2': 1e-4,
            'dropout': 0.05,
            # 'maxpool': True,
        },
    ],
    'outro_layers': [
        'GlobalAveragePooling2D()(x)',
        "Dropout(0.2)(x)",
    ]
}

cifar_pixel_shuffle_inception = {
    'v': 'inception',
    'stem_layer': {
        'filters': 32,
        # 'stride': 4,
        'kernel': 7,
        # 'upscale_factor': (2, 2),
        'l2': 1e-4,
        'dropout': 0.05,
    },
    'stages': [
        {
            'n_blocks': 2,
            'filters': [16, 32, 16, 32],
            'maxpool': True,
            'dropout': 0.05,
            'l2': 1e-4,
        },
        {
            'n_blocks': 2,
            'filters': [32, 64, 32, 64],
            # 'maxpool': True,
            'dropout': 0.05,
            'l2': 1e-4,
        },

    ],
    'outro_layers': [
        'GlobalAveragePooling2D()(x)',
        "Dropout(0.1)(x)",
    ]
}

cifar_vgg = {
    'v': 'vgg',
    'stages': [
        {
            'n_blocks': 3,
            'filters': 32,
            'kernel': 3,
            'maxpool': True,

            'l2': 1e-4,
        },
        {
            'n_blocks': 3,
            'filters': 64,
            'kernel': 3,
            'maxpool': True,

            'l2': 1e-4,
        },
        {
            'n_blocks': 3,
            'filters': 128,
            'kernel': 3,

            'l2': 1e-4,
        },

    ],
    'outro_layers': [
        'GlobalAveragePooling2D()(x)',
        "Dropout(0.2)(x)",
    ],
}

cifar_resnet_v1 = {
    'v': 1,
    'stem_layer': {
        'filters': 16,
        'kernel': 7,
        'stride': 2,
        #
        'l2': 1e-4,
    },
    'stages': [
        {
            'n_blocks': 3,
            'filters': 32,
            'kernel': 3,

            'l2': 1e-4,
        },
        {
            'n_blocks': 3,
            'filters': 64,
            'kernel': 3,
            'stride': 2,

            'l2': 1e-4,
        },
    ],
    'outro_layers': [
        'GlobalAveragePooling2D()(x)',
        "Dropout(0.2)(x)",
    ],
}

cifar_resnet_v2 = {
    'v': 2,
    'stem_layer': {
        'filters': 32,
        'kernel': 7,
        'stride': 1,
        'l2': 1e-4,
        'dropout': 0.05,
    },
    'stages': [
        {
            'n_blocks': 2,
            'filters': 16,
            'kernel': 3,
            'l2': 1e-4,
            'dropout': 0.05,
        },
        {
            'n_blocks': 2,
            'filters': 32,
            'kernel': 3,
            'stride': 2,
            'l2': 1e-4,
            'dropout': 0.05,
        },
        {
            'n_blocks': 2,
            'filters': 64,
            'kernel': 3,
            'stride': 2,
            'l2': 1e-4,
            'dropout': 0.05,
        },
    ],
    'outro_layers': [
        'GlobalAveragePooling2D()(x)',
        "Dropout(0.2)(x)",
    ],
}
cifar_pixel_shuffle_resnet_v2 = {
    'v': 2,
    'stem_layer': {
        'filters': 32,
        'stride': 4,
        'kernel': 4,
        'padding': 'same',
        'upscale_factor': (4, 4),
        'l2': 1e-4,
        'dropout': 0.05,
    },
    'stages': [
        {
            'n_blocks': 2,
            'filters': 32,
            'kernel': 3,
            'l2': 1e-4,
            'dropout': 0.05,
        },
        {
            'n_blocks': 2,
            'filters': 64,
            'kernel': 3,
            'stride': 2,
            'l2': 1e-4,
            'dropout': 0.05,
        },
    ],
    'outro_layers': [
        'GlobalAveragePooling2D()(x)',
        "Dropout(0.2)(x)",
    ],
}
cifar_pixel_shuffle_resnet_v1 = {
    'v': 1,
    'stem_layer': {
        'filters': 32,
        'stride': 4,
        'kernel': 4,
        'upscale_factor': (2, 2),
        'l2': 1e-4,
        'dropout': 0.05,
    },
    'stages': [
        {
            'n_blocks': 3,
            'filters': 32,
            'kernel': 3,
            'l2': 1e-4,
            'dropout': 0.05,
        },
        {
            'n_blocks': 3,
            'filters': 64,
            'kernel': 3,
            'stride': 2,
            'l2': 1e-4,
            'dropout': 0.1,
        },
    ],
    'outro_layers': [
        'GlobalAveragePooling2D()(x)',
        "Dropout(0.2)(x)",
    ],
}

cifar_model = {
    ModelType.PIXEL_SHUFFLE_VGG.value: cifar_pixel_shuffle,
    ModelType.VGG.value: cifar_vgg,
    ModelType.RESNET_V1.value: cifar_resnet_v1,
    ModelType.RESNET_V2.value: cifar_resnet_v2,
    ModelType.PIXEL_SHUFFLE_INCEPTION.value: cifar_pixel_shuffle_inception,
    ModelType.PIXEL_SHUFFLE_RESNET_V1.value: cifar_pixel_shuffle_resnet_v1,
    ModelType.PIXEL_SHUFFLE_RESNET_V2.value: cifar_pixel_shuffle_resnet_v2
}

mnist_model = {
    ModelType.PIXEL_SHUFFLE_VGG.value: mnist_pixel_shuffle,
    ModelType.VGG.value: mnist_vgg,
    ModelType.RESNET_V1.value: mnist_resnet_v1,
    ModelType.RESNET_V2.value: mnist_resnet_v2,
    ModelType.PIXEL_SHUFFLE_INCEPTION.value: mnist_pixel_shuffle_inception,
    ModelType.PIXEL_SHUFFLE_RESNET_V1.value: mnist_pixel_shuffle_resnet_v1,
    ModelType.PIXEL_SHUFFLE_RESNET_V2.value: mnist_pixel_shuffle_resnet_v2
}
