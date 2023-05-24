from enums import ModelType


def get_config(model_type):
    configs = {
        ModelType.ADAPTATION_VGG: adaptation_vgg(
            filters=[128, 32, 64],
            n=3,
        ),
        ModelType.ADAPTATION_INCEPTION: adaptation_inception(
            filters=[128, [32, 64, 32, 64]],
            n=3,
        ),
        ModelType.ADAPTATION_RESNET_V2: adaptation_resnet_v2(
            filters=[128, 32, 64],
            n=3,
        ),
        ModelType.CONV_MIXER: conv_mixer(
            filters=196,
            n=6,
        ),
    }
    return configs[model_type]


def conv_mixer(filters, n):
    return {
        'v': 'conv-mixer',
        'stem_layer': {
            'filters': filters,
            'stride': 4,
            'kernel': 4,
            'dropout': 0.1,
        },
        'stages': [
            {
                'n_blocks': n,
                'filters': filters,
                'dropout': 0.1,
                'kernel': 5,
            },
        ],
        'outro_layers': [
            'GlobalAveragePooling2D()(x)',
        ],
    }


def adaptation_resnet_v2(filters, n):
    return {
        'v': 2,
        'stem_layer': {
            'filters': filters[0],
            'stride': 4,
            'kernel': 4,
            'dropout': 0.05,
        },
        'stages': [
            {
                'n_blocks': n,
                'filters': filters[1],
                'kernel': 3,
                'dropout': 0.05,
            },
            {
                'n_blocks': n,
                'filters': filters[2],
                'kernel': 3,
                'stride': 2,
                'dropout': 0.05,
            },
        ],
        'outro_layers': [
            'GlobalAveragePooling2D()(x)',
        ],
    }


def adaptation_inception(filters, n):
    return {
        'v': 'inception',
        'stem_layer': {
            'filters': filters[0],
            'stride': 4,
            'kernel': 4,
            'l2': 1e-4,
            'dropout': 0.05,
        },
        'stages': [
            {
                'n_blocks': n,
                'filters': filters[1],
                'dropout': 0.05
            },
        ],
        'outro_layers': [
            'GlobalAveragePooling2D()(x)',
        ]
    }


def adaptation_vgg(filters, n):
    return {
        'v': 'vgg',
        'stem_layer': {
            'filters': filters[0],
            'kernel': 4,
            'stride': 4,
            'dropout': 0.1,
        },
        'stages': [
            {
                'n_blocks': n,
                'filters': filters[1],
                'kernel': 3,
                'dropout': 0.05,
                'maxpool': True,
            },
            {
                'n_blocks': n,
                'filters': filters[2],
                'kernel': 3,
                'dropout': 0.05,
            },
        ],
        'outro_layers': [
            'GlobalAveragePooling2D()(x)',
        ]
    }
