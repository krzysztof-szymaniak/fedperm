from enum import Enum


class Aggregation(Enum):
    CONCATENATE = 'concat'
    ADD = 'add'
    AVERAGE = 'avg'


class Overlap(Enum):  # scheme, base_grid
    CENTER = ('center', False)
    CROSS = ('cross', False)
    EDGES = ('edges', False)
    CORNERS = ('corners', False)
    FULL = ('full', True)


class PermSchemas(Enum):
    FULL = None
    BS_16_3 = (16, 16, 3)
    BS_8_3 = (8, 8, 3)
    BS_4_3 = (4, 4, 3)
    BS_14_1 = (14, 14, 1)
    BS_7_1 = (7, 7, 1)
    BS_2_3 = (2, 2, 3)
    BS_2_1 = (2, 2, 1)


class ModelType(Enum):
    PIXEL_SHUFFLE_VGG = 'ps-vgg'
    VGG = 'vgg'
    RESNET_V1 = 'resnet-v1'
    RESNET_V2 = 'resnet-v2'
    PIXEL_SHUFFLE_INCEPTION = 'ps-inception'
    PIXEL_SHUFFLE_RESNET_V1 = 'ps-resnet-v1'
    PIXEL_SHUFFLE_RESNET_V2 = 'ps-resnet-v2'
    VISION_TRANSFORMER = 'vis-trans'
    CONV_MIXER = 'conv-mixer'
    MLP = 'mlp'
