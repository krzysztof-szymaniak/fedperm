from enum import Enum


class Aggregation(Enum):
    STRIP_CONCAT = 'strip_concat'
    CONCAT = 'concat'
    ADD = 'add'
    AVERAGE = 'avg'


class Overlap(Enum):  # scheme, base_grid
    CENTER = ('center', True)
    CROSS = ('cross', False)
    EDGES = ('edges', False)
    CORNERS = ('corners', False)
    FULL = ('full', True)
    NONE = ('none', True)


class PermSchemas(Enum):
    FULL = None
    BS_16_3 = (16, 16, 3)
    BS_8_3 = (8, 8, 3)
    BS_4_3 = (4, 4, 3)
    BS_14_1 = (14, 14, 1)
    BS_7_1 = (7, 7, 1)
    BS_2_3 = (2, 2, 3)
    BS_2_1 = (2, 2, 1)
    IDENTITY = 'identity'


class ModelType(Enum):
    ADAPTATION_VGG = 'ps-vgg'
    VGG = 'vgg'
    RESNET_V1 = 'resnet-v1'
    RESNET_V2 = 'resnet-v2'
    ADAPTATION_INCEPTION = 'ps-inception'
    ADAPTATION_RESNET_V1 = 'ps-resnet-v1'
    ADAPTATION_RESNET_V2 = 'ps-resnet-v2'
    VISION_TRANSFORMER = 'vis-trans'
    CONV_MIXER = 'conv-mixer'
    MLP = 'mlp'
