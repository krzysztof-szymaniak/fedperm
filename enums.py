from enum import Enum


class Aggregation(Enum):
    STRIP_CONCAT = 'strip_concat'
    CONCAT = 'concat'
    ADD = 'add'
    AVERAGE = 'avg'


class Overlap(Enum):
    #       (scheme_name, apply_basic_grid)
    CENTER = ('center', True)
    CROSS = ('cross', False)
    EDGES = ('edges', False)
    CORNERS = ('corners', False)
    FULL = ('full', True)
    NONE = ('none', True)


class PermSchemas(Enum):
    NAIVE = "naive"
    BS_16 = (16, 16,)  # blockwise shuffle size
    BS_14 = (14, 14,)
    BS_8 = (8, 8,)
    BS_7 = (7, 7,)
    BS_4 = (4, 4,)
    BS_2 = (2, 2,)
    IDENTITY = 'identity'


class ModelType(Enum):
    # ADAPTATION_VGG = 'ps-vgg'
    # VGG = 'vgg'
    # RESNET_V1 = 'resnet-v1'
    # RESNET_V2 = 'resnet-v2'
    # ADAPTATION_INCEPTION = 'ps-inception'
    # ADAPTATION_RESNET_V1 = 'ps-resnet-v1'
    # ADAPTATION_RESNET_V2 = 'ps-resnet-v2'
    # VISION_TRANSFORMER = 'vis-trans'
    CONV_MIXER = 'conv-mixer'
