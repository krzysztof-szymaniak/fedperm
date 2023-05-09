from enum import Enum


class Aggregation(Enum):
    CONCATENATE = 'concat'
    ADD = 'add'
    AVERAGE = 'avg'
    CONCAT_STRIP = 'concat_strip'


class Overlap(Enum):
    CENTER = 'center'
    CROSS = 'cross'
    EDGES = 'edges'
    CORNERS = 'corners'
    FULL = 'full'


class PermSchemas(Enum):
    WHOLE = (0, False)
    SHUFFLE_WITHIN_BLOCKS_2 = (2, False)
    SHUFFLE_WITHIN_BLOCKS_4 = (4, False)
    SHUFFLE_BLOCKS_2 = (2, True)
    SHUFFLE_BLOCKS_4 = (4, True)
