from enum import Enum


class Aggregation(Enum):
    CONCATENATE = 'concat'
    ADD = 'add'
    AVERAGE = 'avg'


class Overlap(Enum):
    CENTER = 'center'
    CROSS = 'cross'
    EDGES = 'edges'
    CORNERS = 'corners'
    FULL = 'full'
