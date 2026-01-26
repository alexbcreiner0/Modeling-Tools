from dataclasses import dataclass, field
from numpy import array, ndarray


@dataclass
class Params:
    new_param: float
    l: ndarray = field(default_factory= lambda: array([0.2, 0.8]))
    L: ndarray = field(default_factory= lambda: array([11.0, 24.0]))
    A: ndarray = field(default_factory= lambda: array([[0.8, 0.2], [0.0, 0.0]]))
    b: ndarray = field(default_factory= lambda: array([0.0, 0.3475]))
    q: ndarray = field(default_factory= lambda: array([55.0, 30.0]))
    p: ndarray = field(default_factory= lambda: array([2.2, 0.8]))
    init_MELT: int = 1
    output_rate: float = 0.1
    labor_rate: float = 0.034
    T: int = 70
    stop_halfway: bool = False
