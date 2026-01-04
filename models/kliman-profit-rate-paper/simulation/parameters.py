from numpy import array, ndarray
from dataclasses import dataclass, field

@dataclass
class Params:
    l: ndarray = field(default_factory=lambda: array([0.2, 0.8]))
    L: ndarray = field(default_factory=lambda: array([11, 24]))
    A: ndarray = field(default_factory=lambda: array([[0.8, 0.2], [0.0, 0.0]])) # <- do this if you need to have a matrix parameter default
    b: ndarray = field(default_factory=lambda: array([0.0, 0.3475]))
    q: ndarray = field(default_factory=lambda: array([55,30]))
    p: ndarray = field(default_factory=lambda: array([2.2, 0.8]))
    init_MELT: float= 1
    output_rate: float = 0.1
    labor_rate: float = 0.034
    T: int = 70
    stop_halfway: bool = False

