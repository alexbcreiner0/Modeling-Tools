from dataclasses import dataclass, field
from numpy import array, ndarray


@dataclass
class Params:
    A: ndarray = field(default_factory=lambda: array([[0.0, 0.40832602, 0.40832602, 0.10208151, 0.0], [0.10208151, 0.0, 0.10208151, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.20416301], [0.0, 0.40832602, 0.0, 0.0, 0.0], [0.71457054, 0.0, 0.0, 0.0, 0.91873356]]))
    l: ndarray = field(default_factory=lambda: array([1.0, 1.0, 1.0, 1.0, 1.0]))
    T: int = 200
    tau: float = 16.0
    avg_interval: float = 50.0
    scenario: str = 'divergent_current_algo'
