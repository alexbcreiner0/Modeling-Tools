from dataclasses import dataclass, field
from numpy import array, ndarray


@dataclass
class Params:
    n_agents: int = 100
    T: int = 1000
