from numpy import array, ndarray
from dataclasses import dataclass, field

@dataclass
class Params:
    # Define your parameters here like so:
    # Defaults are not required but may cause crashes if your params.yml file in the data folder fails to define a parameter that doesn't have one.
    x: float = 5 # name: type = default
    A: ndarray = field(default_factory=lambda: array("<your matrix goes here>") # <- do this if you need to have a matrix parameter default

