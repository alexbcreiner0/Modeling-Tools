from numpy import array, ndarray
from dataclasses import dataclass, asdict, is_dataclass
import numpy as np

@dataclass
class Params:
    y1i: float
    y2i: float
    e: float
    a: float
    k1: float
    k2: float
    r: float # reproduction rate for employed workers
    s: float # attrition rate for unemployed workers
    N0: int # initial number of workers
    T: int
    dt: float = 0.01
    balanced: bool= False
    balanced_indep: str= "y1" # options are y1 and y2, only matters if balanced= True
    money_reinvestment: bool= False
    analytic_soln: bool= True
    enforce_nonneg: bool= False
    enforce_population_constraints: bool= False

