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
    w: float
    r: float # reproduction rate for employed workers
    s: float # attrition rate for unemployed workers
    N0: int # initial number of workers
    T: int

# def params_from_mapping(map: dict):
#     return Params(
#         y1i = float(map["y1"]),
#         y2i = float(map["y2"]),
#         e = float(map["e"]),
#         a = float(map["a"]),
#         k1 = float(map["k1"]),
#         k2 = float(map["k2"]),
#         w = float(map["w"]),
#         r = float(map["r"]),
#         s = float(map["s"]),
#         N0 = float(map["N0"]),
#         T = float(map["T"])
#     )

def to_plain(obj): # opaque as fuck chatgpt code for converting the parameters dataclass to a yaml-friendly dictionary
    """Recursively convert dataclass / numpy types to YAML-friendly Python types."""
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


