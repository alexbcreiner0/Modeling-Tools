from numpy import array, ndarray
from dataclasses import dataclass, asdict, is_dataclass
import numpy as np

# This dataclass contains all of the settings which are needed to specify the model as well as anything else of significance to the simulation
# (such as tolerances, time steps, etcetera)

@dataclass
class Params:
    A: ndarray
    l: ndarray
    b_bar: ndarray
    c_bar: ndarray 
    alpha_w: float
    alpha_c: float
    alpha_L: float
    kappa: ndarray
    eta: ndarray
    eta_w: float
    eta_r: float
    L: float
    w0: float
    r0: float
    q0: ndarray
    p0: ndarray
    s0: ndarray
    m_w0: float
    s_floor: float = 1e-5
    eps_u: float = 1e-8
    eps_m: float = 1e-8
    T: int = 100

def params_from_mapping(map: dict):
    return Params(
        A=np.array(map["A"], dtype= float),
        L=float(map["L"]),
        l=np.array(map["l"]),
        b_bar=np.array(map["b_bar"]),
        c_bar=np.array(map["c_bar"]),
        alpha_w=float(map["alpha_w"]),
        alpha_c=float(map["alpha_c"]),
        alpha_L=float(map["alpha_L"]),
        kappa=np.array(map["kappa"]),
        eta=np.array(map["eta"]),
        eta_w=float(map["eta_w"]),
        eta_r=float(map["eta_r"]),
        w0=float(map["w0"]),
        r0=float(map["r0"]),
        q0=np.array(map["q0"]),
        p0=np.array(map["p0"]),
        s0=np.array(map["s0"]),
        m_w0=float(map["m_w0"]),
        T=int(map["T"]),
    )

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


