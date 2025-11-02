from numpy import array, ndarray
from dataclasses import dataclass

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
        A=array(map["A"], dtype= float),
        L=float(map["L"]),
        l=array(map["l"]),
        b_bar=array(map["b_bar"]),
        c_bar=array(map["c_bar"]),
        alpha_w=float(map["alpha_w"]),
        alpha_c=float(map["alpha_c"]),
        alpha_L=float(map["alpha_L"]),
        kappa=array(map["kappa"]),
        eta=array(map["eta"]),
        eta_w=float(map["eta_w"]),
        eta_r=float(map["eta_r"]),
        w0=float(map["w0"]),
        r0=float(map["r0"]),
        q0=array(map["q0"]),
        p0=array(map["p0"]),
        s0=array(map["s0"]),
        m_w0=float(map["m_w0"]),
        T=int(map["T"]),
    )
