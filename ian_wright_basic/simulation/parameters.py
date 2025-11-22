from numpy import array, ndarray
from dataclasses import dataclass, asdict, is_dataclass
import numpy as np

# This dataclass contains all of the settings which are needed to specify the model as well as anything else of significance to the simulation
# (such as tolerances, time steps, etcetera)

@dataclass
class Params:
    A: ndarray # Initial requirements matrix. The number of commodities is inferred to be the dimension of this matrix, assume it equals n
    l: ndarray # Initial living labor n-vector.
    b_bar: ndarray # Worker consumption baseline. Workers will consume a scalar multiple of this n-vector each period using their savings
    c_bar: ndarray # The same as b_bar but for capitalists
    alpha_w: float # Worker propensity to consume. Workers will spend exactly this proportion of their savings on as many multiples of b_bar as that can purchase
    alpha_c: float # Capitalist propensity to consume. 
    alpha_L: float # The rate of growth of the labor pool
    kappa: ndarray # n-vector of elasticity constants. ith entry controls how dramatically ith output fluctuates with respect to the fluctuation of profit rates
    eta: ndarray # n-vector of elascitity constants. ith entry controls how dramatically ith price fluctuates with respect to changes in supply
    eta_w: float # controls how dramatically wages change with respect to employment and size of the reserve army
    eta_r: float # controls how dramatically the interest rate changes with respect to capitalist savings
    L: float # initial pool of available labor
    w: float # initial hourly wage
    r: float # initial interest rate. if set to zero, it will never change, i.e. the model will not have a credit system and capitalists draw their means of production from a free communal pool
    q: ndarray # initial output n-vector
    p: ndarray # initial price n-vector
    s: ndarray # initial supply n-vector
    m_w: float # initial worker savings. implicitly, there is a parameter M=1=total money in circulation. All money is posessed by either workers or capitalists. Thus whatever m_w0 is, initial capitalist savings will be 1-m_w
    init_tssi_melt: float = 1
    alpha_l: float = 0.0 # rate of technological innovation. Each period the living labor vector will get scaled down by 1-this proportion
    # the remainder of these constants are purely technical in that they don't directly pertain to the economic scenario
    s_floor: float = 1e-5 # this and the next two constants just prevent the model from accidentally dividing by zero, they are not economically relevant
    eps_u: float = 1e-8
    eps_m: float = 1e-8
    T: int = 100 # number of time steps to simulate
    res: int = 3 # resolution, i.e. the number of points to sample per time step

# I don't think any of this still matters
# def params_from_mapping(map: dict):
#     return Params(
#         A=np.array(map["A"], dtype= float),
#         L=float(map["L"]),
#         l=np.array(map["l"]),
#         b_bar=np.array(map["b_bar"]),
#         c_bar=np.array(map["c_bar"]),
#         alpha_w=float(map["alpha_w"]),
#         alpha_c=float(map["alpha_c"]),
#         alpha_L=float(map["alpha_L"]),
#         alpha_l = float(map["alpha_l"]),
#         kappa=np.array(map["kappa"]),
#         eta=np.array(map["eta"]),
#         eta_w=float(map["eta_w"]),
#         eta_r=float(map["eta_r"]),
#         w0=float(map["w0"]),
#         r0=float(map["r0"]),
#         q0=np.array(map["q0"]),
#         p0=np.array(map["p0"]),
#         s0=np.array(map["s0"]),
#         m_w0=float(map["m_w0"]),
#         T=int(map["T"]),
#         res=int(map["res"])
#     )

# def to_plain(obj): # opaque as fuck chatgpt code for converting the parameters dataclass to a yaml-friendly dictionary
#     """Recursively convert dataclass / numpy types to YAML-friendly Python types."""
#     if is_dataclass(obj):
#         obj = asdict(obj)
#     if isinstance(obj, dict):
#         return {k: to_plain(v) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)):
#         return [to_plain(v) for v in obj]
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     if isinstance(obj, (np.floating,)):
#         return float(obj)
#     if isinstance(obj, (np.integer,)):
#         return int(obj)
#     if isinstance(obj, (np.bool_,)):
#         return bool(obj)
#     return obj


