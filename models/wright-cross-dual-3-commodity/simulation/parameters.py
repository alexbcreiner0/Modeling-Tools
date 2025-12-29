from numpy import array, ndarray
from dataclasses import dataclass, field
import numpy as np

# This dataclass contains all of the settings which are needed to specify the model as well as anything else of significance to the simulation
# (such as tolerances, time steps, etcetera)

@dataclass
class Params:
    A: ndarray = field(default_factory=lambda: array([[0.2, 0.0, 0.4], [0.2, 0.8, 0.0], [0.0, 0.1, 0.1]])) # Initial requirements matrix. The number of commodities is inferred to be the dimension of this matrix, assume it equals n
    l: ndarray = field(default_factory=lambda: array([0.7, 0.6, 0.3])) # Initial living labor n-vector.
    b_bar: ndarray = field(default_factory=lambda: array([0.6, 0.0, 0.2])) # Worker consumption baseline. Workers will consume a scalar multiple of this n-vector each period using their savings
    c_bar: ndarray = field(default_factory=lambda: array([0.2, 0.0, 0.4])) # The same as b_bar but for capitalists
    alpha_w: float = 0.8 # Worker propensity to consume. Workers will spend exactly this proportion of their savings on as many multiples of b_bar as that can purchase
    alpha_c: float = 0.7 # Capitalist propensity to consume. 
    alpha_L: float = 0.0 # The rate of growth of the labor pool
    kappa: ndarray = field(default_factory=lambda: array([1, 1, 1])) # n-vector of elasticity constants. ith entry controls how dramatically ith output fluctuates with respect to the fluctuation of profit rates
    eta: ndarray = field(default_factory=lambda: array([2, 2, 2])) # n-vector of elascitity constants. ith entry controls how dramatically ith price fluctuates with respect to changes in supply
    eta_w: float = 0.25 # controls how dramatically wages change with respect to employment and size of the reserve army
    eta_r: float = 2 # controls how dramatically the interest rate changes with respect to capitalist savings
    L: float = 1 # initial pool of available labor
    w: float = 0.5 # initial hourly wage
    r: float = 0.03 # initial interest rate. if set to zero, it will never change, i.e. the model will not have a credit system and capitalists draw their means of production from a free communal pool
    q: ndarray = field(default_factory=lambda: array([0.01, 0.1, 0.1])) # initial output n-vector
    p: ndarray = field(default_factory=lambda: array([1.0, 0.8, 0.5])) # initial price n-vector
    s: ndarray = field(default_factory=lambda: array([0.01, 0.1, 0.25])) # initial supply n-vector
    m_w: float = 0.5 # initial worker savings. implicitly, there is a parameter M=1=total money in circulation. All money is posessed by either workers or capitalists. Thus whatever m_w0 is, initial capitalist savings will be 1-m_w
    M: float = 1.0 # initial money supply
    M_change_type: str = "none" # options are none and discrete
    money_injection_target: str = "capitalists" # options are 'capitalists' and 'workers'
    M_change_interval: int = 20 # interval in which to inject money into economy
    M_change_duration: int = 1 # money injection must be continuous for it to have any effect. this dictates for how long the easing will go on
    delta_M: float = 1 # how much money to inject
    shock_type: str = "culs" # options are 'culs', 'cslu', 'cs', or 'ls'
    change_type: str = "none" # options are 'discrete', 'cts', and 'none'
    economy_type: str = "unrestricted" # options (so far) are 'unrestricted' and 'fixed_real_wage'
    wage_deflation: float = 0.0 # atrophy rate of the wage
    gamma_L: float = 0.3 # Maximum rate of unemployment
    atrophy_with_unemployment: str = "always" # options are 'always' and 'unemployment'
    supply_shock_mag: float = 0.5 # supply shock magnitude
    supply_shock_interval: int = 50 # how often the supply shocks are applied
    supply_shock_setting: str = "none" # other option are 'periodic' and 'cts'
    change_interval: int = 20
    shock_mag: float = 0.03
    cost_tradeoff: float = 1e-2
    stop_cts_changes_halfway: bool = False
    fix_sector_receiving_change: int = -1 # options are -1 (random) 0 (corn) 1 (iron) (2) sugar
    init_tssi_melt: float = 1
    output_equation: str = "absolute" # options are relative, absolute, and cross-dual
    employment_guardrails: bool = False # actively prevent employment from exceeding the labor force
    # the remainder of these constants are purely technical in that they don't directly pertain to the economic scenario
    s_floor: float = 1e-5 # this and the next two constants just prevent the model from accidentally dividing by zero, they are not economically relevant
    mu_L: float = 0.5
    eps_u: float = 1e-8
    eps_m: float = 1e-8
    T: int = 100 # number of time steps to simulate
    res: int = 3 # resolution, i.e. the number of points to sample per time step

