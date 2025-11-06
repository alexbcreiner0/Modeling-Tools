import numpy as np
import copy
from scipy.integrate import solve_ivp
from typing import Callable, Dict, Any
from .parameters import Params

# This is where the app will look for a get_trajectories function. 
def get_trajectories_perturbed(params, rtol=1e-6, atol=1e-9, method="BDF") -> tuple[Dict[str, np.ndarray], np.ndarray]:
    n = params.A.shape[0]
    y0 = np.concatenate
    # fundamentally we are solving for q, p, s, m and (in my improved version) L
    y0 = np.concatenate([params.q0, params.p0, params.s0, np.array([params.m_w0]), np.array([params.L])])
    t1 = np.linspace(0,params.T//2, params.T//2)

    dydt1 = get_dydt(params)

    print(f"Simulating from 0 to {params.T//2}")
    sol1 = solve_ivp(dydt1, (0, params.T//2), y0, method=method, rtol=rtol, atol=atol, dense_output=True)
    
    y1 = sol1.sol(t1) # returns a callable function representing the solution (only here because of dense_output)
    q1 = y1[0:n, :].T
    p1 = y1[n:2*n, :].T
    s1 = y1[2*n:3*n, :].T
    m_w1 = y1[3*n, :]
    L1 = y1[3*n+1, :]
   
    traj1 = {"q": q1, "p": p1, "s": s1, "m_w": m_w1, "L": L1}

    get_dependent_plots(params, traj1, t1)

    new_params = copy.deepcopy(params)
    setattr(new_params, "l", 0.5*params.l)

    dydt2 = get_dydt(new_params)

    print(f"Simulating from {params.T//2} to {params.T}")
    sol2 = solve_ivp(dydt2, (params.T//2, params.T), y1[:,-1], method=method, rtol=rtol, atol=atol, dense_output=True)

    t2 = np.linspace(params.T//2,params.T, params.T//2)
    y2 = sol2.sol(t2)
    q2 = y2[0:n, :].T
    p2 = y2[n:2*n, :].T
    s2 = y2[2*n:3*n, :].T
    m_w2 = y2[3*n, :]
    L2 = y2[3*n+1, :]
   
    traj2 = {"q": q2, "p": p2, "s": s2, "m_w": m_w2, "L": L2}

    get_dependent_plots(new_params, traj2, t2)

    traj = {}
    for key in traj1:
        new_val = []
        for val in traj1[key]:
            new_val.append(val)
        for val in traj2[key]:
            new_val.append(val)
        traj[key] = np.array(new_val)

    t = np.hstack((t1, t2))

    return traj, t

def get_trajectories(params, rtol=1e-6, atol=1e-9, method="RK45") -> tuple[Dict[str, np.ndarray], np.ndarray]:
    n = params.A.shape[0]
    y0 = np.concatenate
    # fundamentally we are solving for q, p, s, m and (in my improved version) L
    y0 = np.concatenate([params.q0, params.p0, params.s0, np.array([params.m_w0]), np.array([params.L])])
    t = np.linspace(0,params.T, params.T)

    dydt = get_dydt(params)

    sol = solve_ivp(dydt, (0, params.T), y0, method=method, rtol=rtol, atol=atol, dense_output=True)
    
    y = sol.sol(t) # returns a callable function representing the solution (only here because of dense_output)
    q = y[0:n, :].T
    p = y[n:2*n, :].T
    s = y[2*n:3*n, :].T
    m_w = y[3*n, :]
    L = y[3*n+1, :]

    traj = {"q": q, "p": p, "s": s, "m_w": m_w, "L": L}
    get_dependent_plots(params, traj, t)

    return traj, t

# returns a function representing change in y wrt t
def get_dydt(params: Params) -> Callable[[float, np.ndarray], np.ndarray]:
    A = params.A
    l = params.l
    b_bar = params.b_bar
    c_bar = params.c_bar
    kappa = params.kappa
    eta = params.eta
    alpha_w = params.alpha_w
    alpha_c = params.alpha_c
    L = params.L
    alpha_L = params.alpha_L

    # Creates the right hand side of the equation dy/dt = f(t,y)
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        n = A.shape[0]
        q = y[0:n]
        p = y[n:2*n]
        s = y[2*n:3*n]
        m_w = float(y[3*n])
        L = float(y[3*n+1])

        delta_L = alpha_L*L

        w = get_hourly_wage(params, q, L)
        r = get_interest_rate(params, m_w)

        p_dot_b = max(p.dot(b_bar), 1e-12)
        p_dot_c = max(p.dot(c_bar), 1e-12)

        total_labor = l.dot(q)
        delta_m_w = total_labor * w - alpha_w * m_w

        b = (b_bar * alpha_w * m_w) / p_dot_b
        c = (c_bar * alpha_c * (1.0 - m_w)) / p_dot_c
        total_demand = A@q + b + c

        delta_s = q - total_demand

        s_safe = np.maximum(s, params.s_floor)
        delta_p = -eta * delta_s * (p / s_safe)

        unit_cost = A.T@p+w*l
        revenue = p * total_demand # sectoral revenue vector (not a dot product)
        total_cost = unit_cost * (1.0 + r) * q 
        profit = revenue - total_cost # sectoral profit vector

        denom = np.maximum(unit_cost * (1.0 + r), 1e-12)
        delta_q = kappa * (profit / denom)

        return np.concatenate([delta_q, delta_p, delta_s, np.array([delta_m_w]), np.array([delta_L])])

    return rhs

# just a clean up function that fills in a bunch of things I'll want to plot using the soln
def get_dependent_plots(params, traj, t):
    n = params.A.shape[0]
    Tn = len(t)

    w_list = []
    for i in range(Tn):
        q_i, L_i = traj["q"][i], traj["L"][i]
        w_list.append(get_hourly_wage(params, q_i, L_i))
    traj["w"] = np.array(w_list)

    traj["r"] = np.array([get_interest_rate(params, mw_i) for mw_i in traj["m_w"]])

    b, c = [], []
    for i in range(Tn):
        b.append((params.alpha_w * traj["m_w"][i])/(traj["p"][i].dot(params.b_bar))*params.b_bar)
        c.append((params.alpha_c * (1-traj["m_w"][i]))/(traj["p"][i].dot(params.c_bar))*params.c_bar)
    traj["b"], traj["c"] = np.array(b), np.array(c)

    traj["total_labor_employed"] = traj["q"]@params.l

    outputs = traj["q"]
    subsistence_bundles = traj["b"]
    values = np.linalg.inv(np.eye(n)-params.A.T)@params.l
    traj["values"] = values

    total_value_produced = np.array([outputs[i].dot(values) for i in range(len(outputs))])
    values_ms = np.array([subsistence_bundles[i].dot(values) for i in range(len(outputs))])
    traj["values_ms"] = values_ms

    cc_vals = []
    for i in range(Tn):
        cc_vals.append(values.dot(params.A@outputs[i]))
    traj["cc_vals"] = np.array(cc_vals)

    traj["compos_of_capital"] = cc_vals/values_ms

    surplus_values = total_value_produced - values_ms - cc_vals
    traj["surplus_vals"] = surplus_values

    traj["e"] = surplus_values / values_ms

    # e = []
    # for i in range(len(t)):
    #     e.append(surplus_values[i]/values_ms[i])
    # traj["e"] = np.array(e)
    epr_profit_rates, epr_prices = [], []
    actual_prices = traj["p"]
    wages = traj["w"]
    hourly_b_vecs = [wages[i] / (actual_prices[i].dot(params.b_bar)) * params.b_bar for i in range(Tn)]

    for i,b_vec in enumerate(hourly_b_vecs):
        r_hat, p = get_equilibrium_info(params.A, params.l, b_vec)
        epr_profit_rates.append(1/r_hat-1)
        actual_p = actual_prices[i]
        scalar = np.linalg.norm(actual_p)/np.linalg.norm(p)
        epr_prices.append(scalar*p)
    traj["epr_profit_rates"] = np.array(epr_profit_rates)
    traj["epr_prices"] = np.array(epr_prices)
    mop_unit_costs = np.array([params.A.T@p_i for p_i in traj["p"]])
    labor_unit_costs = np.array([w_i*params.l for w_i in traj["w"]])
    unit_costs = mop_unit_costs + labor_unit_costs
    sectoral_profit_rates = []
    for i in range(len(traj["p"])):
        sectoral_profit_rates.append(traj["p"][i] - unit_costs[i])
        for j in range(len(sectoral_profit_rates[i])):
            sectoral_profit_rates[i][j] /= unit_costs[i][j]
    traj["profit_rates"] = np.array(sectoral_profit_rates)
    value_rops = surplus_values / (values_ms + cc_vals)
    value_rops2 = traj["e"] / (traj["compos_of_capital"] + 1)
    traj["value_rops2"] = value_rops2
    traj["value_rops"] = value_rops
    employment = traj["total_labor_employed"]
    demployment_dt = np.gradient(employment, t, axis=0)
    traj["labor_demand"] = demployment_dt / employment
    wage_values = np.array([np.array(values) for i in range(len(t))])

    for i,w in enumerate(wages):
        wage_values[i] *= w
    traj["wage_values"] = wage_values
    traj["reserve_army_size"] = traj["L"]-traj["total_labor_employed"]
    traj["m_c"] = np.array([1-m_wi for m_wi in traj["m_w"]])

def get_equilibrium_info(A, l, b):
    M = A+np.linalg.outer(b,l)
    evals, evecs = np.linalg.eig(M.T)
    index = np.argmax(evals.real)
    r_hat = evals[index]
    p = evecs[:,index].real
    if p[0] < 0:  p *= -1
    return r_hat, p

def get_hourly_wage(params: Params, q: np.ndarray, L: float) -> float:
    initial_total_labor = float(params.l.dot(params.q0))
    total_labor = float(params.l.dot(q))
    denom = max(L - total_labor, params.eps_u)
    num = max(1 - initial_total_labor, params.eps_u)
    return params.w0 * (num / denom) ** params.eta_w

def get_interest_rate(params: Params, mw_scalar: float) -> float:
    denom = max(1.0 - float(mw_scalar), params.eps_m)
    num = max(1.0 - params.m_w0, params.eps_m)
    return params.r0 * (num / denom) ** params.eta_r


