import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Dict, Any
from .parameters import Params

# This is where the app will look for a get_trajectories function. 

def get_trajectories(params) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    This is the only function which needs to exist in order for the app to work. 
    You pass this function to the main.py file, which tries to use it to create a dictionary 
    of trajectories to use in it's plots. 
    """
    try:
        res = simulate(params)
        t = np.linspace(0,params.T,params.T)
    except ValueError:
        print("Error, your society probably ran out of supplies to keep going.")
        return {}, np.ndarray([])

    traj = res["unpack"](t)
    get_dependent_plots(params, traj, t)
    return traj, t

def simulate(params, rtol=1e-6, atol=1e-9, method="RK45") -> Dict[str, Any]:
    n = params.A.shape[0]
    y0 = np.concatenate([params.q0, params.p0, params.s0, np.array([params.m_w0]), np.array([params.L])])
    f_ty = make_rhs(params)

    # required positional args:
    # first input is a function f(t,y) representing the time derivative of y wrt t. most of the time dynamics are constant so it is more f(y);
    # f(t,y) must return a vector of the same dim as y. 
    # second input is t_span, the interval of integration. 
    # third arg is the initial condition for y
    # fourth arg is method, self explanatory
    # outputs:
    
    sol = solve_ivp(f_ty, (0.0, params.T), y0, method=method, rtol=rtol, atol=atol, dense_output=True)
    print(sol)

    def unpack(t_array: np.ndarray):
        Y = sol.sol(t_array) # a callable function representing the solution (only here because of the dense_output option)
        q = Y[0:n, :].T
        p = Y[n:2*n, :].T
        s = Y[2*n:3*n, :].T
        m_w = Y[3*n, :]
        L = Y[3*n+1, :]
        w_list = []
        for i in range(len(q)):
            q_i, L_i = q[i], L[i]
            w_list.append(get_hourly_wage(params, q_i, L_i))
        w = np.array(w_list)
        # w = np.array([get_hourly_wage(params, q_i) for q_i in q])
        r = np.array([get_interest_rate(params, mw_i) for mw_i in m_w])
        b, c = [], []
        for i in range(len(q)):
            b.append((params.alpha_w * m_w[i])/(p[i].dot(params.b_bar))*params.b_bar)
            c.append((params.alpha_c * (1-m_w[i]))/(p[i].dot(params.c_bar))*params.c_bar)
        b, c = np.array(b), np.array(c)
        total_labor_employed = q@params.l
        return dict(t=t_array, q=q, p=p, s=s, m_w=m_w, w=w, r=r, total_labor_employed=total_labor_employed, b=b, c=c, L=L)

    return {"sol": sol, "unpack": unpack}

def make_rhs(params: Params) -> Callable[[float, np.ndarray], np.ndarray]:
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
    # l = params.l

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

def get_dependent_plots(params, traj, t):
    outputs = traj["q"]
    subsistence_bundles = traj["b"]
    values = np.linalg.inv(np.eye(3)-params.A.T)@params.l
    traj["values"] = values
    total_value_produced = np.array([outputs[i].dot(values) for i in range(len(outputs))])
    values_ms = np.array([subsistence_bundles[i].dot(values) for i in range(len(outputs))])
    traj["values_ms"] = values_ms
    cc_vals = []
    for i in range(len(t)):
        cc_vals.append(values.dot(params.A@outputs[i]))
    traj["cc_vals"] = np.array(cc_vals)
    surplus_values = total_value_produced - values_ms - cc_vals
    traj["surplus_vals"] = surplus_values
    e = []
    for i in range(len(t)):
        e.append(surplus_values[i]/values_ms[i])
    traj["e"] = np.array(e)
    epr_profit_rates = []
    epr_prices = []
    actual_prices = traj["p"]
    wages = traj["w"]
    hourly_b_vecs = []
    for i in range(params.T):
        hourly_b_vecs.append(wages[i]/(actual_prices[i].dot(params.b_bar))*params.b_bar)
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
    traj["value_rops"] = value_rops
    employment = traj["total_labor_employed"]
    demployment_dt = np.gradient(employment, t, axis=0)
    traj["labor_demand"] = demployment_dt / employment
    wage_values = np.array([np.array(values) for i in range(len(t))])
    for i,w in enumerate(wages):
        wage_values[i] *= w
    corn_values, iron_values, sugar_values = wage_values[:,0], wage_values[:,1], wage_values[:,2]
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


