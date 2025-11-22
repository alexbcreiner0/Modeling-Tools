# from parameters import Params
import numpy as np
from scipy.integrate import solve_ivp
import sys
import random

class CapitalistEconomy:
    """Basic capitalist economy"""
    def __init__(self, params):
        self.params = params 
        self.n = self.params.A.shape[0]
        self.current_t = 0
        self.t = [0.0]
        # Simulation solves the equation dy/dt = f(t,y), where y is a vector of every relevant independent quantity dumped into one place
        # note what y consists of - all other numbers can be considered dependent variables and can be determined from just these
        self.y = np.concatenate([params.q, params.p, params.s, params.l, np.array([params.m_w]), np.array([params.L])])
        self.exo_supply_deduction = np.zeros(self.n) # carries a to-be-applied exogenous shock to supply

        # initialize the table of trajectories
        self.traj = {
            "p": np.array([params.p]),
            "q": np.array([params.q]), 
            "s": np.array([params.s]),
            "m_w": np.array([params.m_w]), 
            "L": np.array([params.L]), 
            "w": np.array([params.w]),
            "r": np.array([params.r]),
            "l": np.array([params.l])
        }

        # dependent variables
        self.traj["total_labor_employed"] = np.array([self._get_employment(params.q, params.l)])
        b, c = self._get_consumption(params.m_w, params.p)
        self.traj["b"], self.traj["c"] = np.array([b]), np.array([c])
        values = self._get_values(self.params.A, self.params.l)
        self.traj["values"] = np.array([values])
        self.traj["wage_values"] = np.array([params.w * values])
        self.traj["reserve_army_size"] = np.array([self.params.L - self.traj["total_labor_employed"]])
        self.traj["m_c"] = np.array([1-params.m_w])
        val_ms, val_cc, surplus_val, e, value_rop = self._get_value_split(values, b, self.params.q, self.params.A)
        self.traj["values_ms"], self.traj["cc_vals"], self.traj["surplus_vals"], self.traj["e"], self.traj["value_rops"] = np.array([val_ms]), np.array([val_cc]), np.array([surplus_val]), np.array([e]), np.array([value_rop])

        epr, eq_p = self._get_equilibrium_info(params.p, params.w, params.A, params.l)
        hourly_b, ms_vals, c_vals, v_vals, s_vals = self._get_composition_info(self.params.p, self.params.w, values, self.params.A, self.params.l, e)
        self.traj["hourly_b"], self.traj["val_ms"], self.traj["c_vals"], self.traj["v_vals"], self.traj["s_vals"] = np.array([hourly_b]), np.array([ms_vals]), np.array([c_vals]), np.array([v_vals]), np.array([s_vals])
        self.traj["sectoral_comps"] = np.array([c_vals / v_vals])

        self.traj["epr"], self.traj["epr_prices"] = np.array([epr]), np.array([eq_p])
        self.traj["compos_of_capital"] = np.array([val_cc / val_ms])
        profit_rates = self._get_profit_rates(params.A, params.p, params.w, params.l)
        self.traj["profit_rates"] = np.array([profit_rates])

        num = params.l.dot(params.q) - (params.p@params.b_bar)*(params.l.dot(params.q)) 
        M = params.A+np.linalg.outer(params.b_bar,params.l)
        den = params.p.dot(M@params.q)
        self.traj["kliman_profit_rate"] = np.array([num/den])

        values = self.traj["values"][0]
        MELT = params.p.dot(params.q) / values.dot(params.q)
        self.traj["TSSI_MELT"] = np.array([self.params.init_tssi_melt])
        # self.traj["TSSI_MELT"] = np.array([MELT])
        self.traj["MELT"] = np.array([MELT])
        self.traj["MELT_values"] = np.array([MELT*values])

        max_rop_epr, max_rop_eq_p = self._get_equilibrium_info(params.p, 0, params.A, params.l)
        self.traj["max_rop"] = np.array([max_rop_epr])

        # calculate initial f(t,y) 
        self.dydt = self._get_dydt(self.params)

    def step(self):

        self._step_traj(self.y)
        self.t.append(self.current_t)

        t_eval = np.linspace(self.current_t, self.current_t+1, self.params.res+1)[1:]
        sol = solve_ivp(
            self.dydt,
            (float(self.current_t), float(self.current_t+1)), 
            self.y,
            method= "BDF", 
            rtol= 1e-6, 
            atol=1e-9,
            t_eval=t_eval, 
            max_step=1.0
        )
        if not sol.success or not np.all(np.isfinite(sol.y)):
            print("Simulation failed.")
            print(f"  success = {sol.success}")
            print(f"  status  = {sol.status}")
            print(f"  message = {sol.message}")
            print(f"  nfev    = {sol.nfev}, njev = {getattr(sol, 'njev', None)}, nlu = {getattr(sol, 'nlu', None)}")

            if sol.t.size > 0:
                t_last = sol.t[-1]
                y_last = sol.y[:, -1].copy()
                print(f"  last t = {t_last}")
                print(f"  last y = {y_last}")

                # Evaluate derivative at last state
                try:
                    f_last = self.dydt(t_last, y_last)
                    print(f"  |f_last|_inf = {np.max(np.abs(f_last))}")
                    print(f"  f_last = {f_last}")
                except Exception as e:
                    print(f"  error evaluating dydt at last state: {e}")
                    # How far did we get?
                    if sol.t.size > 0:
                        print(f"  last time reached = {sol.t[-1]}")
                        y_last = sol.y[:, -1]
                        print(f"  last state (y_last) = {y_last}")

            # Where did non-finite values first appear?
            if not np.all(np.isfinite(sol.y)):
                bad_mask = ~np.isfinite(sol.y)
                idx_t, idx_var = np.where(bad_mask)
                first = np.argmin(idx_t)  # earliest time index with a bad value
                t_bad = sol.t[idx_t[first]]
                var_bad = idx_var[first]
                print(f"  first non-finite at t = {t_bad}, variable index = {var_bad}")
                print(f"  value there = sol.y[{var_bad}, {idx_t[first]}] = {sol.y[var_bad, idx_t[first]]}")

        self.exo_supply_deduction = np.zeros(self.n) # if there was an exogenous supply shock, it will have been applied by now, so set it back to zero

        # log the initial point *once per step* if you want
        # (optional) this is sol.y[:,0] == initial self.y
        self.y = sol.y[:, 0]
        self._step_traj(self.y)
        # don't append time if already present, or do it guarded:
        if len(self.t) == 0 or self.t[-1] != float(t_eval[0]):
            self.t.append(float(t_eval[0]))

        m = sol.y.shape[1]

        # now log the evolved subpoints
        for i in range(m):
            self.y = sol.y[:, i]
            self._step_traj(self.y)
            self.current_t = sol.t[i]
            self.t.append(float(self.current_t))

        self.dydt = self._get_dydt(self.params)

#         # increment all trajectories
#         for i in range(self.params.res):

#             self.y = sol.y[:,i]
#             self._step_traj(self.y)
#             self.current_t = t_eval[i]
#             self.t.append(float(self.current_t))

#         # recalculate f(t,y)
#         self.dydt = self._get_dydt(self.params)

    def change_param(self, param_name, new_val):
        setattr(self.params, param_name, new_val)

    def implement_culs_shock(self, beta, cu= True, epsilon= 1e-2):
        """Implements a sudden new labor saving, capital using, super-profit generating innovation. beta is the proportion by which to reduce the living labor input by"""
        i = random.randint(0,self.n-1)
        print(f"Improving sector {i}")
        q, p, s, l, m_w, L = self._split_state(self.y)

        a_i, w = self.params.A[:,i].copy(), self._get_hourly_wage(self.y)
        cost_ratio = a_i.dot(p) / (w*l[i])
        old_cost = a_i.dot(p) + w*l[i]
        print(f"Old cost of commodity {i}: {old_cost}")
        gamma = cost_ratio + epsilon # bigger epsilon => more dramatic superprofits (i think)
        alpha = beta / gamma 
        
        l[i] *= 1-beta
        # TODO: change to only effect a single (random) input
        if cu:
            self.params.A[:,i] *= 1+alpha

        eigvals = np.linalg.eigvals(self.params.A)
        if np.max(np.abs(eigvals)) >= 1:
            print("Warning! A matrix is not productive!")
            print("Scaling matrix to safe values")
            self.params.A /= (np.max(np.abs(eigvals)) + epsilon)

        new_a_i = self.params.A[:,i]
        new_cost = new_a_i.dot(p) + w*l[i]
        print(f"New cost of commodity {i}: {new_cost}")

        self.y = np.concatenate([q, p, s, l, np.array([m_w]), np.array([L])])
        return q, p, s, l, m_w, L

    def implement_cslu_shock(self, alpha, lu= True, epsilon= 1e-2):
        """Implements a sudden new labor saving, capital using, super-profit generating innovation. beta is the proportion by which to reduce the living labor input by"""
        i = random.randint(0,self.n-1)
        q, p, s, l, m_w, L = self._split_state(self.y)

        a_i, w = self.params.A[:,i].copy(), self._get_hourly_wage(self.y)
        cost_ratio = a_i.dot(p) / (w*l[i])
        old_cost = a_i.dot(p) + w*l[i]
        print(f"Old cost of commodity {i}: {old_cost}")
        beta = alpha * cost_ratio * (1 - epsilon) # bigger epsilon => more dramatic superprofits (i think)

        self.params.A[:,i] *= 1-alpha
        
        # TODO: change to only effect a single (random) input
        if lu:
            l[i] *= 1+beta

        eigvals = np.linalg.eigvals(self.params.A)
        if np.max(np.abs(eigvals)) >= 1:
            print("Warning! A matrix is not productive!")
            print("Scaling matrix to safe values")
            self.params.A /= (np.max(np.abs(eigvals)) + epsilon)

        new_a_i = self.params.A[:,i]
        new_cost = new_a_i.dot(p) + w*l[i]
        print(f"New cost of commodity {i}: {new_cost}")

        self.y = np.concatenate([q, p, s, l, np.array([m_w]), np.array([L])])
        return q, p, s, l, m_w, L



    def implement_culs_shock_corrected(self, beta, epsilon=1e-2):
        """Implements a sudden new labor-saving, capital-using innovation.
           beta is the proportion by which to reduce the living labor input."""
        i = random.randint(0, self.n-1)

        # Current state and prices/wage
        q, p, s, l, m_w, L = self._split_state(self.y)
        a_i = self.params.A[:, i].copy()
        w = self.traj["w"][-1]
        l_i = l[i]

        # Old unit cost
        old_cost = a_i.dot(p) + w * l_i
        print(f"Old cost of commodity {i}: {old_cost}")

        # Choose alpha to guarantee cost reduction
        cost_ratio = a_i.dot(p) / (w * l_i)   # scalar
        gamma = cost_ratio + epsilon
        alpha = beta / gamma                  # scalar

        # Apply shock: more capital, less labor
        l[i] = (1 - beta) * l_i
        self.params.A[:, i] = (1 + alpha) * a_i

        # Optional: keep A productive
        eigvals = np.linalg.eigvals(self.params.A)
        rho = np.max(np.abs(eigvals))
        if rho >= 1:
            print("Warning! A matrix is not productive!")
            print("Scaling matrix to safe values")
            self.params.A /= (rho + epsilon)

        # New cost at same p, w
        a_i_new = self.params.A[:, i]
        l_i_new = l[i]
        new_cost = a_i_new.dot(p) + w * l_i_new
        print(f"New cost of commodity {i}: {new_cost}")

        # Update state vector
        self.y = np.concatenate([q, p, s, l, np.array([m_w]), np.array([L])])
        return q, p, s, l, m_w, L

    def _step_traj(self, y):
        q, p, s, l, m_w, L = self._split_state(y)
        self.traj["q"] = np.append(self.traj["q"], [q], axis=0)
        self.traj["p"] = np.append(self.traj["p"], [p], axis=0)
        self.traj["s"] = np.append(self.traj["s"], [s], axis=0)
        self.traj["m_w"] = np.append(self.traj["m_w"], m_w)
        self.traj["L"] = np.append(self.traj["L"], L)
        self.traj["l"] = np.append(self.traj["l"], [l], axis=0)

        w = self._get_hourly_wage(y)
        self.traj["w"] = np.append(self.traj["w"], w)
        r = self._get_interest_rate(m_w)
        self.traj["r"] = np.append(self.traj["r"], r)
        employment = self._get_employment(q, l)
        self.traj["total_labor_employed"] = np.append(self.traj["total_labor_employed"], employment)
        b, c = self._get_consumption(m_w, p)
        self.traj["b"] = np.append(self.traj["b"], [b], axis=0)
        self.traj["c"] = np.append(self.traj["c"], [c], axis=0)

        values = self._get_values(self.params.A, l)
        self.traj["values"] = np.append(self.traj["values"], [values], axis=0)
        self.traj["wage_values"] = np.append(self.traj["wage_values"], [w * values], axis=0)
        self.traj["reserve_army_size"] = np.append(self.traj["reserve_army_size"], L - employment)
        self.traj["m_c"] = np.append(self.traj["m_c"], 1-m_w)

        val_ms, val_cc, surplus_val, e, value_rop = self._get_value_split(values, b, q, self.params.A)
        self.traj["values_ms"] = np.append(self.traj["values_ms"], val_ms)
        self.traj["cc_vals"] = np.append(self.traj["cc_vals"], val_cc)
        self.traj["surplus_vals"] = np.append(self.traj["surplus_vals"], surplus_val)
        self.traj["e"] = np.append(self.traj["e"], e)
        self.traj["value_rops"] = np.append(self.traj["value_rops"], value_rop)
        self.traj["compos_of_capital"] = np.append(self.traj["compos_of_capital"], val_cc / val_ms)

        epr, eq_p = self._get_equilibrium_info(p, w, self.params.A, l)
        self.traj["epr"] = np.append(self.traj["epr"], epr)
        self.traj["epr_prices"] = np.append(self.traj["epr_prices"], [eq_p], axis=0)

        epr, eq_p = self._get_equilibrium_info(p, w, self.params.A, l)

        hourly_b, ms_val, c_vals, v_vals, s_vals = self._get_composition_info(p, w, values, self.params.A, l, e)
        self.traj["hourly_b"] = np.append(self.traj["hourly_b"], [hourly_b], axis= 0)
        self.traj["val_ms"] = np.append(self.traj["val_ms"], ms_val)
        self.traj["c_vals"] = np.append(self.traj["c_vals"], [c_vals], axis= 0)
        self.traj["v_vals"] = np.append(self.traj["v_vals"], [v_vals], axis= 0)
        self.traj["s_vals"] = np.append(self.traj["s_vals"], [s_vals], axis= 0)
        self.traj["sectoral_comps"] = np.append(self.traj["sectoral_comps"], [c_vals / v_vals], axis= 0)

        profit_rates = self._get_profit_rates(self.params.A, p, w, l)
        self.traj["profit_rates"] = np.append(self.traj["profit_rates"], [profit_rates], axis=0)

        MELT = p.dot(q) / values.dot(q)
        self.traj["MELT_values"] = np.append(self.traj["MELT_values"], [MELT*values], axis= 0)
        self.traj["MELT"] = np.append(self.traj["MELT"], MELT)
        OLD_TSSI_MELT = self.traj["TSSI_MELT"][-1]
        old_l = self.traj["l"][-2]
        old_p = self.traj["p"][-2]
        old_epr, old_eqp = self._get_pf_info(self.params.A, old_l, self.params.b_bar)
        epr, eqp = self._get_pf_info(self.params.A, l, self.params.b_bar)
        TSSI_MELT = eqp.dot(q) / ((1/OLD_TSSI_MELT) * (self.params.A.T@old_eqp).dot(q) + old_l.dot(q))
        self.traj["TSSI_MELT"] = np.append(self.traj["TSSI_MELT"], TSSI_MELT)
        max_rop_epr, max_rop_eqp = self._get_equilibrium_info(p, 0, self.params.A, l)
        self.traj["max_rop"] = np.append(self.traj["max_rop"], max_rop_epr)

        hourly_b = w / (p.dot(self.params.b_bar)) * self.params.b_bar
        num = TSSI_MELT*(l.dot(q)) - old_eqp.dot(hourly_b)*(l.dot(q))
        M = self.params.A+np.linalg.outer(hourly_b,l)
        den = old_eqp.dot(M@q)
        self.traj["kliman_profit_rate"] = np.append(self.traj["kliman_profit_rate"], num/den)

    def check_supply(self):
        s = self.y[2*self.n:3*self.n]
        return s

    def receive_offer(self):
        # fill in with whatever seems reasonable
        pass

    def make_offer(self):
        # fill in with whatever seems reasonable
        pass

    def exo_supply_shock(self, deduction):
        q, p, s, l, m_w, L = self._split_state(self.y)
        s -= deduction
        self.y = np.concatenate([q, p, s, l, np.array([m_w]), np.array([L])])
        self.exo_supply_deduction = deduction

    def _trade(self):
        # fill in with whatever seems reasonable
        pass

    def _split_state(self, y):
        n = self.n
        q = y[0:n]
        p = y[n:2*n]
        s = y[2*n:3*n]
        l = y[3*n:4*n]
        m_w = y[4*n]
        L  = y[4*n+1]
        return q, p, s, l, m_w, L

    def _get_employment(self, q, l):
        return q@l

    def _get_dydt(self, params):

        # Creates the right hand side of the equation dy/dt = f(t,y)
        # This is what you will want to make alterations to in order to tweak the dynamics of the system. 
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            q, p, s, l, m_w, L = self._split_state(y)

            delta_L, delta_l = self._get_delta_Ls(y)
            w = self._get_hourly_wage(y)
            r = self._get_interest_rate(m_w)
            delta_m_w = self._get_delta_m_w(y, w)
            total_demand = self._get_total_demand(y)
            # from total demand, we obtain change in supply
            delta_s = self._get_delta_s(y, total_demand)
            # from change in supply, we obtain change in prices
            delta_p = self._get_delta_p(y, delta_s)
            # compute profit, and from that delta_q
            delta_q = self._get_delta_q(y, w, r, total_demand, delta_l, delta_L)
            return np.concatenate([
                delta_q, delta_p, delta_s, delta_l, np.array([delta_m_w]), np.array([delta_L])
            ])

        return rhs

    def _get_delta_Ls(self, y):

        alpha_L, alpha_l, L, l = self.params.alpha_L, self.params.alpha_l, float(y[4*self.n+1]), y[3*self.n:4*self.n]
        return alpha_L*L, -1*alpha_l*l

    def _get_delta_m_w(self, y, w):

        alpha_w = self.params.alpha_w
        q, p, s, l, m_w, L = self._split_state(y)
        total_labor = l.dot(q)
        delta_m_w = total_labor * w - alpha_w * m_w
        return delta_m_w

    def _get_total_demand(self, y):

        b_bar, c_bar, alpha_w, alpha_c = self.params.b_bar, self.params.c_bar, self.params.alpha_w, self.params.alpha_c
        q, p, s, l, m_w, L = self._split_state(y)

        p_dot_b = max(p.dot(b_bar), 1e-12)
        p_dot_c = max(p.dot(c_bar), 1e-12)

        b = (b_bar * alpha_w * m_w) / p_dot_b
        c = (c_bar * alpha_c * (1.0 - m_w)) / p_dot_c
        total_demand = self.params.A@q + b + c
        return total_demand

    def _get_delta_s(self, y, total_demand):

        q, p, s, l, m_w, L = self._split_state(y)
        delta_s = q - total_demand - self.exo_supply_deduction
        return delta_s

    def _get_delta_q(self, y, w, r, total_demand, delta_l= None, delta_L= None):
        
        q, p, s, l, m_w, L = self._split_state(y)
        unit_cost = self.params.A.T@p+w*l
        revenue = p * total_demand # sectoral revenue vector (not a dot product)
        total_cost = unit_cost * (1.0 + r) * q 
        profit = revenue - total_cost # sectoral profit vector

        # from profit, we obtain change in output
        denom = np.maximum(unit_cost * (1.0 + r), 1e-12)
        delta_q = self.params.kappa * (profit / denom)
        return delta_q

    def _get_delta_p(self, y, delta_s):

        q, p, s, l, m_w, L = self._split_state(y)
        s_safe = np.maximum(s, self.params.s_floor)
        delta_p = -self.params.eta * delta_s * (p / s_safe)
        return delta_p

    def _get_hourly_wage(self, y):
        """Returns the current hourly wage given the current level of employment and size of reserve army"""
        q, p, s, l, m_w, L = self._split_state(y)
        initial_employment = float(self.params.l.dot(self.params.q))
        employment = float(l.dot(q))
        denom = max(L - employment, self.params.eps_u)
        num = max(self.params.L - initial_employment, self.params.eps_u)
        return self.params.w * (num / denom) ** self.params.eta_w

    def _get_interest_rate(self, m_w: float) -> float:
        """Returns the current interest rate given the current capitalist savings"""
        denom = max(1.0 - float(m_w), self.params.eps_m)
        num = max(1.0 - self.params.m_w, self.params.eps_m)
        return self.params.r * (num / denom) ** self.params.eta_r

    def _get_consumption(self, m_w, p):
        b = (self.params.alpha_w * m_w)/(p.dot(self.params.b_bar))*self.params.b_bar
        c = (self.params.alpha_c * (1-m_w))/(p.dot(self.params.c_bar))*self.params.c_bar
        return b, c

    def _get_values(self, A, l):
        return np.linalg.inv(np.eye(self.n) - A.T)@l

    def _get_value_split(self, values, b, q, A):
        total_value = q.dot(values)
        val_ms = b.dot(values)
        val_cc = values.dot(A@q)
        surplus_val = total_value - val_ms - val_cc
        e = surplus_val / val_ms
        value_rop = surplus_val / (val_ms + val_cc)
        return val_ms, val_cc, surplus_val, e, value_rop

    def _get_equilibrium_info(self, p, w, A, l):
        hourly_b = w / (p.dot(self.params.b_bar)) * self.params.b_bar
        r_hat, eq_p = self._get_pf_info(A, l, hourly_b)
        r_hat = np.real(r_hat)
        epr = 1/r_hat - 1
        scalar = np.linalg.norm(p) / np.linalg.norm(eq_p)
        eq_p = scalar * eq_p
        return epr, eq_p

    def _get_composition_info(self, p, w, values, A, l, e):
        hourly_b = w / (p.dot(self.params.b_bar)) * self.params.b_bar
        val_ms = values.dot(hourly_b)
        c_vals = A.T@values
        v_vals = (val_ms)*l
        s_vals = e*v_vals
        return hourly_b, val_ms, c_vals, v_vals, s_vals

    def _get_pf_info(self, A, l, b):
        M = A+np.linalg.outer(b,l)
        evals, evecs = np.linalg.eig(M.T)
        index = np.argmax(evals.real)
        r_hat = evals[index]
        p = evecs[:,index].real
        if p[0] < 0:  p *= -1
        return r_hat, p

    def _get_profit_rates(self, A, p, w, l):
        unit_costs = A.T@p + w*l
        unit_profit_rates = p - unit_costs
        for i in range(self.n):
            unit_profit_rates[i] /= unit_costs[i]
        return unit_profit_rates

class CapitalistEconomyFixedRealWage(CapitalistEconomy):

    def _get_hourly_wage(self, y):
        q, p, s, l, m_w, L = self._split_state(y)
        return p.dot(self.params.b_bar)

    def _get_delta_m_w(self, y, w):
        return 0

    def _get_consumption(self, m_w, p):
        b = self.params.b_bar
        c = (self.params.alpha_c)/(p.dot(self.params.c_bar))*self.params.c_bar
        return b, c


    def _get_delta_q(self, y, w, r, total_demand, delta_l=None, delta_L=None):
        q, p, s, l, m_w, L = self._split_state(y)

        if delta_l is None:
            delta_l = -self.params.alpha_l * l
        if delta_L is None:
            delta_L = self.params.alpha_L * L

        unit_cost = self.params.A.T @ p + w * l
        revenue = p * total_demand
        total_cost = unit_cost * (1.0 + r) * q
        profit = revenue - total_cost

        denom = np.maximum(unit_cost * (1.0 + r), 1e-12)
        delta_q_unc = self.params.kappa * (profit / denom)

        # Employment and gap
        E = float(l.dot(q))
        G = E - float(L)  # >0 means over-employment

        # Gap derivative under unconstrained dynamics
        Gdot_unc = delta_l.dot(q) + delta_q_unc.dot(l) - delta_L

        eps = 1e-6
        if (G >= -eps) and (Gdot_unc > 0.0):
            # Project so that Gdot = 0  ⇒  E and L move together, but starting from *G≈0*
            l2 = l.dot(l)
            if l2 > 0.0:
                gamma = -(delta_l.dot(q) + delta_q_unc.dot(l) - delta_L) / l2
                delta_q = delta_q_unc + gamma * l
            else:
                delta_q = delta_q_unc
        else:
            delta_q = delta_q_unc

        return delta_q

    # def _get_delta_q(self, y, w, r, total_demand, delta_l= None, delta_L= None):
    #     q, p, s, l, m_w, L = self._split_state(y)

    #     unit_cost = self.params.A.T@p+w*l
    #     revenue = p * total_demand # sectoral revenue vector (not a dot product)
    #     total_cost = unit_cost * (1.0 + r) * q 
    #     profit = revenue - total_cost # sectoral profit vector

    #     denom = np.maximum(unit_cost * (1.0 + r), 1e-12)

    #     # Without wages fluctuating, no cap on hiring. So need to artificially make sure employment does not exceed available labor.
    #     delta_q_unconstrained = self.params.kappa * (profit / denom)
    #     delta_E = delta_l.dot(q) + delta_q_unconstrained.dot(l)
    #     current_E = q.dot(l)

    #     new_E = current_E + delta_E
    #     if new_E > L + delta_L:
    #         delta_q = delta_q_unconstrained - (delta_l.dot(q) + l.dot(delta_q_unconstrained) - delta_L) / l.dot(l) * l
    #     else:
    #         delta_q = delta_q_unconstrained

    #     return delta_q

class CapitalistEconomyFixedMoneyWage(CapitalistEconomy):

    def _get_hourly_wage(self, y):
        return self.traj["w"][-1]

class CapitalistEconomyFixedEmployment(CapitalistEconomy):

    def _get_delta_q(self, y, w, r, total_demand, delta_l= None):
        q, p, s, l, m_w, L = self._split_state(y)

        unit_cost = self.params.A.T@p+w*l
        revenue = p * total_demand # sectoral revenue vector (not a dot product)
        total_cost = unit_cost * (1.0 + r) * q 
        profit = revenue - total_cost # sectoral profit vector

        denom = np.maximum(unit_cost * (1.0 + r), 1e-12)

        # ----- Employment constraint logic -----

        # 1) unconstrained q dynamics
        delta_q_unconstrained = self.params.kappa * (profit / denom)

        # 2) d/dt (l·q) under unconstrained dynamics
        employment_dot_unconstrained = delta_l.dot(q) + l.dot(delta_q_unconstrained)

        # 3) If employment would fall, project onto constant-employment direction
        if employment_dot_unconstrained < 0.0:
            l_norm2 = l.dot(l)
            if l_norm2 > 0.0:
                gamma = (-delta_l.dot(q) - l.dot(delta_q_unconstrained)) / l_norm2
                delta_q = delta_q_unconstrained + gamma * l
            else:
                # pathological case l = 0, just fall back
                delta_q = delta_q_unconstrained
        else:
            # employment not falling, no constraint applied
            delta_q = delta_q_unconstrained

        return delta_q

class CapitalistEconomyFixedMoneyWageFixedEmployment(CapitalistEconomyFixedEmployment):

    def _get_hourly_wage(self, y):
        return self.traj["w"][-1]

class CapitalistEconomyFixedRealWageFixedEmployment(CapitalistEconomyFixedEmployment):

    def _get_hourly_wage(self, y):
        q, p, s, l, m_w, L = self._split_state(y)
        return p.dot(self.params.b_bar)

class CapitalistEconomyFixedExploitationFixedEmployment(CapitalistEconomyFixedMoneyWageFixedEmployment):

    def __init__(self, params, target_e = None):
        super().__init__(params)
        self.target_e = target_e

    def _get_dydt(self, params):
        A = params.A
        b_bar = params.b_bar
        c_bar = params.c_bar
        kappa = params.kappa
        eta = params.eta
        alpha_w = params.alpha_w
        alpha_c = params.alpha_c
        alpha_L = params.alpha_L
        alpha_l = params.alpha_l

        # Creates the right hand side of the equation dy/dt = f(t,y)
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            n = self.n
            q = y[0:n]
            p = y[n:2*n]
            s = y[2*n:3*n]
            l = y[3*n:4*n]
            m_w = float(y[4*n])
            L = float(y[4*n+1])

            delta_L = alpha_L*L
            delta_l = -1*alpha_l*l

            r = self._get_interest_rate(m_w)

            p_dot_b = max(p.dot(b_bar), 1e-12)
            p_dot_c = max(p.dot(c_bar), 1e-12)

            w = self.traj["w"][-1]

            total_labor = l.dot(q)
            if self.target_e:
                delta_m_w = 0
            else:
                delta_m_w = total_labor * w - alpha_w * m_w

            b = (b_bar * alpha_w * m_w) / p_dot_b
            c = (c_bar * alpha_c * (1.0 - m_w)) / p_dot_c
            total_demand = A@q + b + c

            delta_s = q - total_demand - self.exo_supply_deduction

            s_safe = np.maximum(s, params.s_floor)
            delta_p = -eta * delta_s * (p / s_safe)

            unit_cost = A.T@p+w*l
            revenue = p * total_demand # sectoral revenue vector (not a dot product)
            total_cost = unit_cost * (1.0 + r) * q 
            profit = revenue - total_cost # sectoral profit vector

            denom = np.maximum(unit_cost * (1.0 + r), 1e-12)

            # ----- Employment constraint logic -----

            # 1) unconstrained q dynamics
            delta_q_unconstrained = kappa * (profit / denom)

            # 2) d/dt (l·q) under unconstrained dynamics
            employment_dot_unconstrained = delta_l.dot(q) + l.dot(delta_q_unconstrained)

            # 3) If employment would fall, project onto constant-employment direction
            if employment_dot_unconstrained < 0.0:
                l_norm2 = l.dot(l)
                if l_norm2 > 0.0:
                    gamma = (-delta_l.dot(q) - l.dot(delta_q_unconstrained)) / l_norm2
                    delta_q = delta_q_unconstrained + gamma * l
                else:
                    # pathological case l = 0, just fall back
                    delta_q = delta_q_unconstrained
            else:
                # employment not falling, no constraint applied
                delta_q = delta_q_unconstrained

            return np.concatenate(
                [delta_q, delta_p, delta_s, delta_l,
                 np.array([delta_m_w]), np.array([delta_L])]
            )

            delta_q = kappa * (profit / denom)

            return np.concatenate([delta_q, delta_p, delta_s, delta_l, np.array([delta_m_w]), np.array([delta_L])])

        return rhs

    def _step_traj(self, y):
        q, p, s, l, m_w, L = self._split_state(y)
        self.traj["q"] = np.append(self.traj["q"], [q], axis=0)
        self.traj["p"] = np.append(self.traj["p"], [p], axis=0)
        self.traj["s"] = np.append(self.traj["s"], [s], axis=0)

        if self.target_e:
            m_w = self._compute_mw_for_target_e(q, p, l)
            self.traj["m_w"] = np.append(self.traj["m_w"], m_w)
        else:
            self.traj["m_w"] = np.append(self.traj["m_w"], m_w)
        self.traj["L"] = np.append(self.traj["L"], L)
        self.traj["l"] = np.append(self.traj["l"], [l], axis=0)

        w = self.traj["w"][-1]
        self.traj["w"] = np.append(self.traj["w"], w)


        r = self._get_interest_rate(m_w)
        self.traj["r"] = np.append(self.traj["r"], r)
        employment = self._get_employment(q, l)
        self.traj["total_labor_employed"] = np.append(self.traj["total_labor_employed"], employment)
        b, c = self._get_consumption(m_w, p)
        self.traj["b"] = np.append(self.traj["b"], [b], axis=0)
        self.traj["c"] = np.append(self.traj["c"], [c], axis=0)

        values = self._get_values(self.params.A, l)
        self.traj["values"] = np.append(self.traj["values"], [values], axis=0)
        self.traj["wage_values"] = np.append(self.traj["wage_values"], [w * values], axis=0)
        self.traj["reserve_army_size"] = np.append(self.traj["reserve_army_size"], L - employment)
        self.traj["m_c"] = np.append(self.traj["m_c"], 1-m_w)

        val_ms, val_cc, surplus_val, e, value_rop = self._get_value_split(values, b, q, self.params.A)
        self.traj["values_ms"] = np.append(self.traj["values_ms"], val_ms)
        self.traj["cc_vals"] = np.append(self.traj["cc_vals"], val_cc)
        self.traj["surplus_vals"] = np.append(self.traj["surplus_vals"], surplus_val)
        self.traj["e"] = np.append(self.traj["e"], e)
        self.traj["value_rops"] = np.append(self.traj["value_rops"], value_rop)
        macro_s = values.dot(q - b - self.params.A@q)
        macro_e = macro_s / val_ms
        # self.traj["macro_e"] = np.append(self.traj["macro_e"], macro_e)

        epr, eq_p = self._get_equilibrium_info(p, w, self.params.A, l)
        self.traj["epr"] = np.append(self.traj["epr"], epr)
        self.traj["epr_prices"] = np.append(self.traj["epr_prices"], [eq_p], axis=0)

        hourly_b, ms_vals, c_vals, v_vals, s_vals = self._get_composition_info(p, w, values, self.params.A, l, e)
        self.traj["hourly_b"] = np.append(self.traj["hourly_b"], [hourly_b], axis= 0)
        self.traj["val_ms"] = np.append(self.traj["val_ms"], ms_vals)
        self.traj["c_vals"] = np.append(self.traj["c_vals"], [c_vals], axis= 0)
        self.traj["v_vals"] = np.append(self.traj["v_vals"], [v_vals], axis= 0)
        self.traj["s_vals"] = np.append(self.traj["s_vals"], [s_vals], axis= 0)
        self.traj["sectoral_comps"] = np.append(self.traj["sectoral_comps"], [c_vals / v_vals], axis= 0)

        self.traj["compos_of_capital"] = np.append(self.traj["compos_of_capital"], val_cc / val_ms)
        profit_rates = self._get_profit_rates(self.params.A, p, w, l)
        self.traj["profit_rates"] = np.append(self.traj["profit_rates"], [profit_rates], axis=0)

        MELT = p.dot(q) / values.dot(q)
        self.traj["MELT"] = np.append(self.traj["MELT"], MELT)
        self.traj["MELT_values"] = np.append(self.traj["MELT_values"], [MELT*values], axis= 0)
        OLD_TSSI_MELT = self.traj["TSSI_MELT"][-1]
        old_l = self.traj["l"][-2]
        old_p = self.traj["p"][-2]
        hourly_b = w / (p.dot(self.params.b_bar)) * self.params.b_bar
        old_epr, old_eqp = self._get_pf_info(self.params.A, old_l, hourly_b)
        epr, eqp = self._get_pf_info(self.params.A, l, hourly_b)
        TSSI_MELT = eqp.dot(q) / ((1/OLD_TSSI_MELT) * (self.params.A.T@old_eqp).dot(q) + old_l.dot(q))
        self.traj["TSSI_MELT"] = np.append(self.traj["TSSI_MELT"], TSSI_MELT)

        num = TSSI_MELT*(l.dot(q)) - old_eqp.dot(self.params.b_bar)*(l.dot(q))
        M = self.params.A+np.linalg.outer(self.params.b_bar,l)
        den = old_eqp.dot(M@q)
        self.traj["kliman_profit_rate"] = np.append(self.traj["kliman_profit_rate"], num/den)

    def _compute_mw_for_target_e(self, q, p, l):
        A       = self.params.A
        b_bar   = self.params.b_bar
        alpha_w = self.params.alpha_w

        values = self._get_values(A, l)

        # T and c in value terms
        T = float(q.dot(values))
        c = float(values.dot(A @ q))

        # p·b_bar
        p_dot_b = max(float(p.dot(b_bar)), 1e-12)

        # K = (α_w / (p·b_bar)) * (values·b_bar)
        K = (alpha_w / p_dot_b) * float(values.dot(b_bar))

        # desired exploitation rate
        e_tar = self.target_e

        denom = K * (1.0 + e_tar)
        if denom <= 0.0:
            # fall back gracefully, avoid division by zero
            return 0.5   # or clamp / raise

        m_w_star = (T - c) / denom

        # clamp for safety
        m_w_star = max(min(m_w_star, 0.9999), 1e-6)
        return m_w_star

if __name__ == "__main__":
    pass
