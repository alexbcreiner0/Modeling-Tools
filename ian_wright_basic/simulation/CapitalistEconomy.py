# from parameters import Params
import numpy as np
from scipy.integrate import solve_ivp

class CapitalistEconomy:
    """Basic capitalist economy"""
    def __init__(self, params):
        self.params = params # number of commodities
        self.n = self.params.A.shape[0]
        self.current_t = 0
        self.t = [0.0]
        self.y = np.concatenate([params.q0, params.p0, params.s0, np.array([params.m_w0]), np.array([params.L])])
        self.exo_supply_deduction = np.zeros(self.n)

        self.traj = {
            "p": np.array([params.p0]),
            "q": np.array([params.q0]), 
            "s": np.array([params.s0]),
            "m_w": np.array([params.m_w0]), 
            "L": np.array([params.L]), 
            "w": np.array([params.w0]),
            "r": np.array([params.r0])
        }

        self.traj["total_labor_employed"] = np.array([self._get_employment(params.q0, params.l)])
        b, c = self._get_consumption(params.m_w0, params.p0)
        self.traj["b"], self.traj["c"] = np.array([b]), np.array([c])
        values = self._get_values(self.params.A, self.params.l)
        self.traj["values"] = np.array([values])
        self.traj["wage_values"] = np.array([params.w0 * values])
        self.traj["reserve_army_size"] = np.array([self.params.L - self.traj["total_labor_employed"]])
        self.traj["m_c"] = np.array([1-params.m_w0])
        val_ms, val_cc, surplus_val, e, value_rop = self._get_value_split(values, b, self.params.q0, self.params.A)
        self.traj["values_ms"], self.traj["cc_vals"], self.traj["surplus_vals"], self.traj["e"], self.traj["value_rops"] = np.array([val_ms]), np.array([val_cc]), np.array([surplus_val]), np.array([e]), np.array([value_rop])
        epr, eq_p = self._get_equilibrium_info(params.p0, params.w0, params.A, params.l)
        self.traj["epr"], self.traj["epr_prices"] = np.array([epr]), np.array([eq_p])
        self.traj["compos_of_capital"] = np.array([val_cc / val_ms])
        profit_rates = self._get_profit_rates(params.A, params.p0, params.w0, params.l)
        self.traj["profit_rates"] = np.array([profit_rates])

        self.dydt = self._get_dydt(self.params)

    def step(self):
        sol = solve_ivp(self.dydt, (float(self.current_t), float(self.current_t+1)), self.y,
                        method= "BDF", rtol= 1e-6, atol=1e-9,
                        t_eval=[float(self.current_t+1)], max_step=1.0)
        if not sol.success or not np.all(np.isfinite(sol.y)):
            # Do something besides this
            print("Simulation failed.")
        self.exo_supply_deduction = np.zeros(self.n)
        self.y = sol.y[:,-1]
        self._step_traj(self.y)
        self.current_t = self.current_t+1
        self.t.append(float(self.current_t + 1))

        # self.params.q0 , self.params.p0, self.params.s0, self.params.m_w0, self.params.L = new_q, new_p, new_s, new_mw, new_L
        self.dydt = self._get_dydt(self.params)

    def change_param(self, param_name, new_val):
        setattr(self.params, param_name, new_val)

    def _step_traj(self, y):
        q, p, s, m_w, L = self._split_state(y)
        self.traj["q"] = np.append(self.traj["q"], [q], axis=0)
        self.traj["p"] = np.append(self.traj["p"], [p], axis=0)
        self.traj["s"] = np.append(self.traj["s"], [s], axis=0)
        self.traj["m_w"] = np.append(self.traj["m_w"], m_w)
        self.traj["L"] = np.append(self.traj["L"], L)

        w = self._get_hourly_wage(self.params.l, q, L)
        self.traj["w"] = np.append(self.traj["w"], w)
        r = self._get_interest_rate(m_w)
        self.traj["r"] = np.append(self.traj["r"], r)
        employment = self._get_employment(q, self.params.l)
        self.traj["total_labor_employed"] = np.append(self.traj["total_labor_employed"], employment)
        b, c = self._get_consumption(m_w, p)
        self.traj["b"] = np.append(self.traj["b"], [b], axis=0)
        self.traj["c"] = np.append(self.traj["c"], [c], axis=0)

        values = self._get_values(self.params.A, self.params.l)
        self.traj["values"] = np.append(self.traj["values"], [values], axis=0)
        self.traj["wage_values"] = np.append(self.traj["wage_values"], [w * values], axis=0)
        self.traj["reserve_army_size"] = np.append(self.traj["reserve_army_size"], L - employment)
        self.traj["m_c"] = np.append(self.traj["m_c"], 1-self.params.m_w0)

        val_ms, val_cc, surplus_val, e, value_rop = self._get_value_split(values, b, q, self.params.A)
        self.traj["values_ms"] = np.append(self.traj["values_ms"], val_ms)
        self.traj["cc_vals"] = np.append(self.traj["cc_vals"], val_cc)
        self.traj["surplus_vals"] = np.append(self.traj["surplus_vals"], surplus_val)
        self.traj["e"] = np.append(self.traj["e"], e)
        self.traj["value_rops"] = np.append(self.traj["value_rops"], value_rop)

        epr, eq_p = self._get_equilibrium_info(p, w, self.params.A, self.params.l)
        self.traj["epr"] = np.append(self.traj["epr"], epr)
        self.traj["epr_prices"] = np.append(self.traj["epr_prices"], [eq_p], axis=0)

        self.traj["compos_of_capital"] = np.append(self.traj["compos_of_capital"], val_cc / val_ms)
        profit_rates = self._get_profit_rates(self.params.A, p, w, self.params.l)
        self.traj["profit_rates"] = np.append(self.traj["profit_rates"], [profit_rates], axis=0)

    def check_supply(self):
        s = self.y[2*self.n:3*self.n]
        return s

    def receive_offer(self):
        pass

    def make_offer(self):
        pass

    def exo_supply_shock(self, deduction):
        q, p, s, m_w, L = self._split_state(self.y)
        s -= deduction
        self.y = np.concatenate([q, p, s, np.array([m_w]), np.array([L])])
        self.exo_supply_deduction = deduction

    def _trade(self):
        pass

    def _split_state(self, y):
        n = self.n
        q = y[0:n]
        p = y[n:2*n]
        s = y[2*n:3*n]
        m_w = y[3*n]
        L  = y[3*n+1]
        return q, p, s, m_w, L

    def _get_employment(self, q, l):
        return q@l

    def _get_dydt(self, params):
        A = params.A
        l = params.l
        b_bar = params.b_bar
        c_bar = params.c_bar
        kappa = params.kappa
        eta = params.eta
        alpha_w = params.alpha_w
        alpha_c = params.alpha_c
        alpha_L = params.alpha_L

        # Creates the right hand side of the equation dy/dt = f(t,y)
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            n = self.n
            q = y[0:n]
            p = y[n:2*n]
            s = y[2*n:3*n]
            m_w = float(y[3*n])
            L = float(y[3*n+1])

            delta_L = alpha_L*L

            w = self._get_hourly_wage(l, q, L)
            r = self._get_interest_rate(m_w)

            p_dot_b = max(p.dot(b_bar), 1e-12)
            p_dot_c = max(p.dot(c_bar), 1e-12)

            total_labor = l.dot(q)
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
            delta_q = kappa * (profit / denom)

            return np.concatenate([delta_q, delta_p, delta_s, np.array([delta_m_w]), np.array([delta_L])])

        return rhs

    def _get_hourly_wage(self, l: np.ndarray, q: np.ndarray, L: float) -> float:
        """Returns the current hourly wage given the current level of employment and size of reserve army"""
        initial_employment = float(self.params.l.dot(self.params.q0))
        employment = float(l.dot(q))
        denom = max(L - employment, self.params.eps_u)
        num = max(self.params.L - initial_employment, self.params.eps_u)
        return self.params.w0 * (num / denom) ** self.params.eta_w

    def _get_interest_rate(self, m_w: float) -> float:
        """Returns the current interest rate given the current capitalist savings"""
        denom = max(1.0 - float(m_w), self.params.eps_m)
        num = max(1.0 - self.params.m_w0, self.params.eps_m)
        return self.params.r0 * (num / denom) ** self.params.eta_r

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
        epr = 1/r_hat - 1
        scalar = np.linalg.norm(p) / np.linalg.norm(eq_p)
        eq_p = scalar * eq_p
        return epr, eq_p

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
    def __init__(self, params):
        super().__init__(params)
    
    def _get_dydt(self, params):
        A = params.A
        l = params.l
        b_bar = params.b_bar
        c_bar = params.c_bar
        kappa = params.kappa
        eta = params.eta
        alpha_w = params.alpha_w
        alpha_c = params.alpha_c
        alpha_L = params.alpha_L

        # Creates the right hand side of the equation dy/dt = f(t,y)
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            n = self.n
            q = y[0:n]
            p = y[n:2*n]
            s = y[2*n:3*n]
            m_w = float(y[3*n])
            L = float(y[3*n+1])

            delta_L = alpha_L*L

            r = self._get_interest_rate(m_w)

            p_dot_b = max(p.dot(b_bar), 1e-12)
            p_dot_c = max(p.dot(c_bar), 1e-12)

            w = p_dot_b

            total_labor = l.dot(q)
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
            delta_q = kappa * (profit / denom)

            return np.concatenate([delta_q, delta_p, delta_s, np.array([delta_m_w]), np.array([delta_L])])

        return rhs

    def _step_traj(self, y):
        q, p, s, m_w, L = self._split_state(y)
        self.traj["q"] = np.append(self.traj["q"], [q], axis=0)
        self.traj["p"] = np.append(self.traj["p"], [p], axis=0)
        self.traj["s"] = np.append(self.traj["s"], [s], axis=0)
        self.traj["m_w"] = np.append(self.traj["m_w"], m_w)
        self.traj["L"] = np.append(self.traj["L"], L)

        w = p.dot(self.params.b_bar)
        self.traj["w"] = np.append(self.traj["w"], w)
        r = self._get_interest_rate(m_w)
        self.traj["r"] = np.append(self.traj["r"], r)
        employment = self._get_employment(q, self.params.l)
        self.traj["total_labor_employed"] = np.append(self.traj["total_labor_employed"], employment)
        b, c = self._get_consumption(m_w, p)
        self.traj["b"] = np.append(self.traj["b"], [b], axis=0)
        self.traj["c"] = np.append(self.traj["c"], [c], axis=0)

        values = self._get_values(self.params.A, self.params.l)
        self.traj["values"] = np.append(self.traj["values"], [values], axis=0)
        self.traj["wage_values"] = np.append(self.traj["wage_values"], [w * values], axis=0)
        self.traj["reserve_army_size"] = np.append(self.traj["reserve_army_size"], L - employment)
        self.traj["m_c"] = np.append(self.traj["m_c"], 1-self.params.m_w0)

        val_ms, val_cc, surplus_val, e, value_rop = self._get_value_split(values, b, q, self.params.A)
        self.traj["values_ms"] = np.append(self.traj["values_ms"], val_ms)
        self.traj["cc_vals"] = np.append(self.traj["cc_vals"], val_cc)
        self.traj["surplus_vals"] = np.append(self.traj["surplus_vals"], surplus_val)
        self.traj["e"] = np.append(self.traj["e"], e)
        self.traj["value_rops"] = np.append(self.traj["value_rops"], value_rop)

        epr, eq_p = self._get_equilibrium_info(p, w, self.params.A, self.params.l)
        self.traj["epr"] = np.append(self.traj["epr"], epr)
        self.traj["epr_prices"] = np.append(self.traj["epr_prices"], [eq_p], axis=0)

        self.traj["compos_of_capital"] = np.append(self.traj["compos_of_capital"], val_cc / val_ms)
        profit_rates = self._get_profit_rates(self.params.A, p, w, self.params.l)
        self.traj["profit_rates"] = np.append(self.traj["profit_rates"], [profit_rates], axis=0)


if __name__ == "__main__":
    pass
