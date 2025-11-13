from parameters import Params
import numpy as np
from scipy.integrate import solve_ivp

class CapitalistEconomy:

    def __init__(self, params, dydt):
        self.params = params # number of commodities
        self.n = self.params.A.shape[0]
        self.current_t = 0
        self.t = np.array([0.0])
        self.y = np.concatenate([params.q0, params.p0, params.s0, np.array([params.m_w0]), np.array([params.l])])
        self.traj = {}
        self.dydt_func = dydt
        self.dydt = self.dydt_func(self.params)

    def step(self):
        sol = solve_ivp(self.dydt, (float(self.t), float(self.t+1)), self.y,
                        method= "BDF", rtol= 1e-6, atol=1e-9,
                        t_eval=[float(self.t+1)], max_step=1.0)
        if not sol.success or not np.all(np.isfinite(sol.y)):
            # Do something besides this
            print("Simulation failed.")
        self.y = sol.y[:,-1]
        self.t = self.t+1

    def check_supply(self):
        pass

    def receive_offer(self):
        pass

    def make_offer(self):
        pass

    def _trade(self):
        pass

    def _split_state(self, y, n):
        q = y[0:n]
        p = y[n:2*n]
        s = y[2*n:3*n]
        m_w = y[3*n]
        L  = y[3*n+1]
        return q, p, s, m_w, L

