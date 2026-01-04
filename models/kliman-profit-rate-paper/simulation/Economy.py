import numpy as np
from scipy.linalg import eig

class Economy:
    def __init__(self, params):
        self.params = params
        self.traj = {
            "q": np.array([params.q]),
            "l": np.array([params.l]),
            "L": np.array([params.L]),
            "A": np.array([params.A]),
            "b": np.array([params.b]),
            "p": np.array([params.p])
        }

        eqp, epr = self._get_equilibrium_prices(
            params.A, params.b, params.l, params.p
        )

        self.traj["eqp"] = np.array([eqp])
        self.traj["epr"] = np.array([epr])

        MELT = params.init_MELT
        num = params.l.dot(params.q) * (1 - 1/MELT*params.p.dot(params.b))
        M = params.A + np.outer(params.b, params.l)
        den = 1/MELT*(M.T@params.p).dot(params.q)
        tssi_value_rop = num / den
        self.traj["tssi_value_rop"] = np.array([tssi_value_rop])
        self.traj["MELT"] = np.array([MELT])

        comp1 = params.A[0,0] / params.l[0]
        comp2 = params.A[0,1] / params.l[1]
        comp = np.array([comp1, comp2])
        print(comp)
        self.traj["comps"] = np.array([comp])

        self.current_t = 0
        self.T = params.T

    def step(self):
        self.current_t += 1
        q = self.traj["q"][-1].copy()
        l = self.traj["l"][-1].copy()
        L = self.traj["L"][-1].copy()
        A = self.traj["A"][-1].copy()
        b = self.traj["b"][-1].copy()
        p = self.traj["p"][-1].copy()
        MELT = self.traj["MELT"][-1].copy()
        out_rate = self.params.output_rate
        labor_rate = self.params.labor_rate

        # applying 'technology changes'
        if self.params.stop_halfway and self.current_t > self.T // 2:
            new_q = q
            new_L = L
            new_l = l
        else:
            new_q = (1.0 + out_rate)*q
            new_L = (1.0 + labor_rate)*L
            new_l = new_L / new_q

        M = A+np.outer(b,new_l)
        num = new_l.dot(new_q)*(1 - 1/MELT * p.dot(b))
        den = 1/MELT*(M.T@p).dot(new_q)
        tssi_value_rop = num / den
        new_p = (1+tssi_value_rop)*M.T@p
        new_eqp, new_epr = self._get_equilibrium_prices(A, b, new_l, new_p)

        new_MELT = MELT * (new_p.dot(new_q)) / ((A.T@p + MELT*new_l).dot(new_q))
        self.traj["MELT"] = np.append(self.traj["MELT"], new_MELT)

        comp1 = A[0,0] / new_l[0]
        comp2 = A[0,1] / new_l[1]
        comp = np.array([comp1, comp2])
        self.traj["comps"] = np.append(self.traj["comps"], [comp], axis= 0)

        if "eqp" not in self.traj:
            self.traj["eqp"] = np.array([new_eqp])
        else:
            self.traj["eqp"] = np.append(self.traj["eqp"], [new_eqp], axis= 0)
        if "epr" not in self.traj:
            self.traj["epr"] = np.array([new_epr])
        else:
            self.traj["epr"] = np.append(self.traj["epr"], new_epr)
        if "tssi_value_rop" not in self.traj:
            self.traj["tssi_value_rop"] = np.array([tssi_value_rop])
        else:
            self.traj["tssi_value_rop"] = np.append(self.traj["tssi_value_rop"], tssi_value_rop)

        self.traj["q"] = np.append(self.traj["q"], [new_q], axis= 0)
        self.traj["l"] = np.append(self.traj["l"], [new_l], axis= 0)
        self.traj["L"] = np.append(self.traj["L"], [new_L], axis= 0)
        self.traj["b"] = np.append(self.traj["b"], [b], axis= 0)
        self.traj["p"] = np.append(self.traj["p"], [new_p], axis= 0)

    def _get_equilibrium_prices(self, A, b, l, p_num_ref):
        M = A + np.outer(b, l)
        evals, evecs = eig(M.T)
        idx = np.argmax(evals.real)
        r_hat = evals[idx]
        r_hat = np.real(r_hat)
        eqp = evecs[:,idx].real
        epr = 1 / r_hat - 1
        if eqp[0] < 0: eqp *= 1
        # scale = np.linalg.norm(p_num_ref)
        # eqp *= 1/max(scale, 1e-8)
        return eqp, epr



