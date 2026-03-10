import numpy as np
from scipy.linalg import inv
import queue

def get_values(A, l):
    return inv(np.eye(A.shape[0])-A.T)@l

class Economy:
    def __init__(self, params):
        A, self.l, self.T = params.A, params.l, params.T
        self.tau = params.tau
        self.avg_interval = params.avg_interval
        self.scenario = params.scenario
        self.order_queue = queue.Queue()

        self.params = params

        self.traj = {
            "values": np.array([get_values(params.A, params.l)]),
            "estimated_values_cur": np.array([get_values(params.A,params.l)]),
            "estimated_values_upd_indirect": np.array([get_values(params.A, params.l)]),
            "estimated_values_upd_indirect_w_annual_correction": np.array([get_values(params.A, params.l)]),
            "estimated_values_cur_new": np.array([get_values(params.A, params.l)]),
            "ll_times": np.array([params.l]),
            "updated_lls_cur": np.array([params.l]),
            "avg_order_vol": np.array([np.array([300,300,300,300,300])]),
            "interval_t": np.array([0]),
            "updated_lls_proposed": np.array([params.l]),
            "A": np.array([A]),
            "updated_lls_new_proposed": np.array([params.l])
        }
        self.t = np.array([0])

        self.order_vol = np.array([0,0,0,0,0])
        self.total_l = np.array([0,0,0,0,0])

        self.i = 1

    def step(self):
        cur_A = self.traj["A"][-1]
        current_est_ll = self.traj["updated_lls_cur"][-1]
        current_est_ll_prop = self.traj["updated_lls_proposed"][-1]
        current_est_ll_prop_new = self.traj["updated_lls_new_proposed"][-1]

        if self.i % self.params.avg_interval == 0:

            order_vol_safe = np.array([np.maximum(vol, 1e-7) for vol in self.order_vol])
            new_ll_times_safe = np.array([np.maximum(ll, 1e-7) for ll in self.total_l])
            new_ll_times = new_ll_times_safe / order_vol_safe
            new_vals = get_values(cur_A, new_ll_times)

            self.append_vals("avg_order_vol", self.order_vol)
            self.append_vals("ll_times", new_ll_times)
            self.append_vals("values", new_vals)
            self.append_vals("interval_t", self.i)

            new_est_ll = current_est_ll.copy() 
            new_est_ll_prop = current_est_ll_prop.copy()

            self.append_vals("updated_lls_cur", new_est_ll)
            self.append_vals("updated_lls_proposed", new_est_ll_prop)

            new_est_vals = get_values(cur_A, new_est_ll_prop)
            # new_est_vals[choice] = current_est_vals[choice] + (new_est_ll_prop[choice] - current_est_ll_prop[choice])

            self.append_vals("estimated_values_upd_indirect_w_annual_correction", new_est_vals)

            self.order_vol = np.array([0,0,0,0,0])
            self.total_l = np.array([0,0,0,0,0])

        choice, l_c, order_size = self.get_new_order(self.params, self.i, cur_A, current_est_ll) # order arrivest
        window_orders = self.get_queue_total(choice)
        print(window_orders)
        self.order_queue.put((choice, l_c, order_size))
        if self.order_queue.qsize() > self.avg_interval:
            self.order_queue.get()

        self.total_l[choice] += l_c * order_size 
        self.order_vol[choice] += order_size

        new_est_ll = current_est_ll.copy() 
        new_est_ll_prop = current_est_ll_prop.copy()
        new_est_ll_prop_new = current_est_ll_prop_new.copy()

        avg_order_vol = self.traj["avg_order_vol"][-1]
        avg_order_vol_choice_safe = np.maximum(avg_order_vol[choice], 1e-7)

        tau = self.params.tau
        new_est_ll[choice] = (l_c + (tau-1)*new_est_ll[choice]) / tau
        new_est_ll_prop[choice] = new_est_ll_prop[choice] + (order_size / avg_order_vol_choice_safe)*(l_c - new_est_ll_prop[choice])

        num = l_c * order_size + new_est_ll_prop_new[choice] * window_orders
        den = window_orders + order_size
        if den > 0:
            new_est_ll_prop_new[choice] = num / den

        self.append_vals("updated_lls_cur", new_est_ll)
        self.append_vals("updated_lls_proposed", new_est_ll_prop)
        self.append_vals("updated_lls_new_proposed", new_est_ll_prop_new)

        current_est_vals = self.traj["estimated_values_cur"][-1]
        current_est_vals_upd_indirect = self.traj["estimated_values_upd_indirect"][-1]
        current_est_vals_upd_indirect_annual_correct = self.traj["estimated_values_upd_indirect_w_annual_correction"][-1]

        new_est_vals = current_est_vals.copy()
        new_est_vals[choice] = current_est_vals[choice] + (new_est_ll_prop[choice] - current_est_ll_prop[choice])

        new_est_vals_upd_indirect = current_est_vals_upd_indirect.copy()
        new_est_vals_upd_indirect[choice] = current_est_vals_upd_indirect[choice] + (new_est_ll_prop[choice] - current_est_ll_prop[choice])

        new_est_vals_upd_indirect_annual_correct = current_est_vals_upd_indirect_annual_correct.copy()
        new_est_vals_upd_indirect_annual_correct[choice] = current_est_vals_upd_indirect_annual_correct[choice] + (new_est_ll_prop[choice] - current_est_ll_prop[choice])

        if self.i % self.params.avg_interval == 0:
            self.t = np.append(self.t, self.i)
            self.i += 1

            return

        self.append_vals("estimated_values_cur", new_est_vals)

        a_choice_row = cur_A[choice, :]
        for j, a in enumerate(a_choice_row):
            if a > 0:
                new_val_j_upd_indirect = new_est_ll_prop[j]
                new_val_j_upd_indirect_w_annual_correct = new_est_ll_prop[j]
                a_j_col = cur_A[:, j]
                new_val_j_upd_indirect += a_j_col.dot(new_est_vals_upd_indirect)
                new_val_j_upd_indirect_w_annual_correct += a_j_col.dot(new_est_vals_upd_indirect_annual_correct)

                new_est_vals_upd_indirect[j] = new_val_j_upd_indirect
                new_est_vals_upd_indirect_annual_correct[j] = new_val_j_upd_indirect_w_annual_correct

        self.append_vals("estimated_values_upd_indirect", new_est_vals_upd_indirect)
        self.append_vals("estimated_values_upd_indirect_w_annual_correction", new_est_vals_upd_indirect_annual_correct)

        self.t = np.append(self.t, self.i)
        self.i += 1

    def append_vals(self, key, val):
        if isinstance(val, np.ndarray) or isinstance(val, list):
            self.traj[key] = np.append(self.traj[key], [val], axis= 0)
        else:
            self.traj[key] = np.append(self.traj[key], val)

    def get_queue_total(self, choice):
        total = 0
        l_queue = list(self.order_queue.queue)
        for (c, l_c, order_size) in l_queue:
            if c == choice:
                total += order_size

        return total

    def get_new_order(self, params, i, A, l):
        if params.scenario == "divergent_current_algo":
            choice = 1
            if i % 5 == 0:
                order_size = 1
                l_c = 0.05
            else:
                order_size = 20
                l_c = 1

            return choice, l_c, order_size

        choice = np.random.randint(0,5)
        epsilon = np.random.uniform(-0.2, 0.15)

        order_size = np.random.randint(0, 100)
        l_c = l[choice]
        l_c += l_c * epsilon
        l_c = max(l_c, 0)

        return choice, l_c, order_size


