import numpy as np
import copy
from .Economy import Economy
import sys

def get_trajectories(params):
    sim_params = copy.deepcopy(params)
    e = None

    try:
        economy = Economy(sim_params)
    except Exception as error:
        e = error
        return {}, np.array([]), e

    for i in range(params.T):
        try:
            economy.step()
        except Exception as error:
            e = error
            with open("log.txt", "w") as f:
                print(economy.traj, file= f)
            break

    try:
        economy.get_analytic_curves()
    except Exception as error:
        e = error
        with open("log.txt", "w") as f:
            print(economy.traj, file= f)

    traj, t = economy.traj, economy.t
    return traj, t, e

