import numpy as np
import copy
from .Economy import Economy
from typing import Optional, Callable
import sys

def get_trajectories(params, *, should_stop: Optional[Callable[[], bool]]= None, yield_every: int= 1):
    sim_params = copy.deepcopy(params)

    economy = Economy(sim_params)

    for i in range(params.T):
        if should_stop and should_stop():
            break

        economy.step()

        if (i % yield_every) == 0:
            yield economy.traj, economy.t

    economy.get_analytic_curves()

    traj, t = economy.traj, economy.t
    yield traj, t

