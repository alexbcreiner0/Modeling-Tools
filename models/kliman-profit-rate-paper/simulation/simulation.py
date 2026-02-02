from .parameters import Params
from typing import Tuple, Optional, Callable
import numpy as np
from .Economy import Economy

def get_trajectories(params: Params):
    economy = Economy(params)

    t = [0]
    for i in range(1, params.T):
        economy.step()
        t.append(i)
        traj = economy.traj

        yield dict(traj), np.array(t)
