from .parameters import Params
from typing import Tuple
import numpy as np
from .Economy import Economy

def get_trajectories(params: Params) -> Tuple[dict, np.ndarray, Exception]:

    economy = Economy(params)

    e = None
    t = [0]
    try:
        for i in range(1, params.T):
            economy.step()
            t.append(i)
    except Exception as e:
        pass

    traj = economy.traj

    return traj, t, e
