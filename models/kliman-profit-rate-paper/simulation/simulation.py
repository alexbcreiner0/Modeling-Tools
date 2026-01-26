from .parameters import Params
from typing import Tuple, Optional, Callable
import numpy as np
from .Economy import Economy

def get_trajectories(params: Params, *, should_stop: Optional[Callable[[], bool]]= None, yield_every: int= 1):

    economy = Economy(params)

    t = [0]
    for i in range(1, params.T):
        if should_stop and should_stop():
            break

        economy.step()
        t.append(i)
        traj = economy.traj

        if (i % yield_every) == 0:
            yield traj, t

    yield traj, t
