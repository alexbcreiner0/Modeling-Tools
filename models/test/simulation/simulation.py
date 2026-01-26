from .parameters import Params
from typing import Optional, Callable
import numpy as np
import time

MODEL_READY = True # Set this to True when you think it's ready.
HEAVY_COMPUTE = False # Set true if some sims are compute-heavy (app will attempt to boost performance in various ways).

def get_trajectories(params: Params, *, should_stop: Optional[Callable[[], bool]]= None, yield_every: int= 1):
    a = params.a
    b = params.b
    traj = {
        "sin": np.array([]),
        "cos": np.array([]),
        "e_to_x": np.array([])
    }

    t_list = np.linspace(-10,10,300)
    new_t_list = []
    i = 0
    for t in t_list:
        if should_stop and should_stop():
            break

        new_t_list.append(t)
        i += 1
        sin = a*np.sin(b*t)
        cos = a*np.cos(b*t)
        e_to_x = a*np.e**(b*t)
        traj["sin"] = np.append(traj["sin"], sin)
        traj["cos"] = np.append(traj["cos"], cos)
        traj["e_to_x"] = np.append(traj["e_to_x"], e_to_x)

        time.sleep(0.1)
        if (i % yield_every) == 0:
            yield dict(traj), np.array(new_t_list)

    yield dict(traj), np.array(new_t_list)

