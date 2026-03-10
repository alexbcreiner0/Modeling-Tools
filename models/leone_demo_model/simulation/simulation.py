from .parameters import Params
import numpy as np
from .constants import *

def get_trajectories(params: Params):
    t = np.array([0.0])

    a, b, T = params.a, params.b, params.T

    epsilon = 0.03
    for i in range(T):
        
        traj = {
            "cos": a*np.cos(b*t),
            "sin": a*np.tan(b*t)
        }
        t = np.append(t, t[-1]+epsilon)

        yield traj, t
