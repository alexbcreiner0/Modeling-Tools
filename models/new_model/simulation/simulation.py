from .parameters import Params
import numpy as np

MODEL_READY = False # Set this to True when you think it's ready.
HEAVY_COMPUTE = False # Set true if some sims are compute-heavy (app will attempt to boost performance in various ways).

def get_trajectories(params: Params):
    traj = {}
    t = np.array([0])

    a, b, T = params.a, params.b, params.T

    epsilon = 0.03
    t = np.linspace(0,10,300)
    traj = {
        "cos": a*np.cos(b*t),
        "sin": a*np.sin(b*t),
    }
    # t = np.append(t, t[-1] + epsilon)

    yield traj, t
