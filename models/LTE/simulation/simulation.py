from .parameters import Params
from typing import Tuple
import numpy as np

def get_trajectories(params: Params) -> Tuple[dict, np.ndarray, Exception]:
    """
        Run a simulation, get a trajectories dictionary (keys are strings, values are numpy arrays)
        (we'll call it traj) as well as another numpy array (we'll call it t) representing the independent
        variable axis. Run your sim in a try/except block and pass the exception with those
        so that it displays in the GUI as opposed to crashing the program
    """
    e = None
    try:
        pass
        # run your simulation
    except Exception as e:
        pass

    return traj, t, e
