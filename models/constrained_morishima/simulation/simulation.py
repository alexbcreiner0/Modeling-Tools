import numpy as np
from scipy.linalg import logm, expm
from .parameters import Params

def get_trajectories(params):
    k1, k2, e, a = params.k1, params.k2, params.e, params.a
    c1 = k1 / (1 + e + k1)
    v1 = (1 - c1) / (1 + e)
    s1 = (e * v1)
    c2 = k2 / (1 + e + k2)
    v2 = (1 - c2) / (1 + e)
    s2 = e * v2

    denom = 1 - a * s2
    M = np.array([
        [c1, c2],
        [((1 - a) * s1 * c1 + v1) / denom, ((1 - a) * s1 * c2 + v2) / denom]
    ])

    # Invert M
    M_inv = np.linalg.inv(M)
    log_M_inv = logm(M_inv)
    dt = 0.01
    substep_matrix = expm(dt * log_M_inv)

    y1i, y2i, N0, T = params.y1i, params.y2i, params.N0, params.T

    # Initial output vector
    y0 = np.array([y1i, y2i])  # initial values for y1 and y2

    # Time steps
    n_steps = int(T / dt)
    ys = [y0]
    Ns = [N0]
    Es = [0]

    for t in range(n_steps):
        yt = ys[-1]
        Nt = Ns[-1]
        
        yt1 = substep_matrix @ yt
        E = (v1*yt1[0]+v2*yt1[1]) / params.w
        yt1 = np.maximum(yt1, 0)  # enforce non-negativity

        if E > Nt:
            scale = Nt * params.w / (v1*yt1[0]+v2*yt1[1])
            yt1 = yt1 * scale
            E = Nt

        Nt1 = Nt + params.r * E- params.s*(Nt-E)

        ys.append(yt1)
        Es.append(E)
        Ns.append(Nt1)

        if Nt1 <= 0:
            break

    ys = np.array(ys)
    Ns = np.array(Ns)
    Es = np.array(Es)
    t = np.linspace(0, T, len(ys))

    traj = {"y": ys, "Ns": Ns, "Es": Es}
    return traj, t, None
