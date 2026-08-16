"""Wave celerity and travel time between patches."""

import numpy as np

G = 9.81
H_FLOOR = 0.001        # metres; stops celerity being exactly zero on dry ground
TAU_CAP = 1e6          # seconds; a stand-in for "unreachable"


def celerity(depth_per_patch, g=G, h_floor=H_FLOOR):
    """Wave speed in m/s. Deeper water carries a wave faster; dry ground
    is floored rather than zeroed so the division stays finite."""
    return np.sqrt(g * np.maximum(depth_per_patch, h_floor))


def travel_time(distances, c_flat, tau_cap=TAU_CAP):
    """Seconds for a wave to cross between each pair of patches.
    Speed is the average of the two endpoints' celerity (a first version;
    integrating along the path comes later)."""
    c_pair = 0.5 * (c_flat[:, None] + c_flat[None, :])
    tau = distances / c_pair
    return np.minimum(tau, tau_cap)
