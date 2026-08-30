"""Load one SWE-GNN dike-breach simulation as gridded arrays.

Their raw files store one row per timestep, one column per cell, with cells
numbered the way networkx grid_2d_graph(D,D) numbers them: node i sits at
(x, y) = (i // D, i % D). So a flat 4096-vector reshapes to (D, D) row-major.
We return time-first image stacks the transformer can crop.
"""
import numpy as np


def load_sim(raw_dir, sim_id, dim):
    """Return depth, vx, vy as (T, dim, dim) and dem as (dim, dim), metres.

    raw_dir: folder containing DEM/ WD/ VX/ VY/
    sim_id:  integer simulation id
    dim:     grid side (64 for datasets 1-2, 128 for dataset 3)
    """
    dem_xyz = np.loadtxt(f"{raw_dir}/DEM/DEM_{sim_id}.txt")   # (cells, 3): x y z
    wd = np.loadtxt(f"{raw_dir}/WD/WD_{sim_id}.txt")          # (T, cells)
    vx = np.loadtxt(f"{raw_dir}/VX/VX_{sim_id}.txt")
    vy = np.loadtxt(f"{raw_dir}/VY/VY_{sim_id}.txt")

    n = dim * dim
    assert dem_xyz.shape[0] == n, f"DEM has {dem_xyz.shape[0]} cells, expected {n}"
    assert wd.shape[1] == n, f"WD has {wd.shape[1]} cells, expected {n}"

    dem = dem_xyz[:, 2].reshape(dim, dim)                    # row-major (x outer)
    T = wd.shape[0]
    depth = wd.reshape(T, dim, dim)
    velx = vx.reshape(T, dim, dim)
    vely = vy.reshape(T, dim, dim)
    return depth.astype(np.float32), velx.astype(np.float32), \
           vely.astype(np.float32), dem.astype(np.float32)
