"""Does a larger crop restore shallow/deep discrimination in the mask?"""

import numpy as np

from hat.frames import list_frames
from hat.io import read_depth, DT_SECONDS, CELL_SIZE_M
from hat.crops import take_crop
from hat.patches import to_patches
from hat.geometry import patch_centres, pair_distances
from hat.travel import celerity

F = "data/raw/FloodCastBench/High-fidelity flood forecasting/30m/Australia"
CONFIGS = [(128, 8), (256, 16), (512, 32), (768, 48), (1024, 64)]
STEPS = [4, 12, 24, 48]
SHALLOW_T, DEEP_T = 300, 2400          # frame indices: early vs late
ORIGIN = (0, 0)

frames = list_frames(F)


def wet_mean(arr, patch):
    p = to_patches(arr, patch)
    f = p.reshape(p.shape[0], p.shape[1], -1).astype(np.float64)
    m = f > 0.01
    n = m.sum(-1)
    return np.where(n > 0, np.where(m, f, 0).sum(-1) / np.maximum(n, 1), 0.0)


def openness(depth, crop, patch, step):
    c = take_crop(depth, *ORIGIN, size=crop)
    h = wet_mean(c, patch)
    n = h.shape[0]
    D = pair_distances(patch_centres(n, n, patch=patch, cell=CELL_SIZE_M))
    cel = celerity(h).ravel()
    tau = D / (0.5 * (cel[:, None] + cel[None, :]))
    return float((tau <= step * DT_SECONDS).mean()) * 100


sh_frame = read_depth(frames[SHALLOW_T][1])
dp_frame = read_depth(frames[DEEP_T][1])

print(f"{'crop':>5} {'patch':>6} {'km':>6} {'tokens':>7} {'step':>5} "
      f"{'shallow%':>9} {'deep%':>7} {'ratio':>6}")
for crop, patch in CONFIGS:
    n = crop // patch
    km = crop * CELL_SIZE_M / 1000
    for step in STEPS:
        sh = openness(sh_frame, crop, patch, step)
        dp = openness(dp_frame, crop, patch, step)
        print(f"{crop:>5} {patch:>6} {km:>6.1f} {n*n:>7} {step:>5} "
              f"{sh:>9.1f} {dp:>7.1f} {dp/max(sh,1e-9):>6.2f}")
