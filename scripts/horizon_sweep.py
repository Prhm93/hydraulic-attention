"""Choose the prediction horizon: enough signal to learn, and a mask that
still discriminates between shallow and deep water."""

import numpy as np

from hat.frames import list_frames
from hat.io import read_depth, DT_SECONDS
from hat.crops import take_crop, score_slice
from hat.patches import patch_wet_mean
from hat.geometry import patch_centres, pair_distances
from hat.travel import celerity

F = "data/raw/FloodCastBench/High-fidelity flood forecasting/30m/Australia"
STEPS = [1, 2, 4, 6, 12, 24, 48, 96, 144]      # frames ahead
ORIGINS = [(192, 320), (384, 384), (576, 256), (640, 640), (256, 704)]
TIMES = [900, 1400, 1900, 2400]                # frame indices, mid-to-late event

frames = list_frames(F)
D = pair_distances(patch_centres(16, 16))
cache = {}

def frame(i):
    if i not in cache:
        cache[i] = read_depth(frames[i][1]).astype(np.float64)
    return cache[i]

s = score_slice()
print(f"{'step':>5} {'horizon':>9} {'persist':>10} {'velocity':>10} {'skill%':>7} "
      f"{'open_shallow%':>14} {'open_deep%':>11} {'ratio':>6}")

for step in STEPS:
    p_all, v_all, o_shallow, o_deep = [], [], [], []
    for i in TIMES:
        if i + step >= len(frames):
            continue
        a, b, c = frame(i - step), frame(i), frame(i + step)
        for (r, cc) in ORIGINS:
            prev = take_crop(b - a, r, cc)[s, s]
            nxt = take_crop(c - b, r, cc)[s, s]
            p_all.append(np.sqrt((nxt ** 2).mean()))
            v_all.append(np.sqrt(((nxt - prev) ** 2).mean()))

        # How open is the tau <= step*dt mask, for a shallow and a deep crop?
        for (r, cc), bucket in [(ORIGINS[0], o_shallow), (ORIGINS[1], o_deep)]:
            h = patch_wet_mean(take_crop(b, r, cc))
            cel = celerity(h).ravel()
            tau = D / (0.5 * (cel[:, None] + cel[None, :]))
            bucket.append(float((tau <= step * DT_SECONDS).mean()) * 100)

    p, v = float(np.mean(p_all)), float(np.mean(v_all))
    sh, dp = float(np.mean(o_shallow)), float(np.mean(o_deep))
    ratio = dp / max(sh, 1e-9)
    print(f"{step:>5} {step*300:>8}s {p:>10.6f} {v:>10.6f} {100*(1-v/p):>7.1f} "
          f"{sh:>14.1f} {dp:>11.1f} {ratio:>6.2f}")
