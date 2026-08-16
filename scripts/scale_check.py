"""Is the flood the same problem at 30 m and 60 m, once made dimensionless?"""

import numpy as np
from hat.frames import list_frames
from hat.io import read_depth

B = "data/raw/FloodCastBench/High-fidelity flood forecasting"
G = 9.81
TIMES = [300000, 450000, 600000, 700200, 799800]

f30 = dict(list_frames(f"{B}/30m/Australia"))
f60 = dict(list_frames(f"{B}/60m/Australia"))
print("30m frames:", len(f30), " 60m frames:", len(f60))
print("shared timestamps:", len(set(f30) & set(f60)))


print(f"\n{'time':>8} {'p99_30':>8} {'p99_60':>8} {'wet30%':>7} {'wet60%':>7} "
      f"{'Cr30':>7} {'Cr60':>7} {'ratio':>6}")

for t in TIMES:
    d30 = read_depth(f30[t]).astype(np.float64)
    d60 = read_depth(f60[t]).astype(np.float64)
    w30, w60 = d30[d30 > 0.01], d60[d60 > 0.01]
    p30, p60 = np.percentile(w30, 99), np.percentile(w60, 99)
    cr30 = np.sqrt(G * p30) * 300.0 / 30.0
    cr60 = np.sqrt(G * p60) * 600.0 / 60.0
    print(f"{t:>8} {p30:>8.3f} {p60:>8.3f} "
          f"{100*w30.size/d30.size:>7.2f} {100*w60.size/d60.size:>7.2f} "
          f"{cr30:>7.2f} {cr60:>7.2f} {cr60/cr30:>6.3f}")
