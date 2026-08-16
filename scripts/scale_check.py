"""Is the flood the same problem at 30 m and 60 m, once made dimensionless?"""

import numpy as np
from hat.frames import list_frames
from hat.io import read_depth

B = "data/raw/FloodCastBench/High-fidelity flood forecasting"
G = 9.81
TIMES = [600000, 700000, 800000]

f30 = dict(list_frames(f"{B}/30m/Australia"))
f60 = dict(list_frames(f"{B}/60m/Australia"))
print("30m frames:", len(f30), " 60m frames:", len(f60))
print("shared timestamps:", len(set(f30) & set(f60)))
