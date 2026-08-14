import os
import numpy as np
import rasterio

FOLDER = "data/raw/FloodCastBench/High-fidelity flood forecasting/30m/Australia"
WET = 0.01

# Pick 11 frames evenly spaced across the 864000 s event.
stats = np.load("data/australia_frame_stats.npy")
picks = stats[np.linspace(0, len(stats) - 1, 11).astype(int), 0].astype(int)

print(f"{'time_s':>8} {'wet%':>7} {'max_m':>7} {'p50':>6} {'p90':>6} {'p99':>6} {'>4.2m %':>8}")
for t in picks:
    with rasterio.open(os.path.join(FOLDER, f"{t}.tif")) as src:
        d = src.read(1)
    wet = d[d > WET]                       # only cells that actually hold water
    if wet.size == 0:
        continue
    deep = float((wet > 4.2).mean() * 100)  # share of wet cells past the truncation depth
    p50, p90, p99 = np.percentile(wet, [50, 90, 99])
    print(f"{t:>8} {100*wet.size/d.size:>7.2f} {d.max():>7.2f} {p50:>6.2f} {p90:>6.2f} {p99:>6.2f} {deep:>8.2f}")
