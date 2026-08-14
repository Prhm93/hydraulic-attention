import glob
import os
import numpy as np
import rasterio

FOLDER = "data/raw/FloodCastBench/High-fidelity flood forecasting/30m/Australia"
WET = 0.01   # metres; a cell counts as wet above this

# Collect every .tif, then sort by the NUMBER in the filename, not the text.
# Sorting as text would put 100200.tif before 10200.tif and scramble time.
paths = glob.glob(os.path.join(FOLDER, "*.tif"))
paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
print(f"{len(paths)} frames, first={os.path.basename(paths[0])}, last={os.path.basename(paths[-1])}")

# Read each frame and record a few numbers about it.
rows = []
for i, p in enumerate(paths):
    seconds = int(os.path.splitext(os.path.basename(p))[0])
    with rasterio.open(p) as src:
        d = src.read(1)
    rows.append((seconds, float(d.max()), float((d > WET).mean()), float(d.sum())))
    if i % 200 == 0:
        print(f"  {i}/{len(paths)}", flush=True)

# Save so we never have to read 2881 files again.
arr = np.array(rows)
np.save("data/australia_frame_stats.npy", arr)

# Print a short summary.
print("\nmax depth over all frames:", arr[:, 1].max())
print("wet fraction: start", arr[0, 2], " end", arr[-1, 2], " peak", arr[:, 2].max())
print("frame of peak wet fraction:", int(arr[arr[:, 2].argmax(), 0]), "s")
