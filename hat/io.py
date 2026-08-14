"""Reading depth frames. The frames carry no map information, so the cell
size is supplied here and never read from the file."""

import numpy as np
import rasterio

CELL_SIZE_M = 30.0        # metres; from the dataset paper, NOT from the file
DT_SECONDS = 300.0        # seconds between consecutive frames


def read_depth(path):
    """Read one depth frame as a 2-D float32 array of metres."""
    with rasterio.open(path) as src:
        return src.read(1)


def read_pair(frames, i):
    """Return (depth_now, depth_next) for the i-th frame in a frame list."""
    now = read_depth(frames[i][1])
    nxt = read_depth(frames[i + 1][1])
    return now, nxt
