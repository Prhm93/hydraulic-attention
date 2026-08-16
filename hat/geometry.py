"""Distances and directions between patch centres, in metres."""

import numpy as np

from hat.io import CELL_SIZE_M
from hat.patches import PATCH


def patch_centres(n_rows, n_cols, patch=PATCH, cell=CELL_SIZE_M):
    """(row, col) position of each patch centre, in metres."""
    step = patch * cell                      # 8 * 30 = 240 m between centres
    r = (np.arange(n_rows) + 0.5) * step
    c = (np.arange(n_cols) + 0.5) * step
    rr, cc = np.meshgrid(r, c, indexing="ij")
    return np.stack([rr.ravel(), cc.ravel()], axis=1)   # (n_patches, 2)


def pair_distances(centres):
    """Straight-line distance between every pair of patch centres."""
    diff = centres[:, None, :] - centres[None, :, :]    # (N, N, 2)
    return np.sqrt((diff ** 2).sum(axis=-1))            # (N, N)


def patch_centre_cells(grid_h, grid_w, patch=PATCH):
    """Patch centres in DEM cell coordinates, not metres."""
    r = (np.arange(grid_h) + 0.5) * patch
    c = (np.arange(grid_w) + 0.5) * patch
    rr, cc = np.meshgrid(r, c, indexing="ij")
    return np.stack([rr.ravel(), cc.ravel()], axis=1)


def barrier_max(dem, centres, samples=None, chunk=32):
    """Highest ground on the straight line between every pair of patch centres.
    Walked on the fine DEM: averaging first would erase a narrow embankment."""
    n = centres.shape[0]
    if samples is None:
        span = np.hypot(*(centres.max(axis=0) - centres.min(axis=0)))
        samples = int(np.ceil(span)) + 1          # roughly one sample per cell
    t = np.linspace(0.0, 1.0, samples)[None, None, :]
    h, w = dem.shape
    out = np.empty((n, n), dtype=dem.dtype)
    for s in range(0, n, chunk):
        a, b = centres[s:s + chunk, None, :], centres[None, :, :]
        rr = a[..., 0:1] + (b[..., 0:1] - a[..., 0:1]) * t
        cc = a[..., 1:2] + (b[..., 1:2] - a[..., 1:2]) * t
        ri = np.clip(np.rint(rr).astype(np.int32), 0, h - 1)
        ci = np.clip(np.rint(cc).astype(np.int32), 0, w - 1)
        out[s:s + chunk] = dem[ri, ci].max(axis=-1)
    return out
