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
