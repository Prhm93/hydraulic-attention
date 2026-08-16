"""Grouping cells into patches. Each patch becomes one token."""

import numpy as np

PATCH = 8      # cells on a side; 8 * 30 m = 240 m
WET = 0.01     # metres


def to_patches(arr, patch=PATCH):
    """Split a (H, W) grid into (H/patch, W/patch, patch, patch)."""
    h, w = arr.shape
    return arr.reshape(h // patch, patch, w // patch, patch).transpose(0, 2, 1, 3)


def patch_wet_mean(arr, patch=PATCH, wet=WET):
    """Mean depth over WET cells only, per patch.
    Plain averaging would erase a narrow deep channel inside a dry patch."""
    p = to_patches(arr, patch)
    flat = p.reshape(p.shape[0], p.shape[1], -1).astype(np.float64)
    mask = flat > wet
    count = mask.sum(axis=-1)
    total = np.where(mask, flat, 0.0).sum(axis=-1)
    return np.where(count > 0, total / np.maximum(count, 1), 0.0)


def patch_max(arr, patch=PATCH):
    """Deepest cell in each patch."""
    p = to_patches(arr, patch)
    return p.reshape(p.shape[0], p.shape[1], -1).max(axis=-1)
