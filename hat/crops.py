"""Cutting fixed-size square crops out of a full grid."""

import numpy as np

CROP = 128        # cells on a side that the model SEES
SCORE = 64        # cells on a side that the loss is computed on (centred)


def take_crop(arr, row, col, size=CROP):
    """Cut a square whose top-left corner is at (row, col)."""
    return arr[row:row + size, col:col + size]


def score_slice(size=CROP, score=SCORE):
    """The slice picking the central scored region out of a crop."""
    margin = (size - score) // 2
    return slice(margin, margin + score)


def crop_origins(shape, size=CROP, stride=SCORE):
    """Every top-left corner for crops that fit inside the grid.
    A stride of SCORE means the scored regions tile without overlapping."""
    rows = range(0, shape[0] - size + 1, stride)
    cols = range(0, shape[1] - size + 1, stride)
    return [(r, c) for r in rows for c in cols]
