"""Choosing which crops to train on, weighted by how interesting they are."""

import numpy as np

from hat.crops import take_crop, score_slice, CROP, SCORE

WET = 0.01     # metres; a cell counts as wet above this


def crop_interest(depth, row, col, size=CROP, score=SCORE):
    """How useful this crop is for learning, judged on its SCORED region.
    Returns the fraction of scored cells that are wet."""
    c = take_crop(depth, row, col, size)
    s = score_slice(size, score)
    scored = c[s, s]
    return float((scored > WET).mean())


def rank_origins(depth, origins):
    """Score every candidate crop and return them with their wet fractions."""
    return [(r, c, crop_interest(depth, r, c)) for r, c in origins]
