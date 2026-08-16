"""Splitting one region into train and test areas that cannot see each other."""

from hat.crops import crop_origins


def column_split(shape, crop, score, train_end, test_start):
    """Crops whose FULL extent lies west of train_end, and east of test_start.
    The gap between them is the buffer: at least one crop wide, so a training
    crop's context never reaches into test ground."""
    all_origins = crop_origins(shape, crop, score)
    train = [(r, c) for (r, c) in all_origins if c + crop <= train_end]
    test = [(r, c) for (r, c) in all_origins if c >= test_start]
    return train, test


def check_buffer(train_end, test_start, crop, cell=30.0):
    """The buffer must be at least one crop wide, or the split leaks."""
    gap = test_start - train_end
    return {
        "gap_cells": gap,
        "gap_km": gap * cell / 1000,
        "crop_km": crop * cell / 1000,
        "safe": gap >= crop,
    }
