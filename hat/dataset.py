"""One example = one crop at frame i, with the crop's previous and future change.

Takes a pre-loaded region dict (see hat/region.py); never reads disk itself, so
train and test instances share one 10 GB array. Returns the ingredients for the
physics bias - tau itself is assembled on GPU in the training loop.
"""
import numpy as np
from hat.crops import take_crop, score_slice, CROP, SCORE
from hat.patches import patch_wet_mean, PATCH
from hat.io import FRAME_STEP
from hat.config import H_SCALE, DH_SCALE, BED_SCALE, MANNING_SCALE
