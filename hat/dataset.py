"""One training example = one crop, at one timestep, plus the next timestep."""

import numpy as np

from hat.frames import list_frames
from hat.io import read_depth
from hat.crops import take_crop, score_slice, crop_origins, CROP, SCORE
from hat.patches import patch_wet_mean
from hat.sampling import WET


class FloodPairs:
    """Serves (depth now, depth next, target change) for every crop and step."""

    def __init__(self, folder, crop=CROP, score=SCORE, frame_step=1):
        self.frames = list_frames(folder)
        self.crop = crop
        self.score = score
        self.frame_step = frame_step        # 1 = predict the very next frame

        # Work out the grid shape once, from the first frame.
        shape = read_depth(self.frames[0][1]).shape
        self.origins = crop_origins(shape, crop, score)

        # Every (frame, crop) combination that has a "next" frame to predict.
        n_pairs = len(self.frames) - frame_step
        self.index = [(i, o) for i in range(n_pairs) for o in range(len(self.origins))]

        # Remember the last frame read, so consecutive crops don't re-read it.
        self._cache = {}

    def __len__(self):
        return len(self.index)

    def _frame(self, i):
        """Read frame i, reusing it if it is already in the small cache."""
        if i not in self._cache:
            self._cache = {i: read_depth(self.frames[i][1])}   # keep only one
        return self._cache[i]

    def __getitem__(self, k):
        i, o = self.index[k]
        row, col = self.origins[o]

        now = take_crop(self._frame(i), row, col, self.crop)
        nxt = take_crop(self._frame(i + self.frame_step), row, col, self.crop)

        s = score_slice(self.crop, self.score)
        change = nxt.astype(np.float64) - now.astype(np.float64)

        return {
            "h_now": now,                       # what the model sees
            "target": change[s, s],             # what it must predict
            "h_patch": patch_wet_mean(now),     # depth per patch, for celerity
            "time_s": self.frames[i][0],
            "origin": (row, col),
        }
