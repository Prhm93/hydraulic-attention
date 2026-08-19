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


class FloodPairs:
    def __init__(self, region, start, stop, crop=CROP, score=SCORE,
                 frame_step=FRAME_STEP, exclude=None):
        self.r = region
        self.crop, self.score, self.step = crop, score, frame_step
        self.origins = region["origins"]
        self._exclude = exclude if exclude is not None else (None, None)
        # bed per patch from the ABSOLUTE dem, for eta (shares the barrier datum)
        # bed exists everywhere, so a plain per-patch mean - not the wet-masked
        # mean used for depth. Avoids reusing a wet-mask fn for an unmasked purpose.
        from hat.patches import to_patches
        self.bed_patch = {o: to_patches(
            take_crop(region["dem"], o[0], o[1], crop)).reshape(
            crop // PATCH, crop // PATCH, -1).mean(axis=-1) for o in self.origins}
        # index over [start, stop): need i-step to look back and i+step to predict
        lo = max(start, frame_step)
        hi = min(stop, region["depth"].shape[0] - frame_step)
        # optionally carve out an interior window (the held-out test day), with a
        # frame_step guard band so no training pair reaches into the excluded range
        ex0, ex1 = getattr(self, "_exclude", (None, None))
        def ok(i):
            if ex0 is None:
                return True
            return (i + frame_step < ex0) or (i - frame_step >= ex1)
        self.index = [(i, k) for i in range(lo, hi)
                      for k in range(len(self.origins)) if ok(i)]
        self.origin_idx = {o: k for k, o in enumerate(self.origins)}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, j):
        i, k = self.index[j]
        o = self.origins[k]
        d = self.r["depth"]
        now = take_crop(d[i], o[0], o[1], self.crop).astype(np.float32)
        prev = take_crop(d[i - self.step], o[0], o[1], self.crop).astype(np.float32)
        fut = take_crop(d[i + self.step], o[0], o[1], self.crop).astype(np.float32)
        bed_abs = take_crop(self.r["dem"], o[0], o[1], self.crop).astype(np.float32)
        man = take_crop(self.r["manning"], o[0], o[1], self.crop).astype(np.float32)

        bed_detrended = bed_abs - bed_abs.mean()
        chans = np.stack([
            now / H_SCALE,
            (now - prev) / DH_SCALE,
            bed_detrended / BED_SCALE,
            man / MANNING_SCALE,
        ])                                            # (4, 512, 512)

        s = score_slice(self.crop, self.score)
        h_patch = patch_wet_mean(now).ravel().astype(np.float32)   # (256,)
        eta = h_patch + self.bed_patch[o].ravel()                  # absolute datum
        return {
            "x": chans,
            "target": ((fut - now)[s, s]).astype(np.float32),      # (256, 256)
            "h_patch": h_patch,
            "eta_patch": eta.astype(np.float32),
            "origin": o,
            "oidx": self.origin_idx[o],
        }
