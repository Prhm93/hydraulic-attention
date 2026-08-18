"""Load a whole region into RAM once: frames, DEM, Manning, barriers."""
import numpy as np
import rasterio
from hat.frames import list_frames
from hat.io import read_depth
from hat.config import MANNING_BY_CODE
from hat.crops import crop_origins, CROP, SCORE
from hat.geometry import patch_centre_cells, barrier_max
from hat.patches import PATCH


def manning_from_codes(codes):
    """Map land cover class codes to Manning n."""
    out = np.full(codes.shape, np.nan, dtype=np.float32)
    for code, n in MANNING_BY_CODE.items():
        out[codes == code] = n
    if np.isnan(out).any():
        bad = np.unique(codes[np.isnan(out)])
        raise ValueError(f"unmapped land cover codes: {bad.tolist()}")
    return out


def load_region(frames_dir, dem_path, landcover_path, crop=CROP, score=SCORE,
                start=None, stop=None, verbose=True):
    """Read everything once. Returns a dict shared by every FloodPairs instance."""
    meta = list_frames(frames_dir)[start:stop]
    shape = read_depth(meta[0][1]).shape
    depth = np.empty((len(meta),) + shape, dtype=np.float32)
    for k, (_, path) in enumerate(meta):
        depth[k] = read_depth(path)
        if verbose and k % 200 == 0:
            print(f"  {k}/{len(meta)}", flush=True)

    with rasterio.open(dem_path) as s:
        dem = s.read(1).astype(np.float32)
    with rasterio.open(landcover_path) as s:
        manning = manning_from_codes(s.read(1))
    if dem.shape != shape or manning.shape != shape:
        raise ValueError(f"raster shape mismatch: {dem.shape} {manning.shape} vs {shape}")
    return {"times": [t for t, _ in meta], "depth": depth, "dem": dem,
            "manning": manning, "shape": shape,
            "origins": crop_origins(shape, crop, score)}


def add_barriers(region, crop=CROP, patch=PATCH, verbose=True):
    """Max ground elevation between patch centres, per crop origin.

    Uses the ABSOLUTE dem, never the crop-mean-subtracted version: the gate
    compares eta = h + b against these elevations, so both must share a datum.
    """
    g = crop // patch
    centres = patch_centre_cells(g, g, patch)
    out = {}
    for row, col in region["origins"]:
        sub = region["dem"][row:row + crop, col:col + crop]
        out[(row, col)] = barrier_max(sub, centres)
        if verbose:
            print(f"  barrier {(row, col)} done", flush=True)
    region["barrier"] = out
    return region
