"""Finding the simulation frames and putting them in true time order."""

import glob
import os


def frame_seconds(path):
    """Pull the number out of a filename: '100200.tif' -> 100200."""
    name = os.path.basename(path)          # '100200.tif'
    stem = os.path.splitext(name)[0]       # '100200'
    return int(stem)


def list_frames(folder):
    """Return (seconds, path) pairs, sorted by time rather than by name."""
    paths = glob.glob(os.path.join(folder, "*.tif"))
    paths.sort(key=frame_seconds)          # sort by the NUMBER, not the text
    return [(frame_seconds(p), p) for p in paths]
