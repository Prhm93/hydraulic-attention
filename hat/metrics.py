"""Hydrological evaluation metrics for flood surrogates.

These are the metrics water-resources journals expect, and they measure
things all-cell RMSE cannot see: where water is, when it arrives, and
whether the peak is right.
"""
import numpy as np


def contingency(pred, true, thresh):
    """Wet/dry agreement at a depth threshold.

    Returns POD (fraction of truly-wet cells found), FAR (fraction of
    predicted-wet cells that are wrong), and CSI (both combined).
    """
    p = pred > thresh
    t = true > thresh
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    pod = tp / (tp + fn) if (tp + fn) else np.nan
    far = fp / (tp + fp) if (tp + fp) else np.nan
    csi = tp / (tp + fn + fp) if (tp + fn + fp) else np.nan
    return dict(pod=float(pod), far=float(far), csi=float(csi),
                tp=int(tp), fp=int(fp), fn=int(fn))


def arrival_time(depth_seq, thresh, dt_seconds):
    """First time each cell becomes wet, in seconds.

    depth_seq: (T, H, W) depth over time.
    Cells that never get wet return NaN.
    """
    wet = depth_seq > thresh
    ever = wet.any(axis=0)
    first = np.argmax(wet, axis=0).astype(np.float64)
    first[~ever] = np.nan
    return first * dt_seconds


def arrival_error(pred_seq, true_seq, thresh, dt_seconds):
    """Front arrival time error, in seconds, over cells wet in BOTH.

    This is the metric a travel-time mechanism should improve.
    """
    ap = arrival_time(pred_seq, thresh, dt_seconds)
    at = arrival_time(true_seq, thresh, dt_seconds)
    both = np.isfinite(ap) & np.isfinite(at)
    if both.sum() == 0:
        return dict(n=0, mae=np.nan, bias=np.nan, rmse=np.nan)
    d = ap[both] - at[both]
    return dict(n=int(both.sum()), mae=float(np.abs(d).mean()),
                bias=float(d.mean()), rmse=float(np.sqrt((d ** 2).mean())))


def peak_metrics(pred_seq, true_seq, dt_seconds):
    """Peak depth error and time-to-peak error, over cells that ever wet."""
    pv, tv = pred_seq.max(axis=0), true_seq.max(axis=0)
    pt = pred_seq.argmax(axis=0) * dt_seconds
    tt = true_seq.argmax(axis=0) * dt_seconds
    m = tv > 0.01
    if m.sum() == 0:
        return dict(n=0, depth_mae=np.nan, depth_bias=np.nan, time_mae=np.nan)
    return dict(n=int(m.sum()),
                depth_mae=float(np.abs(pv[m] - tv[m]).mean()),
                depth_bias=float((pv[m] - tv[m]).mean()),
                time_mae=float(np.abs(pt[m] - tt[m]).mean()))


def mass_error(pred_seq, true_seq, cell_area_m2):
    """Relative error in total water volume, per timestep.

    A physically implausible model drifts here even when RMSE looks fine.
    Accumulate in float64 - float32 rounding is larger than the signal.
    """
    pv = pred_seq.astype(np.float64).sum(axis=(1, 2)) * cell_area_m2
    tv = true_seq.astype(np.float64).sum(axis=(1, 2)) * cell_area_m2
    rel = np.full_like(tv, np.nan)
    np.divide(pv - tv, tv, out=rel, where=tv > 0)
    return dict(mean_rel=float(np.nanmean(np.abs(rel))),
                max_rel=float(np.nanmax(np.abs(rel))))
