"""Autoregressive rollout: chain the model's own predictions forward instead of
scoring one real-input step, and watch where it and the baseline degrade.

Each step is one FRAME_STEP horizon (7,200 s), matching the model's training
target. Starting from one real depth frame in the stress window:

  model:    next_depth = clamp(current_depth + model(current_input), min=0)
            next input's Delta-h channel = the model's own predicted Delta-h
            (not recomputed from the clamped depth - literal model output)
  baseline: next_depth = clamp(current_depth + frozen_dh, min=0)
            frozen_dh is the real Delta-h at the start frame, never updated -
            velocity persistence extended forward at a constant rate.

Bed and Manning channels never change - only depth and Delta-h are rebuilt.
Only the model's own forward passes see this drift; ground truth at each step
is the real recorded depth frame_true = start_frame + step * FRAME_STEP.

Compared on the scored centre (hat.crops.score_slice) only, since that is the
region every other script in this repo trusts.

CROP-BOUNDARY WARNING: the model's output beyond the scored centre never had a
training loss on it. Feeding that untrained margin back in lets error grow
inward from the edges each step. ~6 steps is the point this repo has adopted
as the limit before that contamination reaches the scored centre - do not
push far past it without re-deriving the number.
"""
import argparse
import numpy as np
import torch

from hat.region import load_region, add_barriers
from hat.model import HydraulicTransformer
from hat.crops import CROP, SCORE, score_slice, take_crop
from hat.patches import PATCH, to_patches, patch_wet_mean
from hat.config import H_SCALE, DH_SCALE, BED_SCALE, MANNING_SCALE
from hat.io import FRAME_STEP
from hat.metrics import contingency
from scripts.train import (build_gpu_statics, assemble_tau, make_physics,
                           load_ckpt, git_hash)

STEP_WARN = 6
THRESHOLDS = (0.05, 0.30)


def pick_origin(region, frame):
    """Origin whose scored centre has the highest wet fraction at `frame` -
    a rollout on an almost-dry crop would be trivially easy for both sides."""
    s = score_slice(CROP, SCORE)
    best, best_frac = None, -1.0
    for o in region["origins"]:
        crop = take_crop(region["depth"][frame], o[0], o[1], CROP)[s, s]
        frac = float((crop > 0.01).mean())
        if frac > best_frac:
            best, best_frac = o, frac
    return best, best_frac


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


@torch.no_grad()
def rollout(ckpt, variant, start_frame, steps, origin=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = "data/raw/FloodCastBench/Relevant data/"
    F = "data/raw/FloodCastBench/High-fidelity flood forecasting/30m/Australia"
    stop = start_frame + steps * FRAME_STEP + FRAME_STEP
    region = load_region(F, D+"DEM/Australia_DEM.tif",
                         D+"Land use and land cover/Australia.tif",
                         start=0, stop=stop)
    add_barriers(region, verbose=False)
    dist, barrier_all, bed_mean_all = build_gpu_statics(region, device)

    if origin is None:
        origin, wet_frac = pick_origin(region, start_frame)
        print(f"auto-picked origin={origin} wet_frac={wet_frac:.3f}", flush=True)
    oidx = region["origins"].index(origin)
    o = origin

    model = HydraulicTransformer(in_channels=4, depth=6).to(device).eval()
    phys, uses_gate = make_physics(variant, device)
    load_ckpt(ckpt, model, phys, None, variant, git_hash())

    dem_c = take_crop(region["dem"], o[0], o[1], CROP).astype(np.float32)
    man_c = take_crop(region["manning"], o[0], o[1], CROP).astype(np.float32)
    bed_detrended = dem_c - dem_c.mean()
    bed_patch = to_patches(dem_c).reshape(CROP // PATCH, CROP // PATCH, -1).mean(axis=-1)
    ref = float(bed_mean_all[oidx])
    barrier_c = barrier_all[oidx] - ref                          # (256, 256) tensor

    depth0 = take_crop(region["depth"][start_frame], o[0], o[1], CROP).astype(np.float32)
    depth_prev = take_crop(region["depth"][start_frame - FRAME_STEP], o[0], o[1], CROP).astype(np.float32)
    dh0 = depth0 - depth_prev

    cur_depth_model, cur_dh_model = depth0.copy(), dh0.copy()
    cur_depth_base, frozen_dh = depth0.copy(), dh0.copy()

    s = score_slice(CROP, SCORE)
    rows = []
    for k in range(1, steps + 1):
        frame_true = start_frame + k * FRAME_STEP
        true_full = take_crop(region["depth"][frame_true], o[0], o[1], CROP).astype(np.float32)

        x = np.stack([cur_depth_model / H_SCALE, cur_dh_model / DH_SCALE,
                     bed_detrended / BED_SCALE, man_c / MANNING_SCALE])
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)

        bias = None
        if phys is not None:
            h_patch = torch.from_numpy(patch_wet_mean(cur_depth_model).ravel()).float().to(device)
            eta = (h_patch + torch.from_numpy(bed_patch.ravel()).float().to(device)) - ref
            tau = assemble_tau(dist, h_patch.unsqueeze(0))
            bias = phys(tau, eta.unsqueeze(0), barrier_c.unsqueeze(0))

        pred_dh_full = model(x_t, bias)[0, 0].cpu().numpy()      # (512, 512) metres
        next_depth_model = np.clip(cur_depth_model + pred_dh_full, 0, None)
        next_depth_base = np.clip(cur_depth_base + frozen_dh, 0, None)

        pm, bm, tm = next_depth_model[s, s], next_depth_base[s, s], true_full[s, s]
        row = {"step": k, "frame": frame_true}
        for thresh in THRESHOLDS:
            row[f"csi{thresh}_model"] = contingency(pm, tm, thresh)["csi"]
            row[f"csi{thresh}_base"] = contingency(bm, tm, thresh)["csi"]
        row["rmse_model"] = rmse(pm, tm)
        row["rmse_base"] = rmse(bm, tm)
        rows.append(row)

        cur_depth_model, cur_dh_model = next_depth_model, pred_dh_full
        cur_depth_base = next_depth_base

    return rows, origin


def print_table(rows):
    print(f"\n{'step':>4} {'frame':>6} {'csi.05 M':>9} {'csi.05 B':>9} "
          f"{'csi.30 M':>9} {'csi.30 B':>9} {'rmse M':>9} {'rmse B':>9}")
    for r in rows:
        print(f"{r['step']:>4} {r['frame']:>6} "
              f"{r['csi0.05_model']:>9.3f} {r['csi0.05_base']:>9.3f} "
              f"{r['csi0.3_model']:>9.3f} {r['csi0.3_base']:>9.3f} "
              f"{r['rmse_model']:>9.5f} {r['rmse_base']:>9.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True, choices=["1", "4a", "4b"])
    ap.add_argument("--start-frame", type=int, default=1728, help="frame idx: real first step")
    ap.add_argument("--steps", type=int, default=STEP_WARN)
    ap.add_argument("--origin-row", type=int, default=None)
    ap.add_argument("--origin-col", type=int, default=None)
    args = ap.parse_args()
    if args.steps > STEP_WARN:
        print(f"warning: {args.steps} > {STEP_WARN}-step limit this repo has adopted "
              f"for crop-boundary contamination - results past step {STEP_WARN} are suspect",
              flush=True)

    origin = None
    if args.origin_row is not None and args.origin_col is not None:
        origin = (args.origin_row, args.origin_col)

    rows, used_origin = rollout(args.ckpt, args.variant, args.start_frame, args.steps, origin)
    print(f"origin={used_origin} start_frame={args.start_frame} steps={args.steps}")
    print_table(rows)


if __name__ == "__main__":
    main()
