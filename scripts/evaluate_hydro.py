"""Hydrological evaluation of a checkpoint: flood extent (CSI/POD/FAR) and mass
balance, model vs velocity persistence, on the stress test window.

Loads the checkpoint exactly as evaluate.py does (same region, same FloodPairs,
same load_ckpt, same variant switch). For each example it reconstructs metre
depth fields:
  predicted depth = input depth + predicted dh
  true depth      = input depth + target dh
  baseline depth   = input depth + persistence dh   (channel 1 of the input)
then scores those fields with hat/metrics.py.

Single-step only. Arrival-time and peak metrics need a real time sequence per
cell, which this script does not build - that is the rollout script's job.
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from hat.region import load_region, add_barriers
from hat.dataset import FloodPairs
from hat.model import HydraulicTransformer
from hat.crops import CROP, SCORE, score_slice
from hat.config import H_SCALE, DH_SCALE
from hat.io import CELL_SIZE_M
from hat.metrics import contingency, mass_error
from scripts.train import (collate, build_gpu_statics, assemble_tau,
                           make_physics, phi_for_batch, load_ckpt, git_hash)

THRESHOLDS = (0.05, 0.30)   # metres


@torch.no_grad()
def evaluate(ckpt, variant, test_start, test_stop, batch=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = "data/raw/FloodCastBench/Relevant data/"
    F = "data/raw/FloodCastBench/High-fidelity flood forecasting/30m/Australia"
    region = load_region(F, D+"DEM/Australia_DEM.tif",
                         D+"Land use and land cover/Australia.tif",
                         start=0, stop=test_stop)   # load through test window
    add_barriers(region, verbose=False)
    statics = build_gpu_statics(region, device)

    ds = FloodPairs(region, start=test_start, stop=test_stop)
    dl = DataLoader(ds, batch_size=batch, shuffle=False,
                    num_workers=4, collate_fn=collate, drop_last=False)

    model = HydraulicTransformer(in_channels=4, depth=6).to(device).eval()
    phys, uses_gate = make_physics(variant, device)
    load_ckpt(ckpt, model, phys, None, variant, git_hash())

    s = score_slice(CROP, SCORE)
    pred_all, true_all, base_all = [], [], []
    for x, target, h_patch, eta, oidx in dl:
        x = x.to(device); target = target.to(device)
        h_patch = h_patch.to(device); eta = eta.to(device); oidx = oidx.to(device)
        tau = assemble_tau(statics[0], h_patch) if phys is not None else None
        bias = phi_for_batch(phys, uses_gate, tau, eta, oidx, statics[1], statics[2]) \
            if phys is not None else None

        now_m  = x[:, 0, s, s] * H_SCALE                        # input depth, metres
        pred_dh = model(x, bias)[:, 0, s, s]                     # already metres
        base_dh = x[:, 1, s, s] * DH_SCALE                       # channel 1 = prev dh, metres
        true_dh = target                                         # already metres

        # depth cannot be negative; clip the reconstructed fields, not the deltas
        pred_all.append((now_m + pred_dh).clamp(min=0).cpu().numpy())
        base_all.append((now_m + base_dh).clamp(min=0).cpu().numpy())
        true_all.append((now_m + true_dh).clamp(min=0).cpu().numpy())

    pred = np.concatenate(pred_all, axis=0)   # (N, 256, 256) metres
    base = np.concatenate(base_all, axis=0)
    true = np.concatenate(true_all, axis=0)

    extent = {}
    for thresh in THRESHOLDS:
        extent[thresh] = {
            "model": contingency(pred, true, thresh),
            "baseline": contingency(base, true, thresh),
        }

    cell_area = CELL_SIZE_M ** 2
    mass = {
        "model": mass_error(pred, true, cell_area),
        "baseline": mass_error(base, true, cell_area),
    }
    return extent, mass, pred.shape[0]


def print_table(extent, mass, n):
    print(f"\nn examples = {n}\n")
    print(f"{'thresh':>7} {'source':>8} {'POD':>7} {'FAR':>7} {'CSI':>7} "
          f"{'tp':>10} {'fp':>10} {'fn':>10}")
    for thresh in THRESHOLDS:
        for src in ("model", "baseline"):
            c = extent[thresh][src]
            print(f"{thresh:>7.2f} {src:>8} {c['pod']:>7.3f} {c['far']:>7.3f} "
                  f"{c['csi']:>7.3f} {c['tp']:>10d} {c['fp']:>10d} {c['fn']:>10d}")

    print(f"\n{'source':>8} {'mass mean_rel':>14} {'mass max_rel':>14}")
    for src in ("model", "baseline"):
        m = mass[src]
        print(f"{src:>8} {m['mean_rel']:>14.4f} {m['max_rel']:>14.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True, choices=["1", "4a", "4b"])
    ap.add_argument("--test-start", type=int, default=1728, help="frame idx: start of day 7")
    ap.add_argument("--test-stop", type=int, default=2016, help="frame idx: end of day 7")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    extent, mass, n = evaluate(args.ckpt, args.variant, args.test_start, args.test_stop, args.batch)
    print_table(extent, mass, n)


if __name__ == "__main__":
    main()
