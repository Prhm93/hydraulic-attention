"""Evaluate a checkpoint in metres, against velocity persistence on identical crops.

Baseline = channel 1 of the input (the previous dh), which IS the velocity-
persistence prediction. Read from there so model and baseline never diverge.
Skill = 100 * (1 - rmse_model / rmse_baseline), same definition as the horizon sweep.
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from hat.region import load_region, add_barriers
from hat.dataset import FloodPairs
from hat.model import HydraulicTransformer
from hat.io import HORIZON_SECONDS
from hat.crops import CROP, SCORE, score_slice
from hat.config import DH_SCALE
from scripts.train import (collate, build_gpu_statics, assemble_tau,
                           make_physics, phi_for_batch, load_ckpt, git_hash)


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
    opt = torch.optim.AdamW(list(model.parameters()) +
                            (list(phys.parameters()) if phys else []), lr=1e-4)
    load_ckpt(ckpt, model, phys, opt, variant, git_hash())

    s = score_slice(CROP, SCORE)
    se_model, se_base, n = 0.0, 0.0, 0
    for x, target, h_patch, eta, oidx in dl:
        x = x.to(device); target = target.to(device)
        h_patch = h_patch.to(device); eta = eta.to(device); oidx = oidx.to(device)
        tau = assemble_tau(statics[0], h_patch) if phys is not None else None
        bias = phi_for_batch(phys, uses_gate, tau, eta, oidx, statics[1], statics[2]) \
            if phys is not None else None

        pred_m = model(x, bias)[:, 0, s, s] * DH_SCALE          # metres
        base_m = x[:, 1, s, s] * DH_SCALE                       # channel 1 = prev dh, metres
        targ_m = target * DH_SCALE                              # metres

        se_model += float(((pred_m - targ_m) ** 2).sum())
        se_base  += float(((base_m - targ_m) ** 2).sum())
        n += targ_m.numel()

    rmse_model = (se_model / n) ** 0.5
    rmse_base  = (se_base / n) ** 0.5
    skill = 100 * (1 - rmse_model / rmse_base)
    return rmse_model, rmse_base, skill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True, choices=["1", "4a", "4b"])
    ap.add_argument("--test-start", type=int, default=2304, help="frame idx: start of day 9")
    ap.add_argument("--test-stop", type=int, default=2880, help="through day 10")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    rm, rb, sk = evaluate(args.ckpt, args.variant, args.test_start, args.test_stop, args.batch)
    print(f"rmse_model={rm:.5f} m  rmse_baseline={rb:.5f} m  skill={sk:.1f}%")


if __name__ == "__main__":
    main()
