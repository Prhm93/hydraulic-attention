"""Hard look at whether a checkpoint really forecasts, or just predicts small numbers.

Runs six checks on one eval window:
  1. skill overall (the headline)
  2. skill on ACTIVE cells only (|target| above a threshold) - the real test
  3. skill vs a zero-prediction baseline (does the model beat 'nothing moves'?)
  4. error split by how much each cell actually moved (are big changes handled?)
  5. correlation between prediction and truth (shape, not just magnitude)
  6. does the model just output its input? (copy-the-last-frame check)
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from hat.region import load_region, add_barriers
from hat.dataset import FloodPairs
from hat.model import HydraulicTransformer
from hat.crops import CROP, SCORE, score_slice
from hat.config import DH_SCALE
from scripts.train import (collate, build_gpu_statics, assemble_tau,
                           make_physics, phi_for_batch, load_ckpt, git_hash)


@torch.no_grad()
def collect(ckpt, variant, start, stop, batch=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = "data/raw/FloodCastBench/Relevant data/"
    F = "data/raw/FloodCastBench/High-fidelity flood forecasting/30m/Australia"
    region = load_region(F, D+"DEM/Australia_DEM.tif",
                         D+"Land use and land cover/Australia.tif",
                         start=0, stop=stop, verbose=False)
    add_barriers(region, verbose=False)
    statics = build_gpu_statics(region, device)
    ds = FloodPairs(region, start=start, stop=stop)
    dl = DataLoader(ds, batch_size=batch, shuffle=False,
                    num_workers=4, collate_fn=collate, drop_last=False)

    model = HydraulicTransformer(in_channels=4, depth=6).to(device).eval()
    phys, uses_gate = make_physics(variant, device)
    load_ckpt(ckpt, model, phys, None, variant, git_hash())

    s = score_slice(CROP, SCORE)
    P, T, B = [], [], []          # prediction, truth, baseline (all metres)
    for x, target, h_patch, eta, oidx in dl:
        x=x.to(device); target=target.to(device)
        h_patch=h_patch.to(device); eta=eta.to(device); oidx=oidx.to(device)
        tau = assemble_tau(statics[0], h_patch) if phys is not None else None
        bias = phi_for_batch(phys, uses_gate, tau, eta, oidx, statics[1], statics[2]) \
            if phys is not None else None
        pred = model(x, bias)[:, 0, s, s] * DH_SCALE
        P.append(pred.cpu().numpy().ravel())
        T.append((target * DH_SCALE).cpu().numpy().ravel())
        B.append((x[:,1,s,s] * DH_SCALE).cpu().numpy().ravel())
    return np.concatenate(P), np.concatenate(T), np.concatenate(B)


def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def skill(model_err, base_err): return 100 * (1 - model_err / base_err) if base_err > 0 else float('nan')


def report(pred, truth, base, active_thresh=0.01):
    print(f"\n{'='*60}")
    print(f"cells scored: {len(truth):,}")
    print(f"target: mean|dh|={np.abs(truth).mean():.5f}m  max|dh|={np.abs(truth).max():.3f}m")

    # 1. overall
    rm, rb = rmse(pred, truth), rmse(base, truth)
    print(f"\n1. OVERALL      model={rm:.5f}  persist={rb:.5f}  skill={skill(rm,rb):.1f}%")

    # 2. active cells only
    act = np.abs(truth) > active_thresh
    frac = act.mean()
    if act.sum() > 0:
        rma, rba = rmse(pred[act], truth[act]), rmse(base[act], truth[act])
        print(f"2. ACTIVE cells (|dh|>{active_thresh}m, {frac*100:.1f}% of cells)")
        print(f"   model={rma:.5f}  persist={rba:.5f}  skill={skill(rma,rba):.1f}%")
    else:
        print(f"2. ACTIVE cells: NONE above {active_thresh}m - window is static")

    # 3. vs zero prediction
    rz = rmse(np.zeros_like(truth), truth)
    print(f"3. vs ZERO      predicting nothing scores rmse={rz:.5f}")
    print(f"   model beats zero: {rm < rz}  (skill vs zero={skill(rm,rz):.1f}%)")

    # 4. error by movement bucket
    print(f"4. BY MOVEMENT SIZE:")
    edges = [0, 0.001, 0.01, 0.05, 0.2, 10]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (np.abs(truth) >= lo) & (np.abs(truth) < hi)
        if m.sum() > 0:
            print(f"   [{lo:.3f},{hi:.3f})m: {m.sum():>9,} cells  "
                  f"model_rmse={rmse(pred[m],truth[m]):.5f}  persist={rmse(base[m],truth[m]):.5f}")

    # 5. correlation
    if truth.std() > 0 and pred.std() > 0:
        r = float(np.corrcoef(pred, truth)[0,1])
        print(f"5. CORRELATION  pred vs truth r={r:.4f}  (1.0=perfect shape match)")

    # 6. lazy check: how close is prediction to just zero (copy last frame)?
    print(f"6. LAZINESS     pred mean|value|={np.abs(pred).mean():.5f}  "
          f"truth mean|value|={np.abs(truth).mean():.5f}")
    print(f"   ratio={np.abs(pred).mean()/max(np.abs(truth).mean(),1e-9):.2f}  "
          f"(near 0 = model outputs ~nothing; near 1 = matches truth scale)")
    print('='*60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True, choices=["1","4a","4b"])
    ap.add_argument("--test-start", type=int, required=True)
    ap.add_argument("--test-stop", type=int, required=True)
    args = ap.parse_args()
    p, t, b = collect(args.ckpt, args.variant, args.test_start, args.test_stop)
    report(p, t, b)


if __name__ == "__main__":
    main()
