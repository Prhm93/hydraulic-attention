"""Train the Hydraulic Attention Transformer. One script, three variants.

  --variant 1    plain transformer (tau_bias=None)
  --variant 4a   tau only  (Phi with beta frozen at 0)
  --variant 4b   tau + barrier gate (full Phi)

Checkpoints are atomic (tmp->rename), every one kept, tagged with the git hash.
Resume is explicit: --resume PATH, and refuses on an architecture-hash mismatch.
"""
import argparse, os, subprocess, time
import numpy as np
import torch
from torch.utils.data import DataLoader

from hat.region import load_region, add_barriers
from hat.dataset import FloodPairs
from hat.model import HydraulicTransformer, PhysicsBias
from hat.geometry import patch_centres, pair_distances
from hat.travel import celerity, G, H_FLOOR
from hat.io import HORIZON_SECONDS
from hat.patches import PATCH
from hat.crops import CROP, SCORE, score_slice


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "nogit"


def collate(batch):
    x = torch.from_numpy(np.stack([b["x"] for b in batch]))
    target = torch.from_numpy(np.stack([b["target"] for b in batch]))
    h_patch = torch.from_numpy(np.stack([b["h_patch"] for b in batch]))
    eta = torch.from_numpy(np.stack([b["eta_patch"] for b in batch]))
    oidx = torch.tensor([b["oidx"] for b in batch], dtype=torch.long)
    return x, target, h_patch, eta, oidx


def build_gpu_statics(region, device):
    """Things computed once and kept on GPU: pair distances, stacked barriers,
    stacked bed means. Distances never change; barriers depend only on terrain."""
    dist = pair_distances(patch_centres(CROP // PATCH, CROP // PATCH))
    dist = torch.from_numpy(dist).float().to(device)              # (256, 256)
    origins = region["origins"]
    barrier = torch.stack([torch.from_numpy(region["barrier"][o]).float()
                           for o in origins]).to(device)          # (9, 256, 256)
    bed_mean = torch.tensor([region["bed_mean"][o] for o in origins],
                            dtype=torch.float32, device=device)   # (9,)
    return dist, barrier, bed_mean


def assemble_tau(dist, h_patch):
    """State-dependent travel time, rebuilt every batch from the current depth.

    celerity = sqrt(g * max(h, floor)); pair speed is the mean of endpoints;
    tau = distance / speed, normalised by the prediction horizon so tau_norm=1
    means 'a wave arrives in exactly one predicted step'.
    """
    cel = torch.sqrt(G * h_patch.clamp(min=H_FLOOR))              # (b, 256)
    pair = 0.5 * (cel[:, :, None] + cel[:, None, :])             # (b, 256, 256)
    tau = dist[None] / pair                                       # broadcast dist
    return tau / HORIZON_SECONDS


def make_physics(variant, device):
    """Return (physics_module_or_None, uses_gate). The variant switch.

      1   -> (None, False)          plain: forward gets tau_bias=None
      4a  -> (PhysicsBias, False)   tau only: beta FROZEN at 0, not just init 0
      4b  -> (PhysicsBias, True)    full Phi
    """
    if variant == "1":
        return None, False
    phys = PhysicsBias(dt_seconds=HORIZON_SECONDS).to(device)
    if variant == "4a":
        phys.beta.data.zero_()
        phys.beta.requires_grad_(False)      # frozen: cannot drift into 4b
    elif variant != "4b":
        raise ValueError(f"unknown variant {variant!r}")
    return phys, (variant == "4b")


def phi_for_batch(phys, uses_gate, tau, eta, oidx, barrier_all, bed_mean_all):
    """Build the (b, 256, 256) bias for this batch, or None for variant 1.
    Recentres eta and barrier by the crop's bed mean so the gate's sigmoid
    sees metres-above-local-ground, not metres-above-sea-level."""
    if phys is None:
        return None
    b = tau.shape[0]
    ref = bed_mean_all[oidx]                       # (b,)
    barrier = barrier_all[oidx] - ref[:, None, None]   # (b, 256, 256)
    eta_c = eta - ref[:, None]                     # (b, 256)
    return phys(tau, eta_c, barrier)


def train_step(model, phys, uses_gate, batch, statics, opt, device):
    x, target, h_patch, eta, oidx = batch
    x = x.to(device); target = target.to(device)
    h_patch = h_patch.to(device); eta = eta.to(device); oidx = oidx.to(device)
    dist, barrier_all, bed_mean_all = statics

    tau = assemble_tau(dist, h_patch) if phys is not None else None
    bias = phi_for_batch(phys, uses_gate, tau, eta, oidx, barrier_all, bed_mean_all) \
        if phys is not None else None

    pred = model(x, bias)                      # (b, 1, 512, 512), raw depth change
    s = score_slice(CROP, SCORE)
    pred_c = pred[:, 0, s, s]                  # model learns normalised dh directly
    loss = ((pred_c - target) ** 2).mean()     # both sides normalised; metres come at eval

    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss)


def save_ckpt(path, model, phys, opt, step, epoch, variant, ghash):
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "phys": phys.state_dict() if phys is not None else None,
        "opt": opt.state_dict(),
        "step": step, "epoch": epoch,
        "variant": variant, "git": ghash,
    }, tmp)
    os.replace(tmp, path)          # atomic on local fs: path is never partial


def load_ckpt(path, model, phys, opt, variant, ghash):
    ck = torch.load(path, map_location="cpu")
    if ck["variant"] != variant:
        raise SystemExit(f"variant mismatch: ckpt is {ck['variant']}, you asked {variant}")
    if ck["git"] != ghash:
        raise SystemExit(f"git mismatch: ckpt built at {ck['git']}, code now {ghash}. "
                         f"Architecture may differ; refusing to load.")
    model.load_state_dict(ck["model"])
    if phys is not None and ck["phys"] is not None:
        phys.load_state_dict(ck["phys"])
    opt.load_state_dict(ck["opt"])
    return ck["step"], ck["epoch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["1", "4a", "4b"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save-every", type=int, default=500, help="steps between mid-epoch saves")
    ap.add_argument("--train-stop", type=int, default=2304, help="frame index: end of days 1-8")
    ap.add_argument("--exclude-day", type=int, default=None,
                    help="day (1-10) to hold out of training for a stress test")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--tag", default="", help="suffix for checkpoint filenames")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ghash = git_hash()
    print(f"variant={args.variant} batch={args.batch} device={device} git={ghash}", flush=True)

    D = "data/raw/FloodCastBench/Relevant data/"
    F = "data/raw/FloodCastBench/High-fidelity flood forecasting/30m/Australia"
    region = load_region(F, D+"DEM/Australia_DEM.tif",
                         D+"Land use and land cover/Australia.tif",
                         start=0, stop=args.train_stop)
    add_barriers(region, verbose=False)
    statics = build_gpu_statics(region, device)

    exclude = None
    if args.exclude_day is not None:
        d0 = (args.exclude_day - 1) * 288
        exclude = (d0, d0 + 288)
        print(f"holding out day {args.exclude_day}: frames {exclude}", flush=True)
    ds = FloodPairs(region, start=0, stop=args.train_stop, exclude=exclude)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=4, collate_fn=collate, drop_last=True)
    print(f"examples={len(ds)} steps/epoch={len(dl)}", flush=True)

    model = HydraulicTransformer(in_channels=4, depth=6).to(device)
    phys, uses_gate = make_physics(args.variant, device)
    params = list(model.parameters()) + (list(phys.parameters()) if phys else [])
    opt = torch.optim.AdamW(params, lr=args.lr)

    step, start_epoch = 0, 0
    if args.resume:
        step, start_epoch = load_ckpt(args.resume, model, phys, opt, args.variant, ghash)
        print(f"resumed at step={step} epoch={start_epoch}", flush=True)

    def ckpt_path(kind):
        t = f"_{args.tag}" if args.tag else ""
        return f"checkpoints/v{args.variant}{t}_{kind}.pt"

    best = float("inf")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        run, seen, t0 = 0.0, 0, time.time()
        for batch in dl:
            loss = train_step(model, phys, uses_gate, batch, statics, opt, device)
            run += loss; seen += 1; step += 1
            if step % 50 == 0:
                print(f"e{epoch} s{step} loss={run/seen:.5f} "
                      f"{seen/(time.time()-t0):.1f}it/s", flush=True)
            if step % args.save_every == 0:
                save_ckpt(ckpt_path(f"step{step}"), model, phys, opt,
                          step, epoch, args.variant, ghash)
        avg = run / max(seen, 1)
        save_ckpt(ckpt_path(f"epoch{epoch}"), model, phys, opt,
                  step, epoch, args.variant, ghash)
        if avg < best:
            best = avg
            save_ckpt(ckpt_path("best"), model, phys, opt,
                      step, epoch, args.variant, ghash)
        print(f"== epoch {epoch} done avg_loss={avg:.5f} best={best:.5f} ==", flush=True)


if __name__ == "__main__":
    main()
