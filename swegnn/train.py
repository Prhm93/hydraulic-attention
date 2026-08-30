"""Train a HydraulicTransformer surrogate on the SWE-GNN dike-breach sims.

Two output variants, identical backbone (patch embed, biased blocks, depth):

  --variant depth        head -> 1 channel : dh
  --variant directional  head -> 3 channels: dh, qx, qy

The only difference between them is the final Linear head and the loss. Loss is
MSE per channel. For the directional variant, --discharge-weight scales the
qx + qy loss relative to dh so we can probe sensitivity to that term.

Paired seeds: --seed feeds torch, numpy, and a private DataLoader generator, so
the same value gives the same init and the same data order across variants.

Checkpoints are atomic (tmp -> rename), tagged with the git hash, variant, and
seed. No physics bias is used here (tau_bias stays None).

Run from the swegnn/ directory:  python train.py --variant directional --seed 0
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make both this dir (for dataset.py / load_swegnn.py) and the repo root
# (for the hat package) importable regardless of where we are invoked.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import SweGnnDataset
from hat.model import HydraulicTransformer

VARIANT_CHANNELS = {"depth": 1, "directional": 3}
CHANNEL_NAMES = {"depth": ("dh",), "directional": ("dh", "qx", "qy")}


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "nogit"


class GridTransformer(HydraulicTransformer):
    """HydraulicTransformer with a multi-channel output head.

    Backbone is inherited unchanged; we only swap the final Linear so it emits
    `out_channels` patch tiles instead of one, and fold those back into a
    (b, out_channels, H, W) image.
    """

    def __init__(self, in_channels, out_channels, depth=6, dim=192, heads=4, patch=32):
        super().__init__(in_channels, depth=depth, dim=dim, heads=heads, patch=patch)
        self.out_channels = out_channels
        self.head = nn.Linear(dim, out_channels * patch * patch)

    def forward(self, x, tau_bias=None):
        b, _, h, w = x.shape
        gh, gw = h // self.patch, w // self.patch
        bias = self.relpos(gh, gw, x.device).unsqueeze(0)
        if tau_bias is not None:
            bias = bias + tau_bias.unsqueeze(1)
        t = self.embed(x)
        for blk in self.blocks:
            t = blk(t, bias)
        t = self.head(self.norm(t))                       # (b, gh*gw, C*p*p)
        c, p = self.out_channels, self.patch
        t = t.view(b, gh, gw, c, p, p)
        t = t.permute(0, 3, 1, 4, 2, 5).reshape(b, c, gh * p, gw * p)
        return t


def collate(batch):
    x = torch.stack([b["input"] for b in batch])          # (b, 2, D, D)
    dh = torch.stack([b["dh"] for b in batch])            # (b, D, D)
    qx = torch.stack([b["qx"] for b in batch])
    qy = torch.stack([b["qy"] for b in batch])
    target = torch.stack([dh, qx, qy], dim=1)             # (b, 3, D, D)
    return x, target


def parse_ids(spec):
    """'1-60' or '1,2,5' or '1-4,7,9-11' -> sorted list of ints."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def channel_losses(pred, target, names):
    """Per-channel mean-squared error. pred/target are (b, C, D, D); target's
    first `len(names)` channels are the ones this variant predicts."""
    return {n: ((pred[:, i] - target[:, i]) ** 2).mean() for i, n in enumerate(names)}


def total_loss(per_ch, discharge_weight):
    """dh at weight 1; qx, qy scaled by discharge_weight (directional only)."""
    loss = per_ch["dh"]
    for n in ("qx", "qy"):
        if n in per_ch:
            loss = loss + discharge_weight * per_ch[n]
    return loss


def run_epoch(model, dl, names, discharge_weight, device, opt=None):
    """One pass. opt=None -> eval (no grad, no step). Returns mean total loss
    and mean per-channel losses over the pass."""
    train = opt is not None
    model.train(train)
    tot = 0.0
    ch_sum = {n: 0.0 for n in names}
    seen = 0
    torch.set_grad_enabled(train)
    for x, target in dl:
        x = x.to(device)
        target = target.to(device)
        pred = model(x)
        per_ch = channel_losses(pred, target, names)
        loss = total_loss(per_ch, discharge_weight)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot += float(loss)
        for n in names:
            ch_sum[n] += float(per_ch[n])
        seen += 1
    torch.set_grad_enabled(True)
    n = max(seen, 1)
    return tot / n, {k: v / n for k, v in ch_sum.items()}


def save_ckpt(path, model, opt, epoch, args, ghash):
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "epoch": epoch,
        "variant": args.variant,
        "seed": args.seed,
        "git": ghash,
        "discharge_weight": args.discharge_weight,
        "train_sims": args.train_sims,
        "val_sims": args.val_sims,
    }, tmp)
    os.replace(tmp, path)


def fmt_ch(d):
    return "  ".join(f"{k}={v:.6f}" for k, v in d.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=["depth", "directional"])
    ap.add_argument("--discharge-weight", type=float, default=1.0,
                    help="scales qx+qy loss relative to dh (directional only)")
    ap.add_argument("--train-sims", default="1-60", help="e.g. 1-60 or 1,2,5")
    ap.add_argument("--val-sims", default="61-80")
    ap.add_argument("--raw-dir", default="data/extracted/raw_datasets")
    ap.add_argument("--dim", type=int, default=64, help="grid side")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--patch", type=int, default=16,
                    help="patch side; 64/16 -> 4x4 tokens (patch 32 gives only 2x2)")
    ap.add_argument("--model-dim", type=int, default=192)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--model-depth", type=int, default=6)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--tag", default="", help="suffix for checkpoint filenames")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=0,
                    help="paired seed: same value = same init and data order across variants")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ghash = git_hash()
    out_ch = VARIANT_CHANNELS[args.variant]
    names = CHANNEL_NAMES[args.variant]
    print(f"variant={args.variant} out_channels={out_ch} seed={args.seed} "
          f"batch={args.batch} device={device} git={ghash}", flush=True)
    if args.variant == "directional":
        print(f"discharge_weight={args.discharge_weight}", flush=True)

    train_ids = parse_ids(args.train_sims)
    val_ids = parse_ids(args.val_sims)
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise SystemExit(f"train/val sim overlap: {sorted(overlap)}")
    print(f"train sims={args.train_sims} ({len(train_ids)})  "
          f"val sims={args.val_sims} ({len(val_ids)})", flush=True)

    # Seed everything before building the model or the loader.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    train_ds = SweGnnDataset(train_ids, args.raw_dir, dim=args.dim)
    val_ds = SweGnnDataset(val_ids, args.raw_dir, dim=args.dim)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          generator=gen, num_workers=args.num_workers,
                          collate_fn=collate, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)
    print(f"train examples={len(train_ds)} steps/epoch={len(train_dl)}  "
          f"val examples={len(val_ds)}", flush=True)

    model = GridTransformer(in_channels=2, out_channels=out_ch,
                            depth=args.model_depth, dim=args.model_dim,
                            heads=args.heads, patch=args.patch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params={n_params:,}", flush=True)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    def ckpt_path(kind):
        return os.path.join(args.ckpt_dir,
                            f"{args.variant}{tag}_s{args.seed}_{kind}.pt")

    best = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        tr_tot, tr_ch = run_epoch(model, train_dl, names, args.discharge_weight,
                                  device, opt=opt)
        va_tot, va_ch = run_epoch(model, val_dl, names, args.discharge_weight,
                                  device, opt=None)
        dt = time.time() - t0
        print(f"epoch {epoch:3d}  {dt:5.1f}s  "
              f"train total={tr_tot:.6f} [{fmt_ch(tr_ch)}]  "
              f"val total={va_tot:.6f} [{fmt_ch(va_ch)}]", flush=True)

        save_ckpt(ckpt_path(f"epoch{epoch}"), model, opt, epoch, args, ghash)
        if va_tot < best:
            best = va_tot
            save_ckpt(ckpt_path("best"), model, opt, epoch, args, ghash)
    print(f"done. best val total={best:.6f}", flush=True)


if __name__ == "__main__":
    main()
