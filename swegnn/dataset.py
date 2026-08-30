"""PyTorch Dataset over SWE-GNN dike-breach simulations.

One example is a single timestep t from one simulation:

    input  : (2, D, D)  = [depth_t, dem]
    targets: dh (D, D)   = depth_{t+1} - depth_t
             qx (D, D)   = vx_t * depth_t   (directional discharge, x)
             qy (D, D)   = vy_t * depth_t   (directional discharge, y)

We are testing whether predicting directional discharge (qx, qy) alongside
depth change beats predicting depth alone: flow direction is dynamic at the
advancing front and is what tells you where the flood spreads next.

No normalisation is applied -- SWE-GNN uses raw metres / (m/s) throughout.
Each simulation is loaded once at construction and indexed into per item.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from load_swegnn import load_sim


class SweGnnDataset(Dataset):
    def __init__(self, sim_ids, raw_dir, dim=64):
        """sim_ids: list of integer simulation ids to include.
        raw_dir:  folder containing DEM/ WD/ VX/ VY/.
        dim:      grid side (64 for these six sims).
        """
        self.sim_ids = list(sim_ids)
        self.raw_dir = raw_dir
        self.dim = dim

        # Load each simulation exactly once; keep tensors resident.
        self.sims = {}          # sim_id -> dict of (T,D,D) / (D,D) float32 tensors
        self.index = []         # flat list of (sim_id, t) with t+1 in range
        for sid in self.sim_ids:
            depth, vx, vy, dem = load_sim(raw_dir, sid, dim)
            self.sims[sid] = {
                "depth": torch.from_numpy(depth),          # (T, D, D)
                "vx": torch.from_numpy(vx),                # (T, D, D)
                "vy": torch.from_numpy(vy),                # (T, D, D)
                "dem": torch.from_numpy(dem),              # (D, D)
            }
            T = depth.shape[0]
            self.index.extend((sid, t) for t in range(T - 1))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        sid, t = self.index[i]
        sim = self.sims[sid]
        depth_t = sim["depth"][t]                          # (D, D)
        depth_next = sim["depth"][t + 1]                   # (D, D)
        dem = sim["dem"]                                   # (D, D)

        x = torch.stack([depth_t, dem], dim=0)             # (2, D, D)
        dh = depth_next - depth_t                          # (D, D)
        qx = sim["vx"][t] * depth_t                        # (D, D)
        qy = sim["vy"][t] * depth_t                        # (D, D)

        return {
            "input": x,
            "dh": dh,
            "qx": qx,
            "qy": qy,
            "sim_id": sid,
            "t": t,
        }


def _self_test():
    raw_dir = "data/extracted/raw_datasets"
    ds = SweGnnDataset(sim_ids=[1, 2, 3, 4, 5, 6], raw_dir=raw_dir, dim=64)
    print(f"{len(ds)} examples from {len(ds.sim_ids)} simulations")
    for sid in ds.sim_ids:
        T = ds.sims[sid]["depth"].shape[0]
        print(f"  sim {sid}: T={T} -> {T - 1} timestep pairs")

    ex = ds[0]
    print("\nper-example shapes:")
    for k in ("input", "dh", "qx", "qy"):
        print(f"  {k:6s} {tuple(ex[k].shape)}")

    # Aggregate stats + NaN check over the whole dataset.
    acc = {k: [] for k in ("depth_t", "dem", "dh", "qx", "qy")}
    for i in range(len(ds)):
        ex = ds[i]
        acc["depth_t"].append(ex["input"][0])
        acc["dem"].append(ex["input"][1])
        acc["dh"].append(ex["dh"])
        acc["qx"].append(ex["qx"])
        acc["qy"].append(ex["qy"])

    print("\nfield statistics (all examples):")
    print(f"  {'field':8s} {'min':>12s} {'max':>12s} {'mean':>12s}")
    for k, chunks in acc.items():
        v = torch.stack(chunks)
        assert not torch.isnan(v).any(), f"NaN found in {k}"
        assert not torch.isinf(v).any(), f"Inf found in {k}"
        print(f"  {k:8s} {v.min().item():12.5f} {v.max().item():12.5f} {v.mean().item():12.5f}")

    print("\nno NaNs / Infs in any field. self-test passed.")


if __name__ == "__main__":
    _self_test()
