"""Sampling DDPM (reverse process 1000 langkah) untuk FID dan figur trajectory.

    python -m src.sample_ddpm --n 2000
Menyimpan outputs/checkpoints/ddpm_samples.pt (sampel, skala [-1,1]) dan
ddpm_traj.pt (x_t pada beberapa langkah untuk 8 sampel).
"""
import argparse
import os
import time

import torch

from .models.ddpm import DDPM
from .utils import set_seed, get_device, CKPT, save_json

TRAJ = (800, 600, 400, 300, 200, 100, 50, 0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=500)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    set_seed(1234)                                   # seed sampling terpisah dari seed training
    dev = get_device()
    m = DDPM().to(dev)
    m.load_state_dict(torch.load(os.path.join(CKPT, "ddpm.pt"), map_location=dev))
    m.eval()
    _, traj = m.sample(8, dev, return_traj=True, traj_steps=TRAJ)
    torch.save({k: v.cpu() for k, v in traj.items()}, os.path.join(CKPT, "ddpm_traj.pt"))
    out, t0 = [], time.perf_counter()
    for i in range(0, a.n, a.bs):
        out.append(m.sample(min(a.bs, a.n - i), dev).cpu())
        print(f"[ddpm-sample] {i + a.bs}/{a.n}  {time.perf_counter() - t0:.0f}s", flush=True)
    torch.save(torch.cat(out), os.path.join(CKPT, "ddpm_samples.pt"))
    save_json({"n": a.n, "seconds": time.perf_counter() - t0}, "ddpm_sampling_time.json")
