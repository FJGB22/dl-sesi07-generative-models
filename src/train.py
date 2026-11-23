"""Training pipeline untuk keenam model generatif.

Contoh:
    python -m src.train --model ae    --epochs 30
    python -m src.train --model gan   --epochs 50
    python -m src.train --model ddpm  --epochs 12
Riwayat loss per epoch disimpan di outputs/results/history_<model>.json,
checkpoint di outputs/checkpoints/<model>.pt.
"""
import argparse
import copy
import os
import time

import torch
import torch.nn.functional as F

from .data import get_loaders
from .models.autoencoders import VanillaAE, SparseAE, VAE
from .models.gans import MLPGenerator, MLPDiscriminator, DCGenerator, DCDiscriminator, dcgan_init, Z_DIM
from .models.ddpm import DDPM
from .utils import set_seed, get_device, save_json, CKPT

SCALE = {"ae": "01", "sae": "01", "vae": "01", "gan": "pm1", "dcgan": "pm1", "ddpm": "pm1"}


def build(name):
    if name == "ae":
        return VanillaAE()
    if name == "sae":
        return SparseAE(lam=1e-3)
    if name == "vae":
        return VAE()
    if name == "ddpm":
        return DDPM()
    raise ValueError(name)


# ----------------------------------------------------------------- AE / SAE / VAE / DDPM
def train_likelihood(name, epochs, lr, device):
    model = build(name).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr, va, _ = get_loaders(SCALE[name], batch_size=128)
    ema = copy.deepcopy(model).eval() if name == "ddpm" else None
    hist = {"epoch": [], "train": [], "val": [], "time_s": [], "parts": []}
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, n, parts = time.perf_counter(), 0.0, 0, {}
        for x, _ in tr:
            x = x.to(device)
            loss, info = model.loss(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if ema is not None:                       # EMA bobot (decay 0.999) untuk sampling DDPM
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(0.999).add_(pm.detach(), alpha=0.001)
            b = x.size(0)
            tot += loss.item() * b
            n += b
            for k, v in info.items():
                parts[k] = parts.get(k, 0.0) + v * b
        dt = time.perf_counter() - t0
        model.eval()
        with torch.no_grad():
            vt, vn = 0.0, 0
            for x, _ in va:
                x = x.to(device)
                l, _ = (ema or model).loss(x)
                vt += l.item() * x.size(0)
                vn += x.size(0)
        hist["epoch"].append(ep)
        hist["train"].append(tot / n)
        hist["val"].append(vt / vn)
        hist["time_s"].append(dt)
        hist["parts"].append({k: v / n for k, v in parts.items()})
        print(f"[{name}] ep {ep:3d} train {tot/n:.5f} val {vt/vn:.5f} {dt:.1f}s {hist['parts'][-1]}", flush=True)
        save_json(hist, f"history_{name}.json")
        torch.save((ema or model).state_dict(), os.path.join(CKPT, f"{name}.pt"))
    return hist


# ----------------------------------------------------------------- GAN / DCGAN
def grad_norm(module):
    g = [p.grad.detach().flatten() for p in module.parameters() if p.grad is not None]
    return torch.cat(g).norm().item()


def train_gan(name, epochs, lr, device):
    if name == "gan":
        G, D = MLPGenerator().to(device), MLPDiscriminator().to(device)
    else:
        G, D = DCGenerator().to(device), DCDiscriminator().to(device)
        G.apply(dcgan_init)
        D.apply(dcgan_init)
    oG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = F.binary_cross_entropy_with_logits
    tr, _, _ = get_loaders("pm1", batch_size=128)
    fixed = torch.randn(64, Z_DIM, device=device)
    hist = {"epoch": [], "loss_D": [], "loss_G": [], "D_real": [], "D_fake": [],
            "gradG": [], "time_s": [], "iter_loss_D": [], "iter_loss_G": []}
    snaps = {}
    for ep in range(1, epochs + 1):
        G.train(); D.train()
        t0 = time.perf_counter()
        sD = sG = sR = sF = sgn = 0.0
        k = 0
        for x, _ in tr:
            x = x.to(device)
            b = x.size(0)
            z = torch.randn(b, Z_DIM, device=device)
            fake = G(z)
            # --- update D: max log D(x) + log(1 - D(G(z)))
            dr, df = D(x), D(fake.detach())
            lD = bce(dr, torch.ones_like(dr)) + bce(df, torch.zeros_like(df))
            oD.zero_grad(set_to_none=True)
            lD.backward()
            oD.step()
            # --- update G (non-saturating): max log D(G(z))
            dg = D(fake)
            lG = bce(dg, torch.ones_like(dg))
            oG.zero_grad(set_to_none=True)
            lG.backward()
            sgn += grad_norm(G)
            oG.step()
            sD += lD.item(); sG += lG.item()
            sR += torch.sigmoid(dr).mean().item(); sF += torch.sigmoid(df).mean().item()
            k += 1
            if k % 20 == 0:
                hist["iter_loss_D"].append(lD.item())
                hist["iter_loss_G"].append(lG.item())
        dt = time.perf_counter() - t0
        for key, v in zip(["loss_D", "loss_G", "D_real", "D_fake", "gradG"], [sD, sG, sR, sF, sgn]):
            hist[key].append(v / k)
        hist["epoch"].append(ep)
        hist["time_s"].append(dt)
        print(f"[{name}] ep {ep:3d} LD {sD/k:.3f} LG {sG/k:.3f} D(x) {sR/k:.3f} D(G(z)) {sF/k:.3f} "
              f"|gradG| {sgn/k:.3f} {dt:.1f}s", flush=True)
        G.eval()
        if ep in (1, 5, 10, 25, epochs):
            with torch.no_grad():
                snaps[ep] = G(fixed[:16]).cpu()
        save_json(hist, f"history_{name}.json")
        torch.save({"G": G.state_dict(), "D": D.state_dict()}, os.path.join(CKPT, f"{name}.pt"))
    torch.save(snaps, os.path.join(CKPT, f"{name}_snapshots.pt"))
    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["ae", "sae", "vae", "gan", "dcgan", "ddpm"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (CPU); 0 = default")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    set_seed(42)
    dev = get_device()
    if a.model in ("gan", "dcgan"):
        train_gan(a.model, a.epochs, a.lr or 2e-4, dev)
    else:
        train_likelihood(a.model, a.epochs, a.lr or (2e-4 if a.model == "ddpm" else 1e-3), dev)
