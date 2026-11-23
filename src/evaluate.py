"""Evaluasi terstandarisasi pada Test Set (7.000 citra) untuk keenam model.

- Rekonstruksi (Vanilla AE, Sparse AE, VAE): Test MSE, PSNR, SSIM.
- FID (VAE, GAN, DCGAN, DDPM): 2.000 sampel sintetis vs statistik Inception 7.000 citra test.
  Sebagai batas bawah dihitung juga FID 2.000 citra validation (riil) vs test.
- Mode coverage: distribusi kelas sampel menurut MLP Classifier (Bagian 2).
- Latensi inferensi per batch 64 sampel dan training time per epoch.

    python -m src.evaluate            (butuh ddpm_samples.pt dari src.sample_ddpm)
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from .data import get_tensors
from .metrics import mse_psnr_ssim, inception_acts, act_stats, frechet, latency_ms
from .mlp_experiments import MLPClassifier
from .models.autoencoders import VanillaAE, SparseAE, VAE, LATENT
from .models.gans import MLPGenerator, DCGenerator, Z_DIM
from .models.ddpm import DDPM
from .utils import set_seed, CKPT, save_json, load_json, to01

N_FID = 2000


def load_models():
    ld = lambda n: torch.load(os.path.join(CKPT, n), map_location="cpu")
    ae, sae, vae = VanillaAE(), SparseAE(), VAE()
    ae.load_state_dict(ld("ae.pt")); sae.load_state_dict(ld("sae.pt")); vae.load_state_dict(ld("vae.pt"))
    gan, dcg = MLPGenerator(), DCGenerator()
    gan.load_state_dict(ld("gan.pt")["G"]); dcg.load_state_dict(ld("dcgan.pt")["G"])
    ddpm = DDPM(); ddpm.load_state_dict(ld("ddpm.pt"))
    mlp = MLPClassifier(); mlp.load_state_dict(ld("mlp.pt"))
    ms = dict(ae=ae, sae=sae, vae=vae, gan=gan, dcgan=dcg, ddpm=ddpm, mlp=mlp)
    for m in ms.values():
        m.eval()
    return ms


def cached_acts(key, x):
    """Aktivasi Inception di-cache per kunci agar evaluasi ulang tidak menghitung ulang."""
    path = os.path.join(CKPT, f"acts_{key}.npy")
    if os.path.exists(path):
        return np.load(path)
    a = inception_acts(x)
    np.save(path, a)
    return a


def class_coverage(mlp, x01):
    with torch.no_grad():
        p = F.softmax(mlp(x01), 1)
    hist = torch.bincount(p.argmax(1), minlength=10).float()
    q = hist / hist.sum()
    ent = -(q[q > 0] * q[q > 0].log()).sum().item() / np.log(10)   # 1.0 = seragam di 10 kelas
    return {"class_hist": hist.int().tolist(), "norm_class_entropy": ent,
            "mean_max_prob": p.max(1).values.mean().item()}


def main():
    torch.set_num_threads(2)
    set_seed(42)
    d = get_tensors("01")
    Xte, yte = d["test"]
    Xva, _ = d["val"]
    Xtr, _ = d["train"]
    ms = load_models()
    R = {"recon": {}, "fid": {}, "coverage": {}, "latency_ms_batch64": {}, "train_time_per_epoch_s": {}, "sparsity": {}}

    # ---------------- rekonstruksi
    with torch.no_grad():
        rec = {"ae": ms["ae"](Xte)[0], "sae": ms["sae"](Xte)[0], "vae": ms["vae"].decode(ms["vae"].encode(Xte)[0])}
        for k, v in rec.items():
            R["recon"][k] = mse_psnr_ssim(Xte, v)
            print(k, R["recon"][k], flush=True)
        for k in ("ae", "sae"):
            h = ms[k].encode(Xte)
            R["sparsity"][k] = {"zero_frac": (h.abs() < 1e-6).float().mean().item(),
                                "active_per_sample": (h.abs() > 1e-6).float().sum(1).mean().item(),
                                "dead_units": int(((h.abs() > 1e-6).sum(0) == 0).sum().item()),
                                "mean_abs_code": h.abs().mean().item(),
                                "l1_per_sample": h.abs().sum(1).mean().item()}

    # ---------------- sampel generatif (N_FID masing-masing)
    g = torch.Generator().manual_seed(42)
    with torch.no_grad():
        S = {"vae": ms["vae"].decode(torch.randn(N_FID, LATENT, generator=g)),
             "gan": to01(ms["gan"](torch.randn(N_FID, Z_DIM, generator=g))),
             "dcgan": to01(ms["dcgan"](torch.randn(N_FID, Z_DIM, generator=g))),
             }
        ddpm_path = os.path.join(CKPT, "ddpm_samples.pt")
        if os.path.exists(ddpm_path):
            S["ddpm"] = to01(torch.load(ddpm_path)[:N_FID])
        # suplemen: sampling Gaussian-fit pada ruang laten AE deterministik (latent voids) & MLP
        for k in ("ae", "sae"):
            z = ms[k].encode(Xte)
            zr = z.mean(0) + z.std(0) * torch.randn(N_FID, LATENT, generator=g)
            S[f"{k}_gaussfit"] = ms[k].decode(zr)
        S["mlp_latent"] = torch.load(os.path.join(CKPT, "mlp_fig.pt"))["x_samples"][:N_FID]
    torch.save({k: v[:64] for k, v in S.items()}, os.path.join(CKPT, "eval_samples.pt"))

    print("Inception: test reference ...", flush=True)
    mu_ref, s_ref = act_stats(cached_acts("real_test", Xte))
    mu_v, s_v = act_stats(cached_acts("real_val", Xva[:N_FID]))
    R["fid"]["real_val_vs_test"] = frechet(mu_v, s_v, mu_ref, s_ref)
    print("FID real val", R["fid"]["real_val_vs_test"], flush=True)
    for k, v in S.items():
        mu, s = act_stats(cached_acts(k, v))
        R["fid"][k] = frechet(mu, s, mu_ref, s_ref)
        R["coverage"][k] = class_coverage(ms["mlp"], v)
        print("FID", k, R["fid"][k], R["coverage"][k], flush=True)
    R["coverage"]["real_test"] = class_coverage(ms["mlp"], Xte[:N_FID])

    if "ddpm" not in S or os.environ.get("SKIP_LATENCY"):
        save_json(R, "metrics_partial.json")
        print("ddpm_samples.pt belum ada / SKIP_LATENCY: metrik parsial disimpan")
        return

    # ---------------- latensi per batch 64 (CPU, 2 thread)
    x64 = Xte[:64]
    z64 = torch.randn(64, LATENT)
    n64 = torch.randn(64, Z_DIM)
    R["latency_ms_batch64"]["ae"] = latency_ms(lambda: ms["ae"](x64))
    R["latency_ms_batch64"]["sae"] = latency_ms(lambda: ms["sae"](x64))
    R["latency_ms_batch64"]["vae"] = latency_ms(lambda: ms["vae"].decode(z64))
    R["latency_ms_batch64"]["vae_recon"] = latency_ms(lambda: ms["vae"].decode(ms["vae"].encode(x64)[0]))
    R["latency_ms_batch64"]["gan"] = latency_ms(lambda: ms["gan"](n64))
    R["latency_ms_batch64"]["dcgan"] = latency_ms(lambda: ms["dcgan"](n64))
    R["latency_ms_batch64"]["ddpm"] = latency_ms(lambda: ms["ddpm"].sample(64), n_warmup=0, n_runs=2)
    xt, tt = torch.randn(64, 1, 28, 28), torch.zeros(64, dtype=torch.long)
    R["latency_ms_batch64"]["ddpm_single_step"] = latency_ms(lambda: ms["ddpm"].net(xt, tt))
    print(R["latency_ms_batch64"], flush=True)

    # ---------------- training time & parameter
    for k in ("ae", "sae", "vae", "gan", "dcgan", "ddpm"):
        h = load_json(f"history_{k}.json")
        R["train_time_per_epoch_s"][k] = float(np.mean(h["time_s"]))
    R["params"] = {k: sum(p.numel() for p in m.parameters()) for k, m in ms.items()}
    R["n_fid"] = N_FID
    save_json(R, "metrics.json")
    print("saved metrics.json")


if __name__ == "__main__":
    main()
