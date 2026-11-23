"""Metrik evaluasi: MSE, PSNR, SSIM, FID, dan latensi inferensi.

Semua citra dievaluasi pada skala piksel [0, 1].
FID memakai InceptionV3 (pytorch-fid, pool3 2048-d). Citra grayscale 28x28
diduplikasi menjadi 3 kanal dan di-resize bilinear ke 299x299 di dalam model.
"""
import time

import numpy as np
import torch
from scipy import linalg
from skimage.metrics import structural_similarity

_INCEPTION = None


def mse_psnr_ssim(x, x_hat):
    """x, x_hat: tensor (N,1,28,28) di [0,1]. Rata-rata per citra."""
    x = x.detach().cpu().clamp(0, 1)
    x_hat = x_hat.detach().cpu().clamp(0, 1)
    mse_i = ((x - x_hat) ** 2).flatten(1).mean(1)
    psnr_i = 10 * torch.log10(1.0 / mse_i.clamp_min(1e-10))
    a, b = x.squeeze(1).numpy(), x_hat.squeeze(1).numpy()
    ssim_i = np.array([structural_similarity(a[i], b[i], data_range=1.0) for i in range(len(a))])
    return {"mse": mse_i.mean().item(), "psnr": psnr_i.mean().item(), "ssim": float(ssim_i.mean())}


def _inception(device):
    global _INCEPTION
    if _INCEPTION is None:
        from pytorch_fid.inception import InceptionV3
        _INCEPTION = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    return _INCEPTION


@torch.no_grad()
def inception_acts(x, device="cpu", bs=100):
    """x: (N,1,28,28) di [0,1] -> aktivasi (N, 2048)."""
    net = _inception(device)
    out = []
    for i in range(0, len(x), bs):
        b = x[i:i + bs].to(device).clamp(0, 1).repeat(1, 3, 1, 1)
        out.append(net(b)[0].squeeze(-1).squeeze(-1).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def act_stats(acts):
    return acts.mean(0), np.cov(acts, rowvar=False)


def frechet(mu1, s1, mu2, s2):
    """FID = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^{1/2})."""
    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1.dot(s2))          # SciPy >= 1.14 tidak lagi memakai argumen disp
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
        covmean = covmean[0] if isinstance(covmean, tuple) else covmean
    covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


@torch.no_grad()
def latency_ms(fn, n_warmup=3, n_runs=20):
    """Rata-rata waktu (ms) satu pemanggilan fn() setelah warm-up."""
    for _ in range(n_warmup):
        fn()
    ts = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts)), float(np.std(ts))
