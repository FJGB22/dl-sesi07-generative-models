"""Bagian 2: pembuktian batasan bobot dan fitur laten MLP Classifier untuk image generation.

Tahap 1  Latih MLP 784 -> 256 -> 64 -> 10 (Cross-Entropy) hingga akurasi test >= 88%.
Tahap 2  Proyeksi balik vektor one-hot kelas lewat transpos bobot:
         z = ReLU(W3^T y_c), h1 = ReLU(W2^T z), x = Sigmoid(W1^T h1).
Tahap 3  Proyeksi balik z_MLP citra uji: x = Sigmoid(W1^T ReLU(W2^T z_MLP)).
Tahap 4  Bekukan MLP, latih Decoder 64 -> 256 -> 784 (MSE) dari z_MLP; bandingkan dengan Vanilla AE.
Tahap 5  Sampling z ~ N(mu_z, diag(sigma_z^2)) dari statistik z_MLP test set -> Decoder Tahap 4.

Jalankan setelah Vanilla AE dilatih:  python -m src.mlp_experiments
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import get_tensors, get_loaders
from .metrics import mse_psnr_ssim
from .models.autoencoders import Decoder, VanillaAE, VAE
from .utils import set_seed, get_device, save_json, CKPT, SEED


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)   # W1: (256, 784)
        self.fc2 = nn.Linear(256, 64)    # W2: (64, 256)
        self.fc3 = nn.Linear(64, 10)     # W3: (10, 64)

    def features(self, x):               # z_MLP in R^64 (lapisan sebelum klasifikasi)
        return F.relu(self.fc2(F.relu(self.fc1(x.flatten(1)))))

    def forward(self, x):
        return self.fc3(self.features(x))


def accuracy(model, loader, device):
    model.eval()
    c = n = 0
    with torch.no_grad():
        for x, y in loader:
            c += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
            n += len(y)
    return c / n


def stage1(device, epochs=25):
    set_seed(SEED)
    tr, va, te = get_loaders("01", batch_size=128)
    m = MLPClassifier().to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    best, best_state, hist = 0, None, {"epoch": [], "train_loss": [], "val_acc": []}
    for ep in range(1, epochs + 1):
        m.train()
        tot = n = 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(m(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        va_acc = accuracy(m, va, device)
        hist["epoch"].append(ep); hist["train_loss"].append(tot / n); hist["val_acc"].append(va_acc)
        print(f"[mlp] ep {ep:2d} CE {tot/n:.4f} val_acc {va_acc:.4f}", flush=True)
        if va_acc > best:                 # model terbaik dipilih dari validation set, bukan test
            best, best_state = va_acc, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    test_acc = accuracy(m, te, device)
    print(f"[mlp] test accuracy = {test_acc:.4f}")
    torch.save(m.state_dict(), os.path.join(CKPT, "mlp.pt"))
    return m, test_acc, hist


def weight_diagnostics(m):
    """Bukti matematis non-invertibilitas: W^T W vs I, singular value, rank."""
    out = {}
    for name, W in [("W1", m.fc1.weight), ("W2", m.fc2.weight), ("W3", m.fc3.weight)]:
        W = W.detach().cpu().double()                 # (out, in)
        WtW = W.T @ W                                 # (in, in)
        I = torch.eye(W.shape[1], dtype=torch.double)
        s = torch.linalg.svdvals(W)
        out[name] = {"shape": list(W.shape),
                     "rel_fro_WtW_minus_I": ((WtW - I).norm() / I.norm()).item(),
                     "rank": int(torch.linalg.matrix_rank(W).item()),
                     "max_rank_possible": int(min(W.shape)),
                     "null_space_dim_of_input": int(W.shape[1] - torch.linalg.matrix_rank(W).item()),
                     "sv_max": s.max().item(), "sv_min": s.min().item(),
                     "cond": (s.max() / s.min()).item()}
    return out


def main():
    device = get_device()
    d = get_tensors("01")
    Xte, yte = d["test"]
    Xtr, _ = d["train"]
    res = {}

    # ---------------- Tahap 1
    t0 = time.perf_counter()
    m, test_acc, hist = stage1(device)
    res["stage1"] = {"test_acc": test_acc, "history": hist, "train_time_s": time.perf_counter() - t0}
    m.eval()
    for p in m.parameters():
        p.requires_grad = False                      # dibekukan untuk Tahap 2-5
    W1, W2, W3 = (m.fc1.weight.cpu(), m.fc2.weight.cpu(), m.fc3.weight.cpu())
    res["weights"] = weight_diagnostics(m)
    with torch.no_grad():
        pre1 = m.fc1(Xte.flatten(1).to(device))
        pre2 = m.fc2(F.relu(pre1))
        res["relu_zeroed_frac"] = {"layer1": (pre1 < 0).float().mean().item(),
                                   "layer2": (pre2 < 0).float().mean().item()}
        z_te = m.features(Xte.to(device)).cpu()
        z_tr = m.features(Xtr.to(device)).cpu()

    # ---------------- Tahap 2: transpos bobot dari one-hot kelas
    with torch.no_grad():
        Y = torch.eye(10)
        z_hat = F.relu(Y @ W3)                   # (10,64) = ReLU(W3^T y_c) untuk tiap kelas
        h1 = F.relu(z_hat @ W2)                  # (10,256)
        x_cls = torch.sigmoid(h1 @ W1).view(-1, 1, 28, 28)
        class_means = torch.stack([Xte[yte == c].mean(0) for c in range(10)])
    res["stage2"] = {"pixel_min": x_cls.min().item(), "pixel_max": x_cls.max().item(),
                     "pixel_std_mean": x_cls.flatten(1).std(1).mean().item(),
                     "vs_class_mean_image": mse_psnr_ssim(class_means, x_cls)}

    # ---------------- Tahap 3: transpos bobot dari z_MLP citra uji
    with torch.no_grad():
        x_tp = torch.sigmoid(F.relu(z_te @ W2) @ W1).view(-1, 1, 28, 28)
        # Pembanding: pseudo-inverse (Moore-Penrose) juga tidak dapat memulihkan citra,
        # karena ReLU sudah membuang informasi dan dimensi 784 -> 64 tidak injektif.
        x_pinv = (torch.linalg.pinv(W1) @ (torch.linalg.pinv(W2) @ z_te.T)).T.clamp(0, 1).view(-1, 1, 28, 28)
    res["stage3"] = {"transpose": mse_psnr_ssim(Xte, x_tp), "pinv": mse_psnr_ssim(Xte, x_pinv),
                     "pixel_min": x_tp.min().item(), "pixel_max": x_tp.max().item()}

    # ---------------- Tahap 4: frozen latent decoder vs Vanilla AE
    set_seed(SEED)
    dec = Decoder().to(device)
    opt = torch.optim.Adam(dec.parameters(), 1e-3)
    loader = DataLoader(TensorDataset(z_tr, Xtr), 128, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    dh = []
    for ep in range(1, 31):                       # 30 epoch, sama dengan Vanilla AE
        tot = n = 0
        for z, x in loader:
            z, x = z.to(device), x.to(device)
            loss = F.mse_loss(dec(z), x)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(x); n += len(x)
        dh.append(tot / n)
        print(f"[frozen-dec] ep {ep:2d} MSE {tot/n:.5f}", flush=True)
    torch.save(dec.state_dict(), os.path.join(CKPT, "mlp_frozen_decoder.pt"))
    dec.eval()
    ae = VanillaAE()
    ae.load_state_dict(torch.load(os.path.join(CKPT, "ae.pt"), map_location="cpu"))
    ae.eval()
    with torch.no_grad():
        x_fd = dec(z_te.to(device)).cpu()
        x_ae = ae(Xte)[0]

    def intra_class_var_ratio(xr):
        """Rasio varians intra-kelas rekonstruksi terhadap citra asli (1.0 = variasi utuh)."""
        r = []
        for c in range(10):
            idx = yte == c
            r.append((xr[idx].var(0).sum() / Xte[idx].var(0).sum()).item())
        return float(np.mean(r))

    # akurasi MLP pada rekonstruksi: seberapa 'kelas' masih terbaca
    def cls_acc(xr):
        with torch.no_grad():
            return (m(xr.to(device)).argmax(1).cpu() == yte).float().mean().item()

    res["stage4"] = {"frozen_mlp_decoder": {**mse_psnr_ssim(Xte, x_fd), "intra_class_var_ratio": intra_class_var_ratio(x_fd),
                                            "mlp_acc_on_recon": cls_acc(x_fd)},
                     "vanilla_ae": {**mse_psnr_ssim(Xte, x_ae), "intra_class_var_ratio": intra_class_var_ratio(x_ae),
                                    "mlp_acc_on_recon": cls_acc(x_ae)},
                     "decoder_train_mse": dh}

    # ---------------- Tahap 5: sampling dari N(mu_z, diag(sigma_z^2))
    g = torch.Generator().manual_seed(SEED)
    mu, sd = z_te.mean(0), z_te.std(0)
    z_rand = mu + sd * torch.randn(2000, 64, generator=g)
    with torch.no_grad():
        x_s = dec(z_rand.to(device)).cpu()
        # pembanding: VAE dari prior N(0, I)
        vae = VAE(); vae.load_state_dict(torch.load(os.path.join(CKPT, "vae.pt"), map_location="cpu")); vae.eval()
        x_vae = vae.decode(torch.randn(2000, 64, generator=g))

        def conf(xs):
            p = F.softmax(m(xs.to(device)), 1).cpu()
            ent = -(p * p.clamp_min(1e-12).log()).sum(1)
            return {"mean_max_prob": p.max(1).values.mean().item(), "mean_entropy": ent.mean().item(),
                    "class_hist": torch.bincount(p.argmax(1), minlength=10).tolist()}

        # nearest-neighbour jarak z_rand ke z_MLP train (seberapa jauh dari manifold data)
        def nn_dist(q, ref):
            ds = []
            for i in range(0, len(q), 500):
                ds.append(torch.cdist(q[i:i + 500], ref).min(1).values)
            return torch.cat(ds)
        d_rand = nn_dist(z_rand, z_tr).mean().item()
        d_real = nn_dist(z_te[:2000], z_tr).mean().item()
    res["stage5"] = {"latent_zero_frac": (z_te == 0).float().mean().item(),
                     "negative_frac_in_samples": (z_rand < 0).float().mean().item(),
                     "nn_dist_sampled_to_train": d_rand, "nn_dist_test_to_train": d_real,
                     "mlp_conf_samples": conf(x_s), "mlp_conf_vae": conf(x_vae), "mlp_conf_real": conf(Xte[:2000])}
    torch.save({"x_cls": x_cls, "class_means": class_means, "x_tp": x_tp[:16], "x_pinv": x_pinv[:16],
                "x_fd": x_fd[:16], "x_ae": x_ae[:16], "x_orig": Xte[:16], "y_orig": yte[:16],
                "x_samples": x_s, "z_rand": z_rand}, os.path.join(CKPT, "mlp_fig.pt"))
    save_json(res, "mlp_experiments.json")
    print({k: v for k, v in res.items() if k not in ("stage1",)})


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        torch.set_num_threads(int(sys.argv[1]))
    main()
