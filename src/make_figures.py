"""Membuat Figur 0 (arsitektur) dan Figur 1-5 laporan, serta metrik interpolasi laten.

    python -m src.make_figures
Hasil: outputs/figures/fig*.png (300 dpi) dan outputs/results/interpolation.json
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch
import torch.nn.functional as F

from .data import get_tensors, CLASSES
from .evaluate import load_models
from .models.gans import Z_DIM
from .utils import CKPT, FIG, load_json, save_json, to01, set_seed

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.titlesize": 9,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
COL = {"ae": "#1f77b4", "sae": "#2ca02c", "vae": "#9467bd", "gan": "#d62728",
       "dcgan": "#ff7f0e", "ddpm": "#17becf", "mlp": "#7f7f7f"}
NAME = {"ae": "Vanilla AE", "sae": "Sparse AE", "vae": "VAE", "gan": "Vanilla GAN",
        "dcgan": "DCGAN", "ddpm": "DDPM"}


def grid(x, nrow):
    """(N,1,28,28) -> array 2D kisi dengan garis pemisah 1 px."""
    x = x.detach().cpu().clamp(0, 1).squeeze(1).numpy()
    n = len(x)
    ncol = int(np.ceil(n / nrow))
    out = np.ones((nrow * 29 + 1, ncol * 29 + 1))
    for i in range(n):
        r, c = divmod(i, ncol)
        out[1 + r * 29:1 + r * 29 + 28, 1 + c * 29:1 + c * 29 + 28] = x[i]
    return out


def minmax(x):
    f = x.flatten(1)
    lo, hi = f.min(1, keepdim=True).values, f.max(1, keepdim=True).values
    return ((f - lo) / (hi - lo + 1e-8)).view_as(x)


def row_axes(ax, img, label):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8.5)


# ------------------------------------------------------------------ Figur 0
def fig0_arch():
    rows = [
        ("Vanilla AE", ["x 784", "FC 256\nReLU", "z 64\nlinear", "FC 256\nReLU", "x̂ 784\nSigmoid"], "MSE"),
        ("Sparse AE", ["x 784", "FC 256\nReLU", "h 64\nReLU", "FC 256\nReLU", "x̂ 784\nSigmoid"], "MSE + λ‖h‖₁"),
        ("VAE", ["x 784", "FC 256\nReLU", "μ, log σ²\n64+64", "z = μ+σ⊙ε", "FC 256→784\nSigmoid"], "BCE + KL"),
        ("Vanilla GAN", ["z 100", "FC 256-512\n-1024 LReLU", "G(z) 784\nTanh", "D: FC 512\n-256 LReLU", "logit\nD(x)"], "BCE (non-sat.)"),
        ("DCGAN", ["z 100×1×1", "ConvT 7×7\n256 BN ReLU", "ConvT 14×14\n128 BN ReLU", "ConvT 28×28\n1 Tanh", "D: Conv 64\n-128 BN → 1"], "BCE (non-sat.)"),
        ("DDPM", ["x_t, t", "U-Net 28→14→7\nch 32/64/128", "ResBlock +\nt-embed", "skip ↑14→28", "ε̂_θ(x_t,t)"], "‖ε − ε̂‖²"),
        ("MLP\nClassifier", ["x 784", "FC 256\nReLU (W1)", "z_MLP 64\nReLU (W2)", "FC 10 (W3)", "softmax\nP(y|x)"], "Cross-Entropy"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.set_xlim(0, 10.6); ax.set_ylim(0, len(rows)); ax.axis("off")
    cols = ["#dbe9f6", "#e3f1de", "#ece3f3", "#f8dddd", "#fde7d2", "#d9f2f4", "#e8e8e8"]
    for i, (nm, blocks, loss) in enumerate(rows):
        y = len(rows) - i - 0.5
        ax.text(0.05, y, nm, ha="left", va="center", fontsize=8.5, weight="bold")
        for j, b in enumerate(blocks):
            x0 = 1.75 + j * 1.55
            ax.add_patch(FancyBboxPatch((x0, y - 0.32), 1.3, 0.64, boxstyle="round,pad=0.02",
                                        fc=cols[i], ec="#555", lw=0.6))
            ax.text(x0 + 0.65, y, b, ha="center", va="center", fontsize=6.6)
            if j < len(blocks) - 1:
                ax.annotate("", (x0 + 1.55, y), (x0 + 1.3, y), arrowprops=dict(arrowstyle="->", lw=0.7))
        ax.text(9.55, y, loss, ha="left", va="center", fontsize=6.8, style="italic")
    fig.savefig(os.path.join(FIG, "fig0_arsitektur.png"))
    plt.close(fig)


# ------------------------------------------------------------------ Figur 1
def fig1_recon(ms, Xte, yte, M):
    idx = torch.tensor([int((yte == c).nonzero()[0]) for c in range(10)])
    x = Xte[idx]
    with torch.no_grad():
        rows = [("Ground truth", x), ("Vanilla AE", ms["ae"](x)[0]), ("Sparse AE", ms["sae"](x)[0]),
                ("VAE (μ)", ms["vae"].decode(ms["vae"].encode(x)[0]))]
    fig, axs = plt.subplots(4, 1, figsize=(7.0, 3.4))
    for ax, (lab, im) in zip(axs, rows):
        k = {"Vanilla AE": "ae", "Sparse AE": "sae", "VAE (μ)": "vae"}.get(lab)
        if k:
            r = M["recon"][k]
            lab = f"{lab}\nMSE {r['mse']:.4f} | SSIM {r['ssim']:.3f}"
        row_axes(ax, grid(im, 1), lab)
    axs[0].set_title("  ".join(c.split("/")[0] for c in CLASSES), fontsize=6.8)
    fig.savefig(os.path.join(FIG, "fig1_rekonstruksi.png"))
    plt.close(fig)


# ------------------------------------------------------------------ Figur 2
def fig2_samples(M):
    S = torch.load(os.path.join(CKPT, "eval_samples.pt"))
    panels = [("A. VAE  z~N(0,I)", "vae"), ("B. Vanilla GAN  z~N(0,I)", "gan"),
              ("C. DCGAN  z~N(0,I)", "dcgan"), ("D. DDPM  x_T~N(0,I)", "ddpm")]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.2))
    for ax, (t, k) in zip(axs, panels):
        ax.imshow(grid(S[k][:16], 4), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{t}\nFID {M['fid'][k]:.1f}", fontsize=7.5)
        ax.axis("off")
    fig.savefig(os.path.join(FIG, "fig2_sampling.png"))
    plt.close(fig)


# ------------------------------------------------------------------ Figur 3
def fig3_mlp(E):
    D = torch.load(os.path.join(CKPT, "mlp_fig.pt"))
    n = 10
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(7, 1, hspace=0.35)
    s4 = E["stage4"]
    items = [
        ("(1) Ground truth", D["x_orig"][:n]),
        ("(2a) W^T dari one-hot y_c\n(10 kelas, min-max)", minmax(D["x_cls"])),
        ("(2b) W^T dari z_MLP\n(skala asli Sigmoid)", D["x_tp"][:n]),
        ("(2b') pseudo-invers W⁺\ndari z_MLP", D["x_pinv"][:n]),
        (f"(3) Frozen decoder z_MLP\nSSIM {s4['frozen_mlp_decoder']['ssim']:.3f}", D["x_fd"][:n]),
        (f"(3) Vanilla AE z_AE\nSSIM {s4['vanilla_ae']['ssim']:.3f}", D["x_ae"][:n]),
        ("(4) Sampling z~N(μ_z, σ_z²)\nruang laten MLP", D["x_samples"][:10]),
    ]
    for i, (lab, im) in enumerate(items):
        ax = fig.add_subplot(gs[i])
        row_axes(ax, grid(im, 1), lab)
    fig.savefig(os.path.join(FIG, "fig3_mlp_batasan.png"))
    plt.close(fig)


# ------------------------------------------------------------------ Figur 4
def fig4_interp(ms, Xte, yte, Xtr):
    A, B = 1, 8                                       # Trouser vs Bag (siluet sangat kontras)
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    mlp = ms["mlp"]
    with torch.no_grad():
        p_te = F.softmax(mlp(Xte), 1)
    ia = int(torch.where(yte == A, p_te[:, A], torch.zeros(len(yte))).argmax())
    ib = int(torch.where(yte == B, p_te[:, B], torch.zeros(len(yte))).argmax())
    xa, xb = Xte[ia:ia + 1], Xte[ib:ib + 1]
    rows, info = [], {}
    with torch.no_grad():
        for k in ("ae", "sae", "vae"):
            enc = (lambda x: ms["vae"].encode(x)[0]) if k == "vae" else ms[k].encode
            za, zb = enc(xa), enc(xb)
            Z = torch.cat([(1 - a) * za + a * zb for a in alphas])
            imgs = ms[k].decode(Z)
            # latent void: jarak z(alpha) ke kode training terdekat, dinormalisasi dg jarak kode test riil
            ztr = enc(Xtr[:20000])
            nn_path = torch.cdist(Z, ztr).min(1).values
            nn_real = torch.cdist(enc(Xte[:1000]), ztr).min(1).values.mean()
            info[k] = {"void_ratio_per_alpha": (nn_path / nn_real).tolist()}
            rows.append((NAME[k], imgs))
        # GAN tidak punya encoder: z_A, z_B dipilih dari 5.000 z acak yang G(z)-nya paling yakin kelas A / B
        g = torch.Generator().manual_seed(42)
        for k in ("gan", "dcgan"):
            Zc = torch.randn(5000, Z_DIM, generator=g)
            p = F.softmax(mlp(to01(ms[k](Zc))), 1)
            za, zb = Zc[p[:, A].argmax()][None], Zc[p[:, B].argmax()][None]
            Z = torch.cat([(1 - a) * za + a * zb for a in alphas])
            rows.append((NAME[k], to01(ms[k](Z))))
            info[k] = {"pA": p[:, A].max().item(), "pB": p[:, B].max().item()}
        for (lab, imgs), k in zip(rows, ("ae", "sae", "vae", "gan", "dcgan")):
            pr = F.softmax(mlp(imgs), 1)
            ent = -(pr * pr.clamp_min(1e-12).log()).sum(1)
            step = (imgs[1:] - imgs[:-1]).pow(2).flatten(1).mean(1)
            info[k].update({"entropy_per_alpha": ent.tolist(), "pred_per_alpha": pr.argmax(1).tolist(),
                            "max_step_mse": step.max().item(), "mean_step_mse": step.mean().item()})
    fig, axs = plt.subplots(5, 1, figsize=(4.6, 4.6))
    for ax, (lab, imgs) in zip(axs, rows):
        row_axes(ax, grid(imgs, 1), lab)
    axs[0].set_title(f"{CLASSES[A]}  →  {CLASSES[B]}", fontsize=8)
    axs[-1].set_xticks([14.5 + 29 * i for i in range(len(alphas))])
    axs[-1].set_xticklabels([f"α={a:.1f}" for a in alphas], fontsize=7.5)
    fig.savefig(os.path.join(FIG, "fig4_interpolasi.png"))
    plt.close(fig)
    save_json(info, "interpolation.json")


# ------------------------------------------------------------------ Figur 5
def fig5_loss():
    H = {k: load_json(f"history_{k}.json") for k in ("ae", "sae", "vae", "gan", "dcgan", "ddpm")}
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.15], hspace=0.55, wspace=0.32)
    for i, k in enumerate(("ae", "sae", "vae")):
        ax = fig.add_subplot(gs[0, i])
        h = H[k]
        ax.plot(h["epoch"], h["train"], color=COL[k], label="train")
        ax.plot(h["epoch"], h["val"], "--", color=COL[k], alpha=.7, label="val")
        if k == "sae":
            ax.plot(h["epoch"], [p["rec"] for p in h["parts"]], ":", color="k", label="komponen MSE")
        ax.set_title(f"{NAME[k]}: " + {"ae": "MSE", "sae": "MSE + λ‖h‖₁", "vae": "BCE + KL (nat/sampel)"}[k])
        ax.set_xlabel("epoch"); ax.legend(fontsize=6.5); ax.grid(alpha=.3)
        if k != "vae":
            ax.set_yscale("log")
    for i, k in enumerate(("gan", "dcgan")):
        ax = fig.add_subplot(gs[1, i])
        h = H[k]
        it = np.linspace(1, len(h["epoch"]), len(h["iter_loss_D"]))
        ax.plot(it, h["iter_loss_D"], color="#444", lw=.5, alpha=.5)
        ax.plot(it, h["iter_loss_G"], color=COL[k], lw=.5, alpha=.5)
        ax.plot(h["epoch"], h["loss_D"], color="#000", lw=1.4, label="L_D (rata-rata epoch)")
        ax.plot(h["epoch"], h["loss_G"], color=COL[k], lw=1.4, label="L_G (rata-rata epoch)")
        ax.set_title(f"{NAME[k]}: loss G & D (tipis = per 20 iterasi)")
        ax.set_xlabel("epoch"); ax.legend(fontsize=6.5); ax.grid(alpha=.3)
    ax = fig.add_subplot(gs[1, 2])
    h = H["ddpm"]
    ax.plot(h["epoch"], h["train"], color=COL["ddpm"], label="train")
    ax.plot(h["epoch"], h["val"], "--", color=COL["ddpm"], label="val (EMA)")
    ax.set_title("DDPM: ‖ε − ε̂_θ‖² (noise MSE)"); ax.set_xlabel("epoch"); ax.legend(fontsize=6.5); ax.grid(alpha=.3)
    traj = torch.load(os.path.join(CKPT, "ddpm_traj.pt"))
    steps = sorted(traj.keys(), reverse=True)
    ax = fig.add_subplot(gs[2, :])
    cols = [to01(traj[s][:4]) for s in steps]                   # (4,1,28,28) per langkah
    img = torch.stack(cols, 1).reshape(-1, 1, 28, 28)            # baris = sampel, kolom = langkah
    ax.imshow(grid(img, 4), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([14.5 + 29 * i for i in range(len(steps))])
    ax.set_xticklabels([f"t={s}" for s in steps], fontsize=7)
    ax.set_yticks([])
    ax.set_title("DDPM reverse diffusion trajectory  x_T → x_0 (4 sampel)")
    fig.savefig(os.path.join(FIG, "fig5_loss_ddpm.png"))
    plt.close(fig)


def main():
    torch.set_num_threads(2)
    set_seed(42)
    d = get_tensors("01")
    Xte, yte = d["test"]
    Xtr, _ = d["train"]
    ms = load_models()
    M = load_json("metrics.json")
    E = load_json("mlp_experiments.json")
    fig0_arch()
    fig1_recon(ms, Xte, yte, M)
    fig2_samples(M)
    fig3_mlp(E)
    fig4_interp(ms, Xte, yte, Xtr)
    fig5_loss()
    print("figures saved to", FIG)


if __name__ == "__main__":
    main()
