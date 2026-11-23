"""Vanilla AE, Sparse AE, dan VAE (MLP 784 -> 256 -> 64 -> 256 -> 784).

Ketiganya memakai encoder dan decoder dengan ukuran lapisan yang sama
dengan MLP Classifier (784 -> 256 -> 64), sehingga perbandingan pada
Bagian 2 (z_MLP vs z_AE) dan Q1 hanya dipengaruhi fungsi objektifnya.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT = 64
HIDDEN = 256


class Encoder(nn.Module):
    def __init__(self, out_dim=LATENT, code_act=True):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, out_dim)
        self.code_act = code_act

    def forward(self, x):
        h = F.relu(self.fc1(x.flatten(1)))
        z = self.fc2(h)
        return F.relu(z) if self.code_act else z


class Decoder(nn.Module):
    """64 -> 256 -> 784, keluaran Sigmoid (piksel [0,1]).
    Arsitektur ini juga dipakai sebagai Frozen-Latent Decoder (Tahap 4)."""

    def __init__(self, in_dim=LATENT):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, 784)

    def forward(self, z):
        return torch.sigmoid(self.fc2(F.relu(self.fc1(z)))).view(-1, 1, 28, 28)


class VanillaAE(nn.Module):
    """L = ||x - x_hat||^2 (MSE). Kode laten ReLU 64-d, sama seperti Sparse AE,
    sehingga satu-satunya beda dengan Sparse AE adalah penalti L1."""

    def __init__(self):
        super().__init__()
        self.enc, self.dec = Encoder(), Decoder()

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        h = self.enc(x)
        return self.dec(h), h

    def loss(self, x):
        x_hat, h = self(x)
        rec = F.mse_loss(x_hat, x)
        return rec, {"rec": rec.item()}


class SparseAE(VanillaAE):
    """L = MSE(x, x_hat) + lambda * ||h||_1 (rata-rata per sampel)."""

    def __init__(self, lam=1e-3):
        super().__init__()
        self.lam = lam

    def loss(self, x):
        x_hat, h = self(x)
        rec = F.mse_loss(x_hat, x)
        l1 = h.abs().sum(1).mean()
        return rec + self.lam * l1, {"rec": rec.item(), "l1": l1.item()}


class VAE(nn.Module):
    """q(z|x) = N(mu, diag(sigma^2)), p(z) = N(0, I).
    L = BCE(x, x_hat) + KL(q(z|x) || p(z))   (dijumlahkan per piksel, dirata-rata per sampel)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.mu = nn.Linear(HIDDEN, LATENT)
        self.logvar = nn.Linear(HIDDEN, LATENT)
        self.dec = Decoder()

    def encode(self, x):
        h = F.relu(self.fc1(x.flatten(1)))
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)   # reparameterization trick
        return self.dec(z), mu, logvar

    def loss(self, x):
        x_hat, mu, logvar = self(x)
        bce = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return bce + kl, {"bce": bce.item(), "kl": kl.item(),
                          "rec": F.mse_loss(x_hat, x).item()}

    @torch.no_grad()
    def sample(self, n, device="cpu"):
        return self.dec(torch.randn(n, LATENT, device=device))
