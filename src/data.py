"""Dataset Fashion-MNIST dengan split terstandarisasi 80/10/10 (seed 42).

Split bawaan torchvision (60k/10k) digabung dulu menjadi 70.000 citra,
lalu dipermutasi dengan generator ber-seed 42 sehingga semua model memakai
indeks train/val/test yang persis sama (56.000 / 7.000 / 7.000).
Data disimpan sebagai tensor di memori (TensorDataset) agar loading cepat.
"""
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from .utils import ROOT, SEED

DATA_DIR = os.path.join(ROOT, "data")
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_all():
    """Kembalikan X (70000,1,28,28) di [0,1] dan y (70000,)."""
    need_dl = not os.path.exists(os.path.join(DATA_DIR, "FashionMNIST", "raw"))
    tr = datasets.FashionMNIST(DATA_DIR, train=True, download=need_dl)
    te = datasets.FashionMNIST(DATA_DIR, train=False, download=need_dl)
    X = torch.cat([tr.data, te.data]).unsqueeze(1).float() / 255.0
    y = torch.cat([tr.targets, te.targets])
    return X, y


def split_indices(n=70000, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def get_tensors(scale="01"):
    """Dict split -> (X, y). scale='pm1' memetakan piksel ke [-1, 1] (untuk tanh)."""
    X, y = load_all()
    if scale == "pm1":
        X = X * 2 - 1
    tr, va, te = split_indices(len(X))
    return {"train": (X[tr], y[tr]), "val": (X[va], y[va]), "test": (X[te], y[te])}


def get_loaders(scale="01", batch_size=128):
    d = get_tensors(scale)
    g = torch.Generator().manual_seed(SEED)
    mk = lambda s: TensorDataset(*d[s])
    return (DataLoader(mk("train"), batch_size, shuffle=True, generator=g),
            DataLoader(mk("val"), batch_size, shuffle=False),
            DataLoader(mk("test"), batch_size, shuffle=False))
