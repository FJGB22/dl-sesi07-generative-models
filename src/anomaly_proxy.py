"""Eksperimen pendukung Q6(1): sensitivitas reconstruction error terhadap cacat lokal.

Setiap citra test diberi cacat sintetis ala inspeksi manufaktur (bercak terang 5x5 dan goresan
diagonal), lalu galat rekonstruksi per citra dipakai sebagai skor anomali. AUROC mengukur seberapa
baik skor tersebut memisahkan citra normal dan cacat (1,0 = sempurna, 0,5 = acak).

    python -m src.anomaly_proxy
"""
import os

import torch

from .data import get_tensors
from .models.autoencoders import VanillaAE, SparseAE
from .utils import CKPT, save_json, set_seed


def add_defects(x, g):
    x = x.clone()
    n = len(x)
    r = torch.randint(4, 20, (n,), generator=g)
    c = torch.randint(4, 20, (n,), generator=g)
    kind = torch.randint(0, 2, (n,), generator=g)
    for i in range(n):
        if kind[i] == 0:                                  # bercak / noda
            x[i, 0, r[i]:r[i] + 5, c[i]:c[i] + 5] = 1.0
        else:                                             # goresan diagonal sepanjang 10 px
            for k in range(10):
                rr, cc = min(r[i] + k, 27), min(c[i] + k, 27)
                x[i, 0, rr, cc] = 1.0
    return x


def auroc(neg, pos):
    """AUROC lewat statistik Mann-Whitney U."""
    s = torch.cat([neg, pos])
    rank = torch.empty_like(s)
    rank[s.argsort()] = torch.arange(1, len(s) + 1, dtype=s.dtype)
    u = rank[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2
    return (u / (len(neg) * len(pos))).item()


def main():
    set_seed(42)
    X, _ = get_tensors("01")["test"]
    Xd = add_defects(X, torch.Generator().manual_seed(42))
    out = {}
    for name, cls in (("ae", VanillaAE), ("sae", SparseAE)):
        m = cls()
        m.load_state_dict(torch.load(os.path.join(CKPT, f"{name}.pt"), map_location="cpu"))
        m.eval()
        with torch.no_grad():
            e_n = ((m(X)[0] - X) ** 2).flatten(1).mean(1)
            e_d = ((m(Xd)[0] - Xd) ** 2).flatten(1).mean(1)
            # galat hanya di area cacat: seberapa jauh model "memperbaiki" cacat (makin besar makin sensitif)
            mask = (Xd != X)
            rec_d = m(Xd)[0]
            fix = ((rec_d - Xd) ** 2)[mask].mean().item()
        out[name] = {"auroc": auroc(e_n, e_d), "err_normal": e_n.mean().item(), "err_defect": e_d.mean().item(),
                     "ratio": (e_d.mean() / e_n.mean()).item(), "err_on_defect_pixels": fix}
        print(name, out[name])
    save_json(out, "anomaly_proxy.json")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
