"""Utilitas umum: seed, device, timer, dan penyimpanan hasil."""
import json
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
CKPT = os.path.join(OUT, "checkpoints")
FIG = os.path.join(OUT, "figures")
LOG = os.path.join(OUT, "logs")
RES = os.path.join(OUT, "results")
for _d in (CKPT, FIG, LOG, RES):
    os.makedirs(_d, exist_ok=True)


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def timer():
    t = {"s": 0.0}
    start = time.perf_counter()
    yield t
    t["s"] = time.perf_counter() - start


def save_json(obj, name):
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def load_json(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def to01(x):
    """Ubah tensor dari [-1, 1] ke [0, 1]."""
    return (x.clamp(-1, 1) + 1) / 2
