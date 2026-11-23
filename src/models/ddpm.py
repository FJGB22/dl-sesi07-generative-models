"""DDPM (Ho et al., 2020) dengan U-Net kecil untuk citra 28x28 grayscale.

Forward process : q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I)
Objektif        : L_simple = E_{t, x0, eps} || eps - eps_theta(x_t, t) ||^2
Reverse process : x_{t-1} = 1/sqrt(a_t) (x_t - (1-a_t)/sqrt(1-abar_t) eps_theta) + sigma_t z
Jadwal beta linier 1e-4 .. 0.02 dengan T = 1000 langkah.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(temb)[:, :, None, None]          # injeksi informasi langkah waktu t
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """28 -> 14 -> 7 dengan skip connection; kanal (32, 64, 128)."""

    def __init__(self, ch=(32, 64, 128), tdim=128):
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inc = nn.Conv2d(1, ch[0], 3, padding=1)
        self.d1 = ResBlock(ch[0], ch[0], tdim)
        self.down1 = nn.Conv2d(ch[0], ch[1], 4, 2, 1)       # 28 -> 14
        self.d2 = ResBlock(ch[1], ch[1], tdim)
        self.down2 = nn.Conv2d(ch[1], ch[2], 4, 2, 1)       # 14 -> 7
        self.mid1 = ResBlock(ch[2], ch[2], tdim)
        self.mid2 = ResBlock(ch[2], ch[2], tdim)
        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 4, 2, 1)  # 7 -> 14
        self.u2 = ResBlock(ch[1] * 2, ch[1], tdim)
        self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 4, 2, 1)  # 14 -> 28
        self.u1 = ResBlock(ch[0] * 2, ch[0], tdim)
        self.out = nn.Sequential(nn.GroupNorm(8, ch[0]), nn.SiLU(), nn.Conv2d(ch[0], 1, 3, padding=1))

    def forward(self, x, t):
        temb = self.tmlp(timestep_embedding(t, self.tdim))
        h1 = self.d1(self.inc(x), temb)
        h2 = self.d2(self.down1(h1), temb)
        m = self.mid2(self.mid1(self.down2(h2), temb), temb)
        u2 = self.u2(torch.cat([self.up2(m), h2], 1), temb)
        u1 = self.u1(torch.cat([self.up1(u2), h1], 1), temb)
        return self.out(u1)


class DDPM(nn.Module):
    def __init__(self, T=1000, beta1=1e-4, betaT=0.02):
        super().__init__()
        self.T = T
        self.net = UNet()
        betas = torch.linspace(beta1, betaT, T)
        alphas = 1 - betas
        abar = torch.cumprod(alphas, 0)
        abar_prev = torch.cat([torch.ones(1), abar[:-1]])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", abar)
        # varians posterior beta_tilde (Ho et al., Eq. 7)
        self.register_buffer("post_var", betas * (1 - abar_prev) / (1 - abar))

    def q_sample(self, x0, t, eps):
        a = self.abar[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * eps

    def loss(self, x0):
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        eps = torch.randn_like(x0)
        mse = F.mse_loss(self.net(self.q_sample(x0, t, eps), t), eps)
        return mse, {"noise_mse": mse.item()}

    @torch.no_grad()
    def sample(self, n, device="cpu", return_traj=False, traj_steps=()):
        x = torch.randn(n, 1, 28, 28, device=device)
        traj = {self.T: x.clone()} if return_traj else None
        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            eps = self.net(x, t)
            a, ab = self.alphas[i], self.abar[i]
            mean = (x - (1 - a) / (1 - ab).sqrt() * eps) / a.sqrt()
            x = mean + (self.post_var[i].sqrt() * torch.randn_like(x) if i > 0 else 0)
            if return_traj and i in traj_steps:
                traj[i] = x.clone()
        return (x.clamp(-1, 1), traj) if return_traj else x.clamp(-1, 1)
