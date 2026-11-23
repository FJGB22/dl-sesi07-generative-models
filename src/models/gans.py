"""Vanilla GAN (fully connected) dan DCGAN (fully convolutional) untuk 28x28.

Keduanya memakai prior z ~ N(0, I) berdimensi 100, keluaran tanh ([-1,1]),
dan loss non-saturating:
    L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
    L_G = -E[log D(G(z))]
"""
import torch
import torch.nn as nn

Z_DIM = 100


class MLPGenerator(nn.Module):
    def __init__(self, z=Z_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 784), nn.Tanh())

    def forward(self, z):
        return self.net(z.flatten(1)).view(-1, 1, 28, 28)


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1))                      # logit; sigmoid ada di BCEWithLogits

    def forward(self, x):
        return self.net(x.flatten(1)).squeeze(1)


class DCGenerator(nn.Module):
    """z(100x1x1) -> 7x7x256 -> 14x14x128 -> 28x28x1. ConvTranspose + BatchNorm + ReLU, tanh di akhir."""

    def __init__(self, z=Z_DIM, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z, ngf * 4, 7, 1, 0, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class DCDiscriminator(nn.Module):
    """28x28 -> 14x14x64 -> 7x7x128 -> 1. Strided conv (tanpa pooling) + LeakyReLU, BN di lapisan tengah."""

    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, 7, 1, 0))

    def forward(self, x):
        return self.net(x).view(-1)


def dcgan_init(m):
    """Inisialisasi bobot N(0, 0.02) sesuai paper DCGAN (Radford et al., 2016)."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
