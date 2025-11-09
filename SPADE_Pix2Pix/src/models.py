"""
models.py
----------
SPADE UNet generator and Spectral-Normalized PatchGAN discriminator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- SPADE ----------
class SPADE(nn.Module):
    def __init__(self, ch, seg_nc=3, hidden=128):
        super().__init__()
        self.norm = nn.InstanceNorm2d(ch, affine=False)
        self.mlp = nn.Sequential(
            nn.Conv2d(seg_nc, hidden, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden, ch * 2, 3, padding=1),
        )

    def forward(self, x, seg):
        seg = F.interpolate(seg, size=x.shape[-2:], mode="nearest")
        h = self.norm(x)
        gamma, beta = torch.chunk(self.mlp(seg), 2, dim=1)
        return h * (1 + gamma) + beta


# ---------- SPADE Blocks ----------
class SPADEConv(nn.Module):
    def __init__(self, in_c, out_c, seg_nc=3, k=4, s=2, p=1, dropout=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.spade = SPADE(out_c, seg_nc)
        self.act = nn.LeakyReLU(0.2, True)
        self.dp = nn.Dropout(0.5) if dropout else nn.Identity()

    def forward(self, x, seg):
        return self.dp(self.act(self.spade(self.conv(x), seg)))


class SPADEResUp(nn.Module):
    def __init__(self, in_c, out_c, seg_nc=3, dropout=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = SPADEConv(in_c, out_c, seg_nc, k=3, s=1, p=1, dropout=dropout)

    def forward(self, x, seg):
        return self.conv(self.up(x), seg)


# ---------- Generator ----------
class UNetGenerator_SPADE(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, seg_nc=3):
        super().__init__()

        # Encoder
        self.d1 = SPADEConv(in_ch, 64, seg_nc)
        self.d2 = SPADEConv(64, 128, seg_nc)
        self.d3 = SPADEConv(128, 256, seg_nc)
        self.d4 = SPADEConv(256, 512, seg_nc)
        self.d5 = SPADEConv(512, 512, seg_nc)
        self.d6 = SPADEConv(512, 512, seg_nc)
        self.d7 = SPADEConv(512, 512, seg_nc)

        # Decoder
        self.u1 = SPADEResUp(512, 512, seg_nc, dropout=True)
        self.u2 = SPADEResUp(1024, 512, seg_nc, dropout=True)
        self.u3 = SPADEResUp(1024, 512, seg_nc, dropout=True)
        self.u4 = SPADEResUp(1024, 512, seg_nc)
        self.u5 = SPADEResUp(768, 256, seg_nc)
        self.u6 = SPADEResUp(384, 128, seg_nc)
        self.u7 = SPADEResUp(192, 64, seg_nc)

        self.outc = nn.Conv2d(64, out_ch, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, seg):
        d1 = self.d1(seg, seg)
        d2 = self.d2(d1, seg)
        d3 = self.d3(d2, seg)
        d4 = self.d4(d3, seg)
        d5 = self.d5(d4, seg)
        d6 = self.d6(d5, seg)
        d7 = self.d7(d6, seg)

        u1 = self.u1(d7, seg); u1 = torch.cat([u1, d6], 1)
        u2 = self.u2(u1, seg); u2 = torch.cat([u2, d5], 1)
        u3 = self.u3(u2, seg); u3 = torch.cat([u3, d4], 1)
        u4 = self.u4(u3, seg); u4 = torch.cat([u4, d3], 1)
        u5 = self.u5(u4, seg); u5 = torch.cat([u5, d2], 1)
        u6 = self.u6(u5, seg); u6 = torch.cat([u6, d1], 1)
        u7 = self.u7(u6, seg)
        return self.tanh(self.outc(u7))


# ---------- Discriminator ----------
def snconv(ic, oc, k, s, p, bias=True):
    return nn.utils.spectral_norm(nn.Conv2d(ic, oc, k, s, p, bias=bias))


class PatchDiscriminatorSN(nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(snconv(in_ch, 64, 4, 2, 1), nn.LeakyReLU(0.2, True)),
            nn.Sequential(snconv(64, 128, 4, 2, 1, False), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True)),
            nn.Sequential(snconv(128, 256, 4, 2, 1, False), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, True)),
            nn.Sequential(snconv(256, 512, 4, 1, 1, False), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, True)),
        ])
        self.head = snconv(512, 1, 4, 1, 1)

    def forward(self, x):
        feats = []
        h = x
        for b in self.blocks:
            h = b(h)
            feats.append(h)
        return self.head(h), feats
