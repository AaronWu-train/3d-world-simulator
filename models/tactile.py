import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ROWS, PAD_COLS, PAD_AREA_MM2 = 12, 32, 22.0 * 62.0
PAD_SAMPLES_PER_PAD = 1500


class TactileParticleHead(nn.Module):
    def __init__(self, C, K=16, hidden=256, reach=1.0, n_tac=2):
        super().__init__()
        self.K = K
        self.reach = reach
        self.mlp = nn.Sequential(
            nn.Linear(27 * (C + n_tac) + 3, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, K * 5)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        b = torch.zeros(K, 5)
        b[:, :3] = torch.atanh(torch.empty(K, 3).uniform_(-0.45, 0.45) / reach)
        b[:, 3] = -3.0
        b[:, 4] = 2.0
        self.mlp[-1].bias.data.copy_(b.reshape(-1))

    @staticmethod
    def gather27(vol, seeds):
        v = F.pad(vol, (1, 1, 1, 1, 1, 1))
        b, z, y, x = seeds.unbind(1)
        return torch.cat([v[b, :, z + dz, y + dy, x + dx] for dz in (0, 1, 2) for dy in (0, 1, 2) for dx in (0, 1, 2)], 1)

    def forward(self, f, tac, seeds, bmin, vsize):
        D, H, W = f.shape[2:]
        b, z, y, x = seeds.unbind(1)
        pos = torch.stack([x / max(W - 1, 1), y / max(H - 1, 1), z / max(D - 1, 1)], 1).float() * 2 - 1
        o = self.mlp(torch.cat([self.gather27(torch.cat([f, tac], 1).float(), seeds), pos], 1)).view(-1, self.K, 5)
        centre = bmin + (torch.stack([x, y, z], 1).float() + 0.5) * vsize
        return centre[:, None] + torch.tanh(o[..., :3]) * self.reach * vsize, o[..., 3], o[..., 4]


def dense_pad(Xp, Sp, n):
    P = Xp.shape[0]
    G = Xp.view(P, PAD_ROWS, PAD_COLS, 3)
    Gs = Sp.view(P, PAD_ROWS, PAD_COLS)
    dv = Xp.device
    r = torch.randint(0, PAD_ROWS - 1, (P, n), device=dv)
    c = torch.randint(0, PAD_COLS - 1, (P, n), device=dv)
    u = torch.rand(P, n, device=dv)
    v = torch.rand(P, n, device=dv)
    pi = torch.arange(P, device=dv)[:, None].expand(P, n)
    w = torch.stack([(1 - u) * (1 - v), (1 - u) * v, u * (1 - v), u * v], -1)
    corners = ((0, 0), (0, 1), (1, 0), (1, 1))
    Y = sum(w[..., i, None] * G[pi, r + dr, c + dc] for i, (dr, dc) in enumerate(corners))
    Ys = sum(w[..., i] * Gs[pi, r + dr, c + dc] for i, (dr, dc) in enumerate(corners))
    return Y.reshape(-1, 3), Ys.reshape(-1)


def surface_loss(
    P, SL, KL, seeds, Y, YS, bmin, vsize, reach, act_thr=0.05, alpha=10.0, sig_w=5.0, rep_w=10.0, rep_k=6, keep_mm=1.0
):
    K = P.shape[1]
    Pf = P.reshape(-1, 3)
    centre = bmin + (torch.stack([seeds[:, 3], seeds[:, 2], seeds[:, 1]], 1).float() + 0.5) * vsize
    R = ((Y[None] - centre[:, None]).abs() < reach * vsize).all(-1)
    ycov = R.any(0)
    elig = R.any(1).repeat_interleave(K)
    d = torch.cdist(Pf, Y) * 1000.0
    fwd_all, nearest = d.min(1)
    zero = d.sum() * 0
    fwd = fwd_all[elig].mean() if elig.any() else zero
    bwd = d[elig][:, ycov].min(0)[0].mean() if (elig.any() and ycov.any()) else zero
    keep = F.binary_cross_entropy_with_logits(KL.reshape(-1), (elig & (fwd_all.detach() < keep_mm)).float())
    rep = zero
    Pe = Pf[elig]
    if len(Pe) > rep_k + 1:
        area = float(ycov.float().mean()) * PAD_AREA_MM2 * (len(Y) // PAD_SAMPLES_PER_PAD)
        h = math.sqrt(max(area, 1.0) / len(Pe))
        knn = (torch.cdist(Pe, Pe) * 1000.0).topk(rep_k + 1, dim=1, largest=False)[0][:, 1:]
        rep = (F.relu(h - knn) / h).pow(2).mean()
    ts = YS[nearest]
    w = (1 + alpha * (ts > act_thr).float()) * elig.float()
    sig = ((torch.sigmoid(SL.reshape(-1)) - ts).abs() * w).sum() / w.sum().clamp(min=1)
    return fwd + bwd + rep_w * rep + keep + sig_w * sig


def fire_color(sig):
    s = torch.clamp(sig, 0, 1)
    return torch.stack(
        [
            torch.where(s < 0.5, 2 * s, torch.ones_like(s)),
            torch.where(s < 0.5, torch.zeros_like(s), 2 * (s - 0.5)),
            torch.zeros_like(s),
        ],
        -1,
    )


def rasterise(X, S, bmin, bext, shape_zyx):
    nz, ny, nx = shape_zyx
    vs = bext / np.array([nx, ny, nz], np.float32)
    idx = np.floor((X - bmin) / vs).astype(np.int64)
    ok = ((idx >= 0) & (idx < np.array([nx, ny, nz]))).all(1)
    idx = idx[ok]
    flat = idx[:, 2] * ny * nx + idx[:, 1] * nx + idx[:, 0]
    sig = np.zeros(nz * ny * nx, np.float32)
    np.maximum.at(sig, flat, S[ok])
    mask = np.zeros(nz * ny * nx, bool)
    mask[flat] = True
    return mask.reshape(nz, ny, nx), sig.reshape(nz, ny, nx)
