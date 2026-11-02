import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class Block(nn.Sequential):

    def __init__(self, a, b):
        super().__init__(
            nn.Conv3d(a, b, 3, padding=1),
            nn.GroupNorm(4, b),
            nn.SiLU(),
            nn.Conv3d(b, b, 3, padding=1),
            nn.GroupNorm(4, b),
            nn.SiLU(),
        )

    def forward(self, x, film=None):
        if film is None:
            return super().forward(x)
        g, b_ = film
        b = g.shape[1] // 2
        h = self[1](self[0](x))
        h = self[2](h * (1 + g[:, :b, None, None, None]) + b_[:, :b, None, None, None])
        h = self[4](self[3](h))
        return self[5](h * (1 + g[:, b:, None, None, None]) + b_[:, b:, None, None, None])


def _film_mlp(inp, hidden, out):
    m = nn.Sequential(nn.Linear(inp, hidden), nn.SiLU(), nn.Linear(hidden, out))
    nn.init.zeros_(m[-1].weight)
    nn.init.zeros_(m[-1].bias)
    return m


class TimeBlock(nn.Module):

    def __init__(self, c, heads=4):
        super().__init__()
        self.h = heads
        self.norm = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * c)
        self.proj = nn.Linear(c, c)
        self.norm2 = nn.LayerNorm(c)
        self.ff = nn.Sequential(nn.Linear(c, 2 * c), nn.GELU(), nn.Linear(2 * c, c))

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, t, 3, self.h, c // self.h).permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, c)
        x = x + self.proj(y)
        return x + self.ff(self.norm2(x))


class CoarseDynamics3D(nn.Module):

    def __init__(
        self,
        latent_channels=8,
        action_dim=14,
        n_history=8,
        n_future=1,
        width=56,
        action_emb_dim=128,
        n_time_blocks=2,
        n_levels=2,
        film=False,
    ):
        super().__init__()
        if n_levels < 1:
            raise ValueError("n_levels must be >= 1")
        if width % 4:
            raise ValueError("width must be divisible by 4 (GroupNorm groups / attention heads)")
        self.C, self.A, self.T, self.n_future = (latent_channels, action_dim, n_history + n_future, n_future)
        w = width
        c = 4 * w
        self.c = c
        self.action_ch = nn.Sequential(nn.Linear(action_dim, action_emb_dim), nn.SiLU(), nn.Linear(action_emb_dim, 8))
        fin = latent_channels + 8 + 1
        ws = [min(w * 2**i, c) for i in range(n_levels)]
        self.n_levels = n_levels
        for i, wi in enumerate(ws):
            setattr(self, f"enc{i}", Block(self.T * fin if i == 0 else ws[i - 1], wi))
        self.mid = Block(ws[-1], c)
        self.lat = nn.Conv3d(fin, c, 1)
        self.pose = nn.Sequential(nn.Linear(action_dim + 1, action_emb_dim), nn.SiLU(), nn.Linear(action_emb_dim, c))
        self.noise = nn.Sequential(nn.Linear(1, action_emb_dim), nn.SiLU(), nn.Linear(action_emb_dim, c))
        self.slot_embed = nn.Parameter(torch.zeros(self.T, c))
        self.temporal = nn.Sequential(*[TimeBlock(c) for _ in range(n_time_blocks)])
        prev = c
        for i in reversed(range(n_levels)):
            setattr(self, f"dec{i}", Block(prev + ws[i], ws[i]))
            prev = ws[i]
        self.out = nn.Conv3d(w, latent_channels, 1)
        self.film = bool(film)
        if self.film:
            self.film_tok = _film_mlp(action_dim + 2, action_emb_dim, 2 * c)
            self.film_dec = nn.ModuleList([_film_mlp(action_dim + 1, action_emb_dim, 4 * ws[i]) for i in range(n_levels)])

    def forward(self, z_seq, external_cond=None, noise_std=None, future_only=False):
        b, t, cch, nz, ny, nx = z_seq.shape
        if t != self.T or cch != self.C:
            raise ValueError("Expected configured history+future frames and latent channels")
        if external_cond is None or external_cond.shape != (t, b, self.A):
            raise ValueError("Expected actions shaped (T,B,action_dim)")
        if noise_std is None:
            noise_std = z_seq.new_zeros(b, t)
        fut, act, nch, *skips = self.encode_tokens(z_seq, external_cond, noise_std)
        out = self.decode_future(fut, act, nch, *skips)
        if future_only:
            return out
        return torch.cat([z_seq[:, : t - self.n_future], out], 1)

    def encode_tokens(self, z_seq, external_cond, noise_std):
        b, t, cch, nz, ny, nx = z_seq.shape
        act = external_cond.permute(1, 0, 2).to(z_seq.dtype)
        ach = self.action_ch(act)
        nch = noise_std.to(z_seq.dtype)[..., None]
        per = torch.cat([z_seq, torch.cat([ach, nch], -1)[..., None, None, None].expand(-1, -1, -1, nz, ny, nx)], 2)
        skips = []
        x = per.flatten(1, 2)
        for i in range(self.n_levels):
            x = getattr(self, f"enc{i}")(x if i == 0 else F.avg_pool3d(x, 2))
            skips.append(x)
        x2 = self.mid(F.avg_pool3d(x, 2))
        sp = x2.shape[-3:]
        times = torch.arange(t, device=z_seq.device, dtype=z_seq.dtype)[None, :, None].expand(b, -1, 1) - (t - self.n_future - 1)
        emb = self.pose(torch.cat([act, times / 10.0], -1)) + self.noise(nch) + self.slot_embed[None].to(z_seq.dtype)
        low = self.lat(F.adaptive_avg_pool3d(per.flatten(0, 1), sp)).reshape(b, t, self.c, *sp)
        tokens = low + x2[:, None] + emb[..., None, None, None]
        if self.film:
            gb = self.film_tok(torch.cat([act, times / 10.0, nch], -1))
            g_, b_ = gb.chunk(2, -1)
            tokens = (
                (low + x2[:, None]) * (1 + g_[..., None, None, None]) + b_[..., None, None, None] + emb[..., None, None, None]
            )
        tokens = self.temporal(tokens.permute(0, 3, 4, 5, 1, 2).reshape(-1, t, self.c))
        fut = tokens.reshape(b, *sp, t, self.c)[:, :, :, :, -self.n_future :].permute(0, 4, 5, 1, 2, 3)
        return (fut, act, nch, *skips)

    def decode_future(self, fut, act, nch, *skips):
        t = act.shape[1]

        def decode(f, films, *sk):
            y = f
            for i in reversed(range(self.n_levels)):
                y = getattr(self, f"dec{i}")(
                    torch.cat([F.interpolate(y, size=sk[i].shape[-3:], mode="trilinear", align_corners=False), sk[i]], 1),
                    film=films[i] if films is not None else None,
                )
            return self.out(y)

        outs = []
        for k in range(self.n_future):
            films = None
            if self.film:
                fa = torch.cat([act[:, t - self.n_future + k], nch[:, t - self.n_future + k]], -1)
                films = [tuple(m(fa).chunk(2, -1)) for m in self.film_dec]
            outs.append(
                checkpoint(decode, fut[:, k], films, *skips, use_reentrant=False)
                if self.training
                else decode(fut[:, k], films, *skips)
            )
        return torch.stack(outs, 1)
