import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from models.seed_upsample import gather_neighborhood_features

LOG_S0 = float(np.log(0.0015))
RAW_DIM = dict(aniso=14, iso=8, surfel=13, capped=14)
LOG_SMAX, LOG_RATIO = (float(np.log(0.02)), float(np.log(4.0)))


def cap_per_seed(cap, seed_cls, S, dev):
    if not isinstance(cap, dict):
        return float(cap)
    d = float(cap.get("default", 3.0))
    r = torch.full((S,), d, device=dev)
    if seed_cls is not None:
        for k, v in cap.items():
            if isinstance(k, int):
                r[seed_cls == k] = float(v)
    return r


_CONSTS = {}


def _cdev(key, make):
    if key not in _CONSTS:
        _CONSTS[key] = make()
    return _CONSTS[key]


def _children(raw, seeds, bmin, vs, child_init, sh1=False, shape="aniso", cap=3.0, seed_cls=None):
    raw.shape[1]
    dev = raw.device
    center = bmin + (seeds[:, [3, 2, 1]].float() + 0.5) * vs
    xyz = (center[:, None] + (torch.tanh(raw[..., :3]) * 0.5 + child_init) * vs).reshape(-1, 3)
    ls = lambda x: (x + LOG_S0).clamp(min=LOG_S0 - 4.0, max=LOG_SMAX)
    if shape == "iso":
        scale = torch.exp(ls(raw[..., 3:4])).expand(-1, -1, 3).reshape(-1, 3)
        quat = torch.tensor([[1.0, 0, 0, 0]], device=dev).expand(xyz.shape[0], 4)
        j = 4
    elif shape == "surfel":
        la = ls(raw[..., 3])
        lb = la + (ls(raw[..., 4]) - la).clamp(-LOG_RATIO, LOG_RATIO)
        lthin = torch.minimum(la, lb) + float(np.log(0.05))
        scale = torch.exp(torch.stack([la, lb, lthin], -1)).reshape(-1, 3)
        quat = F.normalize(raw[..., 5:9] + torch.tensor([1.0, 0, 0, 0], device=dev), dim=-1).reshape(-1, 4)
        j = 9
    elif shape == "capped":
        l3 = ls(raw[..., 3:6])
        m = l3.mean(-1, keepdim=True)
        c = cap_per_seed(cap, seed_cls, raw.shape[0], dev)
        r = float(np.log(c)) / 2 if not torch.is_tensor(c) else (torch.log(c) / 2)[:, None, None]
        rt = r if torch.is_tensor(r) else _cdev(("r", r), lambda: torch.tensor(r, device=dev))
        scale = torch.exp(m + torch.maximum(torch.minimum(l3 - m, rt), -rt)).reshape(-1, 3)
        quat = F.normalize(
            raw[..., 6:10] + _cdev(("q1", str(dev)), lambda: torch.tensor([1.0, 0, 0, 0], device=dev)), dim=-1
        ).reshape(-1, 4)
        j = 10
    else:
        scale = torch.exp(ls(raw[..., 3:6])).reshape(-1, 3)
        quat = F.normalize(raw[..., 6:10] + torch.tensor([1.0, 0, 0, 0], device=dev), dim=-1).reshape(-1, 4)
        j = 10
    g = dict(
        xyz=xyz,
        scale=scale,
        quat=quat,
        opacity=torch.sigmoid(raw[..., j] + 2.0).reshape(-1),
        rgb01=torch.sigmoid(raw[..., j + 1 : j + 4]).reshape(-1, 3),
    )
    if sh1:
        g["sh1"] = raw[..., j + 4 : j + 13].reshape(-1, 3, 3)
    return g


class BigHead(nn.Module):

    def __init__(self, C, K=16, hidden=1024, in_norm=False, shape="aniso", cap=3.0):
        super().__init__()
        self.K = K
        self.shape = shape
        self.D = RAW_DIM[shape]
        self.cap = cap
        self.in_norm = nn.LayerNorm(27 * C) if in_norm else None
        self.net = nn.Sequential(
            nn.Linear(27 * C, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, K * self.D),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.child_init = nn.Parameter((torch.rand(K, 3) * 2 - 1) * 0.4)

    bf16 = False

    def forward(self, feat_2, seeds, bmin, vs, seed_cls=None):
        fm = getattr(self, "fast_mlp", None)
        if fm is not None and (not self.training):
            x = (
                feat_2.gather_nbrs(seeds, bf16=True)
                if hasattr(feat_2, "gather_nbrs")
                else gather_neighborhood_features(feat_2, seeds)
            ).flatten(1)
            return _children(
                fm(x).view(-1, self.K, self.D),
                seeds,
                bmin,
                vs,
                self.child_init,
                shape=self.shape,
                cap=self.cap,
                seed_cls=seed_cls,
            )
        x = (
            feat_2.gather_nbrs(seeds) if hasattr(feat_2, "gather_nbrs") else gather_neighborhood_features(feat_2, seeds)
        ).flatten(1)
        if self.in_norm is not None:
            x = self.in_norm(x.float())
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16 and (not self.training)):
            raw = self.net(x)
        return _children(
            raw.float().view(-1, self.K, self.D),
            seeds,
            bmin,
            vs,
            self.child_init,
            shape=self.shape,
            cap=self.cap,
            seed_cls=seed_cls,
        )


class BigHeadRoundShim(nn.Module):

    def __init__(self, feat_dim, K=16, hidden=1024, shape="aniso", cap=3.0):
        super().__init__()
        self.head = BigHead(feat_dim, K=K, hidden=hidden, in_norm=True, shape=shape, cap=cap)
        self.K = K
        self.p1, self.p2 = (K, 1)

    def forward(self, feat_2, seeds, bmin, vsize, rounds=2, seed_cls=None):
        g = self.head(feat_2, seeds, bmin, vsize, seed_cls=seed_cls)
        S = seeds.shape[0]
        K = self.K
        r = dict(
            xyz=g["xyz"].view(S, K, 3),
            scale=g["scale"].view(S, K, 3),
            opacity=g["opacity"].view(S, K),
            rgb=(g["rgb01"] * 2 - 1).view(S, K, 3),
            quat=g["quat"].view(S, K, 4),
        )
        return (r, r)

    def export_spec(self):
        return dict(
            type="big",
            children=self.K,
            sh1=False,
            in_norm=True,
            shape=self.head.shape,
            cap=self.head.cap,
            state_dict=self.head.state_dict(),
        )
