import torch
import torch.nn as nn
import torch.nn.functional as F
from models.seed_upsample import (
    SeedUpsampler,
    gather_neighborhood_features,
    compute_particle_world_coords,
    local_chamfer_loss,
    _init_particle_offsets,
)


def two_round_chamfer_loss(
    seed_indices,
    head_out,
    gt_xyz,
    gt_rgb,
    bounds_min,
    voxel_size,
    fine_shape,
    rgb_weight=0.1,
    kernel=3,
    seed_weights=None,
    round2_weight=1.0,
    bwd_weight=1.0,
):
    r1_xyz, r1_rgb, r2_xyz, r2_rgb = head_out
    if r1_xyz.shape[0] == 0:
        z = torch.zeros((), device=r1_xyz.device)
        return (z, z, z, z, z, z)
    cd1, f1, b1, g1 = local_chamfer_loss(
        seed_indices,
        r1_xyz,
        r1_rgb,
        gt_xyz,
        gt_rgb,
        bounds_min,
        voxel_size,
        fine_shape=fine_shape,
        rgb_weight=rgb_weight,
        kernel=kernel,
        seed_weights=seed_weights,
        bwd_weight=bwd_weight,
        return_components=True,
    )
    cd2, f2, b2, g2 = local_chamfer_loss(
        seed_indices,
        r2_xyz,
        r2_rgb,
        gt_xyz,
        gt_rgb,
        bounds_min,
        voxel_size,
        fine_shape=fine_shape,
        rgb_weight=rgb_weight,
        kernel=kernel,
        seed_weights=seed_weights,
        bwd_weight=bwd_weight,
        return_components=True,
    )
    return (cd1 + round2_weight * cd2, cd1, cd2, f1 + f2, b1 + b2, g1 + g2)


class TwoRoundGaussianHead(nn.Module):

    def __init__(
        self,
        feat_dim=64,
        hidden=64,
        n_heads=4,
        p1=4,
        p2=4,
        max_offset=1.5,
        child_offset=0.75,
        init_log_scale=-5.8,
        init_opacity_logit=4.6,
        aniso=False,
        n_label=0,
        sh_degree=0,
        scale_max=None,
    ):
        super().__init__()
        self.p1, self.p2 = (p1, p2)
        self.child_offset = child_offset
        self.aniso = bool(aniso)
        import math

        self.log_scale_max = float(math.log(scale_max)) if scale_max else 1.0
        self.sh_degree = int(sh_degree)
        assert self.sh_degree in (0, 1)
        self.round1 = SeedUpsampler(feat_dim=feat_dim, hidden=hidden, n_heads=n_heads, n_particles=p1, max_offset=max_offset)
        ns = 3 if self.aniso else 1
        nq = 4 if self.aniso else 0
        nsh = 9 if self.sh_degree == 1 else 0
        self._ns, self._nsh = (ns, nsh)
        self._gc = ns + 1 + nq
        self._nl = int(n_label)
        self._r2c = 3 + ns + 1 + nq + 3 + nsh + self._nl
        self.r1_gauss = nn.Linear(hidden, self._gc)
        if nsh:
            self.r1_sh = nn.Linear(hidden, nsh)
            nn.init.zeros_(self.r1_sh.weight)
            nn.init.zeros_(self.r1_sh.bias)
        self.child = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(inplace=True), nn.Linear(hidden, p2 * self._r2c)
        )
        self.register_buffer("child_init", _init_particle_offsets(p2, scale=child_offset * 0.5))
        with torch.no_grad():
            self.r1_gauss.bias[:ns] = float(init_log_scale)
            self.r1_gauss.bias[ns] = float(init_opacity_logit)
            b = self.child[-1].bias.view(p2, self._r2c)
            b[:, 3 : 3 + ns] = float(init_log_scale)
            b[:, 3 + ns] = float(init_opacity_logit)
            if self.aniso:
                q1 = ns + 1
                self.r1_gauss.weight[q1 : q1 + 4].zero_()
                self.r1_gauss.bias[q1 : q1 + 4] = 0.0
                self.r1_gauss.bias[q1] = 1.0
                qc = 3 + ns + 1
                self.child[-1].weight.view(p2, self._r2c, -1)[:, qc : qc + 4].zero_()
                b[:, qc : qc + 4] = 0.0
                b[:, qc] = 1.0
            if nsh:
                self.child[-1].weight.view(p2, self._r2c, -1)[:, 3 + ns + 1 + nq + 3 : 3 + ns + 1 + nq + 3 + nsh].zero_()
                b[:, 3 + ns + 1 + nq + 3 : 3 + ns + 1 + nq + 3 + nsh] = 0.0
            if self._nl:
                self.child[-1].weight.view(p2, self._r2c, -1)[:, -self._nl :].zero_()
                b[:, -self._nl :] = -2.94

    def _empty(self, p, device):
        d = dict(
            xyz=torch.zeros(0, p, 3, device=device),
            scale=torch.zeros(0, p, 3, device=device),
            opacity=torch.zeros(0, p, device=device),
            rgb=torch.zeros(0, p, 3, device=device),
        )
        if self.aniso:
            d["quat"] = torch.zeros(0, p, 4, device=device)
        if self.sh_degree == 1:
            d["sh1"] = torch.zeros(0, p, 3, 3, device=device)
        return d

    def forward(self, feat_2, seed_indices, bounds_min, voxel_size, rounds=2):
        device = feat_2.device
        S = seed_indices.shape[0]
        if S == 0:
            e1 = self._empty(self.p1, device)
            return (e1, None) if rounds < 2 else (e1, self._empty(self.p1 * self.p2, device))
        nbr = gather_neighborhood_features(feat_2, seed_indices)
        off1, r1_rgb, feat1 = self.round1(nbr, return_feat=True)
        r1_xyz = compute_particle_world_coords(seed_indices, off1, bounds_min, voxel_size)
        g1 = self.r1_gauss(feat1)
        P2 = self.p1 * self.p2
        if self.aniso:
            r1_scale = torch.exp(g1[..., 0:3].clamp(-10.0, self.log_scale_max))
            r1_op = torch.sigmoid(g1[..., 3])
            r1_quat = F.normalize(g1[..., 4:8], dim=-1)
            round1 = dict(xyz=r1_xyz, scale=r1_scale, opacity=r1_op, rgb=r1_rgb, quat=r1_quat)
        else:
            r1_scale = torch.exp(g1[..., 0:1].clamp(-10.0, self.log_scale_max)).expand(-1, -1, 3)
            r1_op = torch.sigmoid(g1[..., 1])
            round1 = dict(xyz=r1_xyz, scale=r1_scale, opacity=r1_op, rgb=r1_rgb)
        if self.sh_degree == 1:
            round1["sh1"] = self.r1_sh(feat1).view(S, self.p1, 3, 3)
        if rounds < 2:
            return (round1, None)
        raw = self.child(feat1).view(S, self.p1, self.p2, self._r2c)
        coff = torch.tanh(raw[..., :3]) * self.child_offset + self.child_init.view(1, 1, self.p2, 3)
        coff = coff.clamp(-self.child_offset, self.child_offset)
        vs = voxel_size.view(1, 1, 1, 3)
        r2_xyz = (r1_xyz.unsqueeze(2) + coff * vs).reshape(S, P2, 3)
        if self.aniso:
            r2_scale = torch.exp(raw[..., 3:6].clamp(-10.0, self.log_scale_max)).reshape(S, P2, 3)
            r2_op = torch.sigmoid(raw[..., 6]).reshape(S, P2)
            r2_quat = F.normalize(raw[..., 7:11], dim=-1).reshape(S, P2, 4)
            r2_rgb = torch.tanh(raw[..., 11:14]).reshape(S, P2, 3)
            round2 = dict(xyz=r2_xyz, scale=r2_scale, opacity=r2_op, rgb=r2_rgb, quat=r2_quat)
        else:
            r2_scale = torch.exp(raw[..., 3:4].clamp(-10.0, self.log_scale_max)).expand(-1, -1, -1, 3).reshape(S, P2, 3)
            r2_op = torch.sigmoid(raw[..., 4]).reshape(S, P2)
            r2_rgb = torch.tanh(raw[..., 5:8]).reshape(S, P2, 3)
            round2 = dict(xyz=r2_xyz, scale=r2_scale, opacity=r2_op, rgb=r2_rgb)
        if self.sh_degree == 1:
            round2["sh1"] = (raw[..., -9 - self._nl : -self._nl] if self._nl else raw[..., -9:]).reshape(S, P2, 3, 3)
        if self._nl:
            round2["label"] = raw[..., -self._nl :].reshape(S, P2, self._nl)
        round2["feat"] = feat1.unsqueeze(2).expand(-1, -1, self.p2, -1).reshape(S, P2, feat1.shape[-1])
        return (round1, round2)

    def load_state_dict(self, sd, strict=True):
        k_w, k_b = ("child.3.weight", "child.3.bias")
        if self._nl and k_w in sd and (sd[k_w].shape != self.child[-1].weight.shape):
            old_c = sd[k_w].shape[0] // self.p2
            with torch.no_grad():
                w = self.child[-1].weight.detach().clone().view(self.p2, self._r2c, -1)
                b = self.child[-1].bias.detach().clone().view(self.p2, self._r2c)
                w[:, :old_c] = sd[k_w].view(self.p2, old_c, -1).to(w)
                b[:, :old_c] = sd[k_b].view(self.p2, old_c).to(b)
            sd = dict(sd)
            sd[k_w] = w.reshape(self.p2 * self._r2c, -1)
            sd[k_b] = b.reshape(-1)
            print(f"[seed head] adapted checkpoint without label logits ({old_c} -> {self._r2c} per-child channels)", flush=True)
        return super().load_state_dict(sd, strict=strict)
