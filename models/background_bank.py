from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def _voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, grid: float, cap: int, rng):
    if len(xyz) == 0:
        return (xyz, rgb)
    key = np.floor(xyz / grid).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    n = cnt.shape[0]
    sx = np.zeros((n, 3), np.float64)
    sc = np.zeros((n, 3), np.float64)
    np.add.at(sx, inv, xyz)
    np.add.at(sc, inv, rgb)
    xyz_d = (sx / cnt[:, None]).astype(np.float32)
    rgb_d = (sc / cnt[:, None]).astype(np.float32)
    if len(xyz_d) > cap:
        sel = rng.choice(len(xyz_d), cap, replace=False)
        xyz_d, rgb_d = (xyz_d[sel], rgb_d[sel])
    return (xyz_d, rgb_d)


@torch.no_grad()
def build_background_init(
    packed_files,
    bbox,
    n_near: int = 90000,
    n_far: int = 40000,
    max_eps: int = 16,
    far_dist: float = 2.6,
    near_grid: float = 0.008,
    far_grid: float = 0.03,
    stride: int = 1,
    seed: int = 0,
    table_radius: float = 1.4,
    fs_max_depth: float = 3.5,
):
    import h5py, cv2

    rng = np.random.default_rng(seed)
    bb = np.asarray(bbox, np.float32)
    never_valid = {}
    ref_geom = {}
    z_samples = []
    fs_dmax = {}
    files = list(packed_files)[:max_eps]
    for ei, pf in enumerate(files):
        with h5py.File(pf, "r") as p:
            int(p.attrs["H"])
            int(p.attrs["W"])
            T = int(p.attrs["num_frames"])
            fr = int(ei * 9973 % max(T - 1, 1))
            if str(p.attrs.get("depth_semantics", "")) != "full_scene_lrc":
                raise ValueError(f"{pf}: bank init needs packed_v2 full-scene depth (depth_semantics=full_scene_lrc)")
            for c in [int(x) for x in p.attrs["cam_ids"]]:
                z = p[f"img/camera_{c}/depth"][fr].astype(np.float32)
                z[(z < 0.15) | (z > fs_max_depth)] = 0.0
                ck = (c, *z.shape)
                acc = fs_dmax.get(ck)
                fs_dmax[ck] = z if acc is None else np.maximum(acc, z)
            for c in [int(x) for x in p.attrs["cam_ids"]]:
                depth = p[f"img/camera_{c}/depth"][fr].astype(np.float32)
                ck = (c, *depth.shape)
                vm = np.unpackbits(p[f"img/camera_{c}/valid_mask"][fr])[: depth.size].reshape(depth.shape).astype(bool)
                Kir = p[f"cam/camera_{c}/K_ir"][fr].astype(np.float64)
                E = p[f"cam/camera_{c}/E_cam2world"][fr].astype(np.float64)
                vy, vx = np.nonzero(vm)
                if len(vy):
                    ss = slice(None, None, 23)
                    z = depth[vy[ss], vx[ss]]
                    ok = (z > 0.05) & (z < 3.0)
                    x = (vx[ss][ok] - Kir[0, 2]) / Kir[0, 0] * z[ok]
                    y = (vy[ss][ok] - Kir[1, 2]) / Kir[1, 1] * z[ok]
                    pw = (E[:3, :3] @ np.stack([x, y, z[ok]], 1).T).T + E[:3, 3]
                    z_samples.append(pw[:, 2].astype(np.float32))
                nv = never_valid.get(ck)
                never_valid[ck] = ~vm if nv is None else nv & ~vm
                if ck not in ref_geom:
                    g = p[f"render/camera_{c}"]
                    Vr = g["viewmat"][fr].astype(np.float64)
                    if np.isfinite(Vr).all():
                        jpg = np.asarray(g["jpg"][fr], dtype=np.uint8)
                        color = cv2.imdecode(jpg, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
                        ref_geom[ck] = (Kir, E, Vr, g["K"][fr].astype(np.float64), np.ascontiguousarray(color))
    z_table = float(np.percentile(np.concatenate(z_samples), 5.0)) if z_samples else float(bb[4])
    print(f"[bg-bank] table plane z = {z_table:.3f} (bbox z_min {bb[4]:.3f}, src=packed-render)", flush=True)
    tab_xyz, tab_rgb, far_xyz, far_rgb = ([], [], [], [])
    n_meas = n_fb = 0
    for ck, nv in never_valid.items():
        if ck not in ref_geom:
            continue
        c = ck[0]
        Kir, E, Vr, Krgb, color = ref_geom[ck]
        Hc, Wc = color.shape[:2]
        vy, vx = np.nonzero(nv)
        vy, vx = (vy[::stride], vx[::stride])
        d = np.stack([(vx - Kir[0, 2]) / Kir[0, 0], (vy - Kir[1, 2]) / Kir[1, 1], np.ones(len(vx))], 1)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        dw = (E[:3, :3] @ d.T).T
        C = E[:3, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_pl = (z_table - C[2]) / dw[:, 2]
        hit = C[None, :] + t_pl[:, None] * dw
        r2 = hit[:, 0] ** 2 + hit[:, 1] ** 2
        in_bb_xy = (hit[:, 0] > bb[0]) & (hit[:, 0] < bb[1]) & (hit[:, 1] > bb[2]) & (hit[:, 1] < bb[3])
        on_table = np.isfinite(t_pl) & (t_pl > 0.15) & (t_pl < far_dist) & (r2 < table_radius**2) & ~in_bb_xy
        pts = np.where(on_table[:, None], hit, C[None, :] + far_dist * dw)
        keep = np.ones(len(vx), bool)
        if ck in fs_dmax:
            zm = fs_dmax[ck][vy, vx]
            meas = zm > 0
            pm_cam = np.stack([(vx - Kir[0, 2]) / Kir[0, 0] * zm, (vy - Kir[1, 2]) / Kir[1, 1] * zm, zm], 1)
            pm = pm_cam @ E[:3, :3].T + C[None, :]
            pts = np.where(meas[:, None], pm, pts)
            in_bb3 = (
                meas
                & (pts[:, 0] > bb[0])
                & (pts[:, 0] < bb[1])
                & (pts[:, 1] > bb[2])
                & (pts[:, 1] < bb[3])
                & (pts[:, 2] > bb[4])
                & (pts[:, 2] < bb[5])
            )
            keep = ~in_bb3
            r2m = pts[:, 0] ** 2 + pts[:, 1] ** 2
            on_table = np.where(meas, (np.abs(pts[:, 2] - z_table) < 0.03) & (r2m < table_radius**2), on_table)
            n_meas += int((meas & keep).sum())
            n_fb += int((~meas & keep).sum())
        else:
            n_fb += len(vx)
        pc = (Vr[:3, :3] @ pts.T).T + Vr[:3, 3]
        zz = pc[:, 2].clip(1e-06)
        u = (Krgb[0, 0] * pc[:, 0] / zz + Krgb[0, 2]).astype(int)
        v = (Krgb[1, 1] * pc[:, 1] / zz + Krgb[1, 2]).astype(int)
        inb = (pc[:, 2] > 0.05) & (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc)
        col = np.full((len(pts), 3), 0.35, np.float32)
        col[inb] = color[v[inb], u[inb]]
        kt = on_table & keep
        kf = ~on_table & keep
        tab_xyz.append(pts[kt].astype(np.float32))
        tab_rgb.append(col[kt])
        far_xyz.append(pts[kf].astype(np.float32))
        far_rgb.append(col[kf])

    def _cat(a, b, grid, cap):
        if not a:
            return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))
        return _voxel_downsample(np.concatenate(a, 0), np.concatenate(b, 0), grid, cap, rng)

    tab_xyz, tab_rgb = _cat(tab_xyz, tab_rgb, near_grid, n_near)
    far_xyz, far_rgb = _cat(far_xyz, far_rgb, far_grid, n_far)
    xyz = np.concatenate([tab_xyz, far_xyz], 0)
    rgb01 = np.clip(np.concatenate([tab_rgb, far_rgb], 0), 0.0, 1.0)
    scale = np.concatenate(
        [np.full(len(tab_xyz), near_grid * 1.2, np.float32), np.full(len(far_xyz), far_grid * 1.2, np.float32)]
    )
    tot = max(n_meas + n_fb, 1)
    print(
        f"[bg-bank] init: {len(tab_xyz):,} near + {len(far_xyz):,} far = {len(xyz):,} gaussians (candidates: {n_meas:,} FS-measured [{100 * n_meas / tot:.0f}%], {n_fb:,} plane/shell fallback)",
        flush=True,
    )
    if len(xyz) == 0:
        raise RuntimeError("[bg-bank] init produced 0 gaussians — no finite render viewmat found in the sampled episodes")
    return (xyz, rgb01, scale)


class BackgroundGaussians(nn.Module):

    def __init__(self, xyz: np.ndarray, rgb01: np.ndarray, scale_m: np.ndarray, init_opacity_logit: float = 2.2):
        super().__init__()
        self.means = nn.Parameter(torch.from_numpy(np.asarray(xyz, np.float32)))
        self.log_scales = nn.Parameter(torch.log(torch.from_numpy(np.asarray(scale_m, np.float32)).clamp(0.0001)).unsqueeze(1))
        self.opacity_logit = nn.Parameter(torch.full((len(xyz),), float(init_opacity_logit)))
        r = torch.from_numpy(np.asarray(rgb01, np.float32)).clamp(0.02, 0.98)
        self.rgb_logit = nn.Parameter(torch.log(r / (1 - r)))

    max_scale: float = 0.2

    def extra_repr(self):
        return f"n={self.means.shape[0]:,} (isotropic)"

    def gaussians(self) -> dict:
        scale = torch.exp(self.log_scales.clamp(np.log(0.0001), np.log(self.max_scale))).expand(-1, 3)
        return dict(
            xyz=self.means, scale=scale, opacity=torch.sigmoid(self.opacity_logit), rgb=torch.sigmoid(self.rgb_logit) * 2.0 - 1.0
        )

    def load_resized(self, sd):
        with torch.no_grad():
            for name in ("means", "log_scales", "opacity_logit", "rgb_logit"):
                p = getattr(self, name)
                if p.shape != sd[name].shape:
                    p.data = torch.empty(sd[name].shape, dtype=p.dtype, device=p.device)
        self.load_state_dict(sd)

    @staticmethod
    def from_data(packed_files, bbox, n_near=90000, n_far=40000, **kw):
        xyz, rgb01, scale = build_background_init(packed_files, bbox, n_near=n_near, n_far=n_far, **kw)
        return BackgroundGaussians(xyz, rgb01, scale)
