from __future__ import annotations
import torch
import torch.nn.functional as F

try:
    from gsplat import rasterization
except ImportError as e:
    raise ImportError("gsplat is required for the seed-Gaussian rendering loss. Install with: pip install gsplat") from e


def render_gaussians(
    means_world: torch.Tensor,
    scales_world: torch.Tensor,
    opacities: torch.Tensor,
    rgb01: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    near_plane: float = 0.05,
    far_plane: float = 10.0,
    quats: torch.Tensor | None = None,
    sh1: torch.Tensor | None = None,
    packed: bool | None = None,
    with_depth: bool = False,
):
    N = means_world.shape[0]
    if quats is None:
        quats = torch.zeros(N, 4, device=means_world.device, dtype=means_world.dtype)
        quats[:, 0] = 1.0
    if sh1 is not None:
        C0 = 0.28209479177387814
        coeffs = torch.cat([((rgb01 - 0.5) / C0).unsqueeze(1), sh1], dim=1)
        colors, sh_deg = (coeffs, 1)
    else:
        colors, sh_deg = (rgb01, None)
    rendered, alpha, _ = rasterization(
        means=means_world,
        quats=quats,
        scales=scales_world,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_deg,
        near_plane=near_plane,
        far_plane=far_plane,
        rasterize_mode="antialiased",
        **{} if packed is None else dict(packed=packed),
        **dict(render_mode="RGB+ED") if with_depth else {},
    )
    if with_depth:
        return (rendered[..., :3], alpha, rendered[..., 3:4])
    return (rendered, alpha)


def depth_composite(fg, alpha, fg_depth, bg, bg_depth, bg_alpha, margin=0.02, tau=0.005):
    over = fg + (1.0 - alpha) * bg
    with torch.no_grad():
        front = torch.sigmoid((fg_depth - bg_depth[..., None] - margin) / tau) * (alpha > 0.001) * bg_alpha[..., None]
    return (1.0 - front) * over + front * bg


def render_l1_loss(rendered: torch.Tensor, target_rgbs: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    err = (rendered - target_rgbs).abs()
    if valid_mask is None:
        return err.mean()
    err = err.mean(dim=-1)
    m = valid_mask.to(err.dtype)
    return (err * m).sum() / m.sum().clamp(min=1.0)


def _dilate(mask: torch.Tensor, px: int) -> torch.Tensor:
    k = 2 * px + 1
    mf = mask.float().unsqueeze(0).unsqueeze(0)
    mf = F.max_pool2d(mf, kernel_size=k, stride=1, padding=px)
    return mf.squeeze(0).squeeze(0) > 0.5


_ssim_win_cache = {}


def _gaussian_window(ws: int, sigma: float, device, dtype):
    key = (ws, sigma, device, dtype)
    w = _ssim_win_cache.get(key)
    if w is None:
        c = torch.arange(ws, device=device, dtype=dtype) - ws // 2
        g = torch.exp(-(c**2) / (2 * sigma**2))
        g = g / g.sum()
        w = (g[:, None] @ g[None, :]).view(1, 1, ws, ws).expand(3, 1, ws, ws).contiguous()
        _ssim_win_cache[key] = w
    return w


def ssim_masked_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor | None = None,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    x = rendered.permute(0, 3, 1, 2)
    y = target.permute(0, 3, 1, 2)
    w = _gaussian_window(window_size, sigma, x.device, x.dtype)
    pad = window_size // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=3)
    mu_y = F.conv2d(y, w, padding=pad, groups=3)
    mu_x2, mu_y2, mu_xy = (mu_x * mu_x, mu_y * mu_y, mu_x * mu_y)
    sig_x = F.conv2d(x * x, w, padding=pad, groups=3) - mu_x2
    sig_y = F.conv2d(y * y, w, padding=pad, groups=3) - mu_y2
    sig_xy = F.conv2d(x * y, w, padding=pad, groups=3) - mu_xy
    C1, C2 = (0.01**2, 0.03**2)
    ssim = (2 * mu_xy + C1) * (2 * sig_xy + C2) / ((mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2))
    dssim = ((1.0 - ssim) * 0.5).mean(dim=1)
    if weight_map is None:
        return dssim.mean()
    m = weight_map.to(dssim.dtype)
    return (dssim * m).sum() / m.sum().clamp(min=1.0)


def concat_gaussians(fg: dict, bg: dict) -> dict:
    out = {}
    has_quat = "quat" in fg or "quat" in bg
    for k in ("xyz", "scale", "opacity", "rgb"):
        out[k] = torch.cat([fg[k], bg[k]], dim=0)
    if has_quat:

        def _q(d):
            if "quat" in d:
                return d["quat"]
            q = torch.zeros(d["xyz"].shape[0], 4, device=d["xyz"].device, dtype=d["xyz"].dtype)
            q[:, 0] = 1.0
            return q

        out["quat"] = torch.cat([_q(fg), _q(bg)], dim=0)
    if "sh1" in fg or "sh1" in bg:

        def _s(d):
            return d["sh1"] if "sh1" in d else torch.zeros(d["xyz"].shape[0], 3, 3, device=d["xyz"].device, dtype=d["xyz"].dtype)

        out["sh1"] = torch.cat([_s(fg), _s(bg)], dim=0)
    return out


def render_depth_loss(gaussians: dict, depth_cams: list, bg: dict | None = None, gt_minpool: int = 1, min_alpha: float = 0.05):
    g = concat_gaussians(gaussians, bg) if bg is not None else gaussians
    N = g["xyz"].shape[0]
    quats = g.get("quat")
    if quats is None:
        quats = torch.zeros(N, 4, device=g["xyz"].device, dtype=g["xyz"].dtype)
        quats[:, 0] = 1.0
    rgb01 = (g["rgb"].clamp(-1, 1) + 1) * 0.5
    total = g["xyz"].new_zeros(())
    n = 0
    rends = []
    for dc in depth_cams:
        gt = dc["gt"]
        H, W = gt.shape
        if gt_minpool > 0:
            k = 2 * gt_minpool + 1
            gmin = gt.clone()
            gmin[gt <= 0] = float("inf")
            gmin = -F.max_pool2d((-gmin).unsqueeze(0).unsqueeze(0), k, 1, gt_minpool)[0, 0]
            gt = torch.where(torch.isfinite(gmin), gmin, torch.zeros_like(gmin))
        rend, alpha, _ = rasterization(
            means=g["xyz"],
            quats=quats,
            scales=g["scale"],
            opacities=g["opacity"],
            colors=rgb01,
            viewmats=dc["viewmat"].unsqueeze(0),
            Ks=dc["K"].unsqueeze(0),
            width=W,
            height=H,
            sh_degree=None,
            near_plane=0.05,
            far_plane=10.0,
            rasterize_mode="antialiased",
            render_mode="ED",
        )
        D = rend[0, ..., -1]
        a = alpha[0, ..., 0]
        if dc.get("bg_depth") is not None:
            D = a * D + (1.0 - a) * dc["bg_depth"]
            a = a + (1.0 - a) * dc["bg_alpha"]
        m = (gt > 0) & (a > min_alpha)
        if m.any():
            total = total + (D - gt).abs()[m].mean()
            n += 1
        rends.append((D.detach(), a.detach(), gt))
    return (total / max(n, 1), rends)


@torch.no_grad()
def free_space_violators(xyz_world: torch.Tensor, depth_cams: list, margin: float) -> torch.Tensor:
    N = xyz_world.shape[0]
    viol = torch.zeros(N, dtype=torch.bool, device=xyz_world.device)
    for dc in depth_cams:
        vm, K, dmin = (dc["viewmat"], dc["K"], dc["dmin"])
        H, W = dmin.shape
        pc = (vm[:3, :3] @ xyz_world.T + vm[:3, 3:]).T
        z = pc[:, 2]
        zs = z.clamp(min=1e-06)
        u = K[0, 0] * pc[:, 0] / zs + K[0, 2]
        v = K[1, 1] * pc[:, 1] / zs + K[1, 2]
        inb = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        D = dmin[v.long().clamp(0, H - 1), u.long().clamp(0, W - 1)]
        viol |= inb & torch.isfinite(D) & (D - z > margin)
    return viol


LPIPS_BATCHED = False
PERCEP_AUTOCAST = True


def lpips_masked_loss(rendered: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, lpips_fn) -> torch.Tensor:
    m = valid_mask.to(rendered.dtype).unsqueeze(-1)
    r = (rendered * m).permute(0, 3, 1, 2) * 2.0 - 1.0
    t = (target * m).permute(0, 3, 1, 2) * 2.0 - 1.0
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=PERCEP_AUTOCAST):
        return lpips_fn(r, t).float().mean()


def gripper_weight_map(base_weight, gripper_mask, valid, multiplier=1.0):
    if gripper_mask is None or multiplier <= 1:
        return base_weight
    region = gripper_mask.bool() & valid.bool()
    target = region.to(base_weight.dtype) * float(multiplier)
    return torch.maximum(base_weight, target.unsqueeze(0))


def gripper_crop_items(rendered, target, gripper_mask, valid):
    if gripper_mask is None:
        return []
    import cv2
    import numpy as np

    m = (gripper_mask.bool() & valid.bool()).detach().cpu().numpy().astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    items = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 32:
            continue
        x0, y0 = (max(0, x - 8), max(0, y - 8))
        x1, y1 = (min(m.shape[1], x + w + 8), min(m.shape[0], y + h + 8))
        a, b = (rendered[:, y0:y1, x0:x1], target[:, y0:y1, x0:x1])
        mask = valid[y0:y1, x0:x1][None]
        ph, pw = (max(0, 32 - a.shape[1]), max(0, 32 - a.shape[2]))
        if ph or pw:
            a = F.pad(a.permute(0, 3, 1, 2), (0, pw, 0, ph)).permute(0, 2, 3, 1)
            b = F.pad(b.permute(0, 3, 1, 2), (0, pw, 0, ph)).permute(0, 2, 3, 1)
            mask = F.pad(mask, (0, pw, 0, ph))
        items.append((a, b, mask))
    return items


def lpips_batched(items, lpips_fn):
    H = max((a.shape[1] for (a, _, _) in items))
    W = max((a.shape[2] for (a, _, _) in items))
    A, B, M, sc = ([], [], [], [])
    for a, b, m in items:
        ph, pw = (H - a.shape[1], W - a.shape[2])
        sc.append(float(H * W) / float(a.shape[1] * a.shape[2]))
        if ph or pw:
            a = F.pad(a.permute(0, 3, 1, 2), (0, pw, 0, ph)).permute(0, 2, 3, 1)
            b = F.pad(b.permute(0, 3, 1, 2), (0, pw, 0, ph)).permute(0, 2, 3, 1)
            m = F.pad(m, (0, pw, 0, ph))
        A.append(a)
        B.append(b)
        M.append(m)
    A, B, M = (torch.cat(A), torch.cat(B), torch.cat(M))
    mm = M.to(A.dtype).unsqueeze(-1)
    r = (A * mm).permute(0, 3, 1, 2) * 2.0 - 1.0
    t = (B * mm).permute(0, 3, 1, 2) * 2.0 - 1.0
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=PERCEP_AUTOCAST):
        v = lpips_fn(r, t).float().reshape(-1)
    return v * torch.tensor(sc, device=v.device, dtype=v.dtype)


def gripper_crop_lpips(rendered, target, gripper_mask, valid, lpips_fn):
    if gripper_mask is None or lpips_fn is None:
        return rendered.sum() * 0.0
    terms = [
        lpips_masked_loss(a, b, mask, lpips_fn) for (a, b, mask) in gripper_crop_items(rendered, target, gripper_mask, valid)
    ]
    return torch.stack(terms).mean() if terms else rendered.sum() * 0.0


def render_round_loss(
    gaussians: dict,
    cams: list,
    lpips_fn=None,
    obj_alpha: float = 0.0,
    obj_band_px: int = 0,
    band_outside_valid: bool = True,
    use_ssim: bool = False,
    bg: dict | None = None,
    gripper_weight: float = 1.0,
    gripper_lpips_weight: float = 0.0,
    keep_grad: bool = False,
):
    g = concat_gaussians(gaussians, bg) if bg is not None else gaussians
    rgb01 = (g["rgb"].clamp(-1, 1) + 1) * 0.5
    rl1 = g["xyz"].new_zeros(())
    lp = g["xyz"].new_zeros(())
    dss = g["xyz"].new_zeros(())
    rends, alphas = ([], [])
    _lp_full, _lp_grip = ([], [])
    for cam in cams:
        _dt = cam.get("bg_img") is not None and cam.get("bg_depth") is not None
        _r = render_gaussians(
            g["xyz"],
            g["scale"],
            g["opacity"],
            rgb01,
            cam["viewmat"].unsqueeze(0),
            cam["K"].unsqueeze(0),
            cam["W"],
            cam["H"],
            quats=g.get("quat"),
            sh1=g.get("sh1"),
            with_depth=_dt,
        )
        rendered, alpha = (_r[0], _r[1])
        if _dt:
            rendered = depth_composite(rendered, alpha, _r[2], cam["bg_img"], cam["bg_depth"], cam["bg_alpha"])
        elif cam.get("bg_img") is not None:
            rendered = rendered + (1.0 - alpha) * cam["bg_img"].unsqueeze(0)
        valid = cam["valid"].unsqueeze(0)
        if obj_alpha > 0 and cam.get("obj") is not None:
            objd = _dilate(cam["obj"], obj_band_px) if obj_band_px > 0 else cam["obj"]
            support = cam["valid"] | objd if band_outside_valid else cam["valid"]
            wmap = (support.to(rendered.dtype) * (1.0 + obj_alpha * objd.to(rendered.dtype))).unsqueeze(0)
        else:
            wmap = valid
        wmap = gripper_weight_map(wmap, cam.get("gripper"), cam["valid"], gripper_weight)
        rl1 = rl1 + render_l1_loss(rendered, cam["rgb"].unsqueeze(0), valid_mask=wmap)
        if lpips_fn is not None and LPIPS_BATCHED:
            _lp_full.append((rendered, cam["rgb"].unsqueeze(0), valid))
            if gripper_lpips_weight > 0:
                _gi = gripper_crop_items(rendered, cam["rgb"].unsqueeze(0), cam.get("gripper"), cam["valid"])
                _lp_grip.append(_gi)
        elif lpips_fn is not None:
            lp = lp + lpips_masked_loss(rendered, cam["rgb"].unsqueeze(0), valid, lpips_fn)
            if gripper_lpips_weight > 0:
                lp = lp + gripper_lpips_weight * gripper_crop_lpips(
                    rendered, cam["rgb"].unsqueeze(0), cam.get("gripper"), cam["valid"], lpips_fn
                )
        if use_ssim:
            dss = dss + ssim_masked_loss(rendered, cam["rgb"].unsqueeze(0), weight_map=wmap.squeeze(0).unsqueeze(0))
        rends.append(rendered[0] if keep_grad else rendered[0].detach())
        alphas.append(alpha[0].detach())
    if _lp_full:
        if len({(a.shape[1], a.shape[2]) for (a, _, _) in _lp_full}) == 1:
            lp = lp + lpips_batched(_lp_full, lpips_fn).sum()
        else:
            for a, b, m in _lp_full:
                lp = lp + lpips_masked_loss(a, b, m, lpips_fn)
        _flat = [it for g in _lp_grip for it in g]
        if _flat:
            v = lpips_batched(_flat, lpips_fn)
            k = 0
            for g in _lp_grip:
                if g:
                    lp = lp + gripper_lpips_weight * v[k : k + len(g)].mean()
                    k += len(g)
    n = max(len(cams), 1)
    return (rl1 / n, lp / n, dss / n, rends, alphas)
