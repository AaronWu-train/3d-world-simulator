import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.aloha_packed import AlohaPackedPCDataset
from datasets.frames import Frames, warp_depth
from models.background_bank import BackgroundGaussians
from models.big_head import BigHeadRoundShim
from models.build import add_tactile, make_vae, normalize_latent
from models.cached_robot_labels import expand_gripper_grids, load_frame, masked_robot_bce, resolve_robot_labels
from models.class_seeds import class_masks, partition_points, select_class_seeds, slice_head_out
from models.gsplat_render_loss import free_space_violators, render_depth_loss, render_round_loss
from models.losses import (
    compute_object_bce_addon,
    compute_object_rgb_addon,
    compute_occupancy_loss,
    compute_occupancy_targets,
    compute_rgb_target,
    rasterize_object_voxel_masks_multi,
)
from models.seed_upsample_two_round import two_round_chamfer_loss
from models.tactile import PAD_SAMPLES_PER_PAD, dense_pad, fire_color, rasterise, surface_loss
from tasks import TASKS

N_POINTS = 673000
TACTILE_CACHE = {}


def encode(model, xyz, rgb, pos_mean, pos_std):
    return normalize_latent(model.encode(xyz, rgb, pos_mean=pos_mean, pos_std=pos_std)[0], model.latent_split)


def tactile_targets(root, ep):
    key = (str(root), ep)
    if key not in TACTILE_CACHE:
        p = Path(root) / "tactile_targets" / f"episode_{ep}.npz"
        if len(TACTILE_CACHE) > 64:
            TACTILE_CACHE.clear()
        TACTILE_CACHE[key] = dict(np.load(p)) if p.exists() else None
    return TACTILE_CACHE[key]


def tactile_active_alpha(epoch):
    return 10.0 if epoch < 280 else 10.0 * max(0.0, 1.0 - (epoch - 279) / 10)


def tactile_loss(m, out, tac, train, epoch):
    model, device = m.model, m.device
    tac_X, tac_S, tac_Xp, tac_Sp = tac
    bmin, bext = (
        model.bounds_min_world.detach().cpu().numpy(),
        (model.bounds_max_world - model.bounds_min_world).detach().cpu().numpy(),
    )
    X, S = tac_X.cpu().numpy(), tac_S.cpu().numpy()
    occ, sig = torch.zeros((), device=device), torch.zeros((), device=device)
    n = len(model.grid_shapes)
    weights = [0.5] * (n - 1) + [1.0]
    for i, shape in enumerate([(s[2], s[1], s[0]) for s in model.grid_shapes]):
        mask, s = rasterise(X, S, bmin, bext, shape)
        mask, s = torch.from_numpy(mask).to(device), torch.from_numpy(s).to(device)
        lo, ls = out[f"tactile_occ_{i}"][0, 0].float(), out[f"tactile_sig_{i}"][0, 0].float()
        mf = mask.float()
        npos = mf.sum().clamp(min=1.0)
        occ = occ + F.binary_cross_entropy_with_logits(lo, mf, pos_weight=((mf.numel() - npos) / npos).clamp(max=10.0))
        if mask.any():
            w = 1.0 + tactile_active_alpha(epoch) * (s[mask] > 0.05).float()
            sig = sig + weights[i] * ((torch.sigmoid(ls[mask]) - s[mask]).abs() * w).sum() / w.sum()
    loss = 20.0 * occ + 5.0 * sig
    use_gt = not train or torch.rand(1).item() < 0.5
    seed_mask = mask if use_gt or not bool((lo > 0).any()) else lo.detach() > 0
    zyx = torch.nonzero(seed_mask)
    if zyx.shape[0] > 3000:
        zyx = zyx[torch.randperm(zyx.shape[0], device=device)[:3000]]
    if zyx.shape[0] > 0:
        seeds = torch.cat([torch.zeros_like(zyx[:, :1]), zyx], 1)
        vs = (model.bounds_max_world - model.bounds_min_world) / torch.tensor(
            model.grid_shapes[-1], device=device, dtype=model.bounds_min_world.dtype
        )
        logits = torch.cat([out[f"tactile_occ_{n - 1}"], out[f"tactile_sig_{n - 1}"]], 1).float()
        P, SL, KL = model.tactile_particle_head(
            m.features["feat_2"].float().detach(), logits.detach(), seeds, model.bounds_min_world.float(), vs.float()
        )
        Y, YS = dense_pad(tac_Xp, tac_Sp, PAD_SAMPLES_PER_PAD)
        loss = loss + surface_loss(P, SL, KL, seeds, Y, YS, model.bounds_min_world.float(), vs.float(), 2.0)
    return loss


def seed_chamfer(m, seeds, seed_cls, r1, r2, xyz_world, rgb, obj, obj_grids, gripper_grids, robot_grid, chamfer_rgb):
    model, device = m.model, m.device
    nx, ny, nz = model.grid_shapes[-1]
    bmin, vsize = model.bounds_min_world, (model.bounds_max_world - model.bounds_min_world) / torch.tensor(
        [nx, ny, nz], device=device, dtype=model.bounds_min_world.dtype
    )
    z = (seeds[:, 0], 0, seeds[:, 1], seeds[:, 2], seeds[:, 3])
    weights = torch.maximum(
        torch.ones(seeds.shape[0], device=device) + 9.0 * obj_grids[-1][z].float(), gripper_grids[-1][z].float() * 20.0
    )
    labels = partition_points(xyz_world, obj, robot_grid, bmin, vsize)
    cd = torch.zeros((), device=device)
    for c in (0, 1, 2):
        ms, mp = seed_cls == c, labels == c
        if int(ms.sum()) == 0 or int(mp.sum()) == 0:
            continue
        r1c, r2c = slice_head_out(r1, r2, ms)
        loss = two_round_chamfer_loss(
            seeds[ms],
            (r1c["xyz"], r1c["rgb"], r2c["xyz"], r2c["rgb"]),
            xyz_world[:, mp[0]],
            rgb[:, mp[0]],
            bmin,
            vsize,
            fine_shape=(nz, ny, nx),
            rgb_weight=chamfer_rgb,
            kernel=3,
            seed_weights=weights[ms],
            round2_weight=0.0,
            bwd_weight=1.0,
        )[0]
        cd = cd + float(ms.sum()) / float(seeds.shape[0]) * loss
    return cd


def render_losses(m, fg, rcams, dcams):
    if dcams:
        viol = free_space_violators(fg["xyz"].detach().float(), dcams, 0.03)
        fg["rgb"] = torch.where(viol[:, None], fg["rgb"].detach(), fg["rgb"])
    bg = m.bank.gaussians() if len(m.bank.means) else None
    l1, lp, ssim, _, _ = render_round_loss(
        fg,
        rcams,
        keep_grad=False,
        lpips_fn=m.lpips,
        gripper_lpips_weight=2.0,
        obj_alpha=9.0,
        obj_band_px=10,
        gripper_weight=20.0,
        band_outside_valid=False,
        use_ssim=True,
        bg=bg,
    )
    carve = depth = torch.zeros((), device=m.device)
    if dcams:
        nv = viol.sum()
        if int(nv) > 0:
            carve = (fg["opacity"] * viol.to(fg["opacity"].dtype)).sum() / nv
        gts = warp_depth(dcams, rcams)
        for c, gt in zip(rcams, gts):
            gt[~c["valid"]] = 0.0
        depth = render_depth_loss(fg, [dict(K=c["K"], viewmat=c["viewmat"], gt=gt) for c, gt in zip(rcams, gts)], bg=bg)[0]
    return carve, depth, l1 + 0.5 * ssim + 1.0 * lp


def step(m, batch, train, seed_active, frames, epoch, chamfer, chamfer_rgb, hold):
    model, device = m.model, m.device
    m.features.pop("feat_2", None)
    xyz, rgb, obj = (batch[k].to(device, non_blocking=True) for k in ("xyz", "rgb", "is_rope"))
    B, N, _ = xyz.shape
    root, ep, frame, fi = batch["root"][0], int(batch["ep_idx"][0]), int(batch["frame_idx"][0]), int(batch["file_idx"][0])
    if N >= N_POINTS:
        n = int(torch.randint(N_POINTS, min(N_POINTS, N) + 1, (1,)).item())
        idx = torch.randperm(N, device=device)[:n]
        xyz, rgb, obj = xyz[:, idx], rgb[:, idx], obj[:, idx]
    half = torch.randperm(xyz.shape[1], device=device)[: max(1, int(round(xyz.shape[1] * 0.5)))]
    xyz2, rgb2 = xyz[:, half], rgb[:, half]
    pmean, pstd = (torch.as_tensor(p, device=device, dtype=xyz.dtype) for p in (m.pos_mean, m.pos_std))
    xyz_world = xyz * pstd + pmean
    tac = None
    xyz_enc, rgb_enc = xyz, rgb
    tt = tactile_targets(root, ep) if m.tactile else None
    if tt is not None and bool(tt["has_tac"]):
        f = min(frame, len(tt["xyz"]) - 1)
        tac_X, tac_S = torch.from_numpy(tt["xyz"][f].reshape(-1, 3)).to(device), torch.from_numpy(tt["sig"][f].reshape(-1)).to(
            device
        )
        tac = tac_X, tac_S, torch.from_numpy(tt["xyz"][f]).to(device).float(), torch.from_numpy(tt["sig"][f]).to(device).float()
        xi, si = tac_X, tac_S
        active = tac_S > 0.05
        if active.any():
            xr = tac_X[active].repeat(3, 1)
            xi = torch.cat([tac_X, xr + 0.0005 * torch.randn_like(xr)])
            si = torch.cat([tac_S, tac_S[active].repeat(3)])
        xyz_enc = torch.cat([xyz, ((xi - pmean) / pstd)[None].to(xyz.dtype)], 1)
        rgb_enc = torch.cat([rgb, (fire_color(si) * 2 - 1)[None].to(rgb.dtype)], 1)
    mean = encode(model, xyz_enc, rgb_enc, m.pos_mean, m.pos_std)
    z = mean + torch.randn_like(mean) * (torch.rand(1).item() * 0.03) if train else mean
    with torch.no_grad():
        occ_mask = model.encoder._compute_bottleneck_occ_mask(xyz_world.reshape(-1, 3), batch_size=B)
    out = model.decode(z, occ_mask_0=occ_mask, use_gt_mask=True)
    levels = len(model.grid_shapes)
    occ_logits, robot_logits, object_logits = (
        [out[f"{k}_{i}"] for i in range(levels)] for k in ("scene_occ", "tac_occ", "signal")
    )
    bmin, bmax = model.bounds_min_world, model.bounds_max_world
    occ_targets = compute_occupancy_targets(xyz_world, bmin, bmax, model.grid_shapes)
    nx, ny, nz = model.grid_shapes[-1]
    rgb_target = compute_rgb_target(xyz_world, rgb, bmin, bmax, nx, ny, nz)
    object_grids = rasterize_object_voxel_masks_multi(
        xyz_world, obj.bool(), m.bbox_min, m.bbox_extent, [(s[2], s[1], s[0]) for s in model.grid_shapes]
    )
    if m.semantic_supervision:
        labels = load_frame(Path(root) / "robot_label_cache", ep, frame, model.grid_shapes, bmin, bmax, device)
    else:
        # Unlabelled monocular reconstructions: empty regions select the generic
        # scene branch, but MUST NOT become negative semantic training labels.
        labels = {kind: [torch.zeros_like(o, dtype=torch.bool) for o in occ_targets]
                  for kind in ("robot", "gripper")}
    gripper_grids = expand_gripper_grids(labels["gripper"], 6)
    robot_core, robot_valid = [], []
    for level, (r, o, occ) in enumerate(zip(labels["robot"], object_grids, occ_targets)):
        heights = bmin[2] + torch.arange(r.shape[-3], device=device) * (bmax[2] - bmin[2]) / r.shape[-3]
        halo = (6 * model.grid_shapes[level][0] + nx - 1) // nx
        res = resolve_robot_labels(
            r, o, occ > 0.5, object_halo_vox=0, robot_halo_vox=halo, table_cells=(heights <= -0.06)[None, None, :, None, None]
        )
        robot_core.append(res[0])
        robot_valid.append(res[1])
    obj_grids = [(F.max_pool3d(o.float(), 3, 1, 1) > 0.5) for o in object_grids]
    obj_grids = [o & ~g if o.dtype == torch.bool else o * ~g for o, g in zip(obj_grids, gripper_grids)]
    loss = compute_occupancy_loss(occ_logits, occ_targets, out["rgb_2"], rgb_target, occ_targets[-1], rgb_weight=1.0)[0]
    loss = loss + compute_object_bce_addon(occ_logits, occ_targets, obj_grids, 9.0)
    loss = loss + compute_object_rgb_addon(out["rgb_2"], rgb_target, occ_targets[-1], obj_grids[-1], 9.0)
    loss = loss + compute_object_bce_addon(occ_logits, occ_targets, gripper_grids, 19.0)
    loss = loss + compute_object_rgb_addon(out["rgb_2"], rgb_target, occ_targets[-1], gripper_grids[-1], 19.0)
    robot_loss = (
        sum(masked_robot_bce(l, c.float().expand(B, -1, -1, -1, -1), v) for l, c, v in zip(robot_logits, robot_core, robot_valid))
        / levels
    )
    if m.semantic_supervision:
        loss = loss + 10.0 * robot_loss
    cloth = 0.0
    for logit, g in zip(object_logits, object_grids):
        t = g.float().unsqueeze(1) if g.dim() == 4 else g.float()
        pw = ((t.numel() - t.sum()) / t.sum().clamp(min=1.0)).clamp(max=20.0)
        cloth = cloth + F.binary_cross_entropy_with_logits(logit, t, pos_weight=pw.expand_as(logit))
    if m.semantic_supervision:
        loss = loss + 10.0 * cloth / levels
    if tac is not None:
        loss = loss + tactile_loss(m, out, tac, train, epoch)
    carve = depth = render = torch.zeros((), device=device)
    if seed_active:
        use_gt = train and torch.rand(1).item() < 0.5
        seeds, seed_cls = select_class_seeds(
            class_masks(
                occ_logits[-1].detach(),
                robot_logits[-1].detach(),
                object_logits[-1].detach(),
                cloth_dilate=1,
                gt_cloth=object_grids[-1] if use_gt or not m.semantic_supervision else None,
                gt_robot=robot_core[-1] if use_gt or not m.semantic_supervision else None,
            ),
            max_seeds=100000,
        )
        if seeds.shape[0] > 0:
            vsize = (bmax - bmin) / torch.tensor([nx, ny, nz], device=device, dtype=bmin.dtype)
            r1, r2 = m.head(m.features["feat_2"], seeds, bmin, vsize, seed_cls=seed_cls)
            loss = loss + chamfer * seed_chamfer(
                m, seeds, seed_cls, r1, r2, xyz_world, rgb, obj, obj_grids, gripper_grids, robot_core[-1], chamfer_rgb
            )
            rcams = frames.render(fi, frame)
            # Monocular predictions have no measured free-space guarantee. They
            # supervise point reconstruction, never stereo carving or depth loss.
            dcams = [] if m.estimated_depth else frames.depth(fi, frame)
            if train and hold:
                rcams = [c for c in rcams if c["cid"] not in hold]
                dcams = [c for c in dcams if c["cid"] not in hold]
            fg = {
                k: r2[k].reshape(-1, *s)
                for k, s in dict(xyz=(3,), scale=(3,), opacity=(), rgb=(3,), quat=(4,), sh1=(3, 3)).items()
                if k in r2
            }
            carve, depth, render = render_losses(m, fg, rcams, dcams)
            loss = loss + 2.0 * carve + 2.0 * depth + 4.0 * render
    mean2 = encode(model, xyz2, rgb2, m.pos_mean, m.pos_std)
    cons = (
        F.mse_loss(mean[occ_mask.expand_as(mean)], mean2[occ_mask.expand_as(mean2)])
        if occ_mask.any()
        else F.mse_loss(mean, mean2)
    )
    loss = loss + 100.0 * cons
    return loss, float((loss - 2.0 * carve - 2.0 * depth - 4.0 * render).detach())


def run_epoch(m, loader, frames, epoch, train, seed_active, vae_eval, chamfer, chamfer_rgb, hold, n_iters):
    (m.model.eval if vae_eval or not train else m.model.train)()
    (m.head.train if train else m.head.eval)()
    pool = ThreadPoolExecutor(max_workers=1)
    it, total, n = iter(loader), 0.0, 0
    for _ in range(n_iters):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        pool.submit(frames.warm, int(batch["file_idx"][0]), int(batch["frame_idx"][0]))
        with torch.enable_grad() if train else torch.no_grad():
            loss, core = step(m, batch, train, seed_active, frames, epoch, chamfer, chamfer_rgb, hold)
            total, n = total + core, n + 1
            if train:
                m.optimizer.zero_grad(set_to_none=True)
                if not bool(torch.isfinite(loss.detach())):
                    continue
                loss.backward()
                params = [p for g in m.optimizer.param_groups for p in g["params"]]
                if not bool(torch.isfinite(torch.nn.utils.clip_grad_norm_(params, 10.0))):
                    m.optimizer.zero_grad(set_to_none=True)
                    continue
                m.optimizer.step()
    pool.shutdown(wait=True)
    if train:
        m.scheduler.step()
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--n-epochs", type=int, default=500)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--val-iters", type=int, default=20)
    p.add_argument("--val-interval", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--phase-a", type=int, default=10)
    p.add_argument("--phase-b", type=int, default=10)
    p.add_argument("--wrist-start", type=int, default=70)
    p.add_argument("--bg-init-eps", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--norm-params", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    if a.seed is not None:
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
    T = TASKS[a.task]
    device = "cuda"
    out = Path(a.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    roots = [f"{a.data_root}/{d}" for d in (a.datasets or T["datasets"])]
    bounds = [[T["x_range"][0], T["y_range"][0], T["z_range"][0]], [T["x_range"][1], T["y_range"][1], T["z_range"][1]]]
    train_ds = AlohaPackedPCDataset([f"{r}/train" for r in roots], n_points=N_POINTS, point_bounds=bounds)
    val_ds = AlohaPackedPCDataset([f"{r}/val" for r in roots], n_points=N_POINTS, point_bounds=bounds)
    norm = Path(a.norm_params) if a.norm_params else out / "norm_params.npz"
    if norm.exists():
        d = np.load(norm)
        pos_mean, pos_std = d["pos_mean"], d["pos_std"]
    else:
        pos_mean, pos_std = train_ds.compute_norm_params()
        np.savez(out / "norm_params.npz", pos_mean=pos_mean, pos_std=pos_std)
    train_ds.pos_mean, train_ds.pos_std = val_ds.pos_mean, val_ds.pos_std = pos_mean, pos_std
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=a.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=a.num_workers, pin_memory=True)
    bbox_min, bbox_extent = train_ds.world_bounds()
    model = make_vae(T["grid"], T["x_range"], T["y_range"], T["z_range"]).to(device)
    head = BigHeadRoundShim(feat_dim=64, K=16, shape="aniso").to(device)
    features = {}
    model.decoder.up_finest.register_forward_hook(lambda m_, i, o: features.__setitem__("feat_2", o))
    if T["tactile"]:
        add_tactile(model)
        model.to(device)
    bb = tuple(float(v) for i in range(3) for v in (bbox_min[i], bbox_min[i] + bbox_extent[i]))
    if T.get("estimated_depth", False):
        # All visible pixels belong to the reconstructed scene. There is no
        # measured never-valid background or calibrated table plane to seed.
        bank = BackgroundGaussians(np.empty((0, 3), np.float32), np.empty((0, 3), np.float32), np.empty(0, np.float32)).to(device)
    else:
        bank = BackgroundGaussians.from_data(
            train_ds.files, bb, n_near=700000, n_far=300000, max_eps=a.bg_init_eps, near_grid=0.0015, far_grid=0.004
        ).to(device)
    bank.max_scale = 0.2
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=a.lr, weight_decay=0.0001)
    optimizer.add_param_group(dict(params=list(bank.parameters()), lr=0.002, weight_decay=0.0))

    def lr_at(epoch):
        if epoch < a.warmup:
            return (epoch + 1) / max(1, a.warmup)
        cos = 0.5 * (1 + np.cos(np.pi * min(1.0, (epoch - a.warmup) / max(1, a.n_epochs - a.warmup))))
        return (a.min_lr + (a.lr - a.min_lr) * cos) / a.lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    start, best = 0, float("inf")
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        head.load_state_dict(ck["seed_head_state_dict"])
        bank.load_resized(ck["bg_bank_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start, best = int(ck["epoch"]) + 1, float(ck["best_val_loss"])
        for _ in range(start):
            scheduler.step()
    train_frames, val_frames = Frames(train_ds.files, device), Frames(val_ds.files, device)
    lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    lpips_fn.requires_grad_(False)
    lpips_fn.to(memory_format=torch.channels_last)
    m = argparse.Namespace(
        model=model,
        head=head,
        bank=bank,
        lpips=lpips_fn,
        features=features,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        pos_mean=pos_mean,
        pos_std=pos_std,
        tactile=T["tactile"],
        semantic_supervision=T.get("semantic_supervision", True),
        estimated_depth=T.get("estimated_depth", False),
        bbox_min=torch.as_tensor(bbox_min, device=device),
        bbox_extent=torch.as_tensor(bbox_extent, device=device),
    )
    pa, pb = a.phase_a, a.phase_a + a.phase_b
    val = None
    for epoch in range(start, a.n_epochs):
        phase_b, chamfer = pa <= epoch < pb, 1.0 if epoch < pb else 0.3
        model.requires_grad_(not phase_b)
        hold = set(T["wrist_cams"]) if epoch < a.wrist_start else None
        train = run_epoch(m, train_loader, train_frames, epoch, True, epoch >= pa, phase_b, chamfer, 0.1, hold, a.iters)
        if val is None or epoch % a.val_interval == 0 or epoch == a.n_epochs - 1:
            val = run_epoch(m, val_loader, val_frames, epoch, False, True, False, chamfer, 0.1, None, a.val_iters)
        print(f"[ep {epoch}] train {train:.4f} val {val:.4f} lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(dict(epoch=epoch, train=train, val=val)) + "\n")
        ck = dict(
            model_state_dict=model.state_dict(),
            seed_head_state_dict=head.state_dict(),
            new_head=head.export_spec(),
            bg_bank_state_dict=bank.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch,
            best_val_loss=best,
            pos_mean=pos_mean,
            pos_std=pos_std,
            args=dict(grid=T["grid"], x_range=T["x_range"], y_range=T["y_range"], z_range=T["z_range"],
                      semantic_supervision=T.get("semantic_supervision", True),
                      estimated_depth=T.get("estimated_depth", False), task=a.task),
        )
        torch.save(ck, out / "checkpoints" / "latest.pth")
        if val < best:
            best = ck["best_val_loss"] = val
            torch.save(ck, out / "checkpoints" / "best.pth")
        if (epoch + 1) % a.save_interval == 0:
            torch.save(ck, out / "checkpoints" / f"model_{epoch}.pth")


if __name__ == "__main__":
    main()
