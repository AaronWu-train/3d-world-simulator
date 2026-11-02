import torch
import torch.nn.functional as F


def compute_occupancy_targets(xyz_world, bounds_min, bounds_max, shapes):
    B = xyz_world.shape[0]
    device = xyz_world.device
    extent = bounds_max - bounds_min
    targets = []
    for nx, ny, nz in shapes:
        spacing = extent / torch.tensor([nx, ny, nz], device=device, dtype=extent.dtype)
        grid = torch.zeros(B, 1, nz, ny, nx, device=device)
        for b in range(B):
            pts = xyz_world[b]
            normed = (pts - bounds_min) / spacing
            ix = normed[:, 0].long().clamp(0, nx - 1)
            iy = normed[:, 1].long().clamp(0, ny - 1)
            iz = normed[:, 2].long().clamp(0, nz - 1)
            grid[b, 0, iz, iy, ix] = 1.0
        targets.append(grid)
    return targets


def compute_rgb_target(xyz_world, rgb, bounds_min, bounds_max, nx, ny, nz):
    B = xyz_world.shape[0]
    device = xyz_world.device
    extent = bounds_max - bounds_min
    spacing = extent / torch.tensor([nx, ny, nz], device=device, dtype=extent.dtype)
    rgb_grid = torch.zeros(B, 3, nz, ny, nx, device=device)
    count_grid = torch.zeros(B, 1, nz, ny, nx, device=device)
    for b in range(B):
        pts = xyz_world[b]
        normed = (pts - bounds_min) / spacing
        ix = normed[:, 0].long().clamp(0, nx - 1)
        iy = normed[:, 1].long().clamp(0, ny - 1)
        iz = normed[:, 2].long().clamp(0, nz - 1)
        for c in range(3):
            rgb_grid[b, c].index_put_((iz, iy, ix), rgb[b, :, c], accumulate=True)
        count_grid[b, 0].index_put_((iz, iy, ix), torch.ones(len(ix), device=device), accumulate=True)
    rgb_grid = rgb_grid / count_grid.clamp(min=1)
    return rgb_grid


def compute_occupancy_loss(
    occ_logits_list,
    occ_targets_list,
    rgb_pred,
    rgb_target,
    occ_target_fine,
    level_weights=(1.0, 1.0, 1.0),
    rgb_weight=1.0,
    empty_rgb_target=1.0,
):
    device = occ_logits_list[0].device
    loss_dict = {}
    total = torch.tensor(0.0, device=device)
    level_weights = tuple(level_weights)[-len(occ_logits_list) :]
    for i, (logits, target, w) in enumerate(zip(occ_logits_list, occ_targets_list, level_weights)):
        n_pos = target.sum().clamp(min=1)
        n_neg = (1 - target).sum().clamp(min=1)
        pos_weight = (n_neg / n_pos).clamp(max=20)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight.expand_as(logits))
        total = total + w * bce
        loss_dict[f"occ_bce_{i}"] = bce.item()
    loss_dict["occ_bce"] = loss_dict.get("occ_bce_2", loss_dict.get("occ_bce_0", 0.0))
    occ_mask = occ_target_fine.expand_as(rgb_pred) > 0.5
    if occ_mask.any():
        rgb_loss = (rgb_pred[occ_mask] - rgb_target[occ_mask]).abs().mean()
    else:
        rgb_loss = torch.tensor(0.0, device=device)
    total = total + rgb_weight * rgb_loss
    loss_dict["rgb"] = rgb_loss.item()
    empty_mask = occ_target_fine.expand_as(rgb_pred) < 0.5
    if empty_mask.any():
        empty_target = torch.full_like(rgb_pred, empty_rgb_target)
        rgb_empty = (rgb_pred[empty_mask] - empty_target[empty_mask]).abs().mean()
    else:
        rgb_empty = torch.tensor(0.0, device=device)
    total = total + rgb_weight * 0.1 * rgb_empty
    loss_dict["rgb_empty"] = rgb_empty.item()
    loss_dict["total"] = total.item()
    return (total, loss_dict)


def rasterize_point_voxel_mask(xyz, mask, bbox_min, bbox_extent, grid_shape_zyx):
    B, M = mask.shape
    nz, ny, nx = grid_shape_zyx
    device = xyz.device
    dtype = xyz.dtype
    n_xyz = torch.tensor([nx, ny, nz], device=device, dtype=dtype)
    voxel_size = bbox_extent.to(device, dtype) / n_xyz
    bmn = bbox_min.to(device, dtype)
    out = torch.zeros(B, nz * ny * nx, device=device, dtype=torch.bool)
    for b in range(B):
        valid = mask[b]
        if not valid.any():
            continue
        p = xyz[b][valid]
        idx_f = torch.floor((p - bmn) / voxel_size).long()
        ix = idx_f[:, 0].clamp(0, nx - 1)
        iy = idx_f[:, 1].clamp(0, ny - 1)
        iz = idx_f[:, 2].clamp(0, nz - 1)
        flat = iz * (ny * nx) + iy * nx + ix
        out[b].scatter_(0, flat, torch.ones_like(flat, dtype=torch.bool))
    return out.view(B, 1, nz, ny, nx)


def rasterize_object_voxel_masks_multi(xyz_world, obj_mask, bbox_min, bbox_extent, grid_shapes_zyx):
    return [
        rasterize_point_voxel_mask(xyz_world, obj_mask, bbox_min, bbox_extent, (nz, ny, nx)) for (nz, ny, nx) in grid_shapes_zyx
    ]


def compute_object_bce_addon(occ_logits_list, occ_targets_list, obj_grids, obj_alpha, level_weights=(1.0, 1.0, 1.0)):
    if obj_alpha <= 0.0:
        return torch.tensor(0.0, device=occ_logits_list[0].device)
    device = occ_logits_list[0].device
    total = torch.tensor(0.0, device=device)
    for i, (logits, target, w) in enumerate(zip(occ_logits_list, occ_targets_list, level_weights)):
        if i >= len(obj_grids) or obj_grids[i] is None:
            continue
        n_pos = target.sum().clamp(min=1)
        n_neg = (1 - target).sum().clamp(min=1)
        pos_weight = (n_neg / n_pos).clamp(max=20)
        bce_per_voxel = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight.expand_as(logits), reduction="none"
        )
        obj_w = obj_grids[i].to(bce_per_voxel.dtype)
        total = total + w * obj_alpha * (bce_per_voxel * obj_w).mean()
    return total


def compute_object_rgb_addon(rgb_pred, rgb_target, occ_target_fine, obj_grid_fine, rgb_obj_alpha):
    if rgb_obj_alpha <= 0.0:
        return torch.tensor(0.0, device=rgb_pred.device)
    occ_mask = occ_target_fine.expand_as(rgb_pred) > 0.5
    if not occ_mask.any():
        return torch.tensor(0.0, device=rgb_pred.device)
    obj_w = obj_grid_fine.to(rgb_pred.dtype).expand_as(rgb_pred)
    err = (rgb_pred - rgb_target).abs() * obj_w
    return rgb_obj_alpha * err[occ_mask].mean()
