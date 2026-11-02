import torch, torch.nn.functional as F

N_CLASS = 3
CLOTH, ROBOT, TABLE = (0, 1, 2)


def class_masks(scene_logit, robot_logit, cloth_logit, cloth_dilate=1, gt_cloth=None, gt_robot=None):
    scene = scene_logit > 0.0
    cloth = gt_cloth.bool() if gt_cloth is not None else torch.sigmoid(cloth_logit) > 0.5
    robot = gt_robot.bool() if gt_robot is not None else torch.sigmoid(robot_logit) > 0.5
    table = scene & ~cloth & ~robot
    if cloth_dilate > 0:
        cloth = F.max_pool3d(cloth.float(), 2 * cloth_dilate + 1, 1, cloth_dilate) > 0.5
    cloth = cloth & (scene | (gt_cloth is not None and gt_cloth.bool()))
    robot = robot & scene
    return {CLOTH: cloth, ROBOT: robot, TABLE: table}


def select_class_seeds(masks, max_seeds=None, counts=None):
    seeds, cls = ([], [])
    for c, m in masks.items():
        s = m.squeeze(1).nonzero()
        if counts is not None:
            counts[c] = int(s.shape[0])
        if s.shape[0]:
            seeds.append(s)
            cls.append(torch.full((s.shape[0],), c, dtype=torch.long, device=s.device))
    if not seeds:
        return (
            torch.zeros((0, 4), dtype=torch.long, device=masks[CLOTH].device),
            torch.zeros((0,), dtype=torch.long, device=masks[CLOTH].device),
        )
    seeds = torch.cat(seeds)
    cls = torch.cat(cls)
    if max_seeds is not None and seeds.shape[0] > max_seeds:
        sel = torch.randperm(seeds.shape[0], device=seeds.device)[:max_seeds]
        seeds, cls = (seeds[sel], cls[sel])
    return (seeds, cls)


def partition_points(xyz_world, obj, robot_grid, bmin, vsize):
    idx = torch.floor((xyz_world - bmin) / vsize).long()
    D, H, W = robot_grid.shape[-3:]
    ok = (idx[..., 0] >= 0) & (idx[..., 0] < W) & (idx[..., 1] >= 0) & (idx[..., 1] < H) & (idx[..., 2] >= 0) & (idx[..., 2] < D)
    idc = idx.clamp(min=0)
    idc[..., 0] = idc[..., 0].clamp(max=W - 1)
    idc[..., 1] = idc[..., 1].clamp(max=H - 1)
    idc[..., 2] = idc[..., 2].clamp(max=D - 1)
    rob = torch.zeros(xyz_world.shape[:2], dtype=torch.bool, device=xyz_world.device)
    for b in range(xyz_world.shape[0]):
        rob[b] = robot_grid[b, 0, idc[b, :, 2], idc[b, :, 1], idc[b, :, 0]] > 0.5
    rob = rob & ok
    cls = torch.full(xyz_world.shape[:2], TABLE, dtype=torch.long, device=xyz_world.device)
    cls[rob] = ROBOT
    cls[obj.bool()] = CLOTH
    return cls


def slice_head_out(r1g, r2g, m):
    return (
        {k: v[m] if torch.is_tensor(v) and v.dim() > 0 and (v.shape[0] == m.shape[0]) else v for (k, v) in r1g.items()},
        {k: v[m] if torch.is_tensor(v) and v.dim() > 0 and (v.shape[0] == m.shape[0]) else v for (k, v) in r2g.items()},
    )
