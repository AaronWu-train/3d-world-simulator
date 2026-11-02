from pathlib import Path
import h5py
import numpy as np
import torch


def load_frame(root, episode, frame, shapes, bmin, bmax, device):
    result = {}
    with h5py.File(Path(root) / f"episode_{episode}.h5", "r") as f:
        if not np.allclose(f.attrs["bounds"], torch.stack((bmin, bmax)).detach().cpu().numpy()):
            raise ValueError("Robot label cache bounds do not match VAE bounds")
        for kind in ("robot", "gripper"):
            grids = []
            by_shape = {tuple(f[f"{kind}/{lv}"].attrs["shape"]): lv for lv in f[kind]}
            for level, xyz in enumerate(shapes):
                zyx = tuple(reversed(xyz))
                if zyx not in by_shape:
                    raise ValueError(f"Robot label cache resolution does not match VAE: need {zyx}, cache has {list(by_shape)}")
                d = f[f"{kind}/{by_shape[zyx]}"]
                mask = np.unpackbits(d[frame], count=int(np.prod(zyx))).reshape(zyx).copy()
                grids.append(torch.from_numpy(mask).to(device=device, dtype=torch.bool)[None, None])
            result[kind] = grids
    return result


def resolve_robot_labels(robot_region, object_cells, occupied, object_halo_vox=1, robot_halo_vox=None, table_cells=None):
    import torch.nn.functional as F

    region = robot_region.bool()
    obj = object_cells.bool() & occupied.bool()
    occ = occupied.bool()
    radius = int(object_halo_vox)
    if radius < 0:
        raise ValueError("Object halo must be nonnegative")
    near_object = obj.float()
    for axis in range(3):
        kernel = [1, 1, 1]
        padding = [0, 0, 0]
        kernel[axis] = 2 * radius + 1
        padding[axis] = radius
        near_object = F.max_pool3d(near_object, tuple(kernel), 1, tuple(padding))
    near_object = near_object > 0
    near_robot = F.max_pool3d(region.float(), 3, 1, 1) > 0
    interior = F.max_pool3d((~region).float(), 3, 1, 1) == 0
    ambiguous = occ & ~obj & (near_robot & ~interior | region & near_object)
    if robot_halo_vox is not None:
        radius = int(robot_halo_vox)
        if radius < 0:
            raise ValueError("Robot halo must be nonnegative")
        expanded = region.float()
        for axis in range(3):
            kernel = [1, 1, 1]
            padding = [0, 0, 0]
            kernel[axis] = 2 * radius + 1
            padding[axis] = radius
            expanded = F.max_pool3d(expanded, tuple(kernel), 1, tuple(padding))
        interior = expanded > 0
        ambiguous = occ & ~obj & interior & near_object
    target = interior & occ & ~obj & ~ambiguous
    if table_cells is not None:
        target = target & ~table_cells.bool()
        ambiguous = ambiguous & ~table_cells.bool()
    valid = occ & ~ambiguous
    return (target, valid, ambiguous)


def masked_robot_bce(logit, target, valid):
    import torch.nn.functional as F

    t = target.to(logit.dtype)
    v = valid.to(logit.dtype)
    positive = (t * v).sum()
    negative = ((1 - t) * v).sum()
    pw = (negative / positive.clamp(min=1)).clamp(min=1, max=20)
    loss = F.binary_cross_entropy_with_logits(logit, t, pos_weight=pw, reduction="none")
    return (loss * v).sum() / v.sum().clamp(min=1)


def expand_gripper_grids(grids, finest_radius):
    import torch.nn.functional as F

    if finest_radius < 0:
        raise ValueError("Gripper halo must be nonnegative")
    result = []
    for grid in grids:
        radius = (finest_radius * grid.shape[-1] + grids[-1].shape[-1] - 1) // grids[-1].shape[-1]
        expanded = grid.float()
        for axis in range(3):
            kernel = [1, 1, 1]
            padding = [0, 0, 0]
            kernel[axis] = 2 * radius + 1
            padding[axis] = radius
            expanded = F.max_pool3d(expanded, tuple(kernel), 1, tuple(padding))
        result.append(expanded > 0)
    return result
