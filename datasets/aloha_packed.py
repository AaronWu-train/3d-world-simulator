from __future__ import annotations
import glob, os
from pathlib import Path
import cv2
from datasets import color_gain as _color_gain
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _unpack_mask(packed_row, H, W):
    return np.unpackbits(packed_row)[: H * W].reshape(H, W).astype(bool)


def _render_obj_labels(f, c, t, pts_world):
    lab = np.zeros(len(pts_world), np.uint8)
    if len(pts_world) == 0:
        return lab
    g = f[f"render/camera_{c}"]
    Vr = g["viewmat"][t].astype(np.float32)
    if not np.isfinite(Vr).all():
        return lab
    hw = (int(g.attrs["H"]), int(g.attrs["W"]))
    m = np.unpackbits(g["obj_mask"][t])[: hw[0] * hw[1]].reshape(hw)
    K = g["K"][t].astype(np.float32)
    pc = pts_world @ Vr[:3, :3].T + Vr[:3, 3]
    z = pc[:, 2]
    ok = z > 0.01
    u = (K[0, 0] * pc[:, 0] / np.where(ok, z, 1) + K[0, 2]).astype(np.int64)
    v = (K[1, 1] * pc[:, 1] / np.where(ok, z, 1) + K[1, 2]).astype(np.int64)
    ok &= (u >= 0) & (u < hw[1]) & (v >= 0) & (v < hw[0])
    lab[ok] = m[v[ok], u[ok]]
    return lab


def _lift_camera(depth, valid, rgb_in_ir, obj, K_ir, E_c2w):
    vy, vx = np.nonzero(valid)
    if len(vy) == 0:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), np.zeros((0,), np.uint8))
    z = depth[vy, vx].astype(np.float32)
    fx, fy = (float(K_ir[0, 0]), float(K_ir[1, 1]))
    cx, cy = (float(K_ir[0, 2]), float(K_ir[1, 2]))
    x = (vx.astype(np.float32) - cx) * z / fx
    y = (vy.astype(np.float32) - cy) * z / fy
    pts_ir = np.stack([x, y, z], axis=-1)
    R = E_c2w[:3, :3].astype(np.float32)
    t = E_c2w[:3, 3].astype(np.float32)
    pts_world = (pts_ir @ R.T + t).astype(np.float32)
    rgb = rgb_in_ir[vy, vx]
    obj_lab = obj[vy, vx] if obj is not None else np.zeros(len(vy), np.uint8)
    return (pts_world, rgb, obj_lab)


class AlohaPackedPCDataset(Dataset):

    def __init__(
        self,
        packed_dir,
        *,
        n_points=380000,
        pos_mean=None,
        pos_std=None,
        use_sor_filter=True,
        wrist_e_override_dir=None,
        wrist_cam=3,
        point_bounds=None,
    ):
        dirs = packed_dir if isinstance(packed_dir, (list, tuple)) else [packed_dir]
        self.files = sorted(
            (f for d in dirs for f in glob.glob(os.path.join(str(d), "episode_*.hdf5"))),
            key=lambda p: int(Path(p).stem.split("_")[1]),
        )
        if not self.files:
            raise FileNotFoundError(f"no packed episodes under {packed_dir}")
        self.wrist_e_dir = wrist_e_override_dir
        self.wrist_cam = wrist_cam
        self._wrist_e = {}
        self.n_points = n_points
        self.pos_mean = None if pos_mean is None else np.asarray(pos_mean, np.float32)
        self.pos_std = None if pos_std is None else np.asarray(pos_std, np.float32)
        self.use_sor_filter = use_sor_filter
        self.point_bounds = None if point_bounds is None else np.asarray(point_bounds, np.float32).reshape(2, 3)
        self._handles = {}
        self.index = []
        self.ep_meta = []
        for ei, fp in enumerate(self.files):
            with h5py.File(fp, "r") as f:
                T = int(f.attrs["num_frames"])
                self.ep_meta.append(
                    dict(
                        H=int(f.attrs["H"]),
                        W=int(f.attrs["W"]),
                        cam_ids=[int(c) for c in f.attrs["cam_ids"]],
                        has_obj=bool(f.attrs["has_obj_mask"]),
                        has_render_obj=f"render/camera_{int(f.attrs['cam_ids'][0])}/obj_mask" in f,
                        has_filter="filter_kind" in f.attrs,
                        bbox=[float(v) for v in f.attrs["bbox"]],
                        ep_id=int(f.attrs["episode"]),
                        root=str(Path(os.path.realpath(fp)).parent.parent),
                    )
                )
            self.index += [(ei, t) for t in range(T)]
        self.bbox = self.ep_meta[0]["bbox"]

    def world_bounds(self):
        b = self.bbox
        mn = np.array([b[0], b[2], b[4]], np.float32)
        ext = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]], np.float32)
        return (mn, ext)

    def compute_norm_params(self, n_sample=200):
        saved = (self.pos_mean, self.pos_std)
        self.pos_mean = self.pos_std = None
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(self.index), min(n_sample, len(self.index)), replace=False)
        acc = []
        for j in idxs:
            xyz = self[int(j)]["xyz"].numpy()
            acc.append(xyz[rng.choice(len(xyz), min(4000, len(xyz)), replace=False)])
        self.pos_mean, self.pos_std = saved
        pts = np.concatenate(acc, 0)
        return (pts.mean(0).astype(np.float32), (pts.std(0) + 1e-06).astype(np.float32))

    def _h(self, ei):
        if ei not in self._handles:
            self._handles[ei] = h5py.File(self.files[ei], "r")
        return self._handles[ei]

    def __len__(self):
        return len(self.index)

    def _subsample(self, n_have, n_want, rng):
        if n_have == 0:
            return np.zeros(n_want, np.int64)
        if n_have >= n_want:
            return rng.choice(n_have, n_want, replace=False)
        return np.concatenate([np.arange(n_have), rng.choice(n_have, n_want - n_have, replace=True)])

    def __getitem__(self, i):
        ei, t = self.index[i]
        f = self._h(ei)
        meta = self.ep_meta[ei]
        _H, _W = (meta["H"], meta["W"])
        rng = np.random.default_rng(i)
        pts, rgb, obj = ([], [], [])
        for c in meta["cam_ids"]:
            depth = f[f"img/camera_{c}/depth"][t].astype(np.float32)
            mkey = f"img/camera_{c}/filter_mask" if self.use_sor_filter and meta["has_filter"] else f"img/camera_{c}/valid_mask"
            valid = _unpack_mask(f[mkey][t], *depth.shape)
            rgb_ir = cv2.cvtColor(cv2.imdecode(f[f"img/camera_{c}/rgb_in_ir"][t], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            rgb_ir = _color_gain.apply(rgb_ir, self.files[ei], c)
            objm = f[f"img/camera_{c}/obj_mask"][t] if meta["has_obj"] else None
            K_ir = f[f"cam/camera_{c}/K_ir"][t]
            E = f[f"cam/camera_{c}/E_cam2world"][t]
            if self.wrist_e_dir is not None and c == self.wrist_cam:
                ov = self._wrist_e.get(ei)
                if ov is None:
                    p = os.path.join(self.wrist_e_dir, f"ep_{meta['ep_id']}_tfref.npz")
                    ov = self._wrist_e[ei] = np.load(p)["E_cam2world"] if os.path.exists(p) else False
                if ov is not False and t < len(ov) and np.isfinite(ov[t]).all():
                    E = ov[t].astype(E.dtype)
            p, cgb, o = _lift_camera(depth, valid, rgb_ir, objm, K_ir, E)
            if objm is None and meta.get("has_render_obj"):
                o = _render_obj_labels(f, c, t, p)
            pts.append(p)
            rgb.append(cgb)
            obj.append(o)
        pts = np.concatenate(pts) if pts else np.zeros((0, 3), np.float32)
        rgb = np.concatenate(rgb) if rgb else np.zeros((0, 3), np.uint8)
        obj = np.concatenate(obj) if obj else np.zeros((0,), np.uint8)
        if self.point_bounds is not None and len(pts):
            lo, hi = self.point_bounds
            keep = np.all((pts >= lo) & (pts <= hi), axis=1)
            pts, rgb, obj = (pts[keep], rgb[keep], obj[keep])
        out = {"ep_idx": meta["ep_id"], "frame_idx": int(t), "file_idx": int(ei), "root": meta["root"]}
        idx = np.arange(len(pts)) if self.n_points is None else self._subsample(len(pts), self.n_points, rng)
        xyz = pts[idx].astype(np.float32)
        if self.pos_mean is not None:
            xyz = (xyz - self.pos_mean) / self.pos_std
        col = rgb[idx].astype(np.float32) / 255.0 * 2.0 - 1.0
        lab = obj[idx].astype(np.int64)
        out["xyz"], out["rgb"] = (torch.from_numpy(xyz), torch.from_numpy(col))
        out["obj_label"] = torch.from_numpy(lab)
        out["is_rope"] = out["obj_label"] > 0
        return out
