import os
import threading
from collections import OrderedDict
import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from datasets.color_gain import apply as color_gain
from datasets.color_gain import root_of


class LRU:

    def __init__(self, read, cap):
        self.read, self.cap, self.items, self.lock = (read, cap, OrderedDict(), threading.Lock())

    def __call__(self, key):
        with self.lock:
            v = self.items.get(key)
            if v is None:
                v = self.items[key] = self.read(*key)
                if len(self.items) > self.cap:
                    self.items.popitem(last=False)
            else:
                self.items.move_to_end(key)
            return v


def bits(data, h, w):
    return np.unpackbits(data)[: h * w].reshape(h, w).astype(bool)


class Frames:

    def __init__(
        self, files, device, gripper_dir="gripper_supervision/masks_halo6_surface", box_dir="sam3_masks", disp_dir="fs_disp"
    ):
        self.files, self.device, self.gripper_dir, self.box_dir, self.disp_dir = (
            list(files),
            device,
            gripper_dir,
            box_dir,
            disp_dir,
        )
        self.h5, self.disp = ({}, {})
        self.render_cache, self.depth_cache = (LRU(self.read_render, 256), LRU(self.read_depth, 16))

    def file(self, fi):
        if fi not in self.h5:
            self.h5[fi] = h5py.File(self.files[fi], "r")
        return self.h5[fi]

    def side(self, fi, rel):
        return root_of(self.files[fi]) / rel

    def read_render(self, fi, frame):
        f, cams = (self.file(fi), [])
        ep = int(f.attrs["episode"])
        for c in [int(x) for x in f.attrs.get("render_cam_ids", f.attrs["cam_ids"])]:
            g = f[f"render/camera_{c}"]
            vm = g["viewmat"][frame].astype(np.float32)
            if not np.isfinite(vm).all():
                continue
            rgb = np.ascontiguousarray(cv2.imdecode(np.asarray(g["jpg"][frame], dtype=np.uint8), cv2.IMREAD_COLOR)[..., ::-1])
            rgb = color_gain(rgb, self.files[fi], c)
            h, w = rgb.shape[:2]
            obj = bits(g["obj_mask"][frame], h, w) if "obj_mask" in g else None
            valid = bits(g["valid_mask"][frame], h, w) if "valid_mask" in g else np.ones((h, w), bool)
            if f.attrs.get("semantic_supervision", "") == "none":
                # Missing labels are explicitly declared by the RGB-only converter.
                gripper = np.zeros((h, w), bool)
            else:
                with h5py.File(self.side(fi, self.gripper_dir) / f"episode_{ep}.h5", "r") as side:
                    gripper = bits(side[f"camera_{c}"]["gripper_mask"][frame], h, w)
            box = self.side(fi, self.box_dir) / f"episode_{ep}_camera_{c}.h5"
            if box.exists():
                with h5py.File(box, "r") as side:
                    match = np.flatnonzero(side["frame_indices"][:] == frame)
                    if len(match):
                        x0, y0, x1, y1 = map(int, g.attrs["crop"])
                        obj = side["mask_union"][int(match[0]), y0:y1, x0:x1].astype(bool)
            cams.append(
                dict(
                    rgb=rgb,
                    K=g["K"][frame].astype(np.float32),
                    viewmat=vm,
                    valid=valid,
                    obj=obj,
                    gripper=gripper,
                    H=h,
                    W=w,
                    cid=c,
                )
            )
        return cams

    def warm(self, fi, frame):
        try:
            self.render_cache((fi, frame))
        except Exception:
            pass

    def render(self, fi, frame):
        dev, out = (self.device, [])
        for c in self.render_cache((fi, frame)):
            t = {k: torch.from_numpy(c[k]).to(dev) for k in ("K", "viewmat", "valid")}
            t.update(
                rgb=torch.from_numpy(c["rgb"]).to(dev).float() / 255.0,
                gripper=torch.from_numpy(c["gripper"]).to(dev),
                H=c["H"],
                W=c["W"],
                cid=c["cid"],
            )
            t["obj"] = torch.from_numpy(c["obj"]).to(dev) if c["obj"] is not None else None
            out.append(t)
        return out

    def read_depth(self, fi, frame):
        f, cams = (self.file(fi), [])
        if f.attrs.get("depth_semantics", "") == "monocular_relative_estimate":
            for c in [int(x) for x in f.attrs["cam_ids"]]:
                cams.append(dict(
                    K=f[f"cam/camera_{c}/K_ir"][frame].astype(np.float32),
                    viewmat=np.linalg.inv(f[f"cam/camera_{c}/E_cam2world"][frame].astype(np.float32)),
                    cid=c,
                    estimated_depth=f[f"img/camera_{c}/depth"][frame].astype(np.float32),
                ))
            return cams
        if fi not in self.disp:
            src = str(f.attrs.get("src_episode", os.path.basename(self.files[fi])))
            self.disp[fi] = h5py.File(str(self.side(fi, self.disp_dir) / ("disp_" + src)), "r")
        d = self.disp[fi]
        for c in [int(x) for x in f.attrs["cam_ids"]]:
            try:
                viewmat = np.linalg.inv(f[f"cam/camera_{c}/E_cam2world"][frame].astype(np.float32)).astype(np.float32)
            except np.linalg.LinAlgError:
                continue
            cam = dict(
                K=f[f"cam/camera_{c}/K_ir"][frame].astype(np.float32),
                viewmat=viewmat,
                cid=c,
                disp=d[f"disp/camera_{c}"][frame],
                baseline=float(d[f"meta/camera_{c}/baseline"][frame]),
            )
            if f"disp_right/camera_{c}" in d:
                cam["disp_r"] = d[f"disp_right/camera_{c}"][frame]
            cams.append(cam)
        return cams

    def depth(self, fi, frame, max_depth=3.0, lrc=8.0):
        out, dev = ([], self.device)
        for c in self.depth_cache((fi, frame)):
            if "estimated_depth" in c:
                d = torch.from_numpy(c["estimated_depth"].copy()).to(dev).float()
                d[~torch.isfinite(d) | (d <= 0) | (d > max_depth)] = 0.0
            else:
                disp = torch.from_numpy(np.ascontiguousarray(c["disp"])).to(dev).float()
                d = float(c["K"][0, 0]) * c["baseline"] / disp.clamp(min=1e-06)
                d[(disp <= 0.5) | (d > max_depth)] = 0.0
            if "disp_r" in c:
                dr = torch.from_numpy(np.ascontiguousarray(c["disp_r"])).to(dev).float()
                W = disp.shape[1]
                xr = torch.round(torch.arange(W, device=dev, dtype=disp.dtype).expand_as(disp) - disp).long()
                match = torch.gather(dr, 1, xr.clamp(0, W - 1))
                d[~((xr >= 0) & (xr < W) & ((disp - match).abs() <= lrc))] = 0.0
            d = d.clamp(min=0)
            dmin = d.clone()
            dmin[dmin <= 0] = float("inf")
            dmin = -F.max_pool2d((-dmin)[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
            out.append(
                dict(
                    K=torch.from_numpy(c["K"]).to(dev),
                    viewmat=torch.from_numpy(c["viewmat"]).to(dev),
                    cid=c["cid"],
                    gt=d,
                    dmin=dmin,
                )
            )
        return out


@torch.no_grad()
def warp_depth(dcams, rcams):
    pws = []
    for dc in dcams:
        gt, K = (dc["gt"], dc["K"])
        vs, us = torch.nonzero(gt > 0, as_tuple=True)
        z = gt[vs, us]
        pc = torch.stack([(us.float() - K[0, 2]) / K[0, 0] * z, (vs.float() - K[1, 2]) / K[1, 1] * z, z], 1)
        E = torch.linalg.inv(dc["viewmat"])
        pws.append(pc @ E[:3, :3].T + E[:3, 3])
    pw = torch.cat(pws)
    outs = []
    for cam in rcams:
        vm, K = (cam["viewmat"], cam["K"])
        pc = pw @ vm[:3, :3].T + vm[:3, 3]
        z = pc[:, 2]
        zs = z.clamp(min=1e-06)
        u = (K[0, 0] * pc[:, 0] / zs + K[0, 2]).long()
        v = (K[1, 1] * pc[:, 1] / zs + K[1, 2]).long()
        ok = (z > 0.05) & (u >= 0) & (u < cam["W"]) & (v >= 0) & (v < cam["H"])
        D = torch.full((cam["H"] * cam["W"],), float("inf"), device=pw.device, dtype=pw.dtype)
        D.scatter_reduce_(0, v[ok] * cam["W"] + u[ok], z[ok], reduce="amin")
        D = D.view(cam["H"], cam["W"])
        D[torch.isinf(D)] = 0.0
        outs.append(D)
    return outs
