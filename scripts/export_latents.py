import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.aloha_packed import AlohaPackedPCDataset
from models.build import load_vae, normalize_latent
from models.tactile import fire_color

TOL = 1e-3


def rasterize_obj(pts, is_obj, bmin, bext, grid):
    nx, ny, nz = grid
    occ = np.zeros((nz, ny, nx), np.uint8)
    op = pts[is_obj]
    if len(op):
        idx = np.floor((op - bmin) / bext * np.array([nx, ny, nz], np.float32)).astype(np.int64)
        idx = np.clip(idx, 0, np.array([nx - 1, ny - 1, nz - 1]))
        occ[idx[:, 2], idx[:, 1], idx[:, 0]] = 1
    return occ


@torch.no_grad()
def encode_frame(model, pts, rgb, pm, ps, device):
    xyz = torch.from_numpy(((pts - np.asarray(pm, np.float32)) / np.asarray(ps, np.float32))[None]).to(device)
    mean, _ = model.encode(xyz, torch.from_numpy(rgb[None]).to(device), pos_mean=pm, pos_std=ps)
    return normalize_latent(mean, model.latent_split).to(torch.float16).cpu().numpy()[0]


def mode_fp16(x):
    u = x.view(np.uint16)
    return np.array([np.bincount(u[c], minlength=65536).argmax() for c in range(x.shape[0])], np.uint16).view(np.float16)


def compact(z):
    T, C = z.shape[:2]
    nvox = int(np.prod(z.shape[2:]))
    z = z.reshape(T, C, nvox)
    bg, masks, vals, nnz = np.empty((T, C), np.float16), np.empty((T, nvox // 8), np.uint8), [], np.zeros(T, np.int64)
    for i in range(T):
        bg[i] = mode_fp16(z[i])
        keep = (np.abs(z[i].astype(np.float32) - bg[i].astype(np.float32)[:, None]) > TOL).any(0)
        masks[i] = np.packbits(keep)
        vals.append(np.ascontiguousarray(z[i][:, keep].T))
        nnz[i] = int(keep.sum())
    return dict(z_bg=bg, z_mask=masks, z_off=np.concatenate([[0], np.cumsum(nnz)]), z_vals=np.concatenate(vals, 0))


@torch.no_grad()
def tactile_cells(model, z, device, batch=16):
    T, C, nz, ny, nx = z.shape
    out = np.zeros((T, nz, ny, nx), np.uint8)
    for s in range(0, T, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model.decoder(
                model.conv_decoder(torch.from_numpy(z[s : s + batch].astype(np.float32)).to(device)),
                occ_mask_0=None,
                use_gt_mask=False,
            )
        t = (o[f"tactile_occ_{model.decoder.finest}"].float() > 0).float()
        k = t.shape[2] // nz, t.shape[3] // ny, t.shape[4] // nx
        m = F.max_pool3d(t, k) if k != (1, 1, 1) else t
        out[s : s + batch] = (F.max_pool3d(m, 3, 1, 1)[:, 0] > 0.5).cpu().numpy().astype(np.uint8)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--packed-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--episodes", type=int, nargs="*", default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--epid-offset", type=int, default=0)
    a = p.parse_args()
    device = "cuda"
    ck = torch.load(a.vae_ckpt, map_location=device, weights_only=False)
    model = load_vae(ck, device)
    pm, ps, c = ck["pos_mean"], ck["pos_std"], ck["args"]
    grid = c["grid"] if "grid" in c else (c["nx"], c["ny"], c["nz"])
    bmin = np.array([c["x_range"][0], c["y_range"][0], c["z_range"][0]], np.float32)
    bext = np.array([c["x_range"][1], c["y_range"][1], c["z_range"][1]], np.float32) - bmin
    tactile = any(k.startswith("tactile_particle_head.") for k in ck["model_state_dict"])
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = AlohaPackedPCDataset(a.packed_dir, n_points=None, use_sor_filter=False)
    frames_of = defaultdict(list)
    for gi, (ei, t) in enumerate(ds.index):
        frames_of[ei].append((t, gi))
    rng = np.random.default_rng(0)
    for ei in sorted(frames_of):
        meta = ds.ep_meta[ei]
        if a.episodes is not None and meta["ep_id"] not in a.episodes:
            continue
        path = out / f"episode_{meta['ep_id'] + a.epid_offset}_latents.h5"
        if path.exists():
            continue
        frames = sorted(frames_of[ei])[: a.max_frames]
        T = len(frames)
        with h5py.File(ds.files[ei], "r") as f:
            actions = f["action/cartesian_action"][:].astype(np.float32)
            provenance = {k: f.attrs[k] for k in ("action_semantics", "depth_semantics", "depth_model", "depth_model_revision",
                                                 "camera_calibration", "fps", "source_dataset", "source_episode", "task") if k in f.attrs}
            K = np.stack([f[f"cam/camera_{c_}/K_ir"][:].astype(np.float32) for c_ in meta["cam_ids"]], 1)
            E = np.stack([f[f"cam/camera_{c_}/E_cam2world"][:].astype(np.float32) for c_ in meta["cam_ids"]], 1)
        tac = np.load(Path(meta["root"]) / "tactile_targets" / f"episode_{meta['ep_id']}.npz") if tactile else None
        zs, occ = [], []
        for t, gi in frames:
            s = ds[gi]
            pts, rgb, obj = (
                s["xyz"].numpy().astype(np.float32),
                s["rgb"].numpy().astype(np.float32),
                s["is_rope"].numpy().astype(bool),
            )
            occ.append(rasterize_obj(pts, obj, bmin, bext, grid))
            if tac is not None:
                q = min(t, len(tac["xyz"]) - 1)
                X, S = tac["xyz"][q].reshape(-1, 3).astype(np.float32), tac["sig"][q].reshape(-1).astype(np.float32)
                on = S > 0.05
                if on.any():
                    Xr = np.repeat(X[on], 3, 0)
                    X = np.concatenate([X, Xr + 0.0005 * rng.standard_normal(Xr.shape).astype(np.float32)])
                    S = np.concatenate([S, np.repeat(S[on], 3)])
                pts = np.concatenate([pts, X])
                rgb = np.concatenate([rgb, fire_color(torch.from_numpy(S)).numpy().astype(np.float32) * 2 - 1])
            zs.append(encode_frame(model, pts, rgb, pm, ps, device))
        z = np.stack(zs)
        tmp = f"{path}.tmp"
        with h5py.File(tmp, "w") as g:
            g.attrs.update(provenance)
            for k, v in compact(z).items():
                g.create_dataset(k, data=v)
            g.create_dataset("obj_occ", data=np.stack(occ))
            if tactile:
                g.create_dataset("tac_occ", data=tactile_cells(model, z, device), compression="gzip", compression_opts=1)
            g.create_dataset("action", data=actions[:T])
            g.create_dataset("K", data=K[:T])
            g.create_dataset("E", data=E[:T])
            g.attrs.update(
                dict(
                    compact=1,
                    z_shape=np.asarray(z.shape[1:]),
                    pos_mean=np.asarray(pm, np.float32),
                    pos_std=np.asarray(ps, np.float32),
                    bbox=np.asarray(meta["bbox"], np.float32),
                )
            )
        os.replace(tmp, path)
        print(f"{path.name}: {T} frames", flush=True)


if __name__ == "__main__":
    main()
