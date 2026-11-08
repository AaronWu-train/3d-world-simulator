"""Reconstruct held-out RGB-D frames with a trained Amber VAE and particle head.

Outputs source/reconstruction/error images, an overview, Gaussian-center PLYs,
and optional comparison videos. No optimizer runs and no checkpoint is modified.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.aloha_packed import AlohaPackedPCDataset
from datasets.frames import Frames
from models.big_head import BigHeadRoundShim
from models.build import load_vae, normalize_latent
from models.class_seeds import class_masks, select_class_seeds
from models.gsplat_render_loss import render_gaussians, concat_gaussians


def image_metrics(pred, target, valid):
    error = pred[valid].astype(np.float64) - target[valid].astype(np.float64)
    mse = float(np.mean(error**2))
    return dict(rgb_mae=float(np.mean(np.abs(error))),
                psnr_db=float(-10*np.log10(max(mse, 1e-12))))


def summarize_episode(rows):
    worst = min(rows, key=lambda r: r["psnr_db"])
    return dict(episode=rows[0]["episode"], task=rows[0]["task"], frames=len(rows),
                mean_rgb_mae=float(np.mean([r["rgb_mae"] for r in rows])),
                mean_psnr_db=float(np.mean([r["psnr_db"] for r in rows])),
                min_psnr_db=worst["psnr_db"], worst_frame=worst["frame"],
                worst_time_s=worst["time_s"])


def write_summary(out_dir, report):
    lines = ["# VAE reconstruction evaluation", "",
             f"Checkpoint epoch (zero-based): {report['checkpoint_epoch']}. Evaluated {len(report['frames'])} frames in {len(report['episodes'])} episodes.", "",
             f"Mean frame PSNR: {report['mean_psnr_db']:.2f} dB. Mean RGB MAE (0–1): {report['mean_rgb_mae']:.4f}.", "",
             "| Episode | Task | Frames | Mean PSNR (dB) | Worst PSNR (dB) | Worst frame / time | Outputs |",
             "|---|---|---:|---:|---:|---|---|"]
    for r in report["episodes"]:
        ep = r["episode"]
        links = f"[worst image](episode_{ep}_worst.png)"
        if report["full_video"]:
            links += f" · [video](episode_{ep}_comparison.mp4)"
        lines.append(f"| {ep} | {r['task']} | {r['frames']} | {r['mean_psnr_db']:.2f} | {r['min_psnr_db']:.2f} | {r['worst_frame']} / {r['worst_time_s']:.1f}s | {links} |")
    lines += ["", "Images show recorded RGB, reconstruction, and absolute RGB error on a fixed 0–1 scale.", "",
              "Worst frames are selected by full-image RGB error, not by ground-truth grasp/contact annotations. Background pixels affect these metrics. Review the hand/cloth interactions in the videos as well.", "",
              "This reconstructs current frames using estimated monocular geometry; it does not measure true 3D accuracy or predict future frames.", ""]
    (out_dir/"evaluation.md").write_text("\n".join(lines))


def comparison_image(target, pred, title):
    error = np.abs(pred-target).mean(-1)
    # Fixed 0..1 error scale across frames, without per-frame contrast stretching.
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(error, 0, 1)*255), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    images = [np.uint8(np.clip(x, 0, 1)*255) for x in (target, pred)] + [heat]
    panel = np.concatenate(images, axis=1)
    header = np.full((48, panel.shape[1], 3), 245, np.uint8)
    cv2.putText(header, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (20, 20, 20), 1)
    w = target.shape[1]
    for i, text in enumerate(("Recorded RGB", "VAE reconstruction", "Absolute RGB error (0-1)")):
        cv2.putText(header, text, (i*w+8, 39), cv2.FONT_HERSHEY_SIMPLEX, .45, (20, 20, 20), 1)
    return np.concatenate([header, panel])


def save_rgb(path, rgb):
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write {path}")


def save_ply(path, gaussians):
    xyz = gaussians["xyz"].cpu().numpy()
    rgb = np.uint8(np.clip((gaussians["rgb"].cpu().numpy()+1)*127.5, 0, 255))
    # Deterministic thinning for a manageable preview, not an opacity threshold.
    idx = np.arange(0, len(xyz), max(1, (len(xyz)+49999)//50000))
    with path.open("w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(idx)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(f, np.column_stack([xyz[idx], rgb[idx]]), fmt="%.6f %.6f %.6f %d %d %d")


class Reconstructor:
    def __init__(self, checkpoint, device, chunk_size):
        self.device, self.chunk_size = device, chunk_size
        self.model = load_vae(checkpoint, device)
        self.pm, self.ps = checkpoint["pos_mean"], checkpoint["pos_std"]
        self.unlabelled = not checkpoint["args"].get("semantic_supervision", True)
        spec, state = checkpoint["new_head"], checkpoint["seed_head_state_dict"]
        if spec["type"] != "big":
            raise ValueError("This viewer expects the trained BigHeadRoundShim")
        self.head = BigHeadRoundShim(
            feat_dim=state["head.net.0.weight"].shape[1]//27,
            K=spec["children"], hidden=state["head.net.0.weight"].shape[0],
            shape=spec["shape"], cap=spec["cap"],
        ).to(device).eval()
        self.head.load_state_dict(state)
        self.features = {}
        self.model.decoder.up_finest.register_forward_hook(lambda m, i, o: self.features.update(feat=o))
        self.background = None
        bg = checkpoint.get("bg_bank_state_dict", {})
        if "means" in bg and len(bg["means"]):
            self.background = dict(
                xyz=bg["means"].to(device),
                scale=bg["log_scales"].to(device).clamp(np.log(.0001), np.log(.2)).exp().expand(-1, 3),
                opacity=bg["opacity_logit"].to(device).sigmoid(),
                rgb=bg["rgb_logit"].to(device).sigmoid()*2-1,
            )

    @torch.inference_mode()
    def __call__(self, sample, cam):
        xyz = sample["xyz"][None].to(self.device)
        rgb = sample["rgb"][None].to(self.device)
        mean, _ = self.model.encode(xyz, rgb, pos_mean=self.pm, pos_std=self.ps)
        z = normalize_latent(mean, self.model.latent_split)
        # Decode from z alone; do not pass ground-truth occupancy or labels.
        out = self.model.decode(z, use_gt_mask=False)
        finest = len(self.model.grid_shapes)-1
        scene = out[f"scene_occ_{finest}"]
        empty = torch.zeros_like(scene, dtype=torch.bool) if self.unlabelled else None
        masks = class_masks(scene, out[f"tac_occ_{finest}"], out[f"signal_{finest}"],
                            gt_cloth=empty, gt_robot=empty)
        # Unlike the training cap, this preview uses every predicted occupied cell.
        seeds, classes = select_class_seeds(masks)
        if len(seeds) == 0:
            raise RuntimeError("Decoder predicted no occupied cells")
        model = self.model
        vs = (model.bounds_max_world-model.bounds_min_world)/torch.tensor(model.grid_shapes[-1], device=self.device)
        pieces = []
        for start in range(0, len(seeds), self.chunk_size):
            _, part = self.head(self.features["feat"], seeds[start:start+self.chunk_size], model.bounds_min_world, vs,
                                seed_cls=classes[start:start+self.chunk_size])
            pieces.append({k: part[k].reshape(-1, *shape) for k, shape in
                           dict(xyz=(3,), scale=(3,), opacity=(), rgb=(3,), quat=(4,)).items()})
        gaussians = {k: torch.cat([p[k] for p in pieces]) for k in pieces[0]}
        rendered_g = concat_gaussians(gaussians, self.background) if self.background is not None else gaussians
        image, alpha = render_gaussians(
            rendered_g["xyz"], rendered_g["scale"], rendered_g["opacity"],
            (rendered_g["rgb"].clamp(-1, 1)+1)*.5,
            cam["viewmat"][None], cam["K"][None], cam["W"], cam["H"], quats=rendered_g["quat"],
        )
        return image[0].clamp(0, 1).cpu().numpy(), alpha[0, ..., 0].cpu().numpy(), gaussians


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vae-ckpt", type=Path, default=Path("runs/amber/vae/checkpoints/best.pth"))
    p.add_argument("--packed-dir", type=Path, default=Path("data/amber_cloth_estimated/val"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/amber/reconstruction"))
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--episodes", type=int, nargs="+", default=None)
    selection.add_argument("--all-episodes", action="store_true", help="Evaluate every episode in --packed-dir")
    p.add_argument("--frames", type=int, nargs="+", default=None, help="Default: first, middle, last")
    p.add_argument("--video", action="store_true", help="Reconstruct all frames and create an H.264 comparison video")
    p.add_argument("--n-points", type=int, default=673000, help="Match the training sampling count")
    p.add_argument("--head-chunk", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.n_points <= 0 or a.head_chunk <= 0:
        p.error("Point count and chunk size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Reconstruction needs a visible NVIDIA GPU; run from the CUDA-enabled terminal")
    if a.video and shutil.which("ffmpeg") is None:
        p.error("--video needs ffmpeg for H.264 encoding")
    torch.manual_seed(0); torch.set_num_threads(4); cv2.setNumThreads(1)
    checkpoint = torch.load(a.vae_ckpt, map_location="cpu", weights_only=False)
    reconstruct = Reconstructor(checkpoint, a.device, a.head_chunk)
    cfg = checkpoint["args"]
    bounds = [[cfg[k][i] for k in ("x_range", "y_range", "z_range")] for i in (0, 1)]
    ds = AlohaPackedPCDataset(a.packed_dir, n_points=a.n_points, pos_mean=reconstruct.pm,
                             pos_std=reconstruct.ps, point_bounds=bounds)
    available = {m["ep_id"]: i for i, m in enumerate(ds.ep_meta)}
    episodes = sorted(available) if a.all_episodes else a.episodes or [sorted(available)[i] for i in np.unique(np.linspace(0, len(available)-1, min(3, len(available))).astype(int))]
    if set(episodes)-set(available):
        p.error(f"Unknown episodes; available: {sorted(available)}")
    frames = Frames(ds.files, a.device)
    frame_indices = {(fi, t): gi for gi, (fi, t) in enumerate(ds.index)}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    report = dict(checkpoint=str(a.vae_ckpt.resolve()), checkpoint_sha256=hashlib.sha256(a.vae_ckpt.read_bytes()).hexdigest(),
                  checkpoint_epoch=int(checkpoint["epoch"]), packed_dir=str(a.packed_dir.resolve()),
                  n_points=a.n_points, decode_uses_ground_truth_occupancy=False,
                  full_video=a.video,
                  note="Reconstruction of the current frame, not future prediction; geometry is monocular estimated depth",
                  frames=[], episodes=[])
    overview, worst_overview = [], []
    for episode in episodes:
        fi = available[episode]
        with h5py.File(ds.files[fi], "r") as f:
            n, fps, task = int(f.attrs["num_frames"]), float(f.attrs.get("fps", 5)), str(f.attrs.get("task", ""))
        snapshots = sorted(set(a.frames if a.frames is not None else [0, n//2, n-1]))
        if min(snapshots) < 0 or max(snapshots) >= n:
            p.error(f"Frames must be in 0..{n-1} for episode {episode}")
        selected = list(range(n)) if a.video else snapshots
        episode_rows, worst_panel, worst_score = [], None, float("inf")
        with tempfile.TemporaryDirectory(prefix="reconstruction_", dir=a.out_dir) as work:
            for t in selected:
                cam = frames.render(fi, t)[0]
                target, valid = cam["rgb"].cpu().numpy(), cam["valid"].cpu().numpy()
                pred, alpha, gaussians = reconstruct(ds[frame_indices[(fi, t)]], cam)
                metrics = image_metrics(pred, target, valid)
                metrics.update(episode=episode, frame=t, task=task, gaussians=len(gaussians["xyz"]),
                               alpha_mean=float(alpha.mean()), time_s=t/fps)
                report["frames"].append(metrics)
                episode_rows.append(metrics)
                title = f"Episode {episode} | {task} | frame {t} | MAE {metrics['rgb_mae']:.3f} | PSNR {metrics['psnr_db']:.1f} dB"
                panel = comparison_image(target, pred, title)
                if metrics["psnr_db"] < worst_score:
                    worst_score, worst_panel = metrics["psnr_db"], panel.copy()
                if t in snapshots:
                    stem = f"episode_{episode}_frame_{t:03d}"
                    save_rgb(a.out_dir/f"{stem}.png", panel)
                    save_ply(a.out_dir/f"{stem}.ply", gaussians)
                    overview.append(panel)
                if a.video:
                    save_rgb(Path(work)/f"frame_{t:06d}.png", panel)
                del gaussians
                print(f"episode={episode} frame={t} MAE={metrics['rgb_mae']:.4f} PSNR={metrics['psnr_db']:.2f}dB", flush=True)
            if a.video:
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(Path(work)/"frame_%06d.png"),
                                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                str(a.out_dir/f"episode_{episode}_comparison.mp4")], check=True)
        summary = summarize_episode(episode_rows)
        report["episodes"].append(summary)
        save_rgb(a.out_dir/f"episode_{episode}_worst.png", worst_panel)
        worst_overview.append(worst_panel)
        print(f"Completed episode {episode}: mean PSNR={summary['mean_psnr_db']:.2f}dB, worst frame={summary['worst_frame']}", flush=True)
    save_rgb(a.out_dir/"overview.png", np.concatenate(overview, axis=0))
    save_rgb(a.out_dir/"worst_frames_overview.png", np.concatenate(worst_overview, axis=0))
    report["mean_rgb_mae"] = float(np.mean([r["rgb_mae"] for r in report["frames"]]))
    report["mean_psnr_db"] = float(np.mean([r["psnr_db"] for r in report["frames"]]))
    (a.out_dir/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    write_summary(a.out_dir, report)
    print(f"Saved reconstruction comparison to {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
