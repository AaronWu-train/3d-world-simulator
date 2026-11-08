"""Autoregressive Dynamics rollout videos, conditioned on recorded actions.

Only the initial observed latent history is passed to the predictor. Future
ground-truth latents are read AFTER rollout, solely for reference rendering and
metrics. Run this script yourself in the CUDA-enabled training environment.
"""
import argparse
import hashlib
import json
import re
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
from datasets.frames import Frames
from datasets.latent_seq import read_dense_z, n_frames
from models.dynamics_coarse3d import CoarseDynamics3D
from scripts.train_dynamics import predict, INFER_HIST_NOISE


@torch.inference_mode()
def rollout_latents(initial_history, actions, predict_next, n_steps):
    """Return observed history + predictions; never accepts future observations.

    initial_history: (1, history, C, D, H, W); actions: (history+steps, A).
    The action at the predicted index follows the training script's convention.
    """
    if initial_history.ndim != 6 or initial_history.shape[0] != 1:
        raise ValueError("Expected initial history shaped (1, history, C, D, H, W)")
    history_len = initial_history.shape[1]
    if history_len < 1 or n_steps < 1 or actions.ndim != 2 or len(actions) < history_len+n_steps:
        raise ValueError("Need a nonempty history, positive horizon, and sufficient actions")
    if not bool(torch.isfinite(initial_history).all()) or not bool(torch.isfinite(actions).all()):
        raise ValueError("History and actions must be finite")
    history = initial_history.float().clone()
    outputs = [history.clone()]
    for t in range(history_len, history_len+n_steps):
        pred = predict_next(history, actions[t-history_len:t][None], actions[t:t+1][None]).float()
        if pred.shape != initial_history[:, :1].shape or not bool(torch.isfinite(pred).all()):
            raise ValueError(f"Invalid predicted latent at rollout step {t-history_len+1}")
        outputs.append(pred.clone())
        # No teacher forcing: the next call sees this prediction, not a real z.
        history = torch.cat([history, pred], 1)[:, -history_len:]
    return torch.cat(outputs, 1)


def load_dynamics(checkpoint, device):
    state, cfg = checkpoint["model_state_dict"], checkpoint["args"]
    levels = {int(m.group(1)) for key in state if (m := re.match(r"enc(\d+)\.0\.weight$", key))}
    time_blocks = {int(m.group(1)) for key in state if (m := re.match(r"temporal\.(\d+)\.norm\.weight$", key))}
    history = int(cfg["n_history"])
    if state["slot_embed"].shape[0] != history+1:
        raise ValueError("Expected a one-step Dynamics checkpoint")
    model = CoarseDynamics3D(
        latent_channels=state["out.weight"].shape[0], action_dim=state["action_ch.0.weight"].shape[1],
        n_history=history, n_future=1, width=state["enc0.0.weight"].shape[0],
        action_emb_dim=state["action_ch.0.weight"].shape[0], n_time_blocks=len(time_blocks),
        n_levels=len(levels), film="film_tok.0.weight" in state,
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def normalize_actions(actions, checkpoint):
    mean, std = np.asarray(checkpoint["action_mean"], np.float32), np.asarray(checkpoint["action_std"], np.float32)
    if actions.ndim != 2 or mean.shape != (actions.shape[1],) or std.shape != mean.shape:
        raise ValueError("Action dimensions do not match the trained normalization")
    if not np.isfinite(actions).all() or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid actions or action normalization")
    return (actions-mean)/std


def check_latent_pair(latent, packed, vae, action_dim):
    """Fail on misaligned trajectories / incompatible latent representations."""
    cfg = vae["args"]
    grid = cfg.get("grid", (cfg.get("nx"), cfg.get("ny"), cfg.get("nz")))
    shape = tuple(int(v) for v in latent.attrs["z_shape"])
    if shape[1:] != tuple(reversed(grid)):
        raise ValueError("Latent grid does not match VAE checkpoint")
    if n_frames(latent) != int(packed.attrs["num_frames"]):
        raise ValueError("Packed and latent episode lengths differ")
    for key in ("pos_mean", "pos_std"):
        if not np.allclose(latent.attrs[key], vae[key], rtol=1e-5, atol=1e-6):
            raise ValueError(f"VAE and exported latents disagree on {key}")
    actions = latent["action"][:].astype(np.float32)
    if actions.shape != (n_frames(latent), action_dim):
        raise ValueError("Dynamics action dimensions differ from latent file")
    if not np.allclose(actions, packed["action/cartesian_action"][:], rtol=1e-5, atol=1e-5):
        raise ValueError("Latent actions do not match the packed episode")
    if "source_episode" in latent.attrs and int(latent.attrs["source_episode"]) != int(packed.attrs["episode"]):
        raise ValueError("Latent and packed source episode IDs differ")
    for i, cam in enumerate(packed.attrs["cam_ids"]):
        for exported, source in (("K", f"cam/camera_{int(cam)}/K_ir"), ("E", f"cam/camera_{int(cam)}/E_cam2world")):
            if not np.allclose(latent[exported][:, i], packed[source][:], rtol=1e-5, atol=1e-5):
                raise ValueError("Latent and packed camera calibration differ; use matching exports")
    return actions


def comparison_panel(target, reference, prediction, title, phase):
    error = np.abs(prediction-target).mean(-1)
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(error, 0, 1)*255), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    panels = [np.uint8(np.clip(x, 0, 1)*255) for x in (target, reference, prediction)]+[heat]
    body = np.concatenate(panels, axis=1)
    header = np.full((52, body.shape[1], 3), 245, np.uint8)
    cv2.putText(header, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .43, (20, 20, 20), 1)
    labels = ["Recorded RGB", "VAE reference (current frame)", f"Dynamics: {phase}", "Prediction RGB error (0-1)"]
    for i, text in enumerate(labels):
        cv2.putText(header, text, (i*target.shape[1]+6, 41), cv2.FONT_HERSHEY_SIMPLEX, .42, (20, 20, 20), 1)
    return np.concatenate([header, body])


def write_evaluation(root, report):
    lines = ["# Dynamics autoregressive rollout", "",
             f"Initial observed history: {report['n_history']} frames. All later states are model predictions conditioned on recorded actions.", "",
             "Recorded future RGB and true future latents are used only for evaluation / VAE reference rendering. They never replace the predicted history.", "",
             "Metrics below exclude the initial observed history. VAE reference quality separates reconstruction errors from additional Dynamics errors.", "",
             "| Episode | Task | Predicted frames | Duration ahead (s) | Rollout RGB MAE | Rollout PSNR (dB) | VAE reference PSNR (dB) | Latent MSE | Empty predictions | Video |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in report["episodes"]:
        ep = row["episode"]
        lines.append(f"| {ep} | {row['task']} | {row['predicted_frames']} | {row['duration_ahead_s']:.1f} | {row['mean_rgb_mae']:.4f} | {row['mean_psnr_db']:.2f} | {row['reference_mean_psnr_db']:.2f} | {row['mean_latent_mse']:.6f} | {row['empty_predictions']} | [video](episode_{ep}_rollout.mp4) |")
    lines += ["", "Long rollouts can drift: the training schedule may cover only a few consecutive prediction steps. Future actions come from the demonstration; this is not a learned action policy.", "",
              "Image metrics include the background and are not true 3D / contact accuracy. Old latent files have no VAE hash; normalization, grid and action checks cannot prove exact checkpoint identity. Use the fixed VAE exported for this Dynamics run.", ""]
    (root/"evaluation.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dynamics-ckpt", type=Path, default=Path("runs/amber/dynamics_vae100/checkpoints/best.pth"))
    p.add_argument("--vae-ckpt", type=Path, default=Path("runs/amber/dynamics_vae100/vae_fixed.pth"))
    p.add_argument("--latent-dir", type=Path, default=Path("runs/amber/dynamics_vae100/latents/val"))
    p.add_argument("--packed-dir", type=Path, default=Path("data/amber_cloth_estimated/val"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/amber/dynamics_rollout"))
    group = p.add_mutually_exclusive_group()
    group.add_argument("--episodes", type=int, nargs="+")
    group.add_argument("--all-episodes", action="store_true")
    p.add_argument("--start-frame", type=int, default=0, help="First observed frame, using packed / exported frame numbering")
    p.add_argument("--horizon", type=int, default=25, help="Predicted frames after history; 0 means all remaining frames")
    p.add_argument("--hist-noise", type=float, default=INFER_HIST_NOISE, help="Match Dynamics validation conditioning")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--head-chunk", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if a.start_frame < 0 or a.horizon < 0 or a.head_chunk < 1 or not np.isfinite(a.hist_noise) or a.hist_noise < 0:
        p.error("Invalid start frame, horizon, chunk size or history noise")
    if shutil.which("ffmpeg") is None:
        p.error("ffmpeg is required to encode the H.264 videos")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this command yourself in the CUDA-enabled 3dws environment")
    from amber_data.script.reconstruct_vae import Reconstructor, image_metrics, save_rgb

    torch.set_num_threads(4); cv2.setNumThreads(1)
    dc = torch.load(a.dynamics_ckpt, map_location="cpu", weights_only=False)
    vc = torch.load(a.vae_ckpt, map_location="cpu", weights_only=False)
    dynamics = load_dynamics(dc, a.device)
    renderer = Reconstructor(vc, a.device, a.head_chunk)
    history_len = int(dc["args"]["n_history"])
    files = {int(f.stem.split("_")[1]): f for f in a.latent_dir.glob("episode_*_latents.h5")}
    if not files:
        p.error("No exported latent files found")
    episodes = sorted(files) if a.all_episodes else a.episodes or ([21, 32] if {21, 32} <= files.keys() else sorted(files)[:2])
    if set(episodes)-files.keys():
        p.error(f"Unknown episode IDs; available: {sorted(files)}")
    packed_paths = [a.packed_dir/f"episode_{ep}.hdf5" for ep in episodes]
    frames = Frames(packed_paths, a.device)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    report = dict(dynamics_checkpoint=str(a.dynamics_ckpt.resolve()), dynamics_epoch=int(dc["epoch"]),
                  dynamics_sha256=hashlib.sha256(a.dynamics_ckpt.read_bytes()).hexdigest(),
                  vae_checkpoint=str(a.vae_ckpt.resolve()), vae_epoch=int(vc["epoch"]),
                  vae_sha256=hashlib.sha256(a.vae_ckpt.read_bytes()).hexdigest(),
                  latent_dir=str(a.latent_dir.resolve()), packed_dir=str(a.packed_dir.resolve()),
                  n_history=history_len, requested_horizon=a.horizon, start_frame=a.start_frame,
                  seed=a.seed, hist_noise=a.hist_noise, teacher_forcing=False, action_source="recorded demonstration",
                  metrics_exclude_observed_history=True, episodes=[], frames=[])
    overview = []
    for fi, episode in enumerate(episodes):
        torch.manual_seed(a.seed+episode)
        with h5py.File(files[episode], "r") as latent, h5py.File(packed_paths[fi], "r") as packed:
            raw_actions = check_latent_pair(latent, packed, vc, dynamics.A)
            if int(latent.attrs["z_shape"][0]) != dynamics.C:
                raise ValueError("Latent channels differ from Dynamics checkpoint")
            total, fps = n_frames(latent), float(packed.attrs.get("fps", 5))
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("Invalid packed frame rate")
            remaining = total-a.start_frame-history_len
            steps = remaining if a.horizon == 0 else a.horizon
            if steps < 1 or steps > remaining:
                p.error(f"Episode {episode}: only {remaining} future frames available after the observed history")
            end = a.start_frame+history_len+steps
            task = str(packed.attrs.get("task", ""))
            action_slice = normalize_actions(raw_actions[a.start_frame:end], dc)
            # This is the ONLY source of observed states supplied to Dynamics.
            initial = torch.from_numpy(read_dense_z(latent, range(a.start_frame, a.start_frame+history_len)))[None].to(a.device)
            if not np.isclose(float(latent.attrs.get("fps", fps)), fps):
                raise ValueError("Packed and latent frame rates differ")

            def next_prediction(z, ah, an):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return predict(dynamics, z, ah, an, hist_noise=a.hist_noise).float()

            predicted = rollout_latents(initial, torch.from_numpy(action_slice).to(a.device), next_prediction, steps)
            predicted_cpu = predicted[0].cpu().numpy()
            # Future true latents are read only AFTER the full rollout finished.
            reference = read_dense_z(latent, range(a.start_frame, end))
            episode_rows, selected_panels = [], []
            snapshot_indices = {history_len, history_len+steps//2, history_len+steps-1}
            worst_score, worst_panel = float("inf"), None
            with tempfile.TemporaryDirectory(prefix="rollout_", dir=a.out_dir) as work:
                for relative, t in enumerate(range(a.start_frame, end)):
                    cam = frames.render(fi, t)[0]
                    target, valid = cam["rgb"].cpu().numpy(), cam["valid"].cpu().numpy()
                    ref_img, _, ref_g = renderer.decode_latent(torch.from_numpy(reference[relative:relative+1]).to(a.device), cam, allow_empty=True)
                    del ref_g
                    is_prediction = relative >= history_len
                    if is_prediction:
                        image, alpha, gaussians = renderer.decode_latent(predicted[:, relative], cam, allow_empty=True)
                        empty = len(gaussians["xyz"]) == 0
                        del gaussians
                    else:
                        image, empty = ref_img, False
                    k = relative-history_len+1 if is_prediction else 0
                    pm, rm = image_metrics(image, target, valid), image_metrics(ref_img, target, valid)
                    row = dict(episode=episode, frame=t, task=task, is_prediction=is_prediction,
                               rollout_step=k, seconds_ahead=k/fps, rgb_mae=pm["rgb_mae"], psnr_db=pm["psnr_db"],
                               reference_rgb_mae=rm["rgb_mae"], reference_psnr_db=rm["psnr_db"],
                               latent_mse=float(np.mean((predicted_cpu[relative]-reference[relative])**2)), empty_prediction=empty)
                    report["frames"].append(row)
                    if is_prediction:
                        episode_rows.append(row)
                    phase = f"rollout {k} (+{k/fps:.1f}s)" if is_prediction else "OBSERVED HISTORY"
                    panel = comparison_panel(target, ref_img, image, f"Episode {episode} | {task} | frame {t} | {phase}", phase)
                    save_rgb(Path(work)/f"frame_{relative:06d}.png", panel)
                    if relative in snapshot_indices:
                        save_rgb(a.out_dir/f"episode_{episode}_step_{k:03d}.png", panel)
                        selected_panels.append(panel)
                    if is_prediction and pm["psnr_db"] < worst_score:
                        worst_score, worst_panel = pm["psnr_db"], panel.copy()
                    print(f"episode={episode} frame={t} phase={'rollout' if is_prediction else 'observed'} step={k} PSNR={pm['psnr_db']:.2f}dB", flush=True)
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(Path(work)/"frame_%06d.png"),
                                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                str(a.out_dir/f"episode_{episode}_rollout.mp4")], check=True)
            save_rgb(a.out_dir/f"episode_{episode}_overview.png", np.concatenate(selected_panels, axis=0))
            save_rgb(a.out_dir/f"episode_{episode}_worst.png", worst_panel)
            overview.append(selected_panels[-1])
            np.savez_compressed(a.out_dir/f"episode_{episode}_predicted_latents.npz", z=predicted_cpu.astype(np.float16),
                                frame_indices=np.arange(a.start_frame, end), n_history=history_len)
            report["episodes"].append(dict(
                episode=episode, task=task, predicted_frames=steps, duration_ahead_s=steps/fps,
                mean_rgb_mae=float(np.mean([r["rgb_mae"] for r in episode_rows])),
                mean_psnr_db=float(np.mean([r["psnr_db"] for r in episode_rows])),
                reference_mean_psnr_db=float(np.mean([r["reference_psnr_db"] for r in episode_rows])),
                mean_latent_mse=float(np.mean([r["latent_mse"] for r in episode_rows])),
                empty_predictions=sum(r["empty_prediction"] for r in episode_rows)))
            (a.out_dir/"report.json").write_text(json.dumps(report, indent=2)+"\n")
            write_evaluation(a.out_dir, report)
    save_rgb(a.out_dir/"final_predictions.png", np.concatenate(overview, axis=0))
    print(f"Saved rollout videos and report to {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
