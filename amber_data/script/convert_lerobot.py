"""Convert local LeRobot v3 demonstrations to explicitly approximate packed RGB-D.

No hardware is accessed. No metric calibration or semantic labels are invented.
See amber_data/script/README.md for the assumptions and training restrictions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
BBOX = [-1.0, 1.0, -1.0, 1.0, -1.0, 0.8]


def load_extrinsics(path):
    E = np.loadtxt(path, dtype=np.float64)
    if E.shape != (4, 4) or not np.isfinite(E).all() or not np.allclose(E[3], [0, 0, 0, 1]):
        raise ValueError("Extrinsics must be a finite 4x4 camera-to-base rigid transform")
    if not np.allclose(E[:3, :3].T @ E[:3, :3], np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(E[:3, :3]), 1, atol=1e-4):
        raise ValueError("Camera rotation is not a proper rotation")
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                convention="OpenCV camera-to-robot-base; supplied by user", matrix=E.tolist(),
                note="Recording-camera correspondence supplied by user; intrinsics and depth scale remain assumed")


def load_intrinsics(path):
    data = json.loads(path.read_text())
    K = np.asarray(data["camera_matrix"], np.float64).reshape(3, 3)
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError("Invalid pinhole camera matrix")
    if np.any(np.asarray(data.get("dist_coeffs", [0]), np.float64) != 0):
        raise ValueError("Nonzero distortion needs image undistortion, which this converter does not implement")
    w, h = int(data["image_width"]), int(data["image_height"])
    if min(w, h) <= 0:
        raise ValueError("Invalid calibration image size")
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                matrix=K.tolist(), image_width=w, image_height=h, distortion="none")


def scaled_intrinsics(calibration, width, height):
    return (np.diag([width/calibration["image_width"], height/calibration["image_height"], 1])
            @ np.asarray(calibration["matrix"])).astype(np.float32)


def calibration_name(manifest):
    return ("provided_camera_to_robot_base" if "camera_extrinsics" in manifest else "assumed_camera_centered_world") + "; " + (
        "provided_intrinsics" if "camera_intrinsics" in manifest else "assumed_intrinsics")


def apply_calibration(root, extrinsics=None, intrinsics=None):
    report = root / "conversion.json"
    manifest = json.loads(report.read_text())
    if not manifest["complete"]:
        raise ValueError("Wait for conversion completion before updating all camera poses")
    if extrinsics is not None:
        manifest["camera_extrinsics"] = extrinsics
    if intrinsics is not None:
        manifest["camera_intrinsics"] = intrinsics
    E = np.asarray(manifest["camera_extrinsics"]["matrix"], np.float32) if "camera_extrinsics" in manifest else None
    for entry in manifest["episodes"]:
        with h5py.File(root / entry["path"], "r+") as g:
            if E is not None:
                g["cam/camera_0/E_cam2world"][:] = E
                g["render/camera_0/viewmat"][:] = np.linalg.inv(E)
                g.attrs["extrinsics_sha256"] = manifest["camera_extrinsics"]["sha256"]
            if "camera_intrinsics" in manifest:
                K = scaled_intrinsics(manifest["camera_intrinsics"], int(g.attrs["W"]), int(g.attrs["H"]))
                g["cam/camera_0/K_ir"][:] = K
                g["render/camera_0/K"][:] = K
                g.attrs["intrinsics_sha256"] = manifest["camera_intrinsics"]["sha256"]
            g.attrs["camera_calibration"] = calibration_name(manifest)
            g.attrs["bbox"] = BBOX
    manifest["assumptions"][0] = "Monocular depth scale remains assumed, not physically calibrated"
    manifest["assumptions"][1] = calibration_name(manifest)
    tmp = report.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(report)
    (root / "validation.json").unlink(missing_ok=True)
    print(f"Updated camera calibration in {len(manifest['episodes'])} episodes")


def read_rows(path):
    return pq.read_table(path).to_pylist()


def video_frames(path, timestamps, width, height, fps):
    """Seek on video PTS, then match each requested sample within half a frame."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(timestamps[0] / stream.time_base), stream=stream, backward=True)
        i = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            if t < timestamps[i] - 0.5 / fps:
                continue
            if abs(t - timestamps[i]) > 0.51 / fps:
                raise ValueError(f"Video/row alignment failed: {path}, wanted {timestamps[i]}, got {t}")
            yield cv2.resize(frame.to_ndarray(format="rgb24"), (width, height), interpolation=cv2.INTER_AREA)
            i += 1
            if i == len(timestamps):
                return
        raise ValueError(f"Video ended early: {path}, decoded {i}/{len(timestamps)} samples")


class DepthEstimator:
    def __init__(self, path, device, size):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(path, local_files_only=True).to(device).eval()
        self.size = size

    @torch.inference_mode()
    def __call__(self, rgb):
        inputs = self.processor(images=rgb, return_tensors="pt", size={"height": self.size, "width": self.size})
        pred = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).predicted_depth
        return torch.nn.functional.interpolate(
            pred[:, None], size=rgb.shape[:2], mode="bicubic", align_corners=False
        )[0, 0].cpu().numpy()


def load_source(root):
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError("Only LeRobot v3.0 is supported")
    if set(k for k in info["features"] if k.startswith("observation.images.")) != {"observation.images.overhead"}:
        raise ValueError("Expected one overhead camera")
    if info["features"]["action"]["shape"] != [7]:
        raise ValueError("Expected seven Piper action coordinates")
    rows = []
    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        rows.extend(read_rows(path))
    episodes = []
    for path in sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet")):
        episodes.extend(read_rows(path))
    if len(rows) != info["total_frames"] or len(episodes) != info["total_episodes"]:
        raise ValueError(f"{root}: inconsistent frame/episode counts")
    by_episode = {}
    for row in rows:
        by_episode.setdefault(row["episode_index"], []).append(row)
    mapping = json.loads((root / "meta/source_episode_indices.json").read_text())
    for ep in episodes:
        eid = int(ep["episode_index"])
        samples = sorted(by_episode[eid], key=lambda r: r["frame_index"])
        if len(samples) != ep["length"] or [r["frame_index"] for r in samples] != list(range(ep["length"])):
            raise ValueError(f"{root}: non-contiguous episode {eid}")
        if [r["index"] for r in samples] != list(range(ep["dataset_from_index"], ep["dataset_to_index"])):
            raise ValueError(f"{root}: incorrect global indices in episode {eid}")
        ep["samples"] = samples
        ep["source_episode"] = int(mapping["source_episode_indices"][eid])
        ep["root"] = root
        ep["source_dataset"] = mapping["source_dataset"]
        ep["fps"] = float(info["fps"])
        ep["task"] = root.name
    return sorted(episodes, key=lambda ep: ep["episode_index"])


def video_path(ep):
    key = "videos/observation.images.overhead/"
    return ep["root"] / (f"videos/observation.images.overhead/chunk-{ep[key + 'chunk_index']:03d}/"
                         f"file-{ep[key + 'file_index']:03d}.mp4")


def sample_times(ep, stride):
    samples = ep["samples"][::stride]
    start = ep["videos/observation.images.overhead/from_timestamp"]
    times = np.array([start + r["timestamp"] for r in samples], np.float64)
    return samples, times


def split_episodes(episodes, val_fraction, seed):
    # One source episode can never appear in both train and validation.
    result = []
    for task in sorted({e["task"] for e in episodes}):
        group = [e for e in episodes if e["task"] == task]
        order = np.random.default_rng(seed).permutation(len(group))
        nval = max(1, min(len(group) - 1, round(len(group) * val_fraction)))
        if len(group) < 2:
            raise ValueError("Need at least two episodes per task for a train/val split")
        validation = set(int(i) for i in order[:nval])
        for i, ep in enumerate(group):
            result.append(dict(ep, split="val" if i in validation else "train"))
    keys = [(e["source_dataset"], e["source_episode"]) for e in result]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate source episodes across input datasets")
    return result


def fit_depth_scale(episodes, estimator, args):
    # Fit ONE inverse-depth affine transform using training images only.
    values = []
    train = [e for e in episodes if e["split"] == "train"]
    for ep in train[::max(1, len(train) // 8)][:8]:
        _, times = sample_times(ep, args.stride)
        rgb = next(video_frames(video_path(ep), times[:1], args.width, args.height, ep["fps"]))
        values.append(estimator(rgb).reshape(-1))
    lo, hi = np.percentile(np.concatenate(values), [2, 98])
    if not np.isfinite([lo, hi]).all() or hi - lo < 1e-5:
        raise ValueError("Depth predictions have insufficient variation")
    return {"prediction_p02": float(lo), "prediction_p98": float(hi),
            "assumed_near_m": args.near, "assumed_far_m": args.far,
            "fit": "training-only; single fixed affine inverse-depth mapping; not metric calibration"}


def approximate_depth(pred, scale):
    near, far = scale["assumed_near_m"], scale["assumed_far_m"]
    u = np.clip((pred - scale["prediction_p02"]) / (scale["prediction_p98"] - scale["prediction_p02"]), 0, 1)
    return (1 / (1 / far + u * (1 / near - 1 / far))).astype(np.float32)


def convert_episode(ep, estimator, scale, args, manifest):
    # IDs remain the original recording IDs, even after splitting into tasks.
    dst = args.out / ep["split"] / f"episode_{ep['source_episode']}.hdf5"
    if dst.exists():
        raise FileExistsError(f"Refusing to replace {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    samples, times = sample_times(ep, args.stride)
    n, h, w = len(samples), args.height, args.width
    f = w / (2 * np.tan(np.deg2rad(args.fov) / 2))
    K = np.array([[f, 0, (w-1)/2], [0, f, (h-1)/2], [0, 0, 1]], np.float32)
    if "camera_intrinsics" in manifest:
        K = scaled_intrinsics(manifest["camera_intrinsics"], w, h)
    E = np.diag([1., -1., -1., 1.]).astype(np.float32)
    E[2, 3] = 1.0
    if "camera_extrinsics" in manifest:
        E = np.asarray(manifest["camera_extrinsics"]["matrix"], np.float32)
    tmp = dst.with_suffix(".hdf5.tmp")
    try:
        with h5py.File(tmp, "w") as g:
            g.attrs.update(episode=ep["source_episode"], num_frames=n, H=h, W=w,
                           cam_ids=[0], render_cam_ids=[0], bbox=BBOX, has_obj_mask=False,
                           depth_semantics="monocular_relative_estimate", semantic_supervision="none",
                           camera_calibration="assumed_pinhole_camera_centered_world",
                           fps=ep["fps"]/args.stride, task=ep["task"], source_root=str(ep["root"].resolve()),
                           source_dataset=ep["source_dataset"], source_episode=ep["source_episode"],
                           local_episode=ep["episode_index"], action_semantics="absolute_joint_targets_deg_gripper_mm",
                           depth_model=MODEL, depth_model_revision=REVISION, conversion_fingerprint=manifest["fingerprint"])
            if "camera_extrinsics" in manifest:
                g.attrs["extrinsics_sha256"] = manifest["camera_extrinsics"]["sha256"]
            if "camera_intrinsics" in manifest:
                g.attrs["intrinsics_sha256"] = manifest["camera_intrinsics"]["sha256"]
            g.attrs["camera_calibration"] = calibration_name(manifest)
            for key, column in [("action/joint_action", "action"), ("observation/joint_state", "observation.state")]:
                data = np.asarray([r[column] for r in samples], np.float32)
                if data.shape != (n, 7) or not np.isfinite(data).all():
                    raise ValueError(f"Invalid {column} in {ep['root']}")
                g.create_dataset(key, data=data)
            # Historical key consumed by export_latents; explicit attributes prevent semantic ambiguity.
            g["action/cartesian_action"] = g["action/joint_action"]
            g["action/cartesian_action"].attrs["semantics"] = "Piper joint targets, NOT Cartesian poses"
            g.create_dataset("source/frame_index", data=[r["frame_index"] for r in samples])
            g.create_dataset("source/timestamp", data=[r["timestamp"] for r in samples])
            g.create_dataset("source/video_timestamp", data=times)
            for key, data in [("cam/camera_0/K_ir", K), ("cam/camera_0/E_cam2world", E),
                              ("render/camera_0/K", K), ("render/camera_0/viewmat", np.linalg.inv(E))]:
                g.create_dataset(key, data=np.repeat(data[None], n, axis=0))
            render = g["render/camera_0"]
            render.attrs.update(H=h, W=w, crop=[0, 0, w, h])
            depth = g.create_dataset("img/camera_0/depth", (n, h, w), dtype="float16", chunks=(1, h, w), compression="lzf")
            valid = g.create_dataset("img/camera_0/valid_mask", (n, (h*w+7)//8), dtype="uint8", compression="lzf")
            jpg = g.create_dataset("img/camera_0/rgb_in_ir", (n,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            g["render/camera_0/jpg"] = jpg
            for i, rgb in enumerate(video_frames(video_path(ep), times, w, h, ep["fps"])):
                z = approximate_depth(estimator(rgb), scale)
                if not np.isfinite(z).all() or np.any(z <= 0):
                    raise ValueError("Non-finite or non-positive estimated depth")
                depth[i] = z
                valid[i] = np.packbits(np.isfinite(z) & (z > 0))
                ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                jpg[i] = encoded
        tmp.rename(dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dict(path=str(dst.relative_to(args.out)), task=ep["task"], split=ep["split"],
                source_episode=ep["source_episode"], local_episode=int(ep["episode_index"]), frames=n)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("amber_data/20261017_152333_cloth_folding_split"))
    p.add_argument("--out", type=Path, default=Path("data/amber_cloth_estimated"))
    p.add_argument("--model-path", default=".cache/depth-anything-v2-small")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--inference-size", type=int, default=196)
    p.add_argument("--stride", type=int, default=4, help="20 FPS input -> 5 FPS output")
    p.add_argument("--near", type=float, default=0.4)
    p.add_argument("--far", type=float, default=1.4)
    p.add_argument("--fov", type=float, default=60, help="ASSUMED horizontal field of view in degrees")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-episodes", type=int, default=None, help="Smoke run; leaves completion status false")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--extrinsics", type=Path, help="OpenCV camera-to-base 4x4 text matrix")
    p.add_argument("--intrinsics", type=Path, help="Pinhole intrinsics JSON with original image size")
    p.add_argument("--apply-calibration-only", action="store_true", help="Update a complete dataset without rerunning depth")
    a = p.parse_args()
    if a.apply_calibration_only:
        if a.extrinsics is None and a.intrinsics is None:
            p.error("--apply-calibration-only requires --extrinsics or --intrinsics")
        apply_calibration(a.out, load_extrinsics(a.extrinsics) if a.extrinsics else None,
                          load_intrinsics(a.intrinsics) if a.intrinsics else None)
        return
    if min(a.width, a.height, a.stride, a.threads) <= 0 or not 0 < a.near < a.far or not 1 < a.fov < 179:
        p.error("Invalid dimensions, stride, thread count, depth range, or field of view")
    if not 0 < a.val_fraction < 1 or a.inference_size < 28 or a.inference_size % 14:
        p.error("Need 0 < val-fraction < 1 and inference-size divisible by 14")
    torch.set_num_threads(a.threads)
    cv2.setNumThreads(1)
    all_episodes = []
    for task in ("folding", "unfolding"):
        all_episodes += load_source(a.source / task)
    episodes = split_episodes(all_episodes, a.val_fraction, a.seed)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items() if k not in ("resume", "limit_episodes", "out", "extrinsics", "intrinsics", "apply_calibration_only")}
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    report = a.out / "conversion.json"
    if report.exists():
        if not a.resume:
            raise FileExistsError("Output exists; use --resume with identical settings")
        manifest = json.loads(report.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Resume settings differ from previous conversion")
    else:
        manifest = dict(fingerprint=fingerprint, settings=settings, model=MODEL, revision=REVISION,
                        complete=False, episodes=[], approximate_geometry=True,
                        assumptions=["No metric scale, intrinsics, or camera pose calibration",
                                     "60 degree FOV unless overridden; pseudo-world origin below camera",
                                     "No semantic or tactile labels; train without these losses",
                                     "Single-view surfaces only; occluded geometry is unavailable"])
    for key, path, load in (("camera_extrinsics", a.extrinsics, load_extrinsics), ("camera_intrinsics", a.intrinsics, load_intrinsics)):
        if path is not None:
            calibration = load(path)
            if manifest["episodes"] and manifest.get(key, {}).get("sha256") != calibration["sha256"]:
                raise ValueError("Use --apply-calibration-only to update an existing conversion")
            manifest[key] = calibration
    if "camera_extrinsics" in manifest or "camera_intrinsics" in manifest:
        manifest["assumptions"][0] = "Monocular depth scale remains assumed, not physically calibrated"
        manifest["assumptions"][1] = calibration_name(manifest)
    estimator = DepthEstimator(a.model_path, a.device, a.inference_size)
    if "depth_scale" not in manifest:
        manifest["depth_scale"] = fit_depth_scale(episodes, estimator, a)
    a.out.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = report.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        tmp.replace(report)

    save()
    selected = episodes if a.limit_episodes is None else episodes[:a.limit_episodes]
    done = {e["source_episode"] for e in manifest["episodes"]}
    for i, ep in enumerate(selected):
        if ep["source_episode"] in done:
            dst = a.out / ep["split"] / f"episode_{ep['source_episode']}.hdf5"
            with h5py.File(dst, "r") as g:
                if g.attrs["conversion_fingerprint"] != fingerprint:
                    raise ValueError(f"Mismatched output {dst}")
            continue
        entry = convert_episode(ep, estimator, manifest["depth_scale"], a, manifest)
        manifest["episodes"].append(entry)
        save()
        print(f"[{i+1}/{len(selected)}] {entry['path']}: {entry['frames']} frames ({ep['task']})", flush=True)
    manifest["complete"] = len(manifest["episodes"]) == len(episodes)
    save()
    print(f"Converted {len(manifest['episodes'])}/{len(episodes)} episodes; complete={manifest['complete']}", flush=True)


if __name__ == "__main__":
    main()
