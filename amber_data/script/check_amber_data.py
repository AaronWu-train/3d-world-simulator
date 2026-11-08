"""Validate converted geometry through the real readers; export a preview and PLY."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.aloha_packed import AlohaPackedPCDataset
from datasets.frames import Frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/amber_cloth_estimated"))
    p.add_argument("--preview", action="store_true")
    a = p.parse_args()
    manifest = json.loads((a.root / "conversion.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Conversion is incomplete; resume it before validating")
    expected = {e["path"] for e in manifest["episodes"]}
    actual = {str(f.relative_to(a.root)) for split in ("train", "val") for f in (a.root / split).glob("episode_*.hdf5")}
    if expected != actual:
        raise ValueError("Manifest/files differ")
    seen, counts, frames, source_cache = set(), Counter(), Counter(), {}
    for entry in manifest["episodes"]:
        path = a.root / entry["path"]
        with h5py.File(path, "r") as f:
            key = (f.attrs["source_dataset"], int(f.attrs["source_episode"]))
            if key in seen:
                raise ValueError("Source episode duplicated across outputs")
            seen.add(key)
            n, h, w = int(f.attrs["num_frames"]), int(f.attrs["H"]), int(f.attrs["W"])
            assert n == entry["frames"]
            assert f.attrs["depth_semantics"] == "monocular_relative_estimate"
            assert f.attrs["semantic_supervision"] == "none"
            assert f["action/joint_action"].shape == (n, 7)
            assert f["action/cartesian_action"].id == f["action/joint_action"].id
            assert np.isfinite(f["action/joint_action"][:]).all()
            assert np.isfinite(f["observation/joint_state"][:]).all()
            source_root = str(f.attrs["source_root"])
            if source_root not in source_cache:
                rows = []
                for source_file in sorted((Path(source_root) / "data").glob("chunk-*/file-*.parquet")):
                    rows += pq.read_table(source_file).to_pylist()
                source_cache[source_root] = {(r["episode_index"], r["frame_index"]): r for r in rows}
            source = source_cache[source_root]
            original = [source[(int(f.attrs["local_episode"]), int(t))] for t in f["source/frame_index"][:]]
            np.testing.assert_array_equal(f["action/joint_action"][:], np.asarray([r["action"] for r in original], np.float32))
            np.testing.assert_array_equal(f["observation/joint_state"][:], np.asarray([r["observation.state"] for r in original], np.float32))
            np.testing.assert_allclose(f["source/timestamp"][:], [r["timestamp"] for r in original])
            assert np.all(np.diff(f["source/timestamp"][:]) > 0)
            depth = f["img/camera_0/depth"][:].astype(np.float32)
            assert depth.shape == (n, h, w) and np.isfinite(depth).all() and np.all(depth > 0)
            assert depth.min() >= manifest["depth_scale"]["assumed_near_m"] - 0.002
            assert depth.max() <= manifest["depth_scale"]["assumed_far_m"] + 0.002
            K, E = f["cam/camera_0/K_ir"][:], f["cam/camera_0/E_cam2world"][:]
            assert np.allclose(E @ f["render/camera_0/viewmat"][:], np.eye(4), atol=1e-5)
            if "camera_extrinsics" in manifest:
                assert np.allclose(E, manifest["camera_extrinsics"]["matrix"], atol=1e-6)
            if "camera_intrinsics" in manifest:
                from amber_data.script.convert_lerobot import scaled_intrinsics
                assert np.allclose(K, scaled_intrinsics(manifest["camera_intrinsics"], w, h))
            assert (K[:, 0, 0] > 0).all() and (K[:, 1, 1] > 0).all()
            masks = np.unpackbits(f["img/camera_0/valid_mask"][:], axis=1)[:, :h*w]
            assert masks.all()
            for t in (0, n-1):
                jpg = cv2.imdecode(f["img/camera_0/rgb_in_ir"][t], cv2.IMREAD_COLOR)
                assert jpg.shape == (h, w, 3)
            counts[(entry["split"], entry["task"])] += 1
            frames[entry["split"]] += n
    # Exercise actual training readers for first / middle / last frame per split.
    for split in ("train", "val"):
        ds = AlohaPackedPCDataset(a.root / split, n_points=2048)
        reader = Frames(ds.files, "cpu")
        for i in (0, len(ds)//2, len(ds)-1):
            sample = ds[i]
            assert sample["xyz"].shape == (2048, 3) and sample["xyz"].isfinite().all()
            assert sample["rgb"].min() >= -1 and sample["rgb"].max() <= 1
            fi, t = ds.index[i]
            render = reader.render(fi, t)[0]
            assert not render["gripper"].any() and render["obj"] is None
            depth = reader.depth(fi, t)[0]
            assert depth["gt"].isfinite().all() and depth["gt"].min() > 0
        if a.preview and split == "train":
            preview(a.root, ds)
    report = dict(episodes=len(seen), counts={f"{s}/{t}": n for (s,t), n in counts.items()},
                  frames=dict(frames), checks="All episode depth/action arrays; exact actions/states against source; split isolation; calibration; real point/render/depth readers",
                  gpu_training_tested=False)
    (a.root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def preview(root, ds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Mid-episode image, predicted depth, and oblique point cloud view.
    with h5py.File(ds.files[0], "r") as f:
        t = int(f.attrs["num_frames"]) // 2
        rgb = cv2.cvtColor(cv2.imdecode(f["img/camera_0/rgb_in_ir"][t], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        depth = f["img/camera_0/depth"][t].astype(np.float32)
    full = AlohaPackedPCDataset(Path(ds.files[0]).parent, n_points=None)
    pts = full[t]
    xyz = pts["xyz"].numpy()[::4]
    color = np.clip((pts["rgb"].numpy()[::4] + 1) * 127.5, 0, 255).astype(np.uint8)
    with (root / "preview.ply").open("w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(f, np.column_stack((xyz, color)), fmt="%.5f %.5f %.5f %d %d %d")
    fig = plt.figure(figsize=(13, 4))
    ax = fig.add_subplot(131); ax.imshow(rgb); ax.set_title("Recorded RGB"); ax.axis("off")
    ax = fig.add_subplot(132); im = ax.imshow(depth, cmap="viridis"); ax.set_title("Estimated depth (assumed scale)"); ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.6)
    ax = fig.add_subplot(133, projection="3d")
    ax.scatter(*xyz.T, c=color/255, s=0.4)
    ax.view_init(elev=35, azim=-65)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Robot-base frame; estimated scale")
    fig.tight_layout(); fig.savefig(root / "preview.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
