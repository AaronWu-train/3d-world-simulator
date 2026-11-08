import json
import cv2
import numpy as np
import pytest

from amber_data.script.convert_lerobot import (approximate_depth, load_extrinsics,
                                    load_intrinsics, scaled_intrinsics,
                                    split_episodes, video_frames)


def test_fixed_inverse_depth_is_monotonic_and_clipped():
    scale = dict(prediction_p02=2, prediction_p98=6, assumed_near_m=0.4, assumed_far_m=1.4)
    result = approximate_depth(np.array([-10, 2, 4, 6, 20]), scale)
    np.testing.assert_allclose(result[[0, 1]], 1.4)
    np.testing.assert_allclose(result[[3, 4]], 0.4)
    assert 0.4 < result[2] < 1.4
    # Scaling must not depend on which other pixels are present in this frame.
    assert approximate_depth(np.array([4]), scale)[0] == result[2]


def test_episode_split_is_deterministic_and_source_disjoint():
    episodes = [dict(task=task, source_dataset="original", source_episode=i + offset)
                for task, offset in [("folding", 0), ("unfolding", 10)] for i in range(10)]
    result = split_episodes(episodes, 0.2, 42)
    assert result == split_episodes(episodes, 0.2, 42)
    assert sum(e["split"] == "val" for e in result) == 4
    train = {e["source_episode"] for e in result if e["split"] == "train"}
    val = {e["source_episode"] for e in result if e["split"] == "val"}
    assert train.isdisjoint(val)
    with pytest.raises(ValueError, match="Duplicate source"):
        split_episodes(episodes + [episodes[0]], 0.2, 42)


def test_intrinsics_scale_and_reject_distortion(tmp_path):
    p = tmp_path / "intrinsics.json"
    data = dict(camera_matrix=[600, 0, 320, 0, 580, 240, 0, 0, 1], image_width=640, image_height=480, dist_coeffs=[0, 0, 0, 0, 0])
    p.write_text(json.dumps(data))
    k = scaled_intrinsics(load_intrinsics(p), 320, 240)
    np.testing.assert_allclose(k, [[300, 0, 160], [0, 290, 120], [0, 0, 1]])
    data["dist_coeffs"][0] = 0.1
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="undistortion"):
        load_intrinsics(p)


def test_camera_to_base_is_not_inverted_on_load(tmp_path):
    p = tmp_path / "cam.txt"
    e = np.eye(4); e[:3, 3] = [0.6, 0.1, 0.5]
    np.savetxt(p, e)
    np.testing.assert_allclose(load_extrinsics(p)["matrix"], e)
    e[0, 0] = -1
    np.savetxt(p, e)
    with pytest.raises(ValueError, match="proper rotation"):
        load_extrinsics(p)


def test_video_pts_select_the_requested_source_frames(tmp_path):
    # An episode starts midway through a shared file. Select known colors at
    # requested PTS, independently of decoder packet order or keyframe seeking.
    path = tmp_path / "source.avi"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (32, 24))
    assert out.isOpened()
    for i in range(40):
        out.write(np.full((24, 32, 3), i * 5, np.uint8))
    out.release()
    decoded = list(video_frames(path, np.array([1.0, 1.2, 1.4]), 32, 24, 20))
    np.testing.assert_allclose([f.mean() for f in decoded], [100, 120, 140], atol=3)
    with pytest.raises(ValueError, match="ended early"):
        list(video_frames(path, np.array([3.0]), 32, 24, 20))
