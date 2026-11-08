"""CPU loss-path check using the training environment (no CUDA kernels needed).

Run: conda run -n 3dws python -m unittest discover -s amber_data/script/tests -p test_unlabelled_training.py
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scripts import train_vae


class TinyScene(torch.nn.Module):
    """Small differentiable outputs isolate supervision routing from sparse CUDA."""
    def __init__(self):
        super().__init__()
        self.grid_shapes = [(2, 2, 2)]
        self.bounds_min_world = torch.tensor([-1., -1., -1.])
        self.bounds_max_world = torch.tensor([1., 1., 1.])
        self.scene = torch.nn.Parameter(torch.zeros(1, 1, 2, 2, 2))
        self.robot = torch.nn.Parameter(torch.ones(1, 1, 2, 2, 2))
        self.cloth = torch.nn.Parameter(torch.ones(1, 1, 2, 2, 2))
        self.rgb = torch.nn.Parameter(torch.zeros(1, 3, 2, 2, 2))
        self.latent = torch.nn.Parameter(torch.ones(1, 8, 2, 2, 2))
        self.latent_split = 4
        self.encoder = SimpleNamespace(_compute_bottleneck_occ_mask=lambda *a, **kw: torch.ones(1, 1, 2, 2, 2, dtype=torch.bool))

    def encode(self, *a, **kw):
        return self.latent, None

    def decode(self, *a, **kw):
        return dict(scene_occ_0=self.scene, tac_occ_0=self.robot, signal_0=self.cloth, rgb_2=self.rgb)


class UnlabelledTrainingTest(unittest.TestCase):
    def test_missing_labels_do_not_supervise_semantic_heads(self):
        torch.set_num_threads(1)
        model = TinyScene()
        m = SimpleNamespace(model=model, device="cpu", features={}, tactile=False,
                            pos_mean=np.zeros(3), pos_std=np.ones(3), semantic_supervision=False,
                            estimated_depth=True, bbox_min=model.bounds_min_world,
                            bbox_extent=model.bounds_max_world-model.bounds_min_world)
        batch = dict(xyz=torch.linspace(-0.5, 0.5, 48).reshape(1, 16, 3),
                     rgb=torch.full((1, 16, 3), 0.2), is_rope=torch.zeros(1, 16, dtype=torch.bool),
                     root=["missing-labels"], ep_idx=[0], frame_idx=[0], file_idx=[0])
        with patch.object(train_vae, "load_frame", side_effect=AssertionError("Must not load missing labels")):
            loss, metric = train_vae.step(m, batch, True, False, None, 0, 1., 0.1, None)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.scene.grad.abs().sum()), 0)
        self.assertGreater(float(model.rgb.grad.abs().sum()), 0)
        self.assertIsNone(model.robot.grad)
        self.assertIsNone(model.cloth.grad)


if __name__ == "__main__":
    unittest.main()
