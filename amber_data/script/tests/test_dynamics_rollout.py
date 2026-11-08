"""CPU-only checks: python -m unittest discover -s amber_data/script/tests -p test_dynamics_rollout.py"""
import unittest
import numpy as np
import torch

from amber_data.script.rollout_dynamics import rollout_latents, load_dynamics, normalize_actions
from models.dynamics_coarse3d import CoarseDynamics3D


class RolloutTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_predictions_are_fed_back_and_actions_are_aligned(self):
        initial = torch.tensor([1., 2.]).reshape(1, 2, 1, 1, 1, 1)
        actions = torch.arange(5).float()[:, None]*10
        seen = []

        def predictor(history, past_actions, next_action):
            seen.append((history.flatten().tolist(), past_actions.flatten().tolist(), next_action.item()))
            return history[:, -1:] + (next_action.item()-past_actions[0, -1, 0])

        result = rollout_latents(initial, actions, predictor, 3)
        self.assertEqual(result.flatten().tolist(), [1., 2., 12., 22., 32.])
        self.assertEqual(seen, [([1., 2.], [0., 10.], 20.), ([2., 12.], [10., 20.], 30.), ([12., 22.], [20., 30.], 40.)])
        self.assertEqual(initial.flatten().tolist(), [1., 2.])
        self.assertFalse(result.requires_grad)

    def test_horizon_and_nonfinite_predictions_are_rejected(self):
        initial = torch.ones(1, 2, 1, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, "sufficient actions"):
            rollout_latents(initial, torch.zeros(3, 1), lambda *a: initial[:, :1], 2)
        with self.assertRaisesRegex(ValueError, "Invalid predicted latent"):
            rollout_latents(initial, torch.zeros(3, 1), lambda *a: torch.full_like(initial[:, :1], float("nan")), 1)

    def test_saved_normalization_is_used_without_recomputing(self):
        ck = dict(action_mean=np.array([10., 20.]), action_std=np.array([2., 5.]))
        np.testing.assert_allclose(normalize_actions(np.array([[12., 15.]], np.float32), ck), [[1., -1.]])
        ck["action_std"][0] = 0
        with self.assertRaisesRegex(ValueError, "Invalid actions"):
            normalize_actions(np.ones((1, 2)), ck)

    def test_model_architecture_is_restored_from_weights(self):
        torch.manual_seed(4)
        original = CoarseDynamics3D(latent_channels=2, action_dim=3, n_history=2, width=8,
                                   action_emb_dim=16, n_time_blocks=1, n_levels=2, film=True).eval()
        checkpoint = dict(args=dict(n_history=2), model_state_dict=original.state_dict())
        restored = load_dynamics(checkpoint, "cpu")
        z, actions, noise = torch.randn(1, 3, 2, 4, 4, 4), torch.randn(3, 1, 3), torch.zeros(1, 3)
        with torch.inference_mode():
            expected = original(z, actions, noise_std=noise)
            actual = restored(z, actions, noise_std=noise)
        torch.testing.assert_close(actual, expected)

    def test_empty_scene_is_visible_without_using_ground_truth(self):
        from amber_data.script.reconstruct_vae import Reconstructor

        class EmptyDecoder:
            grid_shapes = [(2, 2, 2)]

            def decode(self, z, **kwargs):
                self.kwargs = kwargs
                return {key: torch.full((1, 1, 2, 2, 2), -10.) for key in ("scene_occ_0", "tac_occ_0", "signal_0")}

        renderer = Reconstructor.__new__(Reconstructor)
        renderer.model, renderer.device, renderer.unlabelled = EmptyDecoder(), "cpu", True
        cam = dict(H=4, W=6)
        rgb, alpha, particles = renderer.decode_latent(torch.zeros(1, 8, 2, 2, 2), cam, allow_empty=True)
        self.assertEqual(renderer.model.kwargs, dict(use_gt_mask=False))
        self.assertEqual(rgb.shape, (4, 6, 3))
        self.assertFalse(rgb.any())
        self.assertFalse(alpha.any())
        self.assertEqual(len(particles["xyz"]), 0)
        with self.assertRaisesRegex(RuntimeError, "no occupied"):
            renderer.decode_latent(torch.zeros(1, 8, 2, 2, 2), cam)


if __name__ == "__main__":
    unittest.main()
