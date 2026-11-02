import torch

from models.particle_vae import ParticleVAE, extend_occ_heads
from models.tactile import TactileParticleHead


def make_vae(grid, x_range, y_range, z_range):
    return ParticleVAE(
        in_channels=6,
        encoder_channels=(16, 32, 64),
        bottleneck_shape=tuple(grid),
        x_range=tuple(x_range),
        y_range=tuple(y_range),
        z_range=tuple(z_range),
        latent_channels=8,
        grid_feat_dim=128,
        decoder_channels=(128, 64),
        final_z_kernel=3,
        conv_decoder_mid=64,
        decoder_level0_blocks=2,
        decoder_n_refine=2,
        decoder_iso_levels=(),
        decoder_up1_scale=2,
        decoder_grad_ckpt=False,
        encoder_feat_skip=True,
        encoder_residual=True,
        latent_split=4,
        color_branch_channels=None,
        color_branch_rgb_only=True,
        decoder_residual=True,
    )


def add_tactile(model, reach=2.0):
    extend_occ_heads(model.decoder)
    model.tactile_particle_head = TactileParticleHead(64, K=16, reach=reach)


def load_vae(ckpt, device):
    a = ckpt["args"]
    state = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("tactile_particle_head.")}
    grid = a["grid"] if "grid" in a else (a["nx"], a["ny"], a["nz"])
    model = make_vae(grid, a["x_range"], a["y_range"], a["z_range"])
    if any(".occ_head_" in k and v.shape[0] == 5 for k, v in state.items()):
        extend_occ_heads(model.decoder)
    model.load_state_dict(state)
    return model.to(device).eval()


def normalize_latent(mean, split=4):
    outs, c0 = [], 0
    for g in (mean.shape[1] - split, split):
        m = mean[:, c0 : c0 + g]
        B, C, D, H, W = m.shape
        f = m.reshape(B, C, -1)
        outs.append((f / (torch.norm(f, dim=1, keepdim=True) + 1e-8)).reshape(B, C, D, H, W))
        c0 += g
    return torch.cat(outs, 1)
