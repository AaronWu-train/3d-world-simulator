from __future__ import annotations
import torch
import torch.nn as nn
from .occupancy_vae import OccupancyVAELocal, LocalHierarchicalOccupancyDecoder


def _zero_init_extra_channels(conv: nn.Conv3d) -> None:
    with torch.no_grad():
        conv.weight[1:].zero_()
        conv.bias[1:].fill_(-2.94)


class _ParticleDecoder(LocalHierarchicalOccupancyDecoder):
    FINEST_ALIAS = 2

    def __init__(self, grid_feat_dim, channels, final_z_kernel=3, **dec_kwargs):
        super().__init__(grid_feat_dim=grid_feat_dim, channels=channels, final_z_kernel=final_z_kernel, **dec_kwargs)
        for i, ch in enumerate(self.channels):
            setattr(self, f"occ_head_{i}", nn.Conv3d(ch, 3, kernel_size=1))
            _zero_init_extra_channels(self.occ_head(i))

    def forward(self, grid_features, occ_mask_0=None, use_gt_mask=True, return_features=False):
        outs, rgb, feat = self._run_levels(grid_features, lambda i, f: self.occ_head(i)(f))
        outputs = {}
        for i, o in enumerate(outs):
            outputs[f"scene_occ_{i}"] = o[:, 0:1]
            outputs[f"tac_occ_{i}"] = o[:, 1:2]
            outputs[f"signal_{i}"] = o[:, 2:3]
            if o.shape[1] >= 5:
                outputs[f"tactile_occ_{i}"] = o[:, 3:4]
                outputs[f"tactile_sig_{i}"] = o[:, 4:5]
        outputs[f"rgb_{self.finest}"] = rgb
        if return_features:
            outputs[f"feat_{self.finest}"] = feat
        if self.finest != self.FINEST_ALIAS:
            for k in (
                ("scene_occ", "tac_occ", "signal", "rgb")
                + (("feat",) if return_features else ())
                + (("tactile_occ", "tactile_sig") if f"tactile_occ_{self.finest}" in outputs else ())
            ):
                outputs[f"{k}_{self.FINEST_ALIAS}"] = outputs[f"{k}_{self.finest}"]
        return outputs


def extend_occ_heads(decoder, n_out=5, extra_bias=-4.0):
    for i, ch in enumerate(decoder.channels):
        old = decoder.occ_head(i)
        if old.out_channels >= n_out:
            continue
        new = nn.Conv3d(old.in_channels, n_out, kernel_size=1).to(old.weight.device, old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.bias.fill_(extra_bias)
            new.weight[: old.out_channels] = old.weight
            new.bias[: old.out_channels] = old.bias
        setattr(decoder, f"occ_head_{i}", new)
    return decoder


class ParticleVAE(OccupancyVAELocal):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        old = self.decoder
        channels = tuple(old.channels)
        gfd = old.level0_conv[0].in_channels
        if isinstance(old.final_z_mix, nn.Identity):
            fzk = 1
        else:
            fzk = old.final_z_mix[0].kernel_size[0]
        self.decoder = _ParticleDecoder(grid_feat_dim=gfd, channels=channels, final_z_kernel=fzk, **self._decoder_kwargs)

    def load_scene_backbone(self, scene_ckpt_path: str | None = None, state_dict: dict | None = None) -> dict:
        if state_dict is None:
            assert scene_ckpt_path is not None
            ck = torch.load(scene_ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ck["model_state_dict"]
        sd = dict(state_dict)
        target = self.state_dict()
        for i in range(len(self.grid_shapes)):
            wkey = f"decoder.occ_head_{i}.weight"
            bkey = f"decoder.occ_head_{i}.bias"
            if wkey in sd and target[wkey].shape != sd[wkey].shape:
                new_w = target[wkey].clone()
                new_b = target[bkey].clone()
                new_w[0] = sd[wkey][0]
                new_b[0] = sd[bkey][0]
                sd[wkey] = new_w
                sd[bkey] = new_b
        info = self.load_state_dict(sd, strict=False)
        return info


OccupancyVAELocalTactileHead = ParticleVAE
