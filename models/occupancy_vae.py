import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv


class VoxelGrid(nn.Module):

    def __init__(self, nx=12, ny=12, nz=4, x_range=(-0.18, 0.92), y_range=(-0.98, 0.76), z_range=(-0.21, 0.34)):
        super().__init__()
        self.nx, self.ny, self.nz = (nx, ny, nz)
        self.n_voxels = nx * ny * nz
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range
        self.register_buffer("bounds_min", torch.tensor([x_min, y_min, z_min]))
        self.register_buffer("bounds_max", torch.tensor([x_max, y_max, z_max]))
        dx = (x_max - x_min) / nx
        dy = (y_max - y_min) / ny
        dz = (z_max - z_min) / nz
        self.register_buffer("spacing", torch.tensor([dx, dy, dz]))
        self.register_buffer("grid_dims", torch.tensor([nx, ny, nz]))
        cx = torch.linspace(x_min + dx / 2, x_max - dx / 2, nx)
        cy = torch.linspace(y_min + dy / 2, y_max - dy / 2, ny)
        cz = torch.linspace(z_min + dz / 2, z_max - dz / 2, nz)
        centers = torch.stack(torch.meshgrid(cx, cy, cz, indexing="ij"), dim=-1)
        self.register_buffer("centers", centers)
        self.register_buffer("centers_flat", centers.reshape(-1, 3))


class VoxelConvDecoder(nn.Module):

    def __init__(self, latent_channels=8, mid_channels=64, out_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(latent_channels, mid_channels, 3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU(),
            nn.Conv3d(mid_channels, mid_channels, 3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU(),
            nn.Conv3d(mid_channels, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class SparseResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, key):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.c1 = spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=False, indice_key=key)
        self.n2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.c2 = spconv.SubMConv3d(out_ch, out_ch, 3, padding=1, bias=False, indice_key=key)
        self.short = spconv.SubMConv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else None

    def forward(self, x):
        h = x.replace_feature(F.silu(self.n1(x.features)))
        h = self.c1(h)
        h = h.replace_feature(F.silu(self.n2(h.features)))
        h = self.c2(h)
        idt = self.short(x) if self.short is not None else x
        return h.replace_feature(h.features + idt.features)


class SparseResStage(nn.Module):

    def __init__(self, in_ch, out_ch, key):
        super().__init__()
        self.b1 = SparseResBlock(in_ch, out_ch, key)
        self.b2 = SparseResBlock(out_ch, out_ch, key)
        self.down = spconv.SparseConv3d(out_ch, out_ch, 3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.down(self.b2(self.b1(x)))


class SparseDownBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(inplace=True),
            spconv.SparseConv3d(out_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SpconvEncoder(nn.Module):

    def __init__(
        self,
        in_channels=6,
        channels=(32, 64, 128, 128),
        latent_channels=4,
        bottleneck_shape=(24, 24, 4),
        spatial_bounds=((-0.18, 0.92), (-0.98, 0.76), (-0.21, 0.34)),
        feat_skip=False,
        residual=False,
        latent_split=0,
        color_channels=None,
        color_rgb_only=False,
    ):
        super().__init__()
        self.feat_skip = bool(feat_skip)
        self.residual = bool(residual)
        self.latent_split = int(latent_split)
        assert self.latent_split < latent_channels
        self.geo_ch = latent_channels - self.latent_split
        self.color_rgb_only = bool(color_rgb_only)
        n_down = len(channels)
        factor = 2**n_down
        bx, by, bz = bottleneck_shape
        input_shape = np.array([bx * factor, by * factor, bz * factor])
        bounds_min = np.array([b[0] for b in spatial_bounds], dtype=np.float32)
        bounds_max = np.array([b[1] for b in spatial_bounds], dtype=np.float32)
        extent = bounds_max - bounds_min
        voxel_sizes = extent / input_shape
        self.register_buffer("bounds_min", torch.tensor(bounds_min))
        self.register_buffer("bounds_max", torch.tensor(bounds_max))
        self.register_buffer("voxel_sizes", torch.tensor(voxel_sizes, dtype=torch.float32))
        self.spatial_shape = [int(input_shape[2]), int(input_shape[1]), int(input_shape[0])]
        self.bottleneck_shape_xyz = [bx, by, bz]
        if residual:
            self.stem = spconv.SparseSequential(spconv.SubMConv3d(in_channels, channels[0], 1, bias=False))
            self.down_blocks = nn.ModuleList()
            prev_ch = channels[0]
            for i, ch in enumerate(channels):
                self.down_blocks.append(SparseResStage(prev_ch, ch, key=f"res{i}"))
                prev_ch = ch
        else:
            self.stem = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, channels[0], 1, bias=False),
                nn.BatchNorm1d(channels[0]),
                nn.SiLU(inplace=True),
                spconv.SubMConv3d(channels[0], channels[0], 3, padding=1, bias=False),
                nn.BatchNorm1d(channels[0]),
                nn.SiLU(inplace=True),
            )
            self.down_blocks = nn.ModuleList()
            prev_ch = channels[0]
            for ch in channels:
                self.down_blocks.append(SparseDownBlock(prev_ch, ch))
                prev_ch = ch
        mid = channels[-1]
        to_latent_in = mid + (in_channels if feat_skip else 0)
        self.to_latent = nn.Sequential(
            nn.Conv3d(to_latent_in, mid, 1, bias=False),
            nn.GroupNorm(min(8, mid), mid),
            nn.SiLU(inplace=True),
            nn.Conv3d(mid, mid, 1, bias=False),
            nn.GroupNorm(min(8, mid), mid),
            nn.SiLU(inplace=True),
            nn.Conv3d(mid, self.geo_ch * 2, 1),
        )
        if self.latent_split > 0:
            cch = tuple(color_channels) if color_channels else tuple(channels)
            assert len(cch) == n_down, "color branch must downsample as often as the main one"
            col_in = 3 if self.color_rgb_only else in_channels
            self.col_stem = spconv.SparseSequential(spconv.SubMConv3d(col_in, cch[0], 1, bias=False))
            self.col_blocks = nn.ModuleList()
            prev = cch[0]
            for i, ch in enumerate(cch):
                if residual:
                    self.col_blocks.append(SparseResStage(prev, ch, key=f"cres{i}"))
                else:
                    self.col_blocks.append(SparseDownBlock(prev, ch))
                prev = ch
            cmid = cch[-1]
            col_ch = self.latent_split
            self.to_latent_col = nn.Sequential(
                nn.Conv3d(cmid + (3 if self.color_rgb_only else in_channels), cmid, 1, bias=False),
                nn.GroupNorm(min(8, cmid), cmid),
                nn.SiLU(inplace=True),
                nn.Conv3d(cmid, col_ch * 2, 1),
            )
            nn.init.zeros_(self.to_latent_col[-1].weight)
            nn.init.zeros_(self.to_latent_col[-1].bias)
            self.to_latent_col_direct = nn.Conv3d(3, col_ch * 2, 1)
            with torch.no_grad():
                self.to_latent_col_direct.weight.zero_()
                self.to_latent_col_direct.bias.zero_()
                for j in range(min(3, col_ch)):
                    self.to_latent_col_direct.weight[j, j, 0, 0, 0] = 1.0
                self.to_latent_col_direct.bias[col_ch:] = -2.0
        print(
            f"SpconvEncoder: input_shape(zyx)={self.spatial_shape}, bottleneck(xyz)={self.bottleneck_shape_xyz}, voxel_sizes(xyz)={[f'{v:.4f}' for v in voxel_sizes]}m, n_down={n_down}"
        )

    def voxelize(self, xyz_world, features, batch_size):
        device = xyz_world.device
        B = batch_size
        N_per_batch = xyz_world.shape[0] // B
        D, H, W = self.spatial_shape
        C = features.shape[1]
        normed = (xyz_world - self.bounds_min) / self.voxel_sizes
        ix = normed[:, 0].long().clamp(0, W - 1)
        iy = normed[:, 1].long().clamp(0, H - 1)
        iz = normed[:, 2].long().clamp(0, D - 1)
        batch_idx = torch.arange(B, device=device).repeat_interleave(N_per_batch)
        voxel_id = batch_idx * (D * H * W) + iz * (H * W) + iy * W + ix
        unique_ids, inverse = voxel_id.unique(return_inverse=True)
        feat_sum = torch.zeros(len(unique_ids), C, device=device, dtype=features.dtype)
        feat_sum.scatter_add_(0, inverse.unsqueeze(1).expand(-1, C), features)
        counts = torch.zeros(len(unique_ids), 1, device=device, dtype=features.dtype)
        counts.scatter_add_(0, inverse.unsqueeze(1), torch.ones_like(features[:, :1]))
        feat_mean = feat_sum / counts.clamp(min=1)
        b_idx = unique_ids // (D * H * W)
        rem = unique_ids % (D * H * W)
        z_idx = rem // (H * W)
        rem2 = rem % (H * W)
        y_idx = rem2 // W
        x_idx = rem2 % W
        unique_indices = torch.stack([b_idx, z_idx, y_idx, x_idx], dim=1).int()
        return spconv.SparseConvTensor(feat_mean, unique_indices, self.spatial_shape, B)

    def _compute_bottleneck_occ_mask(self, xyz_world, batch_size):
        device = xyz_world.device
        B = batch_size
        N_per_batch = xyz_world.shape[0] // B
        bx, by, bz = self.bottleneck_shape_xyz
        extent = self.bounds_max - self.bounds_min
        bn_voxel_sizes = extent / torch.tensor([bx, by, bz], device=device, dtype=extent.dtype)
        normed = (xyz_world - self.bounds_min) / bn_voxel_sizes
        ix = normed[:, 0].long().clamp(0, bx - 1)
        iy = normed[:, 1].long().clamp(0, by - 1)
        iz = normed[:, 2].long().clamp(0, bz - 1)
        batch_idx = torch.arange(B, device=device).repeat_interleave(N_per_batch)
        mask = torch.zeros(B, 1, bz, by, bx, dtype=torch.bool, device=device)
        for b in range(B):
            sel = batch_idx == b
            mask[b, 0, iz[sel], iy[sel], ix[sel]] = True
        return mask

    def forward(self, xyz_world, features, batch_size):
        sp = self.voxelize(xyz_world, features, batch_size)
        sp_in = sp
        sp = self.stem(sp)
        for block in self.down_blocks:
            sp = block(sp)
        dense = sp.dense()
        skip = None
        if getattr(self, "feat_skip", False) or getattr(self, "latent_split", 0) > 0:
            skip = self._bottleneck_feat_grid(xyz_world, features, batch_size)
        if getattr(self, "feat_skip", False):
            dense = torch.cat([dense, skip], dim=1)
        out = self.to_latent(dense)
        mean, logvar = out.chunk(2, dim=1)
        if getattr(self, "latent_split", 0) > 0:
            rgb_only = getattr(self, "color_rgb_only", False)
            spc_in = sp_in.replace_feature(sp_in.features[:, 3:6]) if rgb_only else sp_in
            spc = self.col_stem(spc_in)
            for block in self.col_blocks:
                spc = block(spc)
            cskip = skip[:, 3:6] if rgb_only else skip
            cdense = torch.cat([spc.dense(), cskip], dim=1)
            cout = self.to_latent_col(cdense) + self.to_latent_col_direct(skip[:, 3:6])
            cmean, clogvar = cout.chunk(2, dim=1)
            mean = torch.cat([mean, cmean], dim=1)
            logvar = torch.cat([logvar, clogvar], dim=1)
        return (mean, logvar)

    def _bottleneck_feat_grid(self, xyz_world, features, batch_size):
        device = xyz_world.device
        B = batch_size
        bx, by, bz = self.bottleneck_shape_xyz
        C = features.shape[1]
        extent = self.bounds_max - self.bounds_min
        vs = extent / torch.tensor([bx, by, bz], device=device, dtype=extent.dtype)
        normed = (xyz_world - self.bounds_min) / vs
        ix = normed[:, 0].long().clamp(0, bx - 1)
        iy = normed[:, 1].long().clamp(0, by - 1)
        iz = normed[:, 2].long().clamp(0, bz - 1)
        bidx = torch.arange(B, device=device).repeat_interleave(xyz_world.shape[0] // B)
        flat = ((bidx * bz + iz) * by + iy) * bx + ix
        gsum = torch.zeros(B * bz * by * bx, C, device=device, dtype=features.dtype)
        gsum.scatter_add_(0, flat.unsqueeze(1).expand(-1, C), features)
        cnt = torch.zeros(B * bz * by * bx, 1, device=device, dtype=features.dtype)
        cnt.scatter_add_(0, flat.unsqueeze(1), torch.ones_like(features[:, :1]))
        grid = (gsum / cnt.clamp(min=1)).view(B, bz, by, bx, C).permute(0, 4, 1, 2, 3)
        return grid.contiguous()


class DenseResBlock3d(nn.Module):

    def __init__(self, ch, kernel=(3, 3, 3)):
        super().__init__()
        pad = tuple(((k - 1) // 2 for k in kernel))
        self.n1 = nn.GroupNorm(min(8, ch), ch)
        self.c1 = nn.Conv3d(ch, ch, kernel, padding=pad, bias=False)
        self.n2 = nn.GroupNorm(min(8, ch), ch)
        self.c2 = nn.Conv3d(ch, ch, kernel, padding=pad, bias=False)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class LocalUpsampleBlock(nn.Module):

    def __init__(self, in_ch, out_ch, refine_kernel=(3, 3, 3), n_refine=2, up_scale=2, residual=False):
        super().__init__()
        pad = tuple(((k - 1) // 2 for k in refine_kernel))
        if residual:
            if up_scale >= 2 and in_ch > out_ch:
                self.upsample = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, refine_kernel, padding=pad), nn.Upsample(scale_factor=up_scale, mode="nearest")
                )
            elif up_scale >= 2:
                self.upsample = nn.Sequential(
                    nn.Upsample(scale_factor=up_scale, mode="nearest"), nn.Conv3d(in_ch, out_ch, refine_kernel, padding=pad)
                )
            else:
                self.upsample = nn.Conv3d(in_ch, out_ch, refine_kernel, padding=pad)
            self.refine = nn.Sequential(*[DenseResBlock3d(out_ch, refine_kernel) for _ in range(n_refine)])
        else:
            self.upsample = (
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=up_scale, stride=up_scale)
                if up_scale >= 2
                else nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
            )
            layers = []
            for _ in range(n_refine):
                layers += [
                    nn.Conv3d(out_ch, out_ch, kernel_size=refine_kernel, padding=pad),
                    nn.GroupNorm(min(8, out_ch), out_ch),
                    nn.SiLU(inplace=True),
                ]
            self.refine = nn.Sequential(*layers)

    def forward(self, x):
        return self.refine(self.upsample(x))


class LocalHierarchicalOccupancyDecoder(nn.Module):

    def __init__(
        self,
        grid_feat_dim=128,
        channels=(128, 64, 32),
        final_z_kernel=3,
        level0_blocks=2,
        n_refine=2,
        iso_levels=(),
        up1_scale=2,
        residual=False,
        grad_ckpt=False,
    ):
        super().__init__()
        self.grad_ckpt = bool(grad_ckpt)
        channels = tuple((int(c) for c in channels))
        assert len(channels) >= 2, f"decoder needs >= 2 levels, got channels={channels}"
        self.channels = channels
        self.n_levels = len(channels)
        self.finest = self.n_levels - 1
        self.up_scales = [1] + [int(up1_scale)] + [2] * (self.n_levels - 2)
        ch0 = channels[0]
        assert final_z_kernel >= 1 and final_z_kernel % 2 == 1, f"final_z_kernel must be odd >= 1, got {final_z_kernel}"
        iso_levels = tuple(iso_levels or ())

        def _k(level):
            return (3, 3, 3) if level in iso_levels else (1, 3, 3)

        k0 = _k(0)
        p0 = tuple(((k - 1) // 2 for k in k0))
        if residual:
            layers = [nn.Conv3d(grid_feat_dim, ch0, kernel_size=k0, padding=p0)]
            layers += [DenseResBlock3d(ch0, k0) for _ in range(max(1, level0_blocks - 1))]
        else:
            layers = []
            prev = grid_feat_dim
            for _ in range(level0_blocks):
                layers += [
                    nn.Conv3d(prev, ch0, kernel_size=k0, padding=p0),
                    nn.GroupNorm(min(8, ch0), ch0),
                    nn.SiLU(inplace=True),
                ]
                prev = ch0
        self.level0_conv = nn.Sequential(*layers)
        self.occ_head_0 = nn.Conv3d(ch0, 1, 1)
        for i in range(1, self.n_levels):
            setattr(
                self,
                f"up_{i}",
                LocalUpsampleBlock(
                    channels[i - 1],
                    channels[i],
                    refine_kernel=_k(i),
                    n_refine=n_refine,
                    up_scale=self.up_scales[i],
                    residual=residual,
                ),
            )
            setattr(self, f"occ_head_{i}", nn.Conv3d(channels[i], 1, 1))
        chF = channels[-1]
        if final_z_kernel > 1:
            kz_pad = final_z_kernel // 2
            self.final_z_mix = nn.Sequential(
                nn.Conv3d(chF, chF, kernel_size=(final_z_kernel, 1, 1), padding=(kz_pad, 0, 0)),
                nn.GroupNorm(min(8, chF), chF),
                nn.SiLU(inplace=True),
            )
        else:
            self.final_z_mix = nn.Identity()
        setattr(
            self,
            f"rgb_head_{self.finest}",
            nn.Sequential(
                nn.Conv3d(chF, chF, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.SiLU(inplace=True),
                nn.Conv3d(chF, 3, 1),
                nn.Tanh(),
            ),
        )

    def up(self, i):
        return getattr(self, f"up_{i}")

    def occ_head(self, i):
        return getattr(self, f"occ_head_{i}")

    @property
    def up_finest(self):
        return getattr(self, f"up_{self.finest}")

    @property
    def rgb_head(self):
        return getattr(self, f"rgb_head_{self.finest}")

    def _run_levels(self, grid_features, head_fn):
        feat = self.level0_conv(grid_features)
        outs = [head_fn(0, feat)]
        _use_ckpt = self.grad_ckpt and self.training and torch.is_grad_enabled()
        if self.grad_ckpt and (not getattr(self, "_ckpt_logged", False)):
            print(
                f"[decoder] grad_ckpt={self.grad_ckpt} training={self.training} grad_enabled={torch.is_grad_enabled()} -> ckpt branch: {_use_ckpt}",
                flush=True,
            )
            self._ckpt_logged = _use_ckpt
        if _use_ckpt:
            from torch.utils.checkpoint import checkpoint as _ckpt
        for i in range(1, self.n_levels):
            up = self.up(i)
            if i == self.finest:
                _fn = lambda f, _up=up: self.final_z_mix(_up(f))
            else:
                _fn = up
            feat = _ckpt(_fn, feat, use_reentrant=False) if _use_ckpt else _fn(feat)
            outs.append(_ckpt(self.occ_head(i), feat, use_reentrant=False) if _use_ckpt else head_fn(i, feat))
        rgb = _ckpt(self.rgb_head, feat, use_reentrant=False) if _use_ckpt else self.rgb_head(feat)
        return (outs, rgb, feat)

    def forward(self, grid_features, occ_mask_0=None, use_gt_mask=True, return_features=False):
        outs, rgb, feat = self._run_levels(grid_features, lambda i, f: self.occ_head(i)(f))
        if return_features:
            return (*outs, rgb, feat)
        return (*outs, rgb)

    def extract_point_cloud(self, occ_logits_2, rgb_2, bounds_min, spacing_fine, threshold=0.0):
        B = occ_logits_2.shape[0]
        device = occ_logits_2.device
        results = []
        for b in range(B):
            occ = occ_logits_2[b, 0] > threshold
            iz, iy, ix = torch.where(occ)
            if len(ix) == 0:
                results.append((torch.zeros(0, 3, device=device), torch.zeros(0, 3, device=device)))
                continue
            xyz = torch.stack(
                [
                    bounds_min[0] + (ix.float() + 0.5) * spacing_fine[0],
                    bounds_min[1] + (iy.float() + 0.5) * spacing_fine[1],
                    bounds_min[2] + (iz.float() + 0.5) * spacing_fine[2],
                ],
                dim=-1,
            )
            rgb = rgb_2[b, :, iz, iy, ix].T
            results.append((xyz, rgb))
        return results


class OccupancyVAELocal(nn.Module):

    def __init__(
        self,
        in_channels=6,
        encoder_channels=(4, 4),
        bottleneck_shape=(32, 32, 3),
        x_range=(-0.28, 0.28),
        y_range=(-0.3, 0.26),
        z_range=(0.004, 0.06),
        latent_channels=4,
        grid_feat_dim=128,
        decoder_channels=(128, 64, 32),
        final_z_kernel=3,
        conv_decoder_mid=64,
        decoder_level0_blocks=2,
        decoder_n_refine=2,
        decoder_iso_levels=(),
        decoder_up1_scale=2,
        decoder_grad_ckpt=False,
        encoder_feat_skip=False,
        encoder_residual=False,
        latent_split=0,
        color_branch_channels=None,
        color_branch_rgb_only=False,
        decoder_residual=False,
    ):
        super().__init__()
        self.grid_feat_dim = grid_feat_dim
        self.latent_channels = latent_channels
        self.latent_split = int(latent_split)
        self._decoder_kwargs = dict(
            level0_blocks=decoder_level0_blocks,
            n_refine=decoder_n_refine,
            iso_levels=tuple(decoder_iso_levels or ()),
            up1_scale=decoder_up1_scale,
            residual=decoder_residual,
            grad_ckpt=decoder_grad_ckpt,
        )
        spatial_bounds = (x_range, y_range, z_range)
        self.encoder = SpconvEncoder(
            in_channels=in_channels,
            channels=encoder_channels,
            latent_channels=latent_channels,
            bottleneck_shape=bottleneck_shape,
            spatial_bounds=spatial_bounds,
            feat_skip=encoder_feat_skip,
            residual=encoder_residual,
            latent_split=latent_split,
            color_channels=color_branch_channels,
            color_rgb_only=color_branch_rgb_only,
        )
        bx, by, bz = bottleneck_shape
        self.voxel_grid = VoxelGrid(bx, by, bz, x_range=x_range, y_range=y_range, z_range=z_range)
        self.conv_decoder = VoxelConvDecoder(latent_channels, conv_decoder_mid, grid_feat_dim)
        self.decoder = LocalHierarchicalOccupancyDecoder(
            grid_feat_dim=grid_feat_dim, channels=decoder_channels, final_z_kernel=final_z_kernel, **self._decoder_kwargs
        )
        self.grid_shapes = [(bx, by, bz)]
        for sc in self.decoder.up_scales[1:]:
            px, py, pz = self.grid_shapes[-1]
            self.grid_shapes.append((px * sc, py * sc, pz * sc))
        self.register_buffer("bounds_min_world", torch.tensor([x_range[0], y_range[0], z_range[0]], dtype=torch.float32))
        self.register_buffer("bounds_max_world", torch.tensor([x_range[1], y_range[1], z_range[1]], dtype=torch.float32))
        finest = self.grid_shapes[-1]
        extent = self.bounds_max_world - self.bounds_min_world
        self.register_buffer("spacing_fine", extent / torch.tensor(finest, dtype=torch.float32))
        print(f"Bottleneck VoxelGrid: {bx}×{by}×{bz} = {bx * by * bz} voxels")
        print(f"Decoder levels: {' → '.join((f'{s[0]}×{s[1]}×{s[2]}' for s in self.grid_shapes))}")
        print(
            f"Finest voxel size: {self.spacing_fine[0] * 1000:.1f}×{self.spacing_fine[1] * 1000:.1f}×{self.spacing_fine[2] * 1000:.1f}mm"
        )

    def _prepare_features(self, xyz, rgb, pos_mean=None, pos_std=None):
        B, N, _ = xyz.shape
        device = xyz.device
        if pos_mean is not None:
            pos_mean_t = torch.as_tensor(pos_mean, device=device, dtype=xyz.dtype)
            pos_std_t = torch.as_tensor(pos_std, device=device, dtype=xyz.dtype)
            xyz_world = xyz * pos_std_t + pos_mean_t
        else:
            xyz_world = xyz
        features = torch.cat([xyz, rgb], dim=-1)
        return (xyz_world.reshape(B * N, 3), features.reshape(B * N, 6))

    def encode(self, xyz, rgb, pos_mean=None, pos_std=None):
        B = xyz.shape[0]
        xyz_world, features = self._prepare_features(xyz, rgb, pos_mean, pos_std)
        return self.encoder(xyz_world, features, B)

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mean + torch.randn_like(std) * std
        return mean

    def decode(self, z, occ_mask_0=None, use_gt_mask=True):
        decoded = self.conv_decoder(z)
        return self.decoder(decoded, occ_mask_0=occ_mask_0, use_gt_mask=use_gt_mask)

    def extract_point_cloud(self, occ_logits_2, rgb_2, pos_mean=None, pos_std=None, threshold=0.0):
        results = self.decoder.extract_point_cloud(occ_logits_2, rgb_2, self.bounds_min_world, self.spacing_fine, threshold)
        if pos_mean is not None:
            pos_mean_t = torch.as_tensor(pos_mean, device=occ_logits_2.device, dtype=occ_logits_2.dtype)
            pos_std_t = torch.as_tensor(pos_std, device=occ_logits_2.device, dtype=occ_logits_2.dtype)
            results = [((xyz - pos_mean_t) / pos_std_t, rgb) for (xyz, rgb) in results]
        return results
