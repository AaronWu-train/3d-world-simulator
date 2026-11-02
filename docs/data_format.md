# Data format

One root per dataset. Only the fields below are read.

```
<root>/
  train/episode_<E>.hdf5
  val/episode_<E>.hdf5
  fs_disp/disp_episode_<E>.hdf5
  sam3_masks/episode_<E>_camera_<C>.h5
  gripper_supervision/masks_halo6_surface/episode_<E>.h5
  robot_label_cache/episode_<E>.h5
  tactile_targets/episode_<E>.npz
  packing/color_gain_extra.json      optional
```

`T` is frames in the episode. Cameras are `C` in `cam_ids`. Positions are metres in the world frame. Depth is `H x W`, the render crop is `h x w`. Packed masks are `np.packbits` of the flattened boolean image, one row per frame.

## Episode: `train/episode_<E>.hdf5`

Attributes:

| Name | Value |
|---|---|
| `episode` | episode id |
| `num_frames` | `T` |
| `H`, `W` | depth image size |
| `cam_ids` | camera ids |
| `render_cam_ids` | optional, cameras for the render loss, default `cam_ids` |
| `bbox` | `[xmin, xmax, ymin, ymax, zmin, zmax]` |
| `has_obj_mask` | true if `img/camera_C/obj_mask` exists |
| `filter_kind` | present if `filter_mask` exists |

Datasets:

| Path | Shape | Type | Meaning |
|---|---|---|---|
| `action/cartesian_action` | `T, 14` | float | per arm: position 3, rotation 3, gripper 1 |
| `cam/camera_C/K_ir` | `T, 3, 3` | float32 | depth camera intrinsics |
| `cam/camera_C/E_cam2world` | `T, 4, 4` | float32 | depth camera to world |
| `img/camera_C/depth` | `T, H, W` | float16 | metres |
| `img/camera_C/valid_mask` | `T, HW/8` | uint8 | packed, pixels to lift to points |
| `img/camera_C/filter_mask` | `T, HW/8` | uint8 | packed, `valid_mask` after outlier removal; used in VAE training |
| `img/camera_C/rgb_in_ir` | `T` | bytes | JPEG, colour aligned to the depth image |
| `img/camera_C/obj_mask` | `T, H, W` | uint8 | 0 none, otherwise object id |
| `render/camera_C/jpg` | `T` | bytes | JPEG of the render crop |
| `render/camera_C/K` | `T, 3, 3` | float32 | intrinsics of the crop |
| `render/camera_C/viewmat` | `T, 4, 4` | float32 | world to camera |
| `render/camera_C/obj_mask` | `T, hw/8` | uint8 | packed, optional |
| `render/camera_C/valid_mask` | `T, hw/8` | uint8 | packed, optional |

`render/camera_C` carries attributes `H`, `W` and `crop = [x0, y0, x1, y1]`.

## Side files

**`fs_disp/disp_episode_<E>.hdf5`**: stereo disparity for the depth loss and free-space carving.

| Path | Shape | Meaning |
|---|---|---|
| `disp/camera_C` | `T, H, W` | left disparity, pixels |
| `disp_right/camera_C` | `T, H, W` | optional, right disparity for the left-right check |
| `meta/camera_C/baseline` | `T` | metres |

**`sam3_masks/episode_<E>_camera_<C>.h5`**: `frame_indices` (`N`) and `mask_union` (`N, full H, full W`). Object mask for the render loss, cropped with `crop`.

**`gripper_supervision/masks_halo6_surface/episode_<E>.h5`**: `camera_C/gripper_mask` (`T, hw/8`), packed, with the same `crop` as `render/camera_C`.

**`robot_label_cache/episode_<E>.h5`**: voxel labels at each decoder level. Attribute `bounds` is the 2 x 3 grid bounds and must match the task. `robot/<level>` and `gripper/<level>` hold packed `z, y, x` grids per frame, each with attribute `shape`.

**`tactile_targets/episode_<E>.npz`**: for tasks with tactile pads.

| Key | Shape | Meaning |
|---|---|---|
| `xyz` | `T, 4, 384, 3` | taxel positions, 4 pads of 12 x 32 taxels |
| `sig` | `T, 4, 384` | signal in 0 to 1 |
| `has_tac` | scalar | false if the episode has no tactile data |

## Latents: `episode_<E>_latents.h5`

Written by `scripts/export_latents.py`. Most latent cells hold one empty vector, so each frame stores that vector and the other cells.

| Path | Shape | Type | Meaning |
|---|---|---|---|
| `z_bg` | `T, C` | float16 | empty vector per frame |
| `z_mask` | `T, nvox/8` | uint8 | packed, stored cells |
| `z_off` | `T+1` | int64 | row offsets into `z_vals` |
| `z_vals` | `sum, C` | float16 | stored cell values |
| `obj_occ` | `T, nz, ny, nx` | uint8 | object cells |
| `tac_occ` | `T, nz, ny, nx` | uint8 | tactile cells, tactile tasks |
| `action` | `T, 14` | float32 | as above |
| `K`, `E` | `T, n, 3, 3` / `T, n, 4, 4` | float32 | camera parameters |

Attributes: `compact = 1`, `z_shape = (C, nz, ny, nx)`, `pos_mean`, `pos_std`, `bbox`.

`datasets.latent_seq.read_dense_z` restores dense latents.
