from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def n_frames(f):
    return int(f["z_bg"].shape[0])


def read_dense_z(f, frames=None, dtype=np.float32):
    fr = np.arange(n_frames(f)) if frames is None else np.atleast_1d(np.asarray(frames, dtype=np.int64))
    C, nz, ny, nx = (int(v) for v in f.attrs["z_shape"])
    nvox = nz * ny * nx
    bg, off = (f["z_bg"], f["z_off"])
    out = np.empty((len(fr), C, nvox), np.float16)
    for i, t in enumerate(fr):
        t = int(t)
        keep = np.unpackbits(f["z_mask"][t]).view(bool)[:nvox]
        out[i] = bg[t][:, None]
        a, b = (int(off[t]), int(off[t + 1]))
        if b > a:
            out[i][:, keep] = f["z_vals"][a:b].T
    return out.reshape(len(fr), C, nz, ny, nx).astype(dtype, copy=False)


class LatentSeqDataset(Dataset):

    def __init__(self, latent_dir, seq_len, *, action_mean=None, action_std=None):
        self.files = sorted(
            (str(p) for p in Path(latent_dir).glob("episode_*_latents.h5")), key=lambda p: int(Path(p).stem.split("_")[1])
        )
        self.seq_len = seq_len
        self.handles = {}
        self.index = []
        for ei, fp in enumerate(self.files):
            with h5py.File(fp, "r") as f:
                self.action_dim = int(f["action"].shape[1])
                self.index += [(ei, s) for s in range(max(0, n_frames(f) - seq_len + 1))]
        self.action_mean = None if action_mean is None else np.asarray(action_mean, np.float32)
        self.action_std = None if action_std is None else np.asarray(action_std, np.float32)

    def __len__(self):
        return len(self.index)

    def compute_action_norm_params(self):
        acts = []
        for fp in self.files:
            with h5py.File(fp, "r") as f:
                acts.append(f["action"][:].astype(np.float32))
        a = np.concatenate(acts, 0)
        std = a.std(0)
        std[std < 0.02] = 1.0
        return (a.mean(0).astype(np.float32), std.astype(np.float32))

    def __getitem__(self, i):
        ei, s = self.index[i]
        if ei not in self.handles:
            self.handles[ei] = h5py.File(self.files[ei], "r")
        f = self.handles[ei]
        act = f["action"][s : s + self.seq_len].astype(np.float32)
        if self.action_mean is not None:
            act = (act - self.action_mean) / self.action_std
        out = dict(z_seq=torch.from_numpy(read_dense_z(f, range(s, s + self.seq_len))), actions=torch.from_numpy(act))
        for k in ("obj_occ", "tac_occ"):
            if k in f:
                out[k] = torch.from_numpy(f[k][s : s + self.seq_len].astype(np.float32))
        return out
