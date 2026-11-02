import json, os
from pathlib import Path
import numpy as np

_CACHE = {}


def root_of(path):
    p = Path(path)
    if p.suffix == ".hdf5":
        p = Path(os.path.realpath(p)).parent.parent
    return p


def gains_for(path):
    r = str(root_of(path))
    if r not in _CACHE:
        f = Path(r) / "packing" / "color_gain_extra.json"
        _CACHE[r] = (
            {k: np.asarray(v, np.float32) for (k, v) in json.loads(f.read_text()).items() if k.startswith("camera_")}
            if f.exists()
            else {}
        )
    return _CACHE[r]


def apply(rgb_u8, path, cam):
    g = gains_for(path).get(f"camera_{int(cam)}")
    if g is None:
        return rgb_u8
    return np.clip(rgb_u8.astype(np.float32) * g, 0, 255).astype(np.uint8)
