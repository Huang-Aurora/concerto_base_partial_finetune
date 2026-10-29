"""Dataset: one whole rock scene per item, official Sonata default preprocessing.

Pipeline (identical to sonata.transform.default(), plus `segment` carried for
supervision):
  CenterShift(apply_z=True)
  -> GridSample(0.02, fnv, mode=train, return_grid_coord+inverse)
  -> NormalizeColor
  -> ToTensor
  -> Collect(keys=(coord,grid_coord,color,inverse,segment), feat=(coord,color,normal))

Labels are ONLY used as supervision/metrics; the model input `feat` is built
from coord+color+normal. Val scenes use a fixed GridSample seed so the
downsampled subset is identical across epochs.
"""
import numpy as np
import torch
import concerto.transform  # registers all transforms

import config as C
from io_utils import load_scene_raw
from normals import ensure_normals, normals_path


def _t(x):
    if torch.is_tensor(x):
        return x
    return torch.from_numpy(np.ascontiguousarray(x))


class RockSceneDataset(torch.utils.data.Dataset):
    def __init__(self, scene_ids, split):
        assert split in ("train", "val")
        self.scene_ids = list(scene_ids)
        self.split = split
        ensure_normals(self.scene_ids)
        cfg = [
            dict(type="CenterShift", apply_z=True),
            dict(type="GridSample", grid_size=C.GRID_SIZE, hash_type="fnv",
                 mode="train", return_grid_coord=True, return_inverse=True),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "grid_coord", "color", "inverse", "segment"),
                 feat_keys=("coord", "color", "normal")),
        ]
        self.transform = concerto.transform.Compose(cfg)

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, idx):
        sid = self.scene_ids[idx]
        coord, color, label = load_scene_raw(sid)
        normal = np.load(normals_path(sid))
        assert normal.shape == coord.shape
        if self.split == "val":
            np.random.seed(C.VAL_SEED_BASE + sid)  # deterministic val sampling
        data = dict(coord=coord.copy(), color=color.copy(),
                    normal=normal.copy(), segment=label.copy())
        data = self.transform(data)
        point = {k: _t(v) for k, v in data.items()}
        return dict(point=point, scene_id=sid, coord_orig=coord,
                    color_orig=color, label_orig=label)
