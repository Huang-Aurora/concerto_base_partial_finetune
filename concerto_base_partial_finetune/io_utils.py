"""Raw scene loading with a local npy cache (original txt/npy are never touched)."""
import os

import numpy as np

import config as C


def load_scene_raw(sid):
    """Return coord (N,3) float32, color (N,3) float32 in [0,255], label (N,) int64.

    coord/color are cached as npy under the probe cache dir after the first
    read of data/<sid>.txt; labels are read from label/<sid>.npy every call
    (cheap) so that any label update is picked up.
    """
    coord_f = os.path.join(C.RAW_CACHE_DIR, f"{sid}_coord.npy")
    color_f = os.path.join(C.RAW_CACHE_DIR, f"{sid}_color.npy")
    if os.path.exists(coord_f) and os.path.exists(color_f):
        coord = np.load(coord_f)
        color = np.load(color_f)
    else:
        os.makedirs(C.RAW_CACHE_DIR, exist_ok=True)
        arr = np.loadtxt(os.path.join(C.DATA_DIR, f"{sid}.txt"),
                         skiprows=1, dtype=np.float32)
        coord, color = arr[:, :3], arr[:, 3:6]
        np.save(coord_f, coord)
        np.save(color_f, color)
    label_f = os.path.join(C.LABEL_DIR, f"{sid}.npy")
    label = np.load(label_f).astype(np.int64) if os.path.exists(label_f) else None
    if label is not None:
        assert label.shape[0] == coord.shape[0], (
            f"scene {sid}: label {label.shape} vs coord {coord.shape}")
    return coord, color, label
