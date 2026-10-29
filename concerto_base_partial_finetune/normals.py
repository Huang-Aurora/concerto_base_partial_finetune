"""Geometric normal estimation (the rock data has no normals; Sonata needs them).

Per scene: kNN with scipy cKDTree (k=16), local PCA on the neighborhoods, the
eigenvector of the smallest eigenvalue is the surface normal. Orientation:
flip so that z >= 0 (standard terrain convention; residual sign ambiguity on
near-vertical surfaces is a documented limitation of this probe).
Normals are unit length, float32 (N,3), cached under cache/normals/<id>.npy.
"""
import os

import numpy as np
from scipy.spatial import cKDTree

import config as C
from io_utils import load_scene_raw


def _estimate(xyz, k):
    xyz64 = np.asarray(xyz, dtype=np.float64)
    _, idx = cKDTree(xyz64).query(xyz64, k=k + 1, workers=-1)
    idx = np.asarray(idx)[:, 1:]  # first neighbor is the point itself
    nb = xyz64[idx]                       # (N,k,3)
    c = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / (k - 1)
    _, evecs = np.linalg.eigh(cov)        # ascending eigenvalues
    n = evecs[:, :, 0].copy()             # smallest-eigenvalue eigenvector
    n[n[:, 2] < 0] *= -1.0                # orient z >= 0
    d = np.linalg.norm(n, axis=1, keepdims=True)
    d[d == 0] = 1.0
    return (n / d).astype(np.float32)


def normals_path(sid):
    return os.path.join(C.NORMAL_CACHE_DIR, f"{sid}.npy")


def ensure_normals(scene_ids, k=C.NORMAL_K, verbose=True):
    os.makedirs(C.NORMAL_CACHE_DIR, exist_ok=True)
    for sid in scene_ids:
        path = normals_path(sid)
        if os.path.exists(path):
            continue
        coord, _, _ = load_scene_raw(sid)
        n = _estimate(coord, k)
        assert n.shape == coord.shape and np.allclose(np.linalg.norm(n, axis=1), 1, atol=1e-3)
        np.save(path, n)
        if verbose:
            print(f"[normals] scene {sid}: {n.shape[0]} pts -> {path}", flush=True)


if __name__ == "__main__":
    ids = C.TRAIN_SCENES + C.VAL_SCENES + C.TEST_SCENES
    print(f"estimating normals for {len(ids)} scenes ...", flush=True)
    ensure_normals(ids)
