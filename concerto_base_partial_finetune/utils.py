"""Shared training helpers (extracted verbatim from the verified probe
train.py so both the frozen and fine-tune folders use one implementation).

Evaluation protocol: validation scenes are grid-sampled with a FIXED seed
per scene (deterministic across epochs); metrics are computed on grid points
(network supervision resolution) AND on the original full cloud via the
GridSample inverse mapping. The FULL-SIZE rock IoU is the checkpoint
selection metric, exactly as in the frozen probe.
"""
import hashlib
import os
import random
import sys

import numpy as np
import torch

import config as C  # noqa: E402

from helper_ply import write_ply  # noqa: E402

# PLY colors following the project convention (0 gray / 1 red / 2 blue)
GRAY = (192, 192, 192)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


def log(msg, log_f):
    print(msg, flush=True)
    log_f.write(msg + "\n")
    log_f.flush()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encoder_fingerprint(model):
    fp = {}
    for name, p in model.encoder.named_parameters():
        h = hashlib.sha1()
        h.update(p.detach().cpu().numpy().tobytes())
        fp[name] = h.hexdigest()
    return fp


def move_batch(sample, device):
    return {k: v.to(device, non_blocking=True)
            for k, v in sample["point"].items() if torch.is_tensor(v)}


@torch.no_grad()
def evaluate(model, val_ids, device):
    """Return (per-scene records, meter_grid, meter_full).

    meter_grid: metrics on grid-sampled points (network supervision res).
    meter_full: metrics on the ORIGINAL full cloud via inverse mapping.
    """
    model.eval()
    from dataset import RockSceneDataset
    from metrics import SegMeter
    ds = RockSceneDataset(val_ids, "val")
    meter_grid, meter_full = SegMeter(), SegMeter()
    records = []
    for i in range(len(ds)):
        sample = ds[i]
        batch = move_batch(sample, device)
        logits, point = model(batch)
        pred_grid = logits.argmax(1).cpu().numpy()
        seg_grid = point.segment.cpu().numpy()
        inv = point.inverse.cpu().numpy()
        pred_orig = pred_grid[inv]
        lab_orig = sample["label_orig"]
        assert len(pred_orig) == len(lab_orig) == len(sample["coord_orig"])

        m_g = (seg_grid < C.IGNORE_LABEL)
        meter_grid.update(seg_grid[m_g], pred_grid[m_g])
        m_f = lab_orig < C.IGNORE_LABEL
        meter_full.update(lab_orig[m_f], pred_orig[m_f])
        records.append(dict(scene=sample["scene_id"], n_orig=len(lab_orig),
                            n_grid=int(m_g.sum()), n_valid=int(m_f.sum())))
    return records, meter_grid, meter_full


def export_ply(model, scene_ids, out_dir, device, log_f):
    os.makedirs(out_dir, exist_ok=True)
    from dataset import RockSceneDataset
    for sid in scene_ids:
        sample = RockSceneDataset([sid], "val")[0]
        batch = move_batch(sample, device)
        with torch.no_grad():
            logits, point = model(batch)
        pred_orig = logits.argmax(1).cpu().numpy()[point.inverse.cpu().numpy()]
        coord, color, lab = (sample["coord_orig"], sample["color_orig"],
                             sample["label_orig"])
        assert len(pred_orig) == len(lab) == len(coord)
        n = len(lab)
        rgb = color.astype(np.uint8)

        gt_col = np.full((n, 3), GRAY, np.uint8)
        gt_col[lab == 1] = RED
        gt_col[lab == 2] = BLUE

        pr_col = np.full((n, 3), GRAY, np.uint8)
        pr_col[pred_orig == 1] = RED

        err_col = np.full((n, 3), GRAY, np.uint8)
        err_col[(pred_orig == 1) & (lab == 0)] = RED   # false positive
        err_col[(pred_orig == 0) & (lab == 1)] = BLUE  # false negative
        alpha = np.where(lab == 2, 0, 255).astype(np.uint8)

        write_ply(f"{out_dir}/{sid}_orig_rgb.ply", [coord, rgb],
                  ["x", "y", "z", "red", "green", "blue"])
        write_ply(f"{out_dir}/{sid}_gt.ply", [coord, gt_col],
                  ["x", "y", "z", "red", "green", "blue"])
        write_ply(f"{out_dir}/{sid}_pred.ply", [coord, pr_col],
                  ["x", "y", "z", "red", "green", "blue"])
        write_ply(f"{out_dir}/{sid}_error.ply", [coord, err_col, alpha],
                  ["x", "y", "z", "red", "green", "blue", "alpha"])
        np.save(f"{out_dir}/{sid}_pred.npy", pred_orig)
        log(f"[export] scene {sid}: {n} pts -> {out_dir}/{{orig_rgb,gt,pred,error}}.ply",
            log_f)
