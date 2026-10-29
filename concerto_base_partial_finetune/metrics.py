"""Metrics with the RandLA-Net project semantics (rows=truth, cols=pred).

IoU/mIoU use IoU_from_confusions VENDORED VERBATIM from
RandLA-Net-Pytorch-New-master/helper_tool.py (DataProcessing static method)
-- the original helper_tool module cannot be imported here because it pulls
in compiled C++ wrappers (cpp_wrappers / nearest_neighbors) that are not
available on Kaggle. The function below is a line-for-line copy; semantics
are unchanged.
P/R/F1/OA follow the same formulas as the RandLA visualize_preds script.
ignore label (2) must be masked out by the caller, exactly like the RandLA
test protocol (labels < 2).
"""
import numpy as np

import config as C


def iou_from_confusions(confusions):
    """Vendored verbatim from RandLA-Net helper_tool.DataProcessing.

    :param confusions: ([..., n_c, n_c] np.float). Can be any dimension, the
        confusion matrices should be described by the last axes.
    :return: ([..., n_c] np.float32) IoU score
    """
    # Compute TP, FP, FN. This assumes that the second to last axis counts
    # the truths (like the first axis of a confusion matrix), and that the
    # last axis counts the predictions (like the second axis of a confusion
    # matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)

    # Compute IoU
    IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

    # Compute mIoU with only the actual classes
    mask = TP_plus_FN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

    # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
    IoU += mask * mIoU
    return IoU


class SegMeter:
    """Accumulates masked (label, pred) pairs over scenes, then computes metrics."""

    def __init__(self):
        self.gts, self.preds = [], []

    def update(self, gt, pred):
        self.gts.append(np.asarray(gt).reshape(-1))
        self.preds.append(np.asarray(pred).reshape(-1))

    def compute(self):
        from sklearn.metrics import confusion_matrix
        if not self.gts:
            return None
        gt = np.concatenate(self.gts)
        pred = np.concatenate(self.preds)
        conf = confusion_matrix(gt, pred, labels=list(range(C.NUM_CLASSES))).astype(np.float64)
        iou = iou_from_confusions(conf[None])[0]  # same code path as RandLA
        tp = np.diag(conf)
        gt_c, pr_c = conf.sum(1), conf.sum(0)  # TP+FN, TP+FP
        P = np.divide(tp, pr_c, out=np.zeros_like(tp), where=pr_c > 0)
        R = np.divide(tp, gt_c, out=np.zeros_like(tp), where=gt_c > 0)
        F1 = np.divide(2 * tp, pr_c + gt_c, out=np.zeros_like(tp), where=(pr_c + gt_c) > 0)
        oa = tp.sum() / max(conf.sum(), 1.0)
        return dict(confusion=conf, oa=float(oa), miou=float(iou.mean()),
                    iou_bg=float(iou[0]), iou_rock=float(iou[1]),
                    precision_bg=float(P[0]), precision_rock=float(P[1]),
                    recall_bg=float(R[0]), recall_rock=float(R[1]),
                    f1_bg=float(F1[0]), f1_rock=float(F1[1]))
