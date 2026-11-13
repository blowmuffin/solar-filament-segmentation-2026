"""Host-exact pooled Panoptic Quality and the secondary statistics."""
import math
from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_util

from filseg.data import stem_of

IOU_HIT = 0.5


def iou_matrix(gt: list[dict], pred: list[dict]) -> np.ndarray:
    """(n_gt, n_pred) float64 IoU of RLE masks; pycocotools returns [] when either side is empty."""
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)))
    return np.asarray(mask_util.iou(gt, pred, [0] * len(pred)))     # rows follow the first argument


def score(gt: dict[str, list[dict]], pred: dict[str, list[dict]]) -> dict:
    """Pooled PQ as the host computes it. gt: record id -> RLEs; pred: stem -> RLEs, matched to every
    record of the stem through data.stem_of. Predictions of a stem without ground truth are ignored (the
    host only visits annotator-images). ious/dices cover all pairs with IoU > 0; gt_degree[i] and
    pred_degree[j] count the overlapping partners (IoU > 0) of each (record, filament) / (record, prediction)."""
    by_stem = defaultdict(list)
    for rid in gt:
        by_stem[stem_of(rid)].append(rid)
    tp_ious, nonzero, gt_deg, pred_deg = [], [], [], []
    fp = fn = 0
    for stem, rids in by_stem.items():
        preds = pred.get(stem, [])
        # the host divides float32 tensors; the cast reproduces its `> 0.5` test and its TP values
        iou = iou_matrix([m for rid in rids for m in gt[rid]], preds).astype(np.float32)
        row = 0
        for rid in rids:
            block = iou[row:row + len(gt[rid])]
            row += len(gt[rid])
            hit = block > IOU_HIT
            tp_ious += block[hit].tolist()
            fp += int((hit.sum(0) == 0).sum())      # predictions with no hit
            fn += int((hit.sum(1) == 0).sum())      # filaments with no hit (all of them when preds is empty)
            nonzero.append(block[block > 0])
            gt_deg += (block > 0).sum(1).tolist()
            pred_deg += (block > 0).sum(0).tolist()
    tp = len(tp_ious)
    den = tp + 0.5 * fp + 0.5 * fn
    total = math.fsum(tp_ious)
    ious = np.concatenate(nonzero).astype(np.float64) if nonzero else np.zeros(0)
    return dict(pq=total / den if den > 0 else 0.0, sq=total / tp if tp else 0.0, rq=tp / den if den > 0 else 0.0,
                tp=tp, fp=fp, fn=fn, tp_ious=tp_ious, ious=ious, dices=2 * ious / (1 + ious),
                gt_degree=np.array(gt_deg, int), pred_degree=np.array(pred_deg, int))
