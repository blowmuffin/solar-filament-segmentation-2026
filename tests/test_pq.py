import ast
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import rect
from pycocotools import mask as mask_util

from filseg import data as D
from filseg import pq

A, B = rect(100, 110, 100, 110), rect(300, 310, 300, 310)


def test_one_tp_two_fp_one_fn():
    s = pq.score({"01-x": [A, B]}, {"x": [A, rect(300, 310, 305, 315), rect(900, 910, 900, 910)]})
    assert (s["tp"], s["fp"], s["fn"]) == (1, 2, 1) and abs(s["pq"] - 0.4) < 1e-9


def test_iou_half_is_not_a_match():
    s = pq.score({"01-x": [A]}, {"x": [rect(100, 105, 100, 110)]})
    assert (s["tp"], s["fp"], s["fn"]) == (0, 1, 1) and s["pq"] == 0.0


def test_pooled_over_records():
    s = pq.score({"01-x": [A], "02-x": [B]}, {"x": [A]})
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 1) and abs(s["pq"] - 0.5) < 1e-9


def test_no_predictions():
    s = pq.score({"01-x": [A, B]}, {})
    assert (s["tp"], s["fp"], s["fn"]) == (0, 0, 2) and s["pq"] == 0.0


def test_secondary_statistics():
    s = pq.score({"01-x": [A, B]}, {"x": [A, rect(300, 310, 305, 315), rect(900, 910, 900, 910)]})
    assert s["tp_ious"] == [1.0] and s["sq"] == 1.0 and abs(s["rq"] - 0.4) < 1e-9
    assert np.allclose(sorted(s["ious"]), [1 / 3, 1.0]) and np.allclose(sorted(s["dices"]), [0.5, 1.0])
    assert list(s["gt_degree"]) == [1, 1] and list(s["pred_degree"]) == [1, 1, 0]


def test_stem_without_gt_is_ignored():      # the host only scores the images that have ground truth
    s = pq.score({"01-x": [A]}, {"x": [A], "y": [B]})
    assert (s["tp"], s["fp"], s["fn"]) == (1, 0, 0) and s["pq"] == 1.0


def test_iou_matrix_shape_and_empty():
    assert pq.iou_matrix([A, B], [A, A, B]).shape == (2, 3)
    assert pq.iou_matrix([A], []).shape == (1, 0) and pq.iou_matrix([], [A]).shape == (0, 1)


@pytest.mark.slow
def test_annotators_agree_at_about_034():       # forum measurement: 0.340 to 0.344 on 598 record pairs
    gt, pred = {}, {}
    for stem, recs in D.load_records().items():
        masks = list(recs.values())
        for i, j in itertools.combinations(range(len(masks)), 2):      # each pair gets its own pseudo-stem
            gt[f"pair-{stem}#{i}{j}"] = masks[i]; pred[f"{stem}#{i}{j}"] = masks[j]
    assert len(gt) == 598 and 0.32 <= pq.score(gt, pred)["pq"] <= 0.36


def _host_namespace():
    """Execute code cells 3 and 4 (pure helper functions) of the host notebook, in place."""
    path = os.environ.get("FILSEG_HOST_NB")
    if not path:
        pytest.skip("FILSEG_HOST_NB is not set")
    import torch
    nb = json.loads((D.REPO / path).read_text(encoding="utf-8"))
    ns = {"pd": pd, "np": np, "torch": torch, "mask_util": mask_util}
    for i in (3, 4):
        src = "".join(nb["cells"][i]["source"])
        assert all(isinstance(n, ast.FunctionDef) for n in ast.parse(src).body), f"cell {i} is not only defs"
        exec(compile(src, f"<host cell {i}>", "exec"), ns)
    return ns


@pytest.mark.slow
def test_matches_host_notebook():
    host = _host_namespace()
    recs = D.load_records()
    stems = sorted(recs)
    by_n = {n: [s for s in stems if len(recs[s]) == n] for n in (1, 2, 3)}
    # 5 real records: two annotators of one stem share a prediction set, three more stems use a neighbour
    gt, pred = {}, {}
    rids = list(recs[by_n[3][0]])
    gt[rids[0]], gt[rids[1]], pred[by_n[3][0]] = recs[by_n[3][0]][rids[0]], recs[by_n[3][0]][rids[1]], recs[by_n[3][0]][rids[2]]
    for s in by_n[2][:2] + by_n[3][1:2]:
        a, b = list(recs[s])[:2]
        gt[a], pred[s] = recs[s][a], recs[s][b]
    assert len(gt) == 5

    def frame(rows):
        return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    to_str = lambda r: r["counts"].decode("ascii")
    gt_df = frame([(f"{rid}_{k}", to_str(r)) for rid, rs in gt.items() for k, r in enumerate(rs, 1)])
    pred_df = frame([(f"{s}_{k}", to_str(r)) for s, rs in pred.items() for k, r in enumerate(rs, 1)])
    overlap = host["get_overlap_df"](gt_df, pred_df)
    host_pq = host["get_pq_score"](overlap)
    hit = [(row.iou_matrix > 0.5) for row in overlap.itertuples(index=False)]
    host_counts = (sum(int(h.sum()) for h in hit), sum(host["fp_count_hit"](h) for h in hit),
                   sum(host["fn_count_hit"](h) for h in hit))

    ours = pq.score(gt, pred)
    assert ours["tp"] > 0 and host_counts == (ours["tp"], ours["fp"], ours["fn"])
    assert abs(ours["pq"] - host_pq) < 1e-6, (ours["pq"], host_pq)
