"""Gate G-F: the host's own scorer reads our submission file exactly as we mean it (no entry spent)."""
import dataclasses
import json

import pandas as pd
import pytest
from test_pq import _host_namespace

from filseg import data as D
from filseg import instances as inst
from filseg import pq
from filseg import preflight as pf

SUBS = (D.REPO / ".." / "subs").resolve()


@pytest.mark.slow
def test_host_scorer_reads_the_oof_file_as_we_mean_it():
    host = _host_namespace()
    if not (inst.RUNS / "classical" / "oof_fold0.npz").exists():
        pytest.skip("run `python -m filseg.baseline` first")
    stems = {s for s, k in json.loads(D.FOLDS_JSON.read_text(encoding="utf-8")).items() if k == 0}
    SUBS.mkdir(exist_ok=True)
    rule, pred_csv, gt_csv = SUBS / "rehearsal_rule.json", SUBS / "rehearsal_oof.csv", SUBS / "rehearsal_gt.csv"
    rule.write_text(json.dumps(dataclasses.asdict(inst.Rule())), encoding="utf-8")
    inst._main(["--run", "classical", "--split", "oof", "--folds", "0", "--rule", str(rule), "--out", str(pred_csv)])
    recs = D.load_records()
    gt = {rid: rles for s in sorted(stems) for rid, rles in recs[s].items()}
    inst.write_submission(gt, gt_csv)                   # host ground-truth format: <annotator>-<stem>_<k>,<counts>

    overlap = host["get_overlap_df"](pd.read_csv(gt_csv), pd.read_csv(pred_csv))
    host_pq = host["get_pq_score"](overlap)
    hit = [row.iou_matrix > 0.5 for row in overlap.itertuples(index=False)]
    host_counts = (sum(int(h.sum()) for h in hit), sum(host["fp_count_hit"](h) for h in hit),
                   sum(host["fn_count_hit"](h) for h in hit))
    preds = pf.read_submission(pred_csv)        # run() proved this equals the masks in memory
    ours = pq.score(gt, preds)
    pre = pf.preflight(pred_csv, stems, preview=False)
    print("G-F", json.dumps(dict(
        host_pq=host_pq, our_pq=ours["pq"], host_tp_fp_fn=host_counts, our_tp_fp_fn=(ours["tp"], ours["fp"], ours["fn"]),
        annotator_images=len(overlap), images=len(stems), images_with_rows=len(preds),
        rows_per_image=pre["rows_per_image"], preflight_checks=pre["checks"], format_error=pre.get("format_error"))))

    assert len(overlap) == len(gt) and ours["tp"] > 0
    assert host_counts == (ours["tp"], ours["fp"], ours["fn"])
    assert abs(ours["pq"] - host_pq) < 1e-6, (ours["pq"], host_pq)
    assert pre["checks"]["format"], pre.get("format_error")
