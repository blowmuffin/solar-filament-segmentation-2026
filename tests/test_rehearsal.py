"""Gate G-F: the host's own scorer reads our submission file exactly as we mean it (no entry spent).

FILSEG_REHEARSAL_RUN (default classical) names the run whose fold-0 OOF maps are used and FILSEG_REHEARSAL_RULE
(default ../runs/classical/rule_classical.json) the keep rule, so the same test reruns on a trained model."""
import json
import os
from pathlib import Path

import pandas as pd
import pytest
from test_pq import _host_namespace

from filseg import data as D
from filseg import instances as inst
from filseg import pq
from filseg import preflight as pf

SUBS = (D.REPO / ".." / "subs").resolve()
# not content checks: no twin exists for a file that is itself OOF; the tree checks follow the repo's state
NOT_APPLICABLE = {"twin_given", "git_tree_clean", "build_tree_clean"}


@pytest.mark.slow
def test_host_scorer_reads_the_oof_file_as_we_mean_it():
    host = _host_namespace()
    run = os.environ.get("FILSEG_REHEARSAL_RUN", "classical")
    rule = Path(os.environ.get("FILSEG_REHEARSAL_RULE", inst.RUNS / "classical" / "rule_classical.json"))
    if not (inst.RUNS / run / "oof_fold0.npz").exists() or not rule.is_file():
        pytest.skip(f"needs {inst.RUNS / run / 'oof_fold0.npz'} (run `python -m filseg.baseline` for classical) and {rule}")
    stems = {s for s, k in json.loads(D.FOLDS_JSON.read_text(encoding="utf-8")).items() if k == 0}
    SUBS.mkdir(exist_ok=True)
    pred_csv, gt_csv = SUBS / "rehearsal_oof.csv", SUBS / "rehearsal_gt.csv"
    inst._main(["--run", run, "--split", "oof", "--folds", "0", "--rule", str(rule), "--out", str(pred_csv)])
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
    pre = pf.preflight(pred_csv, stems, previous=[], preview=False)
    checks = {k: v for k, v in pre["checks"].items() if k not in NOT_APPLICABLE}
    print("G-F", json.dumps(dict(
        run=run, rule=str(rule), host_pq=host_pq, our_pq=ours["pq"], host_tp_fp_fn=host_counts,
        our_tp_fp_fn=(ours["tp"], ours["fp"], ours["fn"]), annotator_images=len(overlap), images=len(stems),
        images_with_rows=len(preds), rows_per_image=pre["rows_per_image"], empty_image_frac=pre["empty_image_frac"],
        preflight_checks=pre["checks"], format_error=pre.get("format_error"))))

    assert len(overlap) == len(gt)
    assert ours["tp"] >= 30, ours["tp"]                 # a gate that rests on a handful of matches proves little
    assert host_counts == (ours["tp"], ours["fp"], ours["fn"])
    assert abs(ours["pq"] - host_pq) < 1e-6, (ours["pq"], host_pq)
    assert all(checks.values()), {k: v for k, v in checks.items() if not v} | {"format_error": pre.get("format_error")}
