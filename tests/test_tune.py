import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import rect

from filseg import data as D
from filseg import instances as inst
from filseg import tune


@pytest.fixture(autouse=True)
def _one_process(monkeypatch):
    monkeypatch.setattr(tune, "WORKERS", 1)         # pytest never spawns a pool


def test_pmap_with_one_worker_stays_in_process():
    assert tune._pmap(lambda x: 2 * x, [1, 2, 3], 1) == [2, 4, 6]       # a lambda cannot be sent to a pool


def test_fast_pq_equals_exact():           # same toys as test_one_tp_two_fp_one_fn and test_pooled_over_records
    kept = pd.DataFrame({"n_rec": [1, 1, 1], "sum_tp_iou": [1.0, 0.0, 0.0]})
    assert abs(tune.fast_pq(kept, n_gt=2) - 0.4) < 1e-9
    kept = pd.DataFrame({"n_rec": [2], "sum_tp_iou": [1.0]})
    assert abs(tune.fast_pq(kept, n_gt=2) - 0.5) < 1e-9


def test_paired_bootstrap():
    a = pd.DataFrame({"stem": list("abcdef"), "s": [1.0] * 6, "n_gt": [4] * 6, "k": [4] * 6})
    b = a.assign(s=2.0)
    gain, share = tune.paired_bootstrap(a, b)
    assert abs(gain - 0.25) < 1e-9 and share == 1.0
    assert tune.paired_bootstrap(a, a) == (0.0, 0.0)


def test_failure_buckets():
    p = np.zeros((2048, 2048), np.float32)
    p[1000:1020, 1000:1020] = 0.9          # filament 1: matched (400 px)
    p[1100:1110, 1000:1010] = 0.9          # filament 2: found, but 100 px is under min_area
    p[1200:1220, 1000:1010] = 0.9          # filament 3 is 20 x 40; the component covers a quarter of it
    gt = [rect(1000, 1020, 1000, 1020), rect(1100, 1110, 1000, 1010), rect(1200, 1220, 1000, 1040), rect(1400, 1420, 1000, 1020)]
    out = tune.failure_buckets_one(p, {"01-x": gt}, inst.Rule(t_lo=0.5, peak=0.7, mean=0.5, min_area=200, max_r=1.0))
    assert out == {"matched": 1, "discarded": 1, "wrong_shape": 1, "never_detected": 1}


def test_bootstrap_ci():
    same = pd.DataFrame({"stem": list("abcdef"), "s": [1.0] * 6, "n_gt": [4] * 6, "k": [4] * 6})
    assert tune.bootstrap_ci(same) == (0.25, 0.25)          # nothing varies, so the interval is a point
    mixed = same.assign(s=[0.0, 1.0, 2.0, 3.0, 4.0, 0.0])
    lo, hi = tune.bootstrap_ci(mixed)
    assert lo < tune._pq(mixed) < hi and tune.bootstrap_ci(mixed) == (lo, hi)      # seed 0 is fixed


def _row(stem, fold, t_lo, peak, mean, area, r, tp_iou, n_rec=1):
    return dict(stem=stem, fold=fold, t_lo=t_lo, id=1, area=area, peak=peak, mean=mean, r_frac=r, n_rec=n_rec,
                n_tp=int(tp_iou > 0), sum_tp_iou=tp_iou)


def _toy_table():
    """5 folds, one stem each: a good component (matches its filament) and junk (peak 0.6, 150 px, a false positive)."""
    rows = []
    for f in range(5):
        rows += [_row(f"s{f}", f, 0.5, 0.9, 0.8, 400, 0.1, 1.0), _row(f"s{f}", f, 0.5, 0.6, 0.5, 150, 0.1, 0.0)]
    return pd.DataFrame(rows)


def _records(table):
    return {s: {f"01-{s}": [rect(0, 5, 0, 5)]} for s in table.stem.unique()}       # one filament each


def test_grid_search_drops_the_junk():
    t = _toy_table()
    rule, v = tune.grid_search(t, n_gt=5)
    assert v == pytest.approx(1.0) and rule.t_lo == 0.5 and rule.tau is None
    assert (t[tune._keep(t, rule)].area == 400).all()               # the junk (150 px, peak 0.6) is dropped
    assert tune.fast_pq(t[tune._keep(t, rule)], 5) == pytest.approx(v)
    sizes = [len(v) for v in tune.GRID.values()]
    assert sizes == [7, 8, 10, 5, 5] and tune.GRID["peak"][-1] == 0.99 and tune.GRID["mean"][-1] == 0.85 and tune.GRID["t_lo"][0] == 0.25


def test_on_edge():
    r = inst.Rule(t_lo=0.25, peak=0.7, mean=0.85, min_area=100, max_r=0.92)
    assert tune._on_edge(asdict(r)) == {"t_lo": "min", "mean": "max", "min_area": "min"}
    assert tune._on_edge(asdict(inst.Rule(t_lo=0.4, peak=0.99, mean=0.6, min_area=300, max_r=1.0))) == {"peak": "max", "max_r": "max"}


def test_nested_pq_tunes_on_the_other_folds():
    t = _toy_table()
    t = pd.concat([t, pd.DataFrame([dict(stem="empty", fold=2, t_lo=np.nan, id=0, area=0, peak=0.0, mean=0.0, r_frac=0.0,
                                         n_rec=1, n_tp=0, sum_tp_iou=0.0)])])     # a stem with no component at all
    recs = _records(t)
    v, per = tune.nested_pq(t, recs)
    assert list(per.columns[:5]) == ["stem", "n_gt", "s", "k", "tp"] and len(per) == 6 and set(per.cv) == set(range(5))
    assert v == pytest.approx(5 / (0.5 * 6 + 0.5 * 5))              # the empty stem is one false negative
    for f in range(5):
        others = t[t.fold != f]
        rule, _ = tune.grid_search(others, n_gt=others.stem.nunique())
        got = per[per.cv == f].iloc[0]
        assert tuple(got[list(tune.RULE_COLS)]) == (rule.t_lo, rule.peak, rule.mean, rule.min_area, rule.max_r)


def test_nested_pq_single_fold_uses_inner_cv():
    t = _toy_table().assign(fold=0)
    v, per = tune.nested_pq(t, _records(t))
    assert v == pytest.approx(1.0) and set(per.cv) == set(range(5))        # 5 inner groups of fold 0's stems


def _prob(rects, value=230):
    m = np.zeros((2048, 2048), np.uint8)
    for r0, r1, c0, c1 in rects:
        m[r0:r1, c0:c1] = value
    return m


def _save(root, run, fold, stems, probs):
    (root / run).mkdir(parents=True, exist_ok=True)
    np.savez(root / run / f"oof_fold{fold}.npz", stems=np.array(stems), prob=np.stack(probs))


S0, S1 = "20110109104734Ch", "20110110104734Ch"


@pytest.fixture
def world(tmp_path, monkeypatch):
    folds = tmp_path / "folds.json"
    folds.write_text(json.dumps({S0: 0, S1: 1}), encoding="utf-8")
    monkeypatch.setattr(D, "FOLDS_JSON", folds)
    root = tmp_path / "runs"
    _save(root, "A", 0, [S0], [_prob([(1000, 1020, 1000, 1020)])])
    _save(root, "A", 1, [S1], [_prob([(1000, 1020, 1000, 1020)])])
    # two annotators: filament IoUs with the 400 px component are 1.0 and 0.6
    recs = {S0: {f"01-{S0}": [rect(1000, 1020, 1000, 1020)], f"02-{S0}": [rect(1000, 1020, 1005, 1025)]},
            S1: {f"01-{S1}": [rect(1000, 1020, 1000, 1020)]}}
    return root, recs


def test_candidate_table_matches_records_and_caches(world, monkeypatch):
    root, recs = world
    t = tune.candidate_table(["A"], [0, 1], records=recs, root=root)
    a = t[t.stem == S0]
    assert a.t_lo.tolist() == list(tune.T_LO_GRID) and set(a.fold) == {0} and (a.area == 400).all()
    assert (a.n_rec == 2).all() and (a.n_tp == 2).all() and a.sum_tp_iou.tolist() == pytest.approx([1.6] * len(tune.T_LO_GRID))
    b = t[t.stem == S1]
    assert (b.n_rec == 1).all() and b.sum_tp_iou.tolist() == pytest.approx([1.0] * len(tune.T_LO_GRID))
    assert (root / "A" / "cands.pkl").exists()

    with monkeypatch.context() as m:
        m.setattr(tune, "_table_chunk", lambda task: 1 / 0)             # a cache hit must not recompute
        pd.testing.assert_frame_equal(tune.candidate_table(["A"], [0, 1], records=recs, root=root), t)
        assert set(tune.candidate_table(["A"], [1], records=recs, root=root).stem) == {S1}

    # maps rewritten (here: nothing detected in S0) -> that fold is rebuilt, the other kept; S0 stays in the table
    _save(root, "A", 0, [S0], [_prob([])])
    t2 = tune.candidate_table(["A"], [0, 1], records=recs, root=root)
    a2 = t2[t2.stem == S0]
    assert len(a2) == 1 and np.isnan(a2.t_lo.iloc[0]) and a2.n_rec.iloc[0] == 2 and a2.sum_tp_iou.iloc[0] == 0
    assert len(t2[t2.stem == S1]) == len(tune.T_LO_GRID)


def test_cache_key_covers_code_folds_and_annotations(world, monkeypatch, tmp_path):
    root, recs = world
    calls, real = [], tune._table_chunk
    monkeypatch.setattr(tune, "_table_chunk", lambda task: calls.append(1) or real(task))
    src = tmp_path / "feature_code.py"
    src.write_text("a = 1\n", encoding="utf-8")
    monkeypatch.setattr(tune, "SOURCES", (src,))

    def builds() -> int:            # rebuilds caused by one call
        before = len(calls)
        tune.candidate_table(["A"], [0, 1], records=recs, root=root)
        return len(calls) - before

    assert builds() == 1 and builds() == 0                          # built once, then a cache hit
    src.write_text("a = 2\n", encoding="utf-8")                      # feature code (instances, data, tune) edited
    assert builds() == 1 and builds() == 0
    folds = Path(D.FOLDS_JSON)
    folds.write_text(folds.read_text(encoding="utf-8") + "\n", encoding="utf-8")        # folds.json changed
    assert builds() == 1 and builds() == 0
    monkeypatch.setattr(D, "TRAIN_JSON_SHA256", "0" * 64)           # another annotation file
    assert builds() == 1 and builds() == 0
    monkeypatch.setattr(tune, "T_LO_GRID", (0.3, 0.5))              # another t_lo grid
    assert builds() == 1 and builds() == 0


def test_candidate_table_averages_runs(world):
    root, recs = world
    _save(root, "B", 0, [S0], [_prob([])])
    _save(root, "B", 1, [S1], [_prob([])])
    t = tune.candidate_table(["B", "A"], [0, 1], records=recs, root=root)       # mean of 230 and 0 is 0.45
    assert (root / "A+B" / "cands.pkl").exists()
    assert sorted(t[t.stem == S0].t_lo) == [0.25, 0.3, 0.35, 0.4, 0.45]


def test_wrong_fold_file_is_refused(world):
    root, recs = world
    _save(root, "A", 0, [S1], [_prob([])])          # S1 belongs to fold 1
    with pytest.raises(ValueError, match="fold"):
        tune.candidate_table(["A"], [0], records=recs, root=root)


def _toy_world(tmp_path, monkeypatch):
    """10 stems in 5 folds (2 each) with maps under tmp_path/runs/toy; D.FOLDS_JSON and D.load_records point at them."""
    stems = [f"201101{i + 1:02d}104734{'Ch' if i % 2 else 'Lh'}" for i in range(10)]
    folds = tmp_path / "folds.json"
    folds.write_text(json.dumps({s: i % 5 for i, s in enumerate(stems)}), encoding="utf-8")
    monkeypatch.setattr(D, "FOLDS_JSON", folds)
    rng = np.random.default_rng(1)
    recs, root = {}, tmp_path / "runs"
    for k in range(5):
        fold_stems = [s for i, s in enumerate(stems) if i % 5 == k]
        probs = []
        for s in fold_stems:
            i = stems.index(s)
            r0, c0 = int(rng.integers(500, 1300)), int(rng.integers(500, 1300))
            big, small = (r0, r0 + 20, c0, c0 + 20), (r0 + 100, r0 + 110, c0, c0 + 10)
            n_rec = 1 + i % 3
            recs[s] = {f"{a + 1:02d}-{s}": [rect(*big), rect(*small)] for a in range(n_rec)}
            p = _prob([big, small], 230)
            p[r0 + 200:r0 + 215, c0:c0 + 10] = 150                                    # junk, 150 px at 0.59
            p[r0 + 300:r0 + 320, c0:c0 + 10] = 230                                    # a false positive, 200 px
            probs.append(p if i != 9 else _prob([]))                                  # the last stem finds nothing
        _save(root, "toy", k, fold_stems, probs)
    monkeypatch.setattr(D, "load_records", lambda: recs)
    return recs, root


def test_cli_end_to_end(tmp_path, monkeypatch, capsys):
    recs, root = _toy_world(tmp_path, monkeypatch)
    repo_rule = Path(tune.__file__).with_name("rule.json")
    before = repo_rule.read_bytes() if repo_rule.exists() else None
    out = tmp_path / "rule.json"
    tune.main(["--run", "toy", "--runs-root", str(root), "--rule-out", str(out), "--workers", "1"])
    text = capsys.readouterr().out
    for word in ("baseline", "best rule", "nested PQ", "SQ", "RQ", "matched", "discarded", "wrong_shape", "never_detected",
                 "annotators", "site", "within 24 h"):
        assert word in text, word
    assert "inner-CV" not in text
    assert (repo_rule.read_bytes() if repo_rule.exists() else None) == before       # nothing written into the repo
    assert (root / "toy" / "rule.json").read_text(encoding="utf-8") == out.read_text(encoding="utf-8")
    summ = json.loads((root / "toy" / "tune_summary.json").read_text(encoding="utf-8"))
    assert summ["best_rule"]["t_lo"] == 0.25 and summ["grid_edges"]["best"]["t_lo"] == "min"     # identical rows at every t_lo: first wins
    assert "WARN: the best rule sits on a grid edge" in text and "t_lo 0.25 (grid min)" in text
    n_gt = sum(len(v) for r in recs.values() for v in r.values())
    assert sum(summ["failure_buckets"].values()) == n_gt
    assert 0.0 < summ["nested"]["pq"] <= 1.0 and summ["nested"]["ci"][0] <= summ["nested"]["pq"] <= summ["nested"]["ci"][1]
    assert abs(summ["exact_check"]["fast_pq"] - summ["exact_check"]["pq_score"]) <= 0.002
    rule = inst.Rule(**json.loads(out.read_text(encoding="utf-8")))
    assert rule.peak >= 0.5 and rule.tau is None


def test_cli_pilot_is_labelled_and_writes_its_own_files(tmp_path, monkeypatch, capsys):
    recs, root = _toy_world(tmp_path, monkeypatch)
    repo_rule = Path(tune.__file__).with_name("rule.json")
    before = repo_rule.read_bytes() if repo_rule.exists() else None
    tune.main(["--run", "toy", "--folds", "0", "--runs-root", str(root), "--workers", "1"])
    text = capsys.readouterr().out
    label = "inner-CV (fold 0 only, not nested OOF)"
    assert text.count(label) >= 3 and "nested PQ" not in text          # headline, ablation table and slices carry it
    assert "NOTE: one fold only" in text
    run = root / "toy"
    assert (run / "tune_summary_folds0.json").exists() and (run / "rule_folds0.json").exists()
    assert not (run / "tune_summary.json").exists() and not (run / "rule.json").exists()
    assert (repo_rule.read_bytes() if repo_rule.exists() else None) == before
    summ = json.loads((run / "tune_summary_folds0.json").read_text(encoding="utf-8"))
    assert summ["label"] == label and summ["nested"]["pilot_inner_cv"] and summ["n_images"] == 2
    inst.Rule(**json.loads((run / "rule_folds0.json").read_text(encoding="utf-8")))
    # a full run next to it keeps its own names
    tune.main(["--run", "toy", "--runs-root", str(root), "--workers", "1"])
    assert (run / "tune_summary.json").exists() and (run / "tune_summary_folds0.json").exists()
