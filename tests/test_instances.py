import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_util

from filseg import instances as inst
from filseg import preflight as pf
from filseg.data import FOLDS_JSON


def test_components_8_connected():
    p = np.zeros((2048, 2048), np.float32); p[1000:1010, 1000:1020] = 0.9; p[1010, 1020] = 0.9
    p[500:505, 500:505] = 0.6
    _, c = inst.candidates(p, t_lo=0.5)
    assert sorted(x["area"] for x in c) == [25, 201]


def test_limb_and_area_rule():
    p = np.zeros((2048, 2048), np.float32)
    p[1014:1034, 1014:1034] = 0.9          # disk centre, 400 px
    p[1014:1034, 1868:1888] = 0.9          # r about 0.95 R
    p[400:410, 1000:1010] = 0.9            # 100 px, too small
    _, c = inst.candidates(p, t_lo=0.5)
    kept = inst.select(c, inst.Rule(t_lo=0.5, peak=0.7, mean=0.5, min_area=200, max_r=0.92))
    assert len(kept) == 1 and [x for x in c if x["id"] == kept[0]][0]["r_frac"] < 0.05


def test_rle_is_column_major_and_roundtrips(tmp_path):
    lab = np.zeros((2048, 2048), np.int32); lab[3, 7:40] = 1
    rles = inst.to_rles(lab, [1]); m = mask_util.decode(rles[0])
    assert m[3, 7:40].all() and m.sum() == 33
    inst.write_submission({"20150125172714Mh": rles, "empty_stem": []}, tmp_path / "s.csv")
    rows = (tmp_path / "s.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "filament_id,segmentation_rle" and len(rows) == 2
    assert rows[1].startswith("20150125172714Mh_1,") and '"' not in rows[1]
    pf.validate_submission(tmp_path / "s.csv", {"20150125172714Mh", "empty_stem"})


def test_candidate_features():
    p = np.zeros((2048, 2048), np.float32); p[100:110, 300:330] = 0.8; p[100, 300] = 0.95
    lab, (c,) = inst.candidates(p, 0.5)
    assert lab.dtype == np.int32 and (lab == c["id"]).sum() == 300
    assert (c["area"], c["w"], c["h"], c["fill"]) == (300, 30, 10, 1.0)
    assert c["peak"] == pytest.approx(0.95) and c["p50"] == pytest.approx(0.8) and c["mean"] == pytest.approx(0.8, abs=1e-3)
    assert (c["cy"], c["cx"]) == (104.5, 314.5) and c["r_frac"] == pytest.approx(np.hypot(314.5 - 1023.1, 104.5 - 1024.1) / 899.7)
    assert c["major"] == pytest.approx(4 * np.sqrt(np.var(np.tile(np.arange(30), 10), ddof=1)), rel=1e-6) and c["major"] > c["minor"]
    assert inst.candidates(np.zeros((2048, 2048), np.float32), 0.5)[1] == []


def test_to_rles_masks_are_exact_and_disjoint():
    lab = np.zeros((2048, 2048), np.int32); lab[10:20, 5:9] = 1; lab[10:20, 9:12] = 2; lab[50:52, 5:9] = 3
    rles = inst.to_rles(lab, [1, 3, 2])
    for r, i in zip(rles, [1, 3, 2]):
        assert np.array_equal(mask_util.decode(r), lab == i)
    assert sum(mask_util.decode(r).astype(int) for r in rles).max() == 1


def test_select_tau_needs_booster():
    c = [dict(id=1, area=300, peak=0.9, mean=0.8, r_frac=0.1), dict(id=2, area=300, peak=0.9, mean=0.8, r_frac=0.1)]
    with pytest.raises(ValueError, match="booster"):
        inst.select(c, inst.Rule(tau=0.5))
    assert inst.select(c, inst.Rule(tau=0.5), booster=lambda cs: [0.9, 0.1]) == [1]
    assert inst.select(c, inst.Rule(), booster=lambda cs: [0.0, 0.0]) == [1, 2]     # tau None: booster off


def _npz(path, stems, probs, compressed=False):
    (np.savez_compressed if compressed else np.savez)(path, stems=np.array(stems), prob=np.stack(probs))


def _flat(v):
    return np.full((2048, 2048), v, np.uint8)


@pytest.mark.parametrize("compressed", [False, True])
def test_iter_probs_averages_every_file_and_run(tmp_path, compressed):
    for run, vals in {"A": (255, 0), "B": (51, 51)}.items():       # test split: both folds hold both stems
        (tmp_path / run).mkdir()
        for k, v in enumerate(vals):
            _npz(tmp_path / run / f"test_fold{k}.npz", ["s2", "s1"], [_flat(v), _flat(v // 3)], compressed)
    got = dict(inst.iter_probs(["A", "B"], "test", tmp_path))
    assert list(got) == ["s2", "s1"] and got["s2"].dtype == np.float32
    assert np.allclose(got["s2"], (255 + 0 + 51 + 51) / 4 / 255) and np.allclose(got["s1"], (85 + 0 + 17 + 17) / 4 / 255)


def test_iter_probs_oof_checks_coverage(tmp_path):
    (tmp_path / "A").mkdir(); (tmp_path / "B").mkdir()
    _npz(tmp_path / "A" / "oof_fold0.npz", ["s1"], [_flat(255)]); _npz(tmp_path / "A" / "oof_fold1.npz", ["s2"], [_flat(0)])
    _npz(tmp_path / "B" / "oof_fold0.npz", ["s1"], [_flat(51)])
    with pytest.raises(ValueError, match="missing"):
        list(inst.iter_probs(["A", "B"], "oof", tmp_path))               # s2 absent from B
    _npz(tmp_path / "A" / "oof_fold1.npz", ["s1"], [_flat(0)])
    with pytest.raises(ValueError, match="more than one"):
        list(inst.iter_probs(["A"], "oof", tmp_path))                    # s1 in two oof folds of A
    with pytest.raises(FileNotFoundError):
        list(inst.iter_probs(["A"], "test", tmp_path))


def test_iter_probs_restricts_to_folds(tmp_path):
    (tmp_path / "A").mkdir()
    blob = np.zeros((2048, 2048), np.uint8); blob[1000:1030, 1000:1030] = 230
    _npz(tmp_path / "A" / "oof_fold0.npz", ["s0"], [blob]); _npz(tmp_path / "A" / "oof_fold1.npz", ["s1"], [blob])
    assert [s for s, _ in inst.iter_probs(["A"], "oof", tmp_path, folds=[1])] == ["s1"]
    assert [s for s, _ in inst.iter_probs(["A"], "oof", tmp_path)] == ["s0", "s1"]
    with pytest.raises(FileNotFoundError):
        list(inst.iter_probs(["A"], "oof", tmp_path, folds=[2]))


def test_run_end_to_end(tmp_path):
    blob = np.zeros((2048, 2048), np.uint8); blob[1000:1030, 1000:1030] = 230      # 900 px, p = 0.90
    (tmp_path / "A").mkdir()
    _npz(tmp_path / "A" / "oof_fold0.npz", ["has_one", "empty_stem"], [blob, np.zeros_like(blob)])
    rule_json = tmp_path / "rule.json"
    rule_json.write_text(json.dumps({"t_lo": 0.4, "peak": 0.7, "mean": 0.55, "min_area": 200, "max_r": 0.92, "tau": None}),
                         encoding="utf-8")
    out = tmp_path / "subs" / "x.csv"
    preds = inst.run(["A"], "oof", inst.Rule(**json.loads(rule_json.read_text(encoding="utf-8"))), out, tmp_path,
                     stems={"has_one", "empty_stem", "other"})
    assert [len(preds[s]) for s in ("has_one", "empty_stem")] == [1, 0]
    rows = out.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2 and rows[1].startswith("has_one_1,")
    assert mask_util.decode({"size": [2048, 2048], "counts": rows[1].split(",")[1]}).sum() == 900
    with pytest.raises(ValueError, match="stems do not match"):                # oof stems must be known stems
        inst.run(["A"], "oof", inst.Rule(), out, tmp_path, stems={"only_this"})
    _npz(tmp_path / "A" / "test_fold0.npz", ["has_one"], [blob])
    with pytest.raises(ValueError, match="stems do not match"):                # test split must cover every image
        inst.run(["A"], "test", inst.Rule(), out, tmp_path, stems={"has_one", "never_predicted"})


# ---- fix round 1: provenance sidecar and fold-count reporting (review I2, I4) ----

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _two_folds(tmp_path, stems=("has_one", "empty_stem")):
    blob = np.zeros((2048, 2048), np.uint8); blob[1000:1030, 1000:1030] = 230
    (tmp_path / "A").mkdir(exist_ok=True)
    _npz(tmp_path / "A" / "oof_fold0.npz", [stems[0]], [blob])
    _npz(tmp_path / "A" / "oof_fold1.npz", [stems[1]], [np.zeros_like(blob)])


def test_run_writes_provenance_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "_git", lambda args, repo: "c0ffee" if args[0] == "rev-parse" else "")
    _two_folds(tmp_path)
    (tmp_path / "A" / "fold0_meta.json").write_text(json.dumps({"commit": "k0", "ok": True}), encoding="utf-8")
    rule_file = tmp_path / "rule.json"; rule_file.write_text(json.dumps(asdict(inst.Rule())), encoding="utf-8")
    out = tmp_path / "s.csv"
    inst.run(["A"], "oof", inst.Rule(), out, tmp_path, stems={"has_one", "empty_stem"}, rule_file=rule_file)
    m = json.loads(Path(str(out) + ".meta.json").read_text(encoding="utf-8"))
    assert m["csv_sha256"] == _sha(out) and (m["rows"], m["images"]) == (1, 2)
    assert (m["commit"], m["dirty"], m["roundtrip_equal"]) == ("c0ffee", False, True)
    assert m["rule"] == asdict(inst.Rule()) and m["rule_sha256"] == _sha(rule_file)
    assert (m["runs"], m["split"], m["folds"], m["folds_found"]) == (["A"], "oof", [0, 1], {"A": [0, 1]})
    f0, f1 = m["files"]
    assert (f0["run"], f0["file"], f0["sha256"]) == ("A", "oof_fold0.npz", _sha(tmp_path / "A" / "oof_fold0.npz")[:12])
    assert f0["meta"] == {"commit": "k0", "ok": True} and f1["meta"] is None


def test_sidecar_records_a_dirty_tree(tmp_path, monkeypatch):
    repo = tmp_path / "r"; repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    monkeypatch.setattr(inst, "REPO", repo)
    _two_folds(tmp_path)
    side = tmp_path / "s.csv.meta.json"
    inst.run(["A"], "oof", inst.Rule(), tmp_path / "s.csv", tmp_path, stems={"has_one", "empty_stem"})
    assert json.loads(side.read_text(encoding="utf-8"))["dirty"] is False
    (repo / "x.txt").write_text("x", encoding="utf-8")
    inst.run(["A"], "oof", inst.Rule(), tmp_path / "s.csv", tmp_path, stems={"has_one", "empty_stem"})
    assert json.loads(side.read_text(encoding="utf-8"))["dirty"] is True


def test_failed_build_leaves_no_stale_sidecar(tmp_path):
    _two_folds(tmp_path)
    inst.run(["A"], "oof", inst.Rule(), tmp_path / "s.csv", tmp_path, stems={"has_one", "empty_stem"})
    assert (tmp_path / "s.csv.meta.json").exists()
    with pytest.raises(ValueError, match="stems do not match"):
        inst.run(["A"], "oof", inst.Rule(), tmp_path / "s.csv", tmp_path, stems={"only_this"})
    assert not (tmp_path / "s.csv.meta.json").exists()


def test_run_reports_and_enforces_fold_count(tmp_path, capsys):
    _two_folds(tmp_path)
    kw = dict(stems={"has_one", "empty_stem"})
    inst.run(["A"], "oof", inst.Rule(), tmp_path / "s.csv", tmp_path, expect_folds=2, **kw)
    assert "A: oof fold files [0, 1]" in capsys.readouterr().out
    with pytest.raises(ValueError, match="expected 3"):
        inst.run(["A"], "oof", inst.Rule(), tmp_path / "s.csv", tmp_path, expect_folds=3, **kw)
    with pytest.raises(ValueError, match="A: 1 oof fold file"):                  # the folds= filter counts too
        inst.run(["A"], "oof", inst.Rule(), tmp_path / "s.csv", tmp_path, folds=[0], expect_folds=2, **kw)


def test_main_records_rule_file_and_enforces_expect_folds(tmp_path, capsys):
    stems = sorted(json.loads(FOLDS_JSON.read_text(encoding="utf-8")))[:2]
    _two_folds(tmp_path, stems)
    rule = tmp_path / "rule.json"; rule.write_text(json.dumps(asdict(inst.Rule())), encoding="utf-8")
    argv = ["--run", "A", "--split", "oof", "--rule", str(rule), "--out", str(tmp_path / "s.csv"), "--runs-root", str(tmp_path)]
    inst._main(argv + ["--expect-folds", "2"])
    assert "A: oof fold files [0, 1]" in capsys.readouterr().out
    assert json.loads((tmp_path / "s.csv.meta.json").read_text(encoding="utf-8"))["rule_sha256"] == _sha(rule)
    with pytest.raises(ValueError, match="expected 3"):
        inst._main(argv + ["--expect-folds", "3"])
