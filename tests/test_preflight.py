import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import rect
from pycocotools import mask as mask_util

from filseg import instances as inst
from filseg import preflight as pf
from filseg.data import FOLDS_JSON


@pytest.fixture(autouse=True)
def _no_real_subs(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "SUBS", tmp_path / "no_subs")       # the default earlier-entries folder is not the real one


def test_validator_rejects_overlap(tmp_path):
    a = np.zeros((2048, 2048), np.uint8); a[10:20, 10:20] = 1
    r = [mask_util.encode(np.asfortranarray(a)) for _ in range(2)]
    inst.write_submission({"s": r}, tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="overlap"):
        pf.validate_submission(tmp_path / "bad.csv", {"s"})


def test_validator_rejects_unknown_stem(tmp_path):
    inst.write_submission({"not_a_test_image": [rect(10, 20, 10, 20)]}, tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="stem"):
        pf.validate_submission(tmp_path / "bad.csv", {"s"})


def test_preflight_flags_near_copy(tmp_path):
    inst.write_submission({"s": [rect(10, 40, 10, 40)]}, tmp_path / "a.csv")
    inst.write_submission({"s": [rect(10, 40, 10, 40)]}, tmp_path / "b.csv")
    assert pf.preflight(tmp_path / "b.csv", {"s"}, previous=[tmp_path / "a.csv"])["near_copy"] is True


def test_preflight_far_from_earlier_entry_is_not_near_copy(tmp_path):
    inst.write_submission({"s": [rect(10, 40, 10, 40)]}, tmp_path / "a.csv")
    inst.write_submission({"s": [rect(10, 40, 10, 40), rect(100, 110, 100, 110)]}, tmp_path / "b.csv")
    res = pf.preflight(tmp_path / "b.csv", {"s"}, previous=[tmp_path / "a.csv"])
    assert res["near_copy"] is False and res["near_copy_pq"][str(tmp_path / "a.csv")] == pytest.approx(2 / 3)


def _good(tmp_path):
    inst.write_submission({"s": [rect(10, 20, 10, 20)]}, tmp_path / "ok.csv")
    return (tmp_path / "ok.csv").read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("mutate, match", [
    (lambda ls: ["filament_id,rle"] + ls[1:], "header"),
    (lambda ls: ls + [""], "empty line"),
    (lambda ls: ls[:1] + [ls[1].replace(",", ',"') + '"'], "quote"),
    (lambda ls: ls[:1] + [ls[1].replace("s_1", "s_2")], "from 1"),
    (lambda ls: ls[:1] + [ls[1].replace("s_1", "s_01")], "integer"),
    (lambda ls: ls[:1] + [ls[1].replace("s_1", "s_1,x")], "two fields"),
    (lambda ls: ls[:1] + [ls[1], ls[1]], "from 1"),
    (lambda ls: ls[:1] + [ls[1][:-1]], "RLE"),
])
def test_validator_rejects_format_faults(tmp_path, mutate, match):
    ls = _good(tmp_path)
    (tmp_path / "bad.csv").write_text("\n".join(mutate(ls)) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match=match):
        pf.validate_submission(tmp_path / "bad.csv", {"s"})


def test_validator_rejects_crlf_and_wrong_size_and_empty_masks(tmp_path):
    ls = _good(tmp_path)
    (tmp_path / "bad.csv").write_bytes(("\r\n".join(ls) + "\r\n").encode())
    with pytest.raises(ValueError, match="carriage return"):
        pf.validate_submission(tmp_path / "bad.csv", {"s"})
    small = mask_util.encode(np.asfortranarray(np.ones((1024, 1024), np.uint8)))      # a 1024 px map by mistake
    inst.write_submission({"s": [small]}, tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="2048"):
        pf.validate_submission(tmp_path / "bad.csv", {"s"})
    inst.write_submission({"s": [mask_util.encode(np.asfortranarray(np.zeros((2048, 2048), np.uint8)))]}, tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="empty mask"):
        pf.validate_submission(tmp_path / "bad.csv", {"s"})


def test_runs_parser_matches_pycocotools():
    m = np.zeros((2048, 2048), np.uint8); m[5:300, 7:90] = 1; m[1000:1003, 0:2048] = 1; m[2047, 2047] = 1
    c = mask_util.encode(np.asfortranarray(m))["counts"].decode()
    runs = pf._runs(c)
    assert sum(runs) == 2048 * 2048 and sum(runs[1::2]) == m.sum()


def test_roundtrip_equal_catches_transposed_encoding(tmp_path):
    m = np.zeros((2048, 2048), np.uint8); m[10:20, 100:300] = 1
    good = [mask_util.encode(np.asfortranarray(m))]
    inst.write_submission({"s": good}, tmp_path / "t.csv")
    assert pf.roundtrip_equal(tmp_path / "t.csv", {"s": good, "empty": []})
    row_major = [mask_util.encode(np.asfortranarray(m.T))]
    assert not pf.roundtrip_equal(tmp_path / "t.csv", {"s": row_major})
    assert not pf.roundtrip_equal(tmp_path / "t.csv", {"s": good, "other": good})


def _rows(n, shift=0):
    return [rect(1000, 1010, 1000 + 20 * i + shift, 1010 + 20 * i + shift) for i in range(n)]


def test_preflight_twin_comparisons(tmp_path):
    inst.write_submission({"s": _rows(4)}, tmp_path / "a.csv")
    inst.write_submission({"t": _rows(4), "u": _rows(4)}, tmp_path / "twin_same.csv")
    inst.write_submission({"t": _rows(8)}, tmp_path / "twin_more.csv")       # twin image u has no row: 8 / 2 = 4 per image
    kw = dict(twin_stems={"t", "u"}, preview=False)
    same = pf.preflight(tmp_path / "a.csv", {"s"}, twin=tmp_path / "twin_same.csv", **kw)["checks"]
    more = pf.preflight(tmp_path / "a.csv", {"s"}, twin=tmp_path / "twin_more.csv", **kw)["checks"]
    assert same["rows_within_20pct_of_twin"] and same["area_within_30pct_of_twin"]
    assert more["rows_within_20pct_of_twin"] is True
    inst.write_submission({"t": _rows(8)}, tmp_path / "twin_few.csv")
    few = pf.preflight(tmp_path / "a.csv", {"s"}, twin=tmp_path / "twin_few.csv", twin_stems={"t", "u", "v", "w"}, preview=False)
    assert few["checks"]["rows_within_20pct_of_twin"] is False          # 2 per image vs 4
    big = {"t": [rect(1000, 1020, 1000 + 30 * i, 1020 + 30 * i) for i in range(4)]}   # 400 px vs 100 px masks
    inst.write_submission(big, tmp_path / "twin_big.csv")
    r = pf.preflight(tmp_path / "a.csv", {"s"}, twin=tmp_path / "twin_big.csv", twin_stems={"t"}, preview=False)
    assert r["checks"]["rows_within_20pct_of_twin"] and r["checks"]["area_within_30pct_of_twin"] is False
    assert pf.preflight(tmp_path / "a.csv", {"s"}, preview=False)["checks"]["twin_given"] is False


def test_preflight_row_count_empty_and_disk_checks(tmp_path):
    inst.write_submission({"s": [rect(0, 10, 0, 10)]}, tmp_path / "a.csv")            # corner: off the disk, 1 row
    res = pf.preflight(tmp_path / "a.csv", {"s", "t"}, preview=False)
    c = res["checks"]
    assert res["rows_per_image"] == 0.5 and res["empty_image_frac"] == 0.5 and res["off_disk"] == ["s_1"]
    assert not (c["rows_per_image_3_to_12"] or c["empty_images_at_most_10pct"] or c["mask_centres_on_disk"])
    inst.write_submission({"s": _rows(13)}, tmp_path / "b.csv")
    assert pf.preflight(tmp_path / "b.csv", {"s"}, preview=False)["checks"]["rows_per_image_3_to_12"] is False
    inst.write_submission({"s": [rect(10, 20, 10, 20)], "t": [rect(10, 20, 10, 20)]}, tmp_path / "c.csv")
    bad = pf.preflight(tmp_path / "c.csv", {"s"}, preview=False)                       # stem t unknown
    assert bad["checks"] == {"format": False} and bad["passed"] is False and "stem" in bad["format_error"]


def test_git_tree_clean_check(tmp_path):
    inst.write_submission({"s": _rows(4)}, tmp_path / "a.csv")
    repo = tmp_path / "r"; repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    kw = dict(preview=False, previous=(), repo=repo)
    assert pf.preflight(tmp_path / "a.csv", {"s"}, **kw)["checks"]["git_tree_clean"] is True
    (repo / "dirty.txt").write_text("x", encoding="utf-8")
    assert pf.preflight(tmp_path / "a.csv", {"s"}, **kw)["checks"]["git_tree_clean"] is False
    assert pf.preflight(tmp_path / "a.csv", {"s"}, **{**kw, "repo": tmp_path / "nowhere"})["checks"]["git_tree_clean"] is False


def test_preview_sheet(tmp_path):
    imgs = tmp_path / "imgs"; imgs.mkdir()
    stems = {f"img{i:02d}" for i in range(14)}
    for s in stems:
        cv2.imwrite(str(imgs / f"{s}.jpeg"), np.full((2048, 2048), 90, np.uint8))
    inst.write_submission({"img00": [rect(1000, 1100, 1000, 1100)], "img01": []}, tmp_path / "sub.csv")
    out = pf.preview_sheet(tmp_path / "sub.csv", stems, imgs)
    assert out == tmp_path / "sub_preview.png" and out.stat().st_size > 5000
    assert pf.preview_sheet(tmp_path / "sub.csv", {"no_such_image"}, imgs) is None
    res = pf.preflight(tmp_path / "sub.csv", stems, img_dir=imgs)
    assert res["preview"] == str(out)


# ---- fix round 1 (review I1, I2, I3) ----

def test_previous_defaults_to_earlier_entries_in_subs(tmp_path):
    subs = tmp_path / "subs"; subs.mkdir()
    for n in ["s1_a.csv", "s2_b.csv", "s3_c_oof.csv", "other.csv", "s_x.csv", "S4_me.csv"]:
        inst.write_submission({"s": [rect(10, 40, 10, 40)]}, subs / n)
    res = pf.preflight(subs / "S4_me.csv", {"s"}, preview=False, subs_dir=subs)
    assert [Path(p).name for p in res["previous_files"]] == ["s1_a.csv", "s2_b.csv"] and res["n_previous_compared"] == 2
    assert res["near_copy"] is True and res["checks"]["not_near_copy"] is False and res["warnings"] == []
    none = pf.preflight(subs / "S4_me.csv", {"s"}, preview=False, subs_dir=tmp_path / "empty")
    assert none["n_previous_compared"] == 0 and any("0 earlier entries" in w for w in none["warnings"])
    assert pf.preflight(subs / "S4_me.csv", {"s"}, preview=False, previous=(), subs_dir=subs)["n_previous_compared"] == 0


def test_cli_states_how_many_earlier_entries_were_compared(tmp_path, monkeypatch, capsys):
    imgs, subs = tmp_path / "imgs", tmp_path / "subs"; imgs.mkdir(); subs.mkdir()
    for s in "ab":
        cv2.imwrite(str(imgs / f"{s}.jpeg"), np.full((2048, 2048), 90, np.uint8))
    monkeypatch.setattr(pf, "TEST_IMG", imgs); monkeypatch.setattr(pf, "SUBS", subs)
    inst.write_submission({"a": _rows(4), "b": _rows(4)}, subs / "s2_new.csv")
    with pytest.raises(SystemExit) as e:
        pf._main([str(subs / "s2_new.csv")])
    out = capsys.readouterr().out
    assert e.value.code == 1 and "compared against 0 earlier entries" in out and "WARN" in out
    inst.write_submission({"a": _rows(4), "b": _rows(4)}, subs / "s1_old.csv")
    with pytest.raises(SystemExit):
        pf._main([str(subs / "s2_new.csv")])
    out = capsys.readouterr().out
    assert "compared against 1 earlier entries" in out and "FAIL  not_near_copy" in out and "s1_old.csv" in out


@pytest.fixture
def built(tmp_path, monkeypatch):
    """A CSV built by instances.run (so with its sidecar) from a 1-fold oof run; git faked as clean."""
    monkeypatch.setattr(pf, "_git", lambda args, repo: "c0ffee" if args[0] == "rev-parse" else "")
    blob = np.zeros((2048, 2048), np.uint8); blob[1000:1030, 1000:1030] = 230
    (tmp_path / "A").mkdir()
    np.savez(tmp_path / "A" / "oof_fold0.npz", stems=np.array(["has_one", "empty_stem"]), prob=np.stack([blob, np.zeros_like(blob)]))
    (tmp_path / "A" / "fold0_meta.json").write_text(json.dumps({"commit": "k0", "ok": True}), encoding="utf-8")
    rule = tmp_path / "rule.json"; rule.write_text(json.dumps({"t_lo": 0.4}), encoding="utf-8")
    stems = {"has_one", "empty_stem"}
    inst.run(["A"], "oof", inst.Rule(), tmp_path / "built.csv", tmp_path, stems=stems, rule_file=rule)
    return tmp_path / "built.csv", stems


def _pre(csv, stems, **kw):
    return pf.preflight(csv, stems, preview=False, previous=(), **kw)


def test_provenance_checks_pass_for_a_fresh_build(built):
    res = _pre(*built)
    c = res["checks"]
    assert c["provenance_sidecar"] and c["decoded_equals_in_memory"] and c["build_tree_clean"]
    assert c["commit_rule_kernels_recorded"] and c["git_tree_clean"]
    assert res["commit"] == "c0ffee" and res["build"]["folds_found"] == {"A": [0]} and res["build"]["files"][0]["meta"]["commit"] == "k0"


def test_provenance_checks_fail_not_vanish(built):
    csv, stems = built
    meta = pf.meta_path(csv); good = json.loads(meta.read_text(encoding="utf-8"))

    def with_meta(**kw):
        meta.write_text(json.dumps({**good, **kw}), encoding="utf-8")
        return _pre(csv, stems)["checks"]
    one_file = dict(files=[dict(run="A", file="f", sha256="0", meta=None)])
    assert with_meta(roundtrip_equal=False)["decoded_equals_in_memory"] is False
    assert with_meta(dirty=True)["build_tree_clean"] is False
    assert with_meta(dirty=None)["build_tree_clean"] is False
    assert with_meta(rule_sha256=None)["commit_rule_kernels_recorded"] is False
    assert with_meta(commit=None)["commit_rule_kernels_recorded"] is False
    assert with_meta(run_meta={"A": None}, **one_file)["commit_rule_kernels_recorded"] is False
    assert with_meta(run_meta={"A": {"commit": "x"}}, **one_file)["commit_rule_kernels_recorded"] is True
    meta.unlink()
    res = _pre(csv, stems)
    c = res["checks"]
    assert [c[k] for k in ("provenance_sidecar", "decoded_equals_in_memory", "build_tree_clean", "commit_rule_kernels_recorded")] == [False] * 4
    assert res["passed"] is False and "missing" in res["provenance_error"]


def test_provenance_fails_when_the_csv_changed_after_the_build(built):
    csv, stems = built
    inst.write_submission({"has_one": [rect(1000, 1031, 1000, 1030)]}, csv)
    res = _pre(csv, stems)
    c = res["checks"]
    assert c["provenance_sidecar"] is False and c["decoded_equals_in_memory"] is False and c["build_tree_clean"] is False
    assert "sha256" in res["provenance_error"]


def test_twin_stems_come_from_the_twin_sidecar_then_the_option(tmp_path):
    folds = json.loads(FOLDS_JSON.read_text(encoding="utf-8"))
    fold0 = sorted(s for s, f in folds.items() if f == 0)
    inst.write_submission({"s": _rows(4)}, tmp_path / "a.csv")
    inst.write_submission({s: [rect(1000, 1010, 1000, 1010)] for s in fold0}, tmp_path / "twin.csv")   # 1 row per fold-0 image
    kw = dict(twin=tmp_path / "twin.csv", preview=False, previous=())
    allf = pf.preflight(tmp_path / "a.csv", {"s"}, **kw)
    assert allf["twin_images"] == len(folds) and allf["checks"]["rows_within_20pct_of_twin"] is False
    opt = pf.preflight(tmp_path / "a.csv", {"s"}, twin_folds=[0], **kw)
    assert opt["twin_images"] == len(fold0) and opt["twin_rows_per_image"] == 1.0
    pf.meta_path(tmp_path / "twin.csv").write_text(json.dumps({"split": "oof", "folds": [0]}), encoding="utf-8")
    side = pf.preflight(tmp_path / "a.csv", {"s"}, **kw)
    assert side["twin_images"] == len(fold0) and side["twin_folds"] == [0]
    assert pf.preflight(tmp_path / "a.csv", {"s"}, twin_folds=[1], **kw)["twin_images"] == len(fold0)      # sidecar first
