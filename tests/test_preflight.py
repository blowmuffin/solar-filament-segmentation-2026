import subprocess

import cv2
import numpy as np
import pytest
from conftest import rect
from pycocotools import mask as mask_util

from filseg import instances as inst
from filseg import preflight as pf


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


def test_git_and_record_checks(tmp_path):
    inst.write_submission({"s": _rows(4)}, tmp_path / "a.csv")
    repo = tmp_path / "r"; repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    rule = tmp_path / "rule.json"; rule.write_text("{}", encoding="utf-8")
    kw = dict(preview=False, repo=repo, rule_json=rule, kernels=["k:v1"])
    c = pf.preflight(tmp_path / "a.csv", {"s"}, **kw)["checks"]
    assert c["git_tree_clean"] is True and c["commit_rule_kernels_recorded"] is False       # no commit in the new repo yet
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
