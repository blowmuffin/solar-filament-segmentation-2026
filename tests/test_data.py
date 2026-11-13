import json
from collections import Counter

import cv2
import numpy as np
from pycocotools import mask as mask_util

from filseg import data as D


def test_counts():
    recs = D.load_records()
    assert len(recs) == 707
    assert sum(len(r) for r in recs.values()) == 1154
    assert sum(len(f) for r in recs.values() for f in r.values()) == 8199
    assert sorted(Counter(len(r) for r in recs.values()).items()) == [(1, 411), (2, 145), (3, 151)]


def test_stem():
    assert D.stem_of("040301-20140609195854Bh") == "20140609195854Bh"


def test_first_polygon_area():          # the JSON gives area 750.0 for this filament
    rle = D.load_records()["20140609195854Bh"]["040301-20140609195854Bh"][0]
    assert abs(int(mask_util.area(rle)) - 750) <= 1


def test_soft_target_levels():
    recs = D.load_records(); stem = next(s for s, r in recs.items() if len(r) == 3)
    t = D.soft_target(recs[stem])
    assert t.shape == (2048, 2048) and set(np.unique(np.round(t * 6)).tolist()) <= {0, 2, 4, 6}


def test_folds_grouped_and_balanced():
    n = {s: len(r) for s, r in D.load_records().items()}; f = D.make_folds(n)
    assert set(f) == set(n) and sorted(Counter(f.values()).values()) == [141, 141, 141, 142, 142]
    per = Counter()
    for s, k in f.items(): per[k] += n[s]
    assert all(225 <= v <= 237 for v in per.values())
    sites = Counter(s[-2:] for s in n)
    for site in (x for x, c in sites.items() if c >= 15):
        assert {f[s] for s in n if s[-2:] == site} == {0, 1, 2, 3, 4}
    assert f == D.make_folds(n)


def test_normalize_at_both_resolutions():
    img = D.load_image(sorted(D.TRAIN_IMG.glob("*.jpeg"))[0])
    for res in (2048, 1024):
        x = D.normalize(cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)); s = res / 2048
        yy, xx = np.mgrid[:res, :res]; r = np.hypot(xx - D.CX * s, yy - D.CY * s)
        assert x.dtype == np.float32 and (x[r > D.R * s] == 0).all()
        inside = x[r < 0.98 * D.R * s]
        assert abs(inside.mean()) < 1e-3 and abs(inside.std() - 1) < 1e-3


def test_committed_folds_are_pinned():           # the frozen file must not drift from make_folds (numpy upgrade, edit)
    n = {s: len(r) for s, r in D.load_records().items()}
    pairs = json.loads(D.FOLDS_JSON.read_text(encoding="utf-8"), object_pairs_hook=list)
    assert sorted(s for s, _ in pairs) == sorted(n)           # every stem exactly once
    assert dict(pairs) == D.make_folds(n)


def test_folds_cli_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "FOLDS_JSON", tmp_path / "folds.json")
    D._main(["folds"])
    first = D.FOLDS_JSON.read_bytes()
    D.FOLDS_JSON.write_bytes(b"{}")
    try:
        D._main(["folds"])
        raise AssertionError("overwrote an existing folds.json without --force")
    except SystemExit:
        assert D.FOLDS_JSON.read_bytes() == b"{}"
    D._main(["folds", "--force"])
    assert D.FOLDS_JSON.read_bytes() == first


def test_mask_orientation_row_is_y_col_is_x():   # a transposed or flipped mask puts the centroid hundreds of px away
    coco = json.loads(D.TRAIN_JSON.read_text(encoding="utf-8"))
    rid = "040301-20140609195854Bh"
    poly = np.array(next(a for a in coco["annotations"] if a["image_id"] == rid)["segmentation"][0]).reshape(-1, 2)
    rows, cols = np.nonzero(mask_util.decode(D.load_records()["20140609195854Bh"][rid][0]))
    assert abs(rows.mean() - poly[:, 1].mean()) < 5 and abs(cols.mean() - poly[:, 0].mean()) < 5
    assert abs(poly[:, 1].mean() - poly[:, 0].mean()) > 100   # the filament is off the diagonal, so the check can tell


def test_build_cache_alignment_and_orientation(tmp_path):      # 3 stems at 1024 px: stand-in for the full cache
    recs = D.load_records()
    stems = sorted(s for s, r in recs.items() if len(r) == 3)[:3]
    D.build_cache(tmp_path, 1024, stems=stems)
    assert json.loads((tmp_path / "stems.json").read_text(encoding="utf-8")) == stems
    img = np.memmap(tmp_path / "img_1024.u8", np.uint8, "r", shape=(3, 1024, 1024))
    soft = np.memmap(tmp_path / "soft_1024.u8", np.uint8, "r", shape=(3, 1024, 1024))
    disk = D._disk_masks(1024)[0]
    for i, s in enumerate(stems):
        want_img = cv2.resize(D.load_image(D.TRAIN_IMG / f"{s}.jpeg"), (1024, 1024), interpolation=cv2.INTER_AREA)
        want_soft = np.round(cv2.resize(D.soft_target(recs[s]), (1024, 1024), interpolation=cv2.INTER_AREA) * 255)
        assert (img[i] == want_img).all() and (soft[i] == want_soft.astype(np.uint8)).all()
        # filaments are dark in H-alpha: under the soft target the image is below the disk median, and darker
        # than under the same target transposed or flipped (a misaligned target fails this)
        on = soft[i] > 127
        assert img[i][on].mean() < np.median(img[i][disk])
        for bad in (on.T, on[::-1], on[:, ::-1]):
            assert img[i][on].mean() < img[i][bad].mean() - 5
    del img, soft


def test_fit_limb_ring_recovers_circle():
    img = np.zeros((2048, 2048), np.uint8)
    cv2.circle(img, (1030, 1020), 880, 200, 1)
    cx, cy, r = D.fit_limb_ring(img)
    assert abs(cx - 1030) < 1 and abs(cy - 1020) < 1 and abs(r - 880) < 1


def test_disk_constants_on_first_image():
    cx, cy, r = D.fit_limb_ring(D.load_image(sorted(D.TRAIN_IMG.glob("*.jpeg"))[0]))
    assert max(abs(cx - D.CX), abs(cy - D.CY), abs(r - D.R)) < 5
