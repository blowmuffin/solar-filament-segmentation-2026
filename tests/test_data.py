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


def test_build_cache_layout_and_scale(tmp_path, monkeypatch):    # two stems, 256 px: fast stand-in for the full cache
    recs = D.load_records()
    stems = sorted(s for s, r in recs.items() if len(r) == 3)[:2]
    monkeypatch.setattr(D, "load_records", lambda: {s: recs[s] for s in stems})
    D.build_cache(tmp_path, 256)
    assert json.loads((tmp_path / "stems.json").read_text(encoding="utf-8")) == stems
    img = np.memmap(tmp_path / "img_256.u8", np.uint8, "r", shape=(2, 256, 256))
    soft = np.memmap(tmp_path / "soft_256.u8", np.uint8, "r", shape=(2, 256, 256))
    want = cv2.resize(D.load_image(D.TRAIN_IMG / f"{stems[1]}.jpeg"), (256, 256), interpolation=cv2.INTER_AREA)
    assert (img[1] == want).all()
    assert abs(soft[0].mean() / 255 - D.soft_target(recs[stems[0]]).mean()) < 1e-3   # area resize keeps the mean
    del img, soft


def test_fit_limb_ring_recovers_circle():
    img = np.zeros((2048, 2048), np.uint8)
    cv2.circle(img, (1030, 1020), 880, 200, 1)
    cx, cy, r = D.fit_limb_ring(img)
    assert abs(cx - 1030) < 1 and abs(cy - 1020) < 1 and abs(r - 880) < 1


def test_disk_constants_on_first_image():
    cx, cy, r = D.fit_limb_ring(D.load_image(sorted(D.TRAIN_IMG.glob("*.jpeg"))[0]))
    assert max(abs(cx - D.CX), abs(cy - D.CY), abs(r - D.R)) < 5
