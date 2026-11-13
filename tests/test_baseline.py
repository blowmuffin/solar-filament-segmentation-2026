import json

import cv2
import numpy as np

from filseg import baseline
from filseg import data as D
from filseg import instances as inst


def _disk_with_bar(row):
    img = np.zeros((2048, 2048), np.uint8)
    yy, xx = np.mgrid[:2048, :2048]
    img[np.hypot(xx - D.CX, yy - D.CY) < D.R] = 150
    img[row:row + 10, 900:1100] = 60
    return img


def test_dark_residual_finds_dark_bar():
    m = baseline.dark_residual(_disk_with_bar(1000))
    assert m.dtype == np.float32 and m.min() >= 0 and m.max() <= 1
    assert m[1005, 1000] > 0.5 and m[600, 1000] < 0.1 and m[5, 5] == 0


def test_run_writes_what_a_kaggle_run_writes(tmp_path):
    imgs = tmp_path / "img"
    imgs.mkdir()
    for s, row in {"b": 700, "a": 1000, "c": 1300}.items():
        assert cv2.imwrite(str(imgs / f"{s}.jpeg"), _disk_with_bar(row))
    baseline.run(tmp_path / "runs", "classical", {"b": 0, "a": 0, "c": 1}, img_dir=imgs)
    out = tmp_path / "runs" / "classical"
    with np.load(out / "oof_fold0.npz") as z:
        assert sorted(z.files) == ["prob", "stems"] and z["stems"].dtype.kind == "U"
        assert list(z["stems"]) == ["a", "b"]                      # that fold's stems, sorted
        p = z["prob"]
        assert p.dtype == np.uint8 and p.shape == (2, 2048, 2048) and p.flags.c_contiguous
        assert not ((p > 0) & (p < 8)).any()                        # weak values are zeroed
    got = dict(inst.iter_probs(["classical"], "oof", tmp_path / "runs"))     # the downstream reader
    assert list(got) == ["a", "b", "c"] and got["a"].dtype == np.float32
    assert got["a"][1005, 1000] > 0.5 and got["a"][600, 1000] < 0.1 and got["a"][5, 5] == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert {"commit", "parameters", "seconds"} <= meta.keys() and meta["parameters"]["sigma"] == 40
