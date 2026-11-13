import numpy as np
from conftest import rect
from pycocotools import mask as mask_util

from filseg import eda

A, B = rect(100, 110, 100, 110), rect(500, 510, 500, 510)


def test_agreement_counts_other_annotators():
    recs = {"x": {"01-x": [A, B], "02-x": [A], "03-x": [rect(100, 110, 105, 115)]}, "solo": {"01-solo": [A]}}
    ag = eda.agreement(recs).set_index(["rid", "area"])["matched"]
    # A: matched by 02 (IoU 1), not by 03 (IoU 1/3) -> 1; B: nobody -> 0; the single-annotator image is skipped
    assert ag["01-x", 100] .tolist() == [1, 0] and len(ag) == 4


def test_filament_table_shape_and_parts():
    recs = {"x": {"01-x": [rect(1000, 1010, 800, 900), mask_util.merge([A, B])]}}
    f = eda.filament_table(recs)
    assert f.area.tolist() == [1000, 200] and f.parts.tolist() == [1, 2]
    assert abs(f.width_in[0] - 10) < 1.5 and f.elong[0] > 8 and f.dx_R[0] > 0


def test_gap_hours_skips_self():
    t = np.array([0., 5., 100.])
    assert eda.gap_hours(t, t, same=True).tolist() == [5, 5, 95] and eda.gap_hours(t[:1], t[1:]).tolist() == [5]
