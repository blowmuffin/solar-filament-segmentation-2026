"""Table writers and the numbers.tex setter on tiny fake summaries (no maps, no data)."""
import numpy as np
import pytest

from filseg import report_figs as rf


def _stat(v):
    return {m: (v, v - 0.01, v + 0.01) for m in ("pq", "sq", "rq")}


def _tabular(s):
    assert s.startswith("\\begin{tabular}") and s.rstrip().endswith("\\end{tabular}")
    rows = [ln for ln in s.splitlines() if "&" in ln]
    assert len({ln.count("&") for ln in rows}) == 1          # every row has the same number of cells
    assert "nan" not in s.lower()
    return rows


def test_split_ablation_pilot_tables():
    fold_of = {f"2020010{i}0000{'Bh' if i % 2 else 'Uh'}": i % 2 for i in range(1, 7)}
    rows = _tabular(rf.tab_split(fold_of, dict.fromkeys(fold_of, 2), dict.fromkeys(fold_of, 1500)))
    assert rows[-1].startswith("all & 6 & 12 & 9{,}000 & 3 & 0 & 0 & 0 & 0 & 3")
    assert len(_tabular(rf.tab_ablation([("a", _stat(0.1)), ("b", _stat(0.4)), ("human", _stat(0.3))]))) == 4
    assert "0.400$\\pm$.010" in rf.tab_ablation([("b", _stat(0.4))])
    pil = [dict(run="A", setting="2048/512/16", pq=(0.446, 0.427, 0.463), vs_a=None, minutes=61.3),
           dict(run="B", setting="1024/512/16", pq=(0.433, 0.412, 0.452), vs_a=(-0.013, 0.014), minutes=20),
           dict(run="A+B", setting="map average", pq=(0.439, 0.419, 0.458), vs_a=(-0.007, 0.06), minutes=None)]
    t = rf.tab_pilot(pil)
    assert len(_tabular(t)) == 4 and "$-$0.013 (1\\%)" in t and "\\midrule" in t.split("A+B")[0].split("B &")[-1]


def test_slices_table_and_macros(tmp_path):
    im = lambda s, n, v: dict(slice=s, n_images=n, n_gt=3 * n, pq=v, lo=v - 0.02, hi=v + 0.02)  # noqa: E731
    fl = lambda s, v: dict(slice=s, n_gt=100, tp=50, fp=10, fn=50, pq=v)  # noqa: E731
    S = dict(slices=dict(annotators=[im("1", 4, 0.4), im("2", 3, 0.5), im("3", 2, 0.45)],
                         size_px=[fl("< 831.00", 0.37), fl("831.00 to 1888.00", 0.46), fl(">= 1888.00", 0.5)],
                         r_frac=[fl("< 0.48", 0.46), fl("0.48 to 0.69", 0.45), fl(">= 0.69", 0.42)],
                         site=[im("Bh", 5, 0.44), im("Uh", 5, 0.47), im("Lh", 5, 0.43)],
                         within_24h=[im("with a training image within 24 h", 4, 0.45), im("without a training image within 24 h", 5, 0.44)]))
    t = rf.tab_slices(S)
    assert len(_tabular(t)) == 14 and "site Uh (best)" in t and "site Lh (worst)" in t and "$<$831 px" in t
    p = tmp_path / "numbers.tex"
    p.write_text("\\newcommand{\\a}{TBD}   % note { x }\n\\newcommand{\\b}{\\url{TBD}}  % link\n\\newcommand{\\c}{1{,}5}\n", encoding="utf-8")
    assert rf.set_macros(p, {"a": "0.5", "b": "\\url{x.org}"}) == []
    assert p.read_text(encoding="utf-8").splitlines()[:2] == ["\\newcommand{\\a}{0.5}   % note { x }", "\\newcommand{\\b}{\\url{x.org}}  % link"]
    with pytest.raises(KeyError):
        rf.set_macros(p, {"zzz": "1"})


def test_boot3_point_values_and_degrees():
    s, tp, den = np.tile([0.6, 0.0, 0.8], 20), np.tile([1, 0, 1], 20), np.tile([1.0, 1.0, 1.5], 20)
    b = rf.boot3(s, tp, den, n=200)
    assert b["pq"][0] == pytest.approx(1.4 / 3.5) and b["sq"][0] == pytest.approx(0.7) and b["rq"][0] == pytest.approx(2 / 3.5)
    assert all(lo <= v <= hi for v, lo, hi in b.values())
    res = dict(gt_degree=np.array([0, 1, 2, 3]), pred_degree=np.array([1, 1, 1, 2]))
    assert rf.degree_shares(res) == (0.5, 0.25)
