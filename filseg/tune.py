"""Tuning and evaluation tools: candidate table, fast PQ, keep-rule grid search, nested OOF PQ, bootstrap
intervals, failure buckets. CLI: python -m filseg.tune --run A [B] [--folds 0]."""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from filseg import data as D
from filseg import instances as _instances
from filseg import pq
from filseg.instances import RUNS, Rule, _Slabs, candidates, select, to_rles

T_LO_GRID = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
GRID = dict(t_lo=T_LO_GRID, peak=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0),   # 1.0 = saturated (255/255)
            mean=(0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
            min_area=(100, 200, 300, 400, 600), max_r=(0.88, 0.90, 0.92, 0.95, 1.00))
OFF = dict(t_lo=(0.5,), peak=(0.0,), mean=(0.0,), min_area=(0,), max_r=(1e9,))     # "no filter", t_lo 0.5
RULE_COLS = ("t_lo", "peak", "mean", "min_area", "max_r")
BUCKETS = ("matched", "discarded", "wrong_shape", "never_detected")
FEATS = ("area", "peak", "mean", "p10", "p50", "cy", "cx", "r_frac", "w", "h", "fill", "major", "minor")
DTYPES = {"id": "int32", "area": "int32", "w": "int16", "h": "int16", "n_rec": "int8", "n_tp": "int16", "fold": "int8"}
WORKERS = min(os.cpu_count() or 1, 12)
SOURCES = (Path(__file__), Path(_instances.__file__), Path(D.__file__))    # the code the candidate rows come from


def _runs_root() -> Path:
    if "FILSEG_RUNS" in os.environ:
        return (D.REPO / os.environ["FILSEG_RUNS"]).resolve()       # absolute values win over REPO
    return RUNS


# ---------------------------------------------------------------- ground truth and component matching

def _gt_info(recs: dict[str, list[dict]]) -> dict:
    """One entry per ground-truth filament over a stem's records: record index, area, centre distance
    (r_frac, as for components), and the bounding-box crop of its mask."""
    rec, area, rf, crops = [], [], [], []
    for r, rles in enumerate(recs.values()):
        for m in rles:
            x, y, w, h = (int(v) for v in mask_util.toBbox(m))
            crop = mask_util.decode(m)[y:y + h, x:x + w].astype(bool)
            yy, xx = np.nonzero(crop)
            rec.append(r), area.append(len(yy)), crops.append((y, x, crop))
            rf.append(float(np.hypot(x + xx.mean() - D.CX, y + yy.mean() - D.CY) / D.R) if len(yy) else np.nan)
    return dict(rec=np.array(rec, int), area=np.array(area, float), r_frac=np.array(rf), crops=crops, n_rec=len(recs))


def _overlap(lab: np.ndarray, cand_area: np.ndarray, gt: dict) -> tuple[np.ndarray, np.ndarray]:
    """(intersection pixels, float32 IoU), both (n_gt, n_comp), from the label map: bounding-box crops, no RLEs.
    IoU is float64 inter / union cast to float32, as pq.score does before its `> 0.5` test."""
    n = len(cand_area)
    inter = np.zeros((len(gt["area"]), n))
    if n:
        for g, (y, x, crop) in enumerate(gt["crops"]):
            inter[g] = np.bincount(lab[y:y + crop.shape[0], x:x + crop.shape[1]][crop], minlength=n + 1)[1:]
    return inter, (inter / (cand_area[None, :] + gt["area"][:, None] - inter)).astype(np.float32)


def _outcomes(lab, cands, gt, rule: Rule) -> tuple[dict, dict]:
    """Per ground-truth filament: failure bucket (index into BUCKETS, first that applies) and matched IoU;
    per false positive (a kept component without a hit in a record, one per record): area and r_frac."""
    area = np.array([c["area"] for c in cands], float)
    rf = np.array([c["r_frac"] for c in cands], float)
    kept = np.isin([c["id"] for c in cands], select(cands, rule))
    inter, iou = _overlap(lab, area, gt)
    hit = iou > pq.IOU_HIT
    hk = hit & kept
    bucket = np.select([hk.any(1), (hit & ~kept).any(1), (inter > 0).any(1)], [0, 1, 2], 3)
    fp = np.concatenate([np.flatnonzero(kept & ~hit[gt["rec"] == r].any(0)) for r in range(gt["n_rec"])] or [[]]).astype(int)
    return (dict(area=gt["area"], r_frac=gt["r_frac"], bucket=bucket, tp_iou=np.where(hk, iou, 0).sum(1)),
            dict(area=area[fp], r_frac=rf[fp]))


def failure_buckets_one(prob: np.ndarray, records: dict[str, list[dict]], rule: Rule) -> dict:
    """One count per ground-truth filament of the stem's records, first bucket that applies: matched (a kept
    component with IoU > 0.5), discarded (an unkept component with IoU > 0.5), wrong_shape (any component
    overlaps it), never_detected."""
    lab, cands = candidates(prob, rule.t_lo)
    gt, _ = _outcomes(lab, cands, _gt_info(records), rule)
    c = Counter(gt["bucket"].tolist())
    return {b: c[i] for i, b in enumerate(BUCKETS)}


# ---------------------------------------------------------------- reading maps, running workers

def _plan(runs: list[str], folds: list[int], root: Path) -> dict[str, list]:
    """stem -> [(path, run, index in file)] in run order. A fold file must hold only stems of its own fold:
    that is what keeps the nested tuning honest."""
    fold_of = json.loads(Path(D.FOLDS_JSON).read_text(encoding="utf-8"))
    where: dict[str, list] = {}
    for run in runs:
        for k in folds:
            path = Path(root) / run / f"oof_fold{k}.npz"
            with np.load(path) as z:
                stems = [str(s) for s in z["stems"]]
            for i, s in enumerate(stems):
                if fold_of[s] != k:
                    raise ValueError(f"{path} holds {s}, which is in fold {fold_of[s]}, not fold {k}")
                where.setdefault(s, []).append((str(path), run, i))
    bad = [s for s, h in where.items() if len(h) != len(runs)]
    if bad:
        raise ValueError(f"stems missing from some run: {bad[:3]}")
    return where


def _maps(items):
    """(stem, float32 prob) for a chunk of the plan, averaged over the runs exactly as instances.iter_probs does."""
    readers = {}
    try:
        for stem, hits in items:
            acc = np.zeros((D.H, D.W), np.float32)
            for path, run, i in hits:
                if path not in readers:
                    readers[path] = _Slabs(Path(path), run)
                acc += readers[path].get(i)
            yield stem, acc / (255.0 * len(hits))
    finally:
        for rd in readers.values():
            rd.close()


def _pmap(fn, tasks: list, workers: int) -> list:
    """fn over tasks, in order; workers <= 1 runs in this process (no pool is spawned)."""
    w = min(workers, len(tasks))
    if w <= 1:
        return [fn(t) for t in tasks]
    pool = mp.Pool(w)
    try:
        out = pool.map(fn, tasks, chunksize=1)
        pool.close()
        return out
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()


def _chunks(plan: dict, records: dict, *extra) -> list[tuple]:
    items = list(plan.items())
    size = max(8, math.ceil(len(items) / (4 * WORKERS)))
    return [(items[i:i + size], {s: records[s] for s, _ in items[i:i + size]}, *extra) for i in range(0, len(items), size)]


def _table_chunk(task) -> list[dict]:
    """Rows for a chunk of stems: every component at every t_lo with its features, n_rec, n_tp, sum_tp_iou."""
    items, recs_of = task
    rows = []
    for stem, prob in _maps(items):
        gt, n_rec, mine = _gt_info(recs_of[stem]), len(recs_of[stem]), []
        for t in T_LO_GRID:
            lab, cands = candidates(prob, t)
            _, iou = _overlap(lab, np.array([c["area"] for c in cands], float), gt)
            hit = iou > pq.IOU_HIT
            tp, n_tp = np.where(hit, iou, 0).astype(float).sum(0), hit.sum(0)
            mine += [dict(stem=stem, t_lo=t, **c, n_rec=n_rec, n_tp=int(n_tp[j]), sum_tp_iou=float(tp[j]))
                     for j, c in enumerate(cands)]
        # a stem with no component still carries its filaments and records: one row that no rule keeps (t_lo NaN)
        rows += mine or [dict(stem=stem, t_lo=np.nan, id=0, **dict.fromkeys(FEATS, 0), n_rec=n_rec, n_tp=0, sum_tp_iou=0.0)]
    return rows


def _code_key() -> str:
    """Hash of everything but the maps that the rows depend on: the source of instances.py (features), data.py
    (records), this file (matching, dtypes), folds.json, the annotation file's SHA-256 and the t_lo grid."""
    h = hashlib.sha256()
    for p in (*SOURCES, Path(D.FOLDS_JSON)):
        h.update(Path(p).read_bytes())
    h.update((D.TRAIN_JSON_SHA256 + repr(T_LO_GRID)).encode())
    return h.hexdigest()


def _sig(runs, k, root, code) -> list:
    """What the cached rows of fold k were built from: each run's map file (size, mtime) and the code key."""
    stats = [(Path(root) / r / f"oof_fold{k}.npz").stat() for r in sorted(runs)]
    return [(r, s.st_size, s.st_mtime_ns) for r, s in zip(sorted(runs), stats)] + [code]


def candidate_table(runs: list[str], folds: list[int], records: dict | None = None, root=None,
                    workers: int | None = None) -> pd.DataFrame:
    """One row per component per t_lo: stem, fold, t_lo, id, features, n_rec, n_tp, sum_tp_iou (the sum over
    the stem's records of the matched IoU when > 0.5, else 0). A stem with no component has one row with
    t_lo NaN. Cached in <root>/<runs joined by +>/cands.pkl, one signature per fold (map files and _code_key):
    a change to the feature code, folds.json, the annotation SHA or a map file rebuilds the fold."""
    root = Path(root) if root else _runs_root()
    cache = root / "+".join(sorted(runs)) / "cands.pkl"
    # the cache is a file this function wrote itself under ../runs, never downloaded data, so pickle is safe here
    old = pickle.loads(cache.read_bytes()) if cache.exists() else dict(sig={}, table=None)
    code = _code_key()
    sig = {k: _sig(runs, k, root, code) for k in folds}
    need = [k for k in folds if old["sig"].get(k) != sig[k]]
    if need:
        records = D.load_records() if records is None else records
        fold_of = json.loads(Path(D.FOLDS_JSON).read_text(encoding="utf-8"))
        parts = _pmap(_table_chunk, _chunks(_plan(runs, need, root), records), WORKERS if workers is None else workers)
        rows = [r for part in parts for r in part]
        new = pd.DataFrame(rows).assign(fold=lambda d: d.stem.map(fold_of)).astype(DTYPES)
        keep = old["table"][~old["table"].fold.isin(need)] if old["table"] is not None else None
        old = dict(sig={**{k: v for k, v in old["sig"].items() if k not in need}, **{k: sig[k] for k in need}},
                   table=new if keep is None else pd.concat([keep, new], ignore_index=True))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.with_suffix(".tmp").write_bytes(pickle.dumps(old))
        os.replace(cache.with_suffix(".tmp"), cache)
    t = old["table"]
    return t[t.fold.isin(folds)].reset_index(drop=True)


# ---------------------------------------------------------------- PQ from the table

def fast_pq(kept: pd.DataFrame, n_gt: int) -> float:
    """Pooled PQ of disjoint predictions: TP + 0.5 FP + 0.5 FN = 0.5 * filaments + 0.5 * prediction-record pairs."""
    den = 0.5 * n_gt + 0.5 * kept.n_rec.sum()
    return float(kept.sum_tp_iou.sum() / den) if den > 0 else 0.0


def _keep(table: pd.DataFrame, r: Rule) -> pd.Series:
    """Rows kept by the rule (the comparisons of instances.select); NaN t_lo rows are never kept."""
    return ((table.t_lo - r.t_lo).abs() < 1e-9) & (table.peak >= r.peak) & (table["mean"] >= r.mean) & \
        (table.area >= r.min_area) & (table.r_frac <= r.max_r)


def grid_search(table: pd.DataFrame, n_gt: int, grid: dict = GRID) -> tuple[Rule, float]:
    """Rule with the highest pooled PQ over the grid (first one on ties); vectorised over the table rows."""
    best_pq, best = -1.0, None
    for t in grid["t_lo"]:
        d = table[(table.t_lo - t).abs() < 1e-9]
        d = d[(d.peak >= min(grid["peak"])) & (d["mean"] >= min(grid["mean"])) & (d.area >= min(grid["min_area"]))
              & (d.r_frac <= max(grid["max_r"]))]                  # rows no rule of the grid could keep
        pk, mn, ar, rf = (d[c].to_numpy() for c in ("peak", "mean", "area", "r_frac"))
        num, den = d.sum_tp_iou.to_numpy(), d.n_rec.to_numpy()
        for p in grid["peak"]:
            kp = pk >= p
            for m in grid["mean"]:
                km = kp & (mn >= m)
                for a in grid["min_area"]:
                    ka = km & (ar >= a)
                    for r in grid["max_r"]:
                        keep = ka & (rf <= r)
                        k = 0.5 * n_gt + 0.5 * den[keep].sum()
                        v = num[keep].sum() / k if k > 0 else 0.0
                        if v > best_pq:
                            best_pq, best = v, Rule(t, p, m, a, r)
    return best, float(best_pq)


def _on_edge(values: dict) -> dict[str, str]:
    """The rule parameters that sit on the smallest or largest value of their grid: {name: "min" | "max"}."""
    return {c: "min" if values[c] == min(GRID[c]) else "max" for c in RULE_COLS if values[c] in (min(GRID[c]), max(GRID[c]))}


def _pq(df: pd.DataFrame) -> float:
    return float(df.s.sum() / (0.5 * df.n_gt.sum() + 0.5 * df.k.sum()))


def _n_gt(records: dict, stems) -> pd.Series:
    return pd.Series({s: sum(len(v) for v in records[s].values()) for s in stems}, dtype=int)


def _per_stem(kept: pd.DataFrame, n_gt: pd.Series) -> pd.DataFrame:
    """stem, n_gt, s (sum_tp_iou of kept), k (sum of n_rec of kept), tp (matches), for every stem of n_gt."""
    g = kept.groupby("stem").agg(s=("sum_tp_iou", "sum"), k=("n_rec", "sum"), tp=("n_tp", "sum"))
    out = n_gt.rename("n_gt").to_frame().join(g).fillna(0)
    return out.astype({"k": int, "tp": int}).rename_axis("stem").reset_index()


def nested_pq(table: pd.DataFrame, records: dict, grid: dict = GRID) -> tuple[float, pd.DataFrame]:
    """Tune on the other folds, apply to each fold, pool every kept component into one PQ. With one fold in the
    table (pilot) the groups are a seeded 5-way split of its stems instead. Returns the PQ and the per-stem
    table (stem, n_gt, s, k, tp, cv group, the rule columns used)."""
    n_gt = _n_gt(records, table.stem.unique())
    cv = table.groupby("stem").fold.first()
    if cv.nunique() < 2:
        order = np.random.default_rng(0).permutation(sorted(cv.index))
        cv = pd.Series(np.arange(len(order)) % 5, index=order)
    parts = []
    for f in sorted(cv.unique()):
        te, tr = cv.index[cv == f], cv.index[cv != f]
        rule, _ = grid_search(table[table.stem.isin(tr)], int(n_gt[tr].sum()), grid)
        rows = table[table.stem.isin(te) & _keep(table, rule)]
        parts.append(_per_stem(rows, n_gt[te]).assign(cv=f, **{c: getattr(rule, c) for c in RULE_COLS}))
    per = pd.concat(parts, ignore_index=True)
    return _pq(per), per


# ---------------------------------------------------------------- bootstrap over images

def _boot(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    s, g, k = (df[c].to_numpy(float) for c in ("s", "n_gt", "k"))
    return s[idx].sum(1) / (0.5 * g[idx].sum(1) + 0.5 * k[idx].sum(1))


def bootstrap_ci(per_stem: pd.DataFrame, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of the pooled PQ over resamples of the images (stems)."""
    idx = np.random.default_rng(seed).integers(0, len(per_stem), (n, len(per_stem)))
    lo, hi = np.percentile(_boot(per_stem, idx), [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """(PQ of b minus PQ of a, share of resamples of the images in which b beats a); same stems in both."""
    a, b = a.sort_values("stem").reset_index(drop=True), b.sort_values("stem").reset_index(drop=True)
    if list(a.stem) != list(b.stem):
        raise ValueError("a and b must cover the same stems")
    idx = np.random.default_rng(seed).integers(0, len(a), (n, len(a)))
    return float(_pq(b) - _pq(a)), float(((_boot(b, idx) - _boot(a, idx)) > 0).mean())


# ---------------------------------------------------------------- second pass: buckets, slices, exact check

def _rule(d: dict) -> Rule:
    return Rule(t_lo=float(d["t_lo"]), peak=float(d["peak"]), mean=float(d["mean"]), min_area=int(d["min_area"]),
                max_r=float(d["max_r"]))


def _eval_chunk(task):
    """Per stem: outcomes under its nested rule, and the RLEs kept by the final rule (for pq.score)."""
    items, recs_of, rules, final = task
    gts, fps, preds = [], [], {}
    for stem, prob in _maps(items):
        mine = _rule(rules[stem])
        lab, cands = candidates(prob, mine.t_lo)
        g, f = _outcomes(lab, cands, _gt_info(recs_of[stem]), mine)
        gts += [(stem, *r) for r in zip(g["area"], g["r_frac"], g["bucket"], g["tp_iou"])]
        fps += [(stem, *r) for r in zip(f["area"], f["r_frac"])]
        if final.t_lo != mine.t_lo:
            lab, cands = candidates(prob, final.t_lo)
        preds[stem] = to_rles(lab, select(cands, final))
    return gts, fps, preds


def _slices_filament(gt: pd.DataFrame, fp: pd.DataFrame) -> dict:
    """PQ by filament size and by distance from the disk centre, thirds of the ground-truth values. A filament
    (TP or FN) is in the slice of its own value, a false positive in the slice of the component's own value."""
    out = {}
    for name, col in (("size_px", "area"), ("r_frac", "r_frac")):
        cuts = np.quantile(gt[col].dropna(), [1 / 3, 2 / 3])
        gb, fb = np.digitize(gt[col].fillna(0), cuts), np.digitize(fp[col].fillna(0), cuts)
        rows = []
        for b, label in enumerate((f"< {cuts[0]:.2f}", f"{cuts[0]:.2f} to {cuts[1]:.2f}", f">= {cuts[1]:.2f}")):
            g, nfp = gt[gb == b], int((fb == b).sum())
            tp = int((g.bucket == 0).sum())
            den = tp + 0.5 * nfp + 0.5 * (len(g) - tp)
            rows.append(dict(slice=label, n_gt=len(g), tp=tp, fp=nfp, fn=len(g) - tp, pq=float(g.tp_iou.sum() / den) if den else 0.0))
        out[name] = rows
    return out


def _slices_image(per: pd.DataFrame, labels: dict[str, str]) -> list[dict]:
    d = per.assign(g=per.stem.map(labels))
    rows = []
    for g, x in sorted(d.groupby("g"), key=lambda kv: str(kv[0])):
        lo, hi = bootstrap_ci(x)
        rows.append(dict(slice=str(g), n_images=len(x), n_gt=int(x.n_gt.sum()), pq=_pq(x), lo=lo, hi=hi))
    return rows


def _near_train(stems, fold_of: dict, hours: float = 24) -> dict[str, bool]:
    """Whether a stem has a stem of another fold within `hours` (times parsed from YYYYMMDDHHMMSS); analysis only."""
    names = sorted(fold_of)
    t = pd.to_datetime([s[:14] for s in names], format="%Y%m%d%H%M%S").to_numpy()
    f = np.array([fold_of[s] for s in names])
    gap = np.abs(t[:, None] - t[None, :]) / np.timedelta64(1, "h")
    near = ((gap <= hours) & (f[:, None] != f[None, :])).any(1)
    return {s: bool(near[names.index(s)]) for s in stems}


# ---------------------------------------------------------------- command line

def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--rule-out", default=None, help="also write the tuned rule here (e.g. filseg/rule.json); "
                    "nothing is written into the repo otherwise")
    ap.add_argument("--workers", type=int, default=WORKERS, help="processes for the passes over the maps; 1 = no pool")
    a = ap.parse_args(argv)
    root = Path(a.runs_root) if a.runs_root else _runs_root()
    runs, S = a.run, {"runs": a.run, "folds": a.folds}
    full = sorted(a.folds) == [0, 1, 2, 3, 4]
    suffix = "" if full else "_folds" + "_".join(str(k) for k in sorted(a.folds))     # a partial run never overwrites a full one
    out_dir = root / "+".join(sorted(runs))

    t0 = time.perf_counter()
    records = D.load_records()
    table = candidate_table(runs, a.folds, records=records, root=root, workers=a.workers)
    S["seconds_table"] = round(time.perf_counter() - t0, 1)
    stems = sorted(table.stem.unique())
    n_gt = _n_gt(records, stems)
    S["n_images"], S["n_filaments"], S["n_rows"] = len(stems), int(n_gt.sum()), len(table)
    pilot = table.fold.nunique() < 2
    if full:
        label, head = "nested OOF", "nested PQ"
    else:
        label = (f"inner-CV (fold {sorted(a.folds)[0]} only, not nested OOF)" if pilot
                 else f"nested over folds {sorted(a.folds)} only, not the 5-fold nested OOF")
        head = f"PQ [{label}]"
    S["label"] = label
    print(f"runs {'+'.join(runs)}, folds {a.folds}: {len(stems)} images, {int(n_gt.sum())} filaments, {len(table)} candidate rows "
          f"({S['seconds_table']} s for the table)")
    if pilot:
        print("NOTE: one fold only (pilot): nested tuning is impossible, so the rule is tuned with an inner 5-fold split of "
              "this fold's stems (grouped by stem, seed 0)")

    t0 = time.perf_counter()
    best, in_sample = grid_search(table, int(n_gt.sum()))
    S["seconds_grid_search"] = round(time.perf_counter() - t0, 1)
    S["best_rule"], S["best_rule_in_sample_pq"] = asdict(best), in_sample

    # one row per filter added, each nested; the first row is the baseline, the last the full grid
    order = ("peak", "mean", "min_area", "max_r", "t_lo")
    names = ("baseline (t_lo 0.5, no filter)", "+ peak", "+ mean", "+ min_area", "+ max_r", "+ t_lo (full grid)")
    S["steps"] = []
    print(f"\n{head}, one row per filter added (rule shown is tuned on all rows)")
    for i, name in enumerate(names):
        g = {**OFF, **{c: GRID[c] for c in order[:i]}}
        v, per = nested_pq(table, records, g)
        r, ins = grid_search(table, int(n_gt.sum()), g) if i < len(order) else (best, in_sample)
        S["steps"].append(dict(step=name, nested_pq=v, in_sample_pq=ins, rule=asdict(r)))
        print(f"  {name:32s} held-out {v:.4f}  in-sample {ins:.4f}  rule t_lo {r.t_lo} peak {r.peak} mean {r.mean} "
              f"min_area {r.min_area} max_r {r.max_r if r.max_r < 1e9 else 'none'}")
        if i == 0:
            base_per, base_pq = per, v
    nested, per_nested = v, per
    bci = bootstrap_ci(base_per)
    print(f"\nbaseline (t_lo 0.5, no filter): PQ {base_pq:.4f}  95% interval {bci[0]:.4f} to {bci[1]:.4f}")
    S["baseline"] = dict(pq=base_pq, ci=bci)
    print(f"best rule (all rows): {asdict(best)}  in-sample PQ {in_sample:.4f}  ({S['seconds_grid_search']} s for the grid search)")
    edge = _on_edge(asdict(best))
    groups = per_nested.drop_duplicates("cv")
    edge_n = {c: int(groups[c].isin([min(GRID[c]), max(GRID[c])]).sum()) for c in RULE_COLS}
    S["grid_edges"] = dict(best=edge, groups=len(groups), per_group_count={c: n for c, n in edge_n.items() if n})
    if edge:
        print("WARN: the best rule sits on a grid edge, the grid may be binding: " +
              ", ".join(f"{c} {asdict(best)[c]} (grid {e})" for c, e in edge.items()))
    if S["grid_edges"]["per_group_count"]:
        print(f"WARN: groups whose tuned rule sits on a grid edge (of {len(groups)}): " +
              ", ".join(f"{c} {n}" for c, n in S["grid_edges"]["per_group_count"].items()))

    lo, hi = bootstrap_ci(per_nested)
    tp, k, ng = int(per_nested.tp.sum()), int(per_nested.k.sum()), int(per_nested.n_gt.sum())
    sq, rq = float(per_nested.s.sum() / tp) if tp else 0.0, tp / (0.5 * k + 0.5 * ng)
    gain, share = paired_bootstrap(base_per, per_nested)
    S["nested"] = dict(pq=nested, ci=(lo, hi), sq=sq, rq=rq, tp=tp, fp=k - tp, fn=ng - tp, pilot_inner_cv=pilot)
    S["gain_over_baseline"] = dict(gain=gain, share_above_zero=share)
    print(f"{head} {nested:.4f}  95% interval {lo:.4f} to {hi:.4f}  SQ {sq:.4f}  RQ {rq:.4f}  TP {tp}  FP {k - tp}  FN {ng - tp}")
    print(f"gain over baseline {gain:+.4f}, paired bootstrap share of resamples above zero {share:.3f} "
          f"(adoption rule: gain >= 0.005 and share >= 0.90)")

    # second pass over the maps: the nested rule of each stem, and the final rule scored exactly
    t0 = time.perf_counter()
    rules = per_nested.set_index("stem")[list(RULE_COLS)].to_dict("index")
    parts = _pmap(_eval_chunk, _chunks(_plan(runs, a.folds, root), records, rules, best), a.workers)
    gt = pd.DataFrame([r for p in parts for r in p[0]], columns=["stem", "area", "r_frac", "bucket", "tp_iou"])
    fp = pd.DataFrame([r for p in parts for r in p[1]], columns=["stem", "area", "r_frac"])
    preds = {s: v for p in parts for s, v in p[2].items()}
    if len(gt) != ng or int((gt.bucket == 0).sum()) != tp or abs(gt.tp_iou.sum() - per_nested.s.sum()) > 1e-6 * ng or len(fp) != k - tp:
        raise AssertionError("the second pass over the maps disagrees with the candidate table: "
                             f"{len(gt)} vs {ng} filaments, {(gt.bucket == 0).sum()} vs {tp} matches, {len(fp)} vs {k - tp} false positives")
    exact = pq.score({rid: r for s in stems for rid, r in records[s].items()}, preds)["pq"]
    fast = fast_pq(table[_keep(table, best)], int(n_gt.sum()))
    S["exact_check"] = dict(fast_pq=fast, pq_score=exact)
    if abs(fast - exact) > 0.002:
        raise AssertionError(f"fast_pq {fast:.5f} and pq.score {exact:.5f} disagree for {asdict(best)}")
    S["seconds_second_pass"] = round(time.perf_counter() - t0, 1)
    print(f"winning rule re-scored with pq.score: {exact:.4f} (fast_pq {fast:.4f}); second pass over the maps "
          f"{S['seconds_second_pass']} s")

    S["failure_buckets"] = {b: int((gt.bucket == i).sum()) for i, b in enumerate(BUCKETS)}
    print("\nfailure buckets (one per ground-truth filament, nested rules): " +
          ", ".join(f"{b} {n} ({n / ng:.1%})" for b, n in S["failure_buckets"].items()))

    fold_of = json.loads(Path(D.FOLDS_JSON).read_text(encoding="utf-8"))
    near = _near_train(stems, fold_of)
    S["slices"] = dict(_slices_filament(gt, fp))
    S["slices"]["annotators"] = _slices_image(per_nested, {s: str(len(records[s])) for s in stems})
    S["slices"]["site"] = _slices_image(per_nested, {s: s[-2:] for s in stems})
    S["slices"]["within_24h"] = _slices_image(per_nested, {s: "with a training image within 24 h" if near[s]
                                                           else "without a training image within 24 h" for s in stems})
    print(f"\nslices ({head})")
    titles = dict(size_px="filament size, px (thirds of the ground truth)", r_frac="distance from disk centre, r / R (thirds)",
                  annotators="annotators", site="site", within_24h="training image within 24 h")
    for key, rows in S["slices"].items():
        print(f"  {titles[key]}")
        for r in rows:
            extra = f"  95% {r['lo']:.4f} to {r['hi']:.4f}  images {r['n_images']}" if "lo" in r else f"  TP {r['tp']} FP {r['fp']} FN {r['fn']}"
            print(f"    {r['slice']:40s} PQ {r['pq']:.4f}  filaments {r['n_gt']}{extra}")

    rule_text = json.dumps(asdict(best), indent=1) + "\n"
    paths = [out_dir / f"rule{suffix}.json"] + ([Path(a.rule_out)] if a.rule_out else [])
    for path in paths:
        path.write_text(rule_text, encoding="utf-8", newline="\n")
    out = out_dir / f"tune_summary{suffix}.json"
    out.write_text(json.dumps(S, indent=1, default=_jsonable) + "\n", encoding="utf-8", newline="\n")
    print("\nwrote " + ", ".join(str(p) for p in [*paths, out]))
    return S


if __name__ == "__main__":
    main()
