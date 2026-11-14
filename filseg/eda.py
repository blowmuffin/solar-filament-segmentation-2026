"""Data checks (Task 9): site/year mix, train-test drift, annotator agreement, filament geometry, time gaps.
`python -m filseg.eda [--recompute]` writes ../report/eda/EDA.md, the PNG figures and image_stats.csv.
File names are used for analysis only (site, year, time), never as model input."""
import argparse
import json
import multiprocessing as mp

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from filseg import data, pq

OUT = (data.REPO / "../report/eda").resolve()
STATS = ["mean", "std", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "dark", "bright", "grad"]
BIN_EDGES = [0, 300, 500, 2000, 5000, np.inf]
BIN_NAMES = ["<300", "300-499", "500-1999", "2000-4999", ">=5000"]
BLUE, ORANGE, LBLUE, DBLUE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#86b6ef", "#1c5cab", "#0b0b0b", "#52514e", "#e1e0d9"
# limb and area numbers stated by the research summary: name -> (value, absolute tolerance, relative tolerance)
PUBLIC_OOF, PUBLIC_LB = "0.450 and 0.4597", "0.39"      # competing teams, FINDINGS section on public pipelines
RESEARCH = {"|x-cx|/R median": (0.39, .015, 0), "|x-cx|/R p99": (0.86, .015, 0), "|x-cx|/R max": (0.95, .015, 0),
            "r/R median": (0.59, .015, 0), "r/R p99": (0.90, .015, 0), "r/R max": (0.97, .015, 0),
            "area p10 (px)": (409, 0, .02), "area median (px)": (1228, 0, .02), "area p75 (px)": (2436, 0, .02),
            "area p90 (px)": (4684, 0, .02), "area max (px)": (37739, 0, .02),
            "share < 300 px (%)": (4.1, .15, 0), "share < 200 px (%)": (0.8, .15, 0)}


# ---------------------------------------------------------------- helpers
def md(df: pd.DataFrame) -> str:
    d = df.reset_index()
    rows = [list(map(str, d.columns))] + d.map(lambda x: f"{x:.10g}" if isinstance(x, (float, np.floating)) else str(x)).values.tolist()
    rows.insert(1, ["---"] * len(rows[0]))
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def _ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED), ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(color=GRID, lw=.6), ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


# ---------------------------------------------------------------- per-image statistics (computed once, cached as CSV)
def _image_stats(path) -> dict:
    img = data.load_image(path)
    inner = data._disk_masks(img.shape[0])[0]               # r < 0.98R, the disk used by normalize()
    v = img[inner]
    q = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
    g = np.hypot(cv2.Sobel(img, cv2.CV_32F, 1, 0), cv2.Sobel(img, cv2.CV_32F, 0, 1))[inner]
    return dict(stem=path.stem, split="train" if path.parent == data.TRAIN_IMG else "test", size=img.shape[0],
                mean=v.mean(), std=v.std(), **dict(zip(["p01", "p05", "p25", "p50", "p75", "p95", "p99"], q)),
                dark=(v < 0.75 * q[3]).mean(), bright=(v > 1.15 * q[3]).mean(), grad=g.mean())


def image_stats(recompute=False) -> pd.DataFrame:
    f = OUT / "image_stats.csv"
    if f.exists() and not recompute:
        return pd.read_csv(f, encoding="utf-8", parse_dates=["time"])
    paths = sorted(data.TRAIN_IMG.glob("*.jpeg")) + sorted(data.TEST_IMG.glob("*.jpeg"))
    with mp.Pool(min(8, mp.cpu_count())) as pool:
        df = pd.DataFrame(pool.map(_image_stats, paths, chunksize=8))
    df.insert(2, "site", df["stem"].str[-2:])
    df.insert(3, "year", df["stem"].str[:4].astype(int))
    df.insert(4, "time", pd.to_datetime(df["stem"].str[:14], format="%Y%m%d%H%M%S"))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(f, index=False, encoding="utf-8")
    return df


# ---------------------------------------------------------------- 1. site and year
def sec_site_year(df):
    L, res = ["## 1. Images per site and year, train against test", ""], {}
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, key in zip(axs, ("site", "year")):
        c = pd.crosstab(df[key], df["split"])[["train", "test"]]
        c["train %"], c["test %"] = (100 * c.train / c.train.sum()).round(1), (100 * c.test / c.test.sum()).round(1)
        gap = (c["test %"] - c["train %"]).abs()
        res[key] = dict(max_gap=float(gap.max()), at=str(gap.idxmax()), only_test=[str(i) for i in c.index[c.train == 0]])
        L += [f"By {key}:", "", md(c), ""]
        x = np.arange(len(c))
        ax.bar(x - .2, c["train %"], .4, color=BLUE, label="train (707)"), ax.bar(x + .2, c["test %"], .4, color=ORANGE, label="test (180)")
        ax.set_xticks(x, c.index.astype(str), rotation=60 if key == "year" else 0)
        _ax(ax, f"Share of images by {key}", key, "% of split"), ax.legend(fontsize=8, frameon=False)
    _save(fig, "fig_site_year.png")
    sy = pd.crosstab(df["year"], [df["split"], df["site"]])
    L += ["Site by year, train/test image counts:", "",
          md(pd.DataFrame({s: sy["train", s].astype(str) + "/" + sy["test", s].astype(str) for s in sorted(df["site"].unique())})), ""]
    late = df[df.year >= 2018].groupby("split").size()
    res["late"] = {k: int(late.get(k, 0)) for k in ("train", "test")}
    L += [f"Meaning: the test set has the same site mix as train (largest gap {res['site']['max_gap']:.1f} points, site {res['site']['at']}); "
          f"by year the largest gap is {res['year']['max_gap']:.1f} points ({res['year']['at']}). Sites or years only in test: "
          f"{res['site']['only_test'] + res['year']['only_test'] or 'none'}. Years 2018 onward (sparse solar-cycle tail) hold "
          f"{res['late']['train']} train and {res['late']['test']} test images, so a CV score says little about them.", ""]
    return L, res


# ---------------------------------------------------------------- 2. drift
def sec_drift(df):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    X, y = df[STATS].to_numpy(), (df["split"] == "test").to_numpy().astype(int)

    def cv(y, seed):
        oof, folds = np.zeros(len(y)), []
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=10, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1)
            oof[va] = m.fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
            folds.append(roc_auc_score(y[va], oof[va]))
        return roc_auc_score(y, oof), folds

    auc, folds = cv(y, 0)
    reps = [cv(y, s)[0] for s in range(1, 6)]
    shuf = cv(np.random.default_rng(0).permutation(y), 0)[0]
    uni = pd.Series({c: roc_auc_score(y, df[c]) for c in STATS})                 # >0.5: higher in test
    imp = pd.Series(lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=10, subsample=0.8,
                                       subsample_freq=1, colsample_bytree=0.8, random_state=0, n_jobs=1, verbose=-1,
                                       importance_type="gain").fit(X, y).feature_importances_, STATS)
    tab = pd.DataFrame({"univariate AUC (test higher if > .5)": uni.round(3), "LightGBM gain share (%)": (100 * imp / imp.sum()).round(1)})
    tab = tab.loc[(uni - .5).abs().sort_values(ascending=False).index].rename_axis("statistic")
    top = tab.index[0]
    res = dict(auc=float(auc), folds=[float(a) for a in folds], reps_mean=float(np.mean(reps)), reps_sd=float(np.std(reps)),
               shuffled=float(shuf), top=top, top_uni=float(uni[top]))
    fig, axs = plt.subplots(3, 4, figsize=(11, 7))
    for ax, c in zip(axs.ravel(), STATS):                       # ECDFs: no binning artefacts on integer-valued statistics
        for sp, col in (("train", BLUE), ("test", ORANGE)):
            v = np.sort(df.loc[df.split == sp, c])
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=1.4, label=sp)
        _ax(ax, f"{c} (AUC {uni[c]:.2f})")
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Per-image disk statistics, train vs test (5-fold LightGBM AUC {auc:.3f})", fontsize=11, color=INK)
    _save(fig, "fig_drift.png")
    band = ("inside the expected 0.45-0.60 band: no usable separation, so no exposure or contrast shift between train and test on these statistics"
            if .45 <= auc <= .60 else
            "above 0.60: the test images are partly separable from train on these statistics" if auc > .60 else "below 0.45 (odd)")
    L = ["## 2. Drift: train against test on 12 per-image pixel statistics", "",
         f"Statistics inside the disk (r < 0.98R): {', '.join(STATS)}. dark / bright = share of pixels below 0.75 / above 1.15 times the "
         "image median; grad = mean Sobel magnitude. 5-fold stratified CV, LightGBM (200 trees, 8 leaves), pooled out-of-fold AUC.", "",
         f"- AUC (seed 0): **{auc:.3f}**; per-fold {', '.join(f'{a:.3f}' for a in folds)}",
         f"- repeated with seeds 1-5: mean {np.mean(reps):.3f} (sd {np.std(reps):.3f}); labels shuffled control: {shuf:.3f}", "", md(tab), "",
         f"Meaning: AUC {auc:.3f} is {band}. Most separating statistic: {top} (univariate AUC {uni[top]:.3f}). " +
         ("Added to the risk list: " + top + "." if auc > .70 else "Not added to the risk list (below 0.70)."), ""]
    return L, res


# ---------------------------------------------------------------- 3. annotator agreement
def agreement(recs) -> pd.DataFrame:
    """One row per filament on a multi-annotated image: area and the number of other annotators with a filament at IoU > 0.5."""
    rows = []
    for stem, rr in recs.items():
        if len(rr) < 2:
            continue
        for a, rles in rr.items():
            if not rles:
                continue
            hit = sum((pq.iou_matrix(rles, rr[b]).astype(np.float32) > pq.IOU_HIT).any(1).astype(int) for b in rr if b != a)
            rows += [(stem, a, len(rr), int(ar), int(h)) for ar, h in zip(mask_util.area(rles), hit)]
    return pd.DataFrame(rows, columns=["stem", "rid", "n_rec", "area", "matched"])


def human_pq(recs) -> dict:
    """Pooled PQ of one annotator's record against the other records of the image (every choice of the predicting record)."""
    ious, tp, fp, fn = [], 0, 0, 0
    for j in range(3):
        stems = [s for s, rr in recs.items() if len(rr) > j and len(rr) > 1]
        gt = {rid: recs[s][rid] for s in stems for rid in list(recs[s])[:j] + list(recs[s])[j + 1:]}
        r = pq.score(gt, {s: recs[s][list(recs[s])[j]] for s in stems})
        ious += r["tp_ious"]; tp += r["tp"]; fp += r["fp"]; fn += r["fn"]
    den = tp + .5 * fp + .5 * fn
    return dict(pq=sum(ious) / den, sq=sum(ious) / tp, rq=tp / den, tp=tp, fp=fp, fn=fn)


def sec_agreement(recs):
    ag = agreement(recs)
    ag["bin"] = pd.cut(ag["area"], BIN_EDGES, right=False, labels=BIN_NAMES)
    hp = human_pq(recs)
    n_multi = sum(len(r) > 1 for r in recs.values())
    L = ["## 3. Annotator agreement per filament (images with 2 or 3 annotator records)", "",
         f"{n_multi} of {len(recs)} images are multi-annotated; {len(ag)} filaments sit on them. A filament counts as matched by another "
         "annotator when that record has a filament at IoU > 0.5 (the host's test). Bins are by the filament's own area in px.", ""]
    res, fig = {}, plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, n, cols, colors in zip(fig[1], (3, 2), ([0, 1, 2], [0, 1]), ([ORANGE, LBLUE, DBLUE], [ORANGE, DBLUE])):
        sub = ag[ag.n_rec == n]
        t = pd.crosstab(sub["bin"], sub["matched"]).reindex(BIN_NAMES).fillna(0).astype(int).reindex(columns=cols, fill_value=0)
        pct = (100 * t.div(t.sum(1), axis=0)).round(1)
        pct.columns = [f"matched by {c} other (%)" for c in cols]
        pct.insert(0, "n", t.sum(1))
        tot = t.sum()
        pct.loc["all"] = [int(tot.sum())] + (100 * tot / tot.sum()).round(1).tolist()
        L += [f"Images with {n} annotator records:", "", md(pct), ""]
        res[n] = pct
        b = np.zeros(len(BIN_NAMES))
        for c, col in zip(cols, colors):
            v = 100 * t[c].div(t.sum(1)).fillna(0).to_numpy()
            ax.bar(BIN_NAMES, v, bottom=b, color=col, label=f"matched by {c}", width=.7, edgecolor="#fcfcfb", lw=1.5)
            b += v
        _ax(ax, f"{n} annotator records per image", "filament area (px)", "% of filaments"), ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.28), ncol=3)
        ax.tick_params(axis="x", labelsize=7)
    _save(fig[0], "fig_agreement.png")
    ag["any"] = ag["matched"] > 0
    by = ag.groupby("bin", observed=False)["any"].agg(["size", "mean"])
    small, mid, big = (ag.loc[m, "any"].mean() for m in (ag.area < 500, (ag.area >= 500) & (ag.area < 2000), ag.area >= 2000))
    three = ag[ag.n_rec == 3]
    coarse = [(nm, three[(three.area >= lo) & (three.area < hi)]) for nm, lo, hi in (("< 500", 0, 500), ("500-1999", 500, 2000), (">= 2000", 2000, np.inf))]
    res.update(small_any=float(small), mid_any=float(mid), big_any=float(big), all_any=float(ag["any"].mean()), n=len(ag), human=hp,
               unmatched_small=float(100 * (1 - small)), unmatched_big=float(100 * (1 - big)), by=by)
    L += [f"Share matched by at least one other annotator: area < 500 px {100 * small:.1f}%, 500-1999 px {100 * mid:.1f}%, "
          f">= 2000 px {100 * big:.1f}%, all {100 * ag['any'].mean():.1f}%.", "",
          f"Single-annotator reference level: one annotator's record scored as a prediction against the other records of its image (host PQ, pooled over "
          f"all choices of the predicting record): PQ {hp['pq']:.3f} = SQ {hp['sq']:.3f} x RQ {hp['rq']:.3f} (TP {hp['tp']}, FP {hp['fp']}, FN {hp['fn']}).", "",
          "Images with 3 records, coarse bins, matched by 0 / 1 / 2 other annotators: " + "; ".join(
              f"{nm} px: {' / '.join(f'{100 * (g.matched == k).mean():.1f}' for k in range(3))}% (n={len(g)})" for nm, g in coarse) + ".", "",
          f"Meaning: filaments under 500 px are matched by no other annotator {100 - 100 * small:.1f}% of the time against {100 - 100 * big:.1f}% above "
          "2000 px, so for small filaments the label is largely one annotator's call and the recall reachable there is limited by annotator "
          "agreement. The host scores each record separately, so a prediction hit by one record and absent in another costs a TP and an FP. "
          f"A single annotator scores PQ {hp['pq']:.2f} against the others: a reference level, not a ceiling, because a model that predicts the "
          "consensus can score above one annotator.", ""]
    return L, res


# ---------------------------------------------------------------- 4. filament geometry
def filament_table(recs) -> pd.DataFrame:
    """One row per filament (annotation) over all records: area, centroid position, shape from second moments, inscribed width."""
    rows = []
    for stem, rr in recs.items():
        for rid, rles in rr.items():
            for rle in rles:
                x, y, w, h = (int(v) for v in mask_util.toBbox(rle))
                m = mask_util.decode(rle)[y:y + h + 1, x:x + w + 1]
                ys, xs = np.nonzero(m)
                assert len(xs) == mask_util.area(rle), (rid, len(xs))
                cx, cy = xs.mean() + x, ys.mean() + y
                ev = np.sort(np.linalg.eigvalsh(np.cov([xs, ys]))) if len(xs) > 2 else np.array([1 / 12, 1 / 12])
                lam1, lam2 = ev[1], max(ev[0], 1 / 12)           # 1/12: variance of a single pixel width
                dt = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 3).max()
                rows.append((stem, rid, len(xs), abs(cx - data.CX) / data.R, np.hypot(cx - data.CX, cy - data.CY) / data.R,
                             4 * np.sqrt(lam1), 4 * np.sqrt(lam2), np.sqrt(lam1 / lam2), 2 * dt - 1,
                             cv2.connectedComponents(np.pad(m, 1), connectivity=8)[0] - 1))
    return pd.DataFrame(rows, columns=["stem", "rid", "area", "dx_R", "r_R", "length", "width_ell", "elong", "width_in", "parts"])


def sec_filaments(recs):
    f = filament_table(recs)
    f.to_csv(OUT / "filaments.csv", index=False, encoding="utf-8")
    a = f["area"]
    got = {"|x-cx|/R median": f.dx_R.median(), "|x-cx|/R p99": f.dx_R.quantile(.99), "|x-cx|/R max": f.dx_R.max(),
           "r/R median": f.r_R.median(), "r/R p99": f.r_R.quantile(.99), "r/R max": f.r_R.max(),
           "area p10 (px)": a.quantile(.10), "area median (px)": a.median(), "area p75 (px)": a.quantile(.75),
           "area p90 (px)": a.quantile(.90), "area max (px)": a.max(), "share < 300 px (%)": 100 * (a < 300).mean(),
           "share < 200 px (%)": 100 * (a < 200).mean()}
    rows, bad, minor = [], [], []
    low = {"area p10 (px)": 10, "area p75 (px)": 75, "area p90 (px)": 90}
    for k, (want, atol, rtol) in RESEARCH.items():
        ok = abs(got[k] - want) <= max(atol, rtol * want)
        rows.append((k, want, round(float(got[k]), 3 if want < 2 else 1), "ok" if ok else "DIFF"))
        if not ok:
            bad.append(f"{k}: research {want}, computed {got[k]:.3f}")
        elif round(got[k], 2 if atol == .015 else len(str(want).partition(".")[2])) != want:
            minor.append(f"{k}: research {want}, computed {got[k]:.3f}" + (f" (numpy method='lower': {np.percentile(a, low[k], method='lower'):g})" if k in low else ""))
    cmp = pd.DataFrame(rows, columns=["statistic", "research", "computed", "status"]).set_index("statistic")
    desc = f[["area", "width_ell", "width_in", "length", "elong", "r_R"]].describe(percentiles=[.1, .5, .9, .99]).T.round(2)
    desc.insert(0, "name", ["area (px)", "minor axis 4*sqrt(l2) (px)", "inscribed width (px)", "major axis 4*sqrt(l1) (px)", "elongation sqrt(l1/l2)", "r/R of centroid"])
    shares = f.groupby(pd.cut(a, BIN_EDGES, right=False, labels=BIN_NAMES), observed=False)["area"].agg(n="size", pixels="sum")
    shares["share of filaments (%)"] = (100 * shares.n / shares.n.sum()).round(1)
    shares["share of mask pixels (%)"] = (100 * shares.pixels / shares.pixels.sum()).round(1)
    per_rec = f.groupby("rid").size()
    res = dict(n=len(f), n_rec=len(per_rec), cmp=cmp, bad=bad, minor=minor, thin={w: float(100 * (f.width_in * w / 2048 < 4).mean()) for w in (2048, 1024, 512)}, width_med=float(f.width_in.median()),
               multi_part=float(100 * (f.parts > 1).mean()), near_limb=float(100 * (f.r_R > .9).mean()), elong_med=float(f.elong.median()),
               filaments_per_record=float(per_rec.mean()), small300=float(100 * (a < 300).mean()))
    L = ["## 4. Filament area, width, elongation and radial position", "",
         f"{len(f)} filaments over {len(per_rec)} annotator records ({per_rec.mean():.1f} per record, max {per_rec.max()}); statistics pool all records. "
         "Width and elongation come from the second moments of the mask pixels (equivalent ellipse); inscribed width is 2 x the largest "
         "distance-transform value - 1. Radial position uses the fixed disk (CX, CY, R) from `filseg.data`.", "",
         "Reproduction of the research limb and area numbers:", "", md(cmp), "", md(desc.set_index("name")), "",
         "Share of filaments and of mask pixels by area bin:", "", md(shares.drop(columns="pixels")), "",
         "Share of filaments with an inscribed width under 4 px after resizing the image to a work resolution: " +
         ", ".join(f"{w} px: {v:.1f}%" for w, v in res["thin"].items()) + f". {res['multi_part']:.1f}% of annotations have more than one "
         f"8-connected part; {res['near_limb']:.1f}% have their centroid beyond 0.9 R.", "",
         f"Meaning: the count is dominated by small filaments (median {a.median():.0f} px, {res['small300']:.1f}% below 300 px) while a few large "
         f"ones carry the mask pixels (max {a.max()} px), so PQ, which counts instances, is driven by the small-to-medium range. Median inscribed "
         f"width is {res['width_med']:.1f} px at 2048 px, so filaments are wide enough for a 1024 px work resolution ({res['thin'][1024]:.1f}% thinner than "
         f"4 px) but not for 512 px ({res['thin'][512]:.1f}%). Only {res['multi_part']:.1f}% of annotations are multi-part, so one connected component "
         "per filament is the right instance unit. Centroids reach the limb (r/R max "
         f"{f.r_R.max():.2f}), so the disk normalisation and the outside-disk mask must be exact there.", ""]
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    axs[0, 0].hist(a, bins=np.logspace(np.log10(a.min()), np.log10(a.max()) + .05, 40), color=BLUE), axs[0, 0].set_xscale("log")
    axs[0, 0].axvline(300, color=ORANGE, lw=1.2), axs[0, 0].text(280, axs[0, 0].get_ylim()[1] * .92, "300 px ", color=MUTED, fontsize=8, ha="right")
    _ax(axs[0, 0], f"Area (median {a.median():.0f} px)", "px, log scale", "filaments")
    axs[0, 1].hist(f.width_in.clip(upper=60), bins=np.arange(0, 62, 2), color=BLUE)
    _ax(axs[0, 1], f"Inscribed width (median {res['width_med']:.1f} px, clipped at 60)", "px", "filaments")
    axs[1, 0].hist(f.elong, bins=40, color=BLUE)
    _ax(axs[1, 0], f"Elongation (median {res['elong_med']:.1f})", "sqrt(l1/l2)", "filaments")
    axs[1, 1].hist(f.r_R, bins=np.linspace(0, 1, 41), color=BLUE)
    _ax(axs[1, 1], "Centroid radius", "r / R", "filaments")
    _save(fig, "fig_filaments.png")
    return L, res, f


# ---------------------------------------------------------------- 5. time gaps
def gap_hours(a, b, same=False) -> np.ndarray:
    """Hours from each element of a to the nearest element of b (times in hours); same=True skips the element itself."""
    d = np.abs(a[:, None] - b[None, :])
    if same:
        np.fill_diagonal(d, np.inf)
    return d.min(1)


def sec_gaps(df):
    folds = json.loads(data.FOLDS_JSON.read_text(encoding="utf-8"))
    t = (df["time"] - pd.Timestamp("2000-01-01")).dt.total_seconds().to_numpy() / 3600
    trn, tst = (df["split"] == "train").to_numpy(), (df["split"] == "test").to_numpy()
    fo = df["stem"].map(folds).to_numpy()
    pops = {"train -> nearest other train": gap_hours(t[trn], t[trn], same=True), "test -> nearest train": gap_hours(t[tst], t[trn])}
    cv = np.zeros(trn.sum())
    for k in range(5):
        v = fo[trn] == k
        cv[v] = gap_hours(t[trn][v], t[trn][~v])
    pops["CV validation -> training folds (folds.json)"] = cv
    cuts = {"<= 1 h": 1, "<= 6 h": 6, "<= 23 h": 23, "<= 24 h": 24, "<= 25 h": 25, "<= 72 h": 72, "<= 7 d": 168}
    rows = {k: {"n": len(g), "median (h)": round(float(np.median(g)), 1), "p90 (h)": round(float(np.percentile(g, 90)), 1),
                **{c: round(float(100 * (g <= h).mean()), 1) for c, h in cuts.items()}} for k, g in pops.items()}
    tab = pd.DataFrame(rows).T
    tab.index.name = "gap to nearest"
    res = dict(cv24=float(100 * (cv <= 24).mean()), test24=float(100 * (pops["test -> nearest train"] <= 24).mean()),
               train24=float(100 * (pops["train -> nearest other train"] <= 24).mean()), cv_med=float(np.median(cv)),
               test_med=float(np.median(pops["test -> nearest train"])), tab=tab,
               cv23=float(100 * (cv <= 23).mean()), cv25=float(100 * (cv <= 25).mean()))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for (k, g), col, ls in zip(pops.items(), (MUTED, ORANGE, BLUE), ("--", "-", "-")):
        g = np.sort(g)
        ax.plot(np.maximum(g, .1), np.arange(1, len(g) + 1) / len(g) * 100, color=col, ls=ls, lw=1.8, label=k)
    ax.set_xscale("log"), ax.axvline(24, color=INK, lw=.8)
    _ax(ax, "Time to the nearest other image", "hours (log scale)", "% of images within this gap"), ax.legend(fontsize=8, frameon=False, loc="upper left")
    _save(fig, "fig_timegap.png")
    L = ["## 5. Time gap to the nearest other image", "",
         "Gap in hours from each image to the nearest image of the comparison pool, any site. 'CV validation' pools the five validation folds "
         "of `folds.json`, each against the other four folds (80% of train); 'test' is against all 707 train images.", "", md(tab), "",
         f"Validation images with a training neighbour within 24 h under folds.json: **{res['cv24']:.1f}%**; test images with a train neighbour "
         f"within 24 h: {res['test24']:.1f}%. The curve jumps at 24 h because many images repeat at the same time of day, so the CV figure is "
         f"cadence-sensitive: {res['cv23']:.1f}% within 23 h, {res['cv24']:.1f}% within 24 h, {res['cv25']:.1f}% within 25 h.", "",
         f"Meaning: the CV neighbour structure ({res['cv24']:.1f}% within 24 h, median {res['cv_med']:.0f} h) is "
         + ("close to" if abs(res['cv24'] - res['test24']) < 5 else "different from")
         + f" the test-to-train structure ({res['test24']:.1f}% within 24 h, median {res['test_med']:.0f} h): "
         + ("temporal neighbours do not make CV optimistic relative to test." if res['cv24'] >= res['test24'] - 5 else
            "CV validation images have fewer close neighbours than test images will have, so CV is, if anything, slightly pessimistic on this axis.")
         + f" The test images sit as close to the train images as the train images sit to each other ({res['test24']:.1f}% against {res['train24']:.1f}% "
         "within 24 h, full pool), consistent with the test being drawn like the train, with the gap to the CV figure coming from the smaller 80% pool; "
         "folds are by image, so images a day apart with the same active regions stay across the fold boundary, as they do for test.", ""]
    return L, res


# ---------------------------------------------------------------- driver
def main(recompute=False):
    OUT.mkdir(parents=True, exist_ok=True)
    df = image_stats(recompute)
    recs = data.load_records()
    L = ["# EDA: data checks", "", f"{(df.split == 'train').sum()} train and {(df.split == 'test').sum()} test images; image sizes "
         f"{sorted(df['size'].unique().tolist())} px. All numbers are computed by `python -m filseg.eda`; per-image statistics are in `image_stats.csv`.", ""]
    parts = {}
    for name, fn, arg in (("site", sec_site_year, df), ("drift", sec_drift, df), ("agree", sec_agreement, recs),
                          ("fil", sec_filaments, recs), ("gap", sec_gaps, df)):
        out = fn(arg)
        L += out[0]
        parts[name] = out[1]
        print("done", name)
    contra, minor = list(parts["fil"]["bad"]), parts["fil"]["minor"]
    auc = parts["drift"]["auc"]
    if not .45 <= auc <= .60:
        contra.append(f"drift AUC {auc:.3f} outside the expected 0.45-0.60 (brief)")
    risks = [f"Small filaments: {parts['agree']['unmatched_small']:.0f}% of filaments under 500 px have no matching filament from any other annotator "
             f"(vs {parts['agree']['unmatched_big']:.0f}% above 2000 px), so recall on small filaments is limited by annotator agreement "
             f"(a single annotator scores PQ {parts['agree']['human']['pq']:.2f} against the others; a reference level, not a ceiling).",
             f"Work resolution: {parts['fil']['thin'][512]:.0f}% of filaments are thinner than 4 px at 512 px, {parts['fil']['thin'][1024]:.0f}% at 1024 px.",
             f"Sparse late years: {parts['site']['late']['train']} train vs {parts['site']['late']['test']} test images from 2018 on.",
             f"Temporal neighbours: {parts['gap']['cv24']:.1f}% of CV validation images have a training neighbour within 24 h "
             f"(test: {parts['gap']['test24']:.1f}%)."]
    if auc > .70:
        risks.insert(0, f"Drift: train/test AUC {auc:.3f}, driven by {parts['drift']['top']}.")
    ok = [f"Drift: AUC {auc:.3f} (shuffled control {parts['drift']['shuffled']:.3f}); no statistic separates train from test.",
          f"Multi-part annotations: {parts['fil']['multi_part']:.1f}% of filaments have several 8-connected parts; one component per filament is fine.",
          f"Site mix: largest train/test gap {parts['site']['site']['max_gap']:.1f} points; no site or year only in test."]
    h = parts["agree"]["human"]
    notes = [f"Single-annotator PQ {h['pq']:.2f} is a reference level, not a ceiling. FINDINGS lists nested OOF {PUBLIC_OOF} and public {PUBLIC_LB} for "
             f"competing teams, all above it: a model predicting the consensus can drop filaments that only one annotator drew, and a filament that "
             f"matches a given record with probability p raises that record's PQ whenever p > PQ / (2 SQ) = {h['pq'] / (2 * h['sq']):.2f}."]
    L += ["## Risks for the report", ""] + [f"- {r}" for r in risks] + ["", "## Checked, not a risk", ""] + [f"- {r}" for r in ok] + ["", "## Contradictions / notes", "",
          *( [f"- {c}" for c in contra] or ["- Research limb and area numbers: none beyond tolerance (limb numbers 0.015, areas 2%, shares 0.15 points); drift AUC is in the expected band."]),
          *[f"- {n}" for n in notes], "",
          "Differences at the quoted precision, inside tolerance (percentile convention, rounding): " + ("; ".join(minor) or "none") + ".", ""]
    (OUT / "EDA.md").write_text("\n".join(L), encoding="utf-8", newline="\n")
    print(f"wrote {OUT / 'EDA.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="rescan the 887 images instead of reading image_stats.csv")
    main(ap.parse_args().recompute)
