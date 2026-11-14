"""Every figure, table and number of the report, from the saved results. CPU only.
CLI (run from repo/): python -m filseg.report_figs [--run A] [--out ../report] [--runs-root ../runs]
Writes <out>/fig/{examples,iou_dice,m2n}.pdf, <out>/tab_{split,ablation,pilot,slices}.tex and fills the values in
<out>/numbers.tex (macros with a source; the others keep their text)."""
import argparse
import inspect
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pycocotools import mask as mask_util  # noqa: E402

from filseg import data as D  # noqa: E402
from filseg import pq, tune  # noqa: E402
from filseg.instances import candidates, select, to_rles  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "font.size": 7, "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": False})
OKABE = ["#0072B2", "#009E73", "#CC79A7"]      # one colour per annotator record
OURS = "#E69F00"
SITES = ("Bh", "Ch", "Lh", "Mh", "Th", "Uh")
EXPERIMENTS = D.REPO.parent / "EXPERIMENTS.md"
NUM = re.compile(r"[\d.]+")
STEP_ORDER = ("peak", "mean", "min_area", "max_r", "t_lo")      # as in tune.main: step i adds STEP_ORDER[i - 1]


# ------------------------------------------------------------------------------------------ formatting

def sgn(x: float, nd: int = 3) -> str:
    """Signed number for LaTeX text: +0.013 / $-$0.005."""
    return f"{x:+.{nd}f}".replace("-", "$-$")


def pm(v: float, lo: float, hi: float) -> str:
    """Value and half the 95% interval as 0.448 +- .009 (a percentile interval, nearly symmetric)."""
    return f"{v:.3f}$\\pm$" + f"{(hi - lo) / 2:.3f}".lstrip("0")


def _tab(head: str, spec: str, body: list[str]) -> str:
    return "\\begin{tabular}{@{}" + spec + "@{}}\\toprule\n" + head + " \\\\ \\midrule\n" + "\n".join(body) + \
        " \\\\ \\bottomrule\n\\end{tabular}\n"


def _lines(rows: list[list[str]], mid_before: set[int] = frozenset()) -> list[str]:
    out = []
    for i, r in enumerate(rows):
        line = " & ".join(r)
        out.append(line + (" \\\\ \\midrule" if i + 1 in mid_before else (" \\\\" if i + 1 < len(rows) else "")))
    return out


# ------------------------------------------------------------------------------------------ table writers (pure)

def tab_split(fold_of: dict, n_rec: dict, n_fil: dict) -> str:
    """Fold, images, records, filaments, images per site; last row all folds."""
    df = pd.DataFrame({"stem": list(fold_of), "fold": list(fold_of.values())})
    df["rec"], df["fil"], df["site"] = df.stem.map(n_rec), df.stem.map(n_fil), df.stem.str[-2:]
    rows = []
    for f, g in list(df.groupby("fold")) + [("all", df)]:
        rows.append([str(f), str(len(g)), f"{g.rec.sum():,}".replace(",", "{,}"), f"{g.fil.sum():,}".replace(",", "{,}")]
                    + [str(int((g.site == s).sum())) for s in SITES])
    return _tab("Fold & Img & Rec & Fil & " + " & ".join(SITES), "r" * 10, _lines(rows, {len(rows) - 1}))


def tab_ablation(rows: list[tuple[str, dict]]) -> str:
    """rows: (label, {pq, sq, rq: (value, lo, hi)}); the last row (human reference) is set off by a rule."""
    body = [[lab] + [pm(*st[m]) for m in ("pq", "sq", "rq")] for lab, st in rows]
    return _tab(" & PQ & SQ & RQ", "lccc", _lines(body, {len(body) - 1}))


def tab_pilot(rows: list[dict]) -> str:
    """rows: run, setting, pq (value, lo, hi), vs_a (gain, share) or None, minutes or None; ensembles after a rule."""
    body = []
    for r in rows:
        d = "--" if r["vs_a"] is None else f"{sgn(r['vs_a'][0])} ({100 * r['vs_a'][1]:.0f}\\%)"
        body.append([r["run"], r["setting"], pm(*r["pq"]), d, "--" if r["minutes"] is None else f"{r['minutes']:.0f}"])
    first_ens = next((i for i, r in enumerate(rows) if "+" in r["run"]), len(rows))
    return _tab("Run & px/crop/batch & PQ & vs A & min", "llccr", _lines(body, {first_ens} if first_ens < len(rows) else set()))


def _band(label: str, unit: str, nd: int) -> str:
    v = re.findall(r"[\d.]+", label)
    f = lambda x: f"{float(x):.{nd}f}"  # noqa: E731
    if label.startswith("<"):
        return f"$<${f(v[0])}{unit}"
    if label.startswith(">="):
        return f"$\\ge${f(v[0])}{unit}"
    return f"{f(v[0])}--{f(v[1])}{unit}"


def tab_slices(S: dict) -> str:
    """Nested OOF PQ by slice from a tune summary: annotators, size third, r/R third, best and worst site, 24 h."""
    sl, rows = S["slices"], []
    img = lambda name, n, r: [name, str(n), pm(r["pq"], r["lo"], r["hi"])]  # noqa: E731
    rows += [img(f"images with {r['slice']} record" + ("s" if r["slice"] != "1" else ""), r["n_images"], r) for r in sl["annotators"]]
    rows += [[f"size {_band(r['slice'], ' px', 0)}", str(r["n_gt"]), f"{r['pq']:.3f}"] for r in sl["size_px"]]
    rows += [[f"$r/R$ {_band(r['slice'], '', 2)}", str(r["n_gt"]), f"{r['pq']:.3f}"] for r in sl["r_frac"]]
    best, worst = max(sl["site"], key=lambda r: r["pq"]), min(sl["site"], key=lambda r: r["pq"])
    rows += [img(f"site {best['slice']} (best)", best["n_images"], best), img(f"site {worst['slice']} (worst)", worst["n_images"], worst)]
    rows += [img("training image within 24 h" if r["slice"].startswith("with a") else "none within 24 h", r["n_images"], r)
             for r in sl["within_24h"]]
    return _tab("Slice & n & PQ", "lrc", _lines(rows, {3, 6, 9, 11}))


def set_macros(path: Path, values: dict[str, str]) -> list[str]:
    """Replace the body of each \\newcommand{\\name}{...} in numbers.tex, keeping the trailing comment. Raises on an
    unknown name; returns the macros that are still TBD."""
    text = Path(path).read_text(encoding="utf-8")
    pat = r"^(\\newcommand\{\\%s\})\{.*?\}(?=\s*(?:%%|$))"
    for name, val in values.items():
        text, n = re.subn(pat % name, lambda m: m.group(1) + "{" + val + "}", text, count=1, flags=re.M)
        if n != 1:
            raise KeyError(f"macro {name} is not in {path}")
    Path(path).write_text(text, encoding="utf-8", newline="\n")
    return re.findall(r"^\\newcommand\{\\(\w+)\}\{(?:\\url\{)?TBD\}?\}", text, flags=re.M)


# ------------------------------------------------------------------------------------------ statistics

def boot3(s, tp, den, n: int = 2000, seed: int = 0) -> dict:
    """PQ, SQ, RQ as (value, lo, hi): 95% percentile intervals over resamples of the images (same draws as
    tune.bootstrap_ci). s = summed matched IoU, tp = matches, den = TP + FP/2 + FN/2, one entry per image."""
    s, tp, den = (np.asarray(x, float) for x in (s, tp, den))
    idx = np.random.default_rng(seed).integers(0, len(s), (n, len(s)))
    S, T, Dn = s[idx].sum(1), tp[idx].sum(1), den[idx].sum(1)
    out = {}
    for name, (num, dd, fn, fd) in dict(pq=(S, Dn, s.sum(), den.sum()), sq=(S, T, s.sum(), tp.sum()),
                                        rq=(T, Dn, tp.sum(), den.sum())).items():
        lo, hi = np.percentile(num / dd, [2.5, 97.5])
        out[name] = (float(fn / fd), float(lo), float(hi))
    return out


def stat_of(per: pd.DataFrame) -> dict:
    """boot3 of a per-image table from tune.nested_pq (columns s, k, tp, n_gt)."""
    return boot3(per.s, per.tp, 0.5 * (per.k + per.n_gt), n=N_BOOT)


N_BOOT = inspect.signature(tune.bootstrap_ci).parameters["n"].default


def human_per_image(records: dict) -> pd.DataFrame:
    """One row per image with 2 or 3 records: matched-IoU sum, TP and PQ denominator when each record in turn is the
    prediction against the others (the pooling of eda.human_pq, kept per image so it can be bootstrapped)."""
    rows = []
    for stem, rr in records.items():
        if len(rr) < 2:
            continue
        rids, s, tp, fp, fn = list(rr), 0.0, 0, 0, 0
        for j, rid in enumerate(rids):
            r = pq.score({o: rr[o] for o in rids if o != rid}, {stem: rr[rid]})
            s, tp, fp, fn = s + math.fsum(r["tp_ious"]), tp + r["tp"], fp + r["fp"], fn + r["fn"]
        rows.append(dict(stem=stem, s=s, tp=tp, fp=fp, fn=fn, den=tp + 0.5 * fp + 0.5 * fn))
    return pd.DataFrame(rows)


def experiments_human() -> dict:
    """PQ, SQ, RQ and the pair count of the annotator-against-annotator row of EXPERIMENTS.md."""
    row = next(ln for ln in EXPERIMENTS.read_text(encoding="utf-8").splitlines() if ln.startswith("| annotator against annotator"))
    pairs = int(re.search(r"\((\d+) record pairs\)", row).group(1))
    v = {k: float(re.search(rf"{k.upper()} ([\d.]+)", row).group(1)) for k in ("pq", "sq", "rq")}
    return dict(pairs=pairs, **v)


# ------------------------------------------------------------------------------------------ nested tables

def nested_tables(run: str, root: Path, records: dict) -> tuple[list[dict], pd.DataFrame]:
    """Per-image nested tables of run `run` for every ablation step (tune.main's steps) and the table itself."""
    table = tune.candidate_table([run], [0, 1, 2, 3, 4], records=records, root=root)
    steps = []
    for i in range(len(STEP_ORDER) + 1):
        g = {**tune.OFF, **{c: tune.GRID[c] for c in STEP_ORDER[:i]}}
        v, per = tune.nested_pq(table, records, g)
        steps.append(dict(pq=v, per=per))
    return steps, table


def pilot_rows(root: Path, records: dict) -> tuple[list[dict], dict]:
    """The four pilot runs and three ensembles (fold 0): value, interval and cost from the saved summaries, and the
    paired bootstrap against A recomputed from the candidate tables."""
    names = ["A", "A2", "B", "C", "A+A2", "A+B", "A+C"]
    per, summ = {}, {}
    for r in names:
        summ[r] = json.loads((root / r / "tune_summary_folds0.json").read_text(encoding="utf-8"))
        table = tune.candidate_table(r.split("+"), [0], records=records, root=root)
        v, per[r] = tune.nested_pq(table, records)
        assert abs(v - summ[r]["nested"]["pq"]) < 1e-9, f"{r}: recomputed {v} != summary {summ[r]['nested']['pq']}"
    cfg = {r: json.loads((root / r.split("+")[0] / "fold0_meta.json").read_text(encoding="utf-8"))["cfg"] for r in names[:4]}
    rows = []
    for r in names:
        pqv, (lo, hi) = summ[r]["nested"]["pq"], summ[r]["nested"]["ci"]
        if "+" in r:
            setting, minutes = "map average", None
        else:
            c = cfg[r]
            same = {k: v for k, v in c.items() if k not in ("run", "seed")} == {k: v for k, v in cfg["A"].items() if k not in ("run", "seed")}
            setting = f"as A, seed {c['seed']}" if same and r != "A" else f"{c['work_res']}/{c['crop']}/{c['batch']}"
            minutes = json.loads((root / r / "fold0_train.json").read_text(encoding="utf-8"))["wall_s"] / 60
        rows.append(dict(run=r, setting=setting, pq=(pqv, lo, hi), minutes=minutes,
                         vs_a=None if r == "A" else tune.paired_bootstrap(per["A"], per[r])))
    return rows, dict(per=per, cfg=cfg, summ=summ)


# ------------------------------------------------------------------------------------------ predictions and figures

def _pred_chunk(task):
    items, _recs, rules = task
    out = {}
    for stem, prob in tune._maps(items):
        r = tune._rule(rules[stem])
        lab, cands = candidates(prob, r.t_lo)
        out[stem] = to_rles(lab, select(cands, r))
    return out


def nested_predictions(run: str, root: Path, records: dict, per_nested: pd.DataFrame) -> dict:
    """Kept RLEs per image under each image's nested rule (the rule tuned without its fold)."""
    rules = per_nested.set_index("stem")[list(tune.RULE_COLS)].to_dict("index")
    parts = tune._pmap(_pred_chunk, tune._chunks(tune._plan([run], [0, 1, 2, 3, 4], root), records, rules), tune.WORKERS)
    return {s: v for p in parts for s, v in p.items()}


def fig_iou_dice(res: dict, out: Path) -> None:
    iou = np.asarray(res["tp_ious"], float)
    dice = 2 * iou / (1 + iou)
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    bins = np.linspace(0.5, 1, 26)
    for v, c, lab in ((iou, "#0072B2", "IoU"), (dice, OURS, "Dice")):
        ax.hist(v, bins=bins, histtype="stepfilled", color=c, alpha=0.55, lw=0, label=f"{lab}, median {np.median(v):.2f}")
        ax.hist(v, bins=bins, histtype="step", color=c, lw=0.8)
    ax.set_xlabel("overlap of matched record-prediction pairs"), ax.set_ylabel("pairs")
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.2)
    fig.tight_layout(pad=0.3), fig.savefig(out), plt.close(fig)


def degree_shares(res: dict) -> tuple[float, float]:
    """Share of record filaments overlapped by 2 or more predictions, share of predictions overlapping 2 or more."""
    return float((res["gt_degree"] > 1).mean()), float((res["pred_degree"] > 1).mean())


def fig_m2n(res: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    cats = ["0", "1", "2", "3+"]
    for k, (deg, c, lab) in enumerate(((res["gt_degree"], "#0072B2", "predictions per record filament"),
                                       (res["pred_degree"], OURS, "record filaments per prediction"))):
        n = np.bincount(np.minimum(deg, 3), minlength=4)
        ax.bar(np.arange(4) + (k - 0.5) * 0.38, n, 0.36, color=c, label=lab)
        for x, v in zip(np.arange(4) + (k - 0.5) * 0.38, n):
            ax.text(x, v, f"{v}", ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(range(4), cats), ax.set_xlabel("overlapping partners (IoU > 0)"), ax.set_ylabel("count")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.4), ax.legend(loc="upper right", fontsize=6.5, handlelength=1.0)
    fig.tight_layout(pad=0.3), fig.savefig(out), plt.close(fig)


def _centre(rle: dict) -> tuple[float, float]:
    x, y, w, h = mask_util.toBbox(rle)
    return y + h / 2, x + w / 2


def _window(c: tuple[float, float], size: int) -> tuple[int, int]:
    return tuple(int(min(max(v - size // 2, 0), D.H - size)) for v in c)


FAIL_TEXT = {1: "a component covered the filament but the keep rule dropped it",
             2: "our component overlaps the largest missed filament at IoU of at most 0.5",
             3: "no component reaches the threshold on the largest missed filament"}


def pick_examples(records: dict, per_nested: pd.DataFrame, seed: int = 0) -> tuple[list[str], str]:
    """Five images drawn by a fixed seed from the images with 2-3 records and at least 4 filaments, and a failure
    image: the worst per-image PQ among 20 further seeded draws."""
    pool = sorted(s for s in records if len(records[s]) >= 2 and sum(map(len, records[s].values())) >= 4)
    order = [pool[i] for i in np.random.default_rng(seed).permutation(len(pool))]
    q = per_nested.set_index("stem")
    score = (q.s / (0.5 * q.n_gt + 0.5 * q.k)).to_dict()
    return order[:5], min(order[5:25], key=lambda s: score[s])


def fig_examples(run: str, root: Path, records: dict, preds: dict, per_nested: pd.DataFrame, out: Path, size: int = 320) -> str:
    """2 x 3 crops: image, each record's contour, our kept components. Returns the failure clause for failureWhat."""
    good, bad = pick_examples(records, per_nested)
    rng = np.random.default_rng(1)
    ctr = {s: _centre(rng.choice([m for rr in records[s].values() for m in rr])) for s in good}
    # failure: centre on the largest filament of the failure image that has no match, class from tune._outcomes
    plan = tune._plan([run], [0, 1, 2, 3, 4], root)
    (_, prob), = tune._maps([(bad, plan[bad])])
    rule = tune._rule(per_nested.set_index("stem")[list(tune.RULE_COLS)].to_dict("index")[bad])
    lab, cands = candidates(prob, rule.t_lo)
    gt = tune._gt_info(records[bad])
    g, _ = tune._outcomes(lab, cands, gt, rule)
    miss = np.flatnonzero(g["bucket"] != 0)
    j = int(miss[np.argmax(g["area"][miss])])
    y, x, crop = gt["crops"][j]
    ctr[bad] = (y + crop.shape[0] / 2, x + crop.shape[1] / 2)
    fig, axes = plt.subplots(2, 3, figsize=(3.33, 2.3))
    for ax, stem, tag in zip(axes.ravel(), good + [bad], "abcdef"):
        r0, c0 = _window(ctr[stem], size)
        sl = (slice(r0, r0 + size), slice(c0, c0 + size))
        ax.imshow(D.load_image(D.TRAIN_IMG / f"{stem}.jpeg")[sl], cmap="gray", vmin=0, vmax=255)
        ours = np.zeros((D.H, D.W), bool)
        for m in preds[stem]:
            ours |= mask_util.decode(m).astype(bool)
        rgba = np.zeros((size, size, 4))
        rgba[ours[sl]] = (*matplotlib.colors.to_rgb(OURS), 0.45)
        ax.imshow(rgba)
        for col, rles in zip(OKABE, records[stem].values()):
            m = mask_util.decode(mask_util.merge(rles))[sl]
            if m.any():
                ax.contour(m, levels=[0.5], colors=col, linewidths=0.7)
        ax.text(0.04, 0.96, f"({tag})", transform=ax.transAxes, va="top", fontsize=6.5, color="white",
                bbox=dict(fc="black", alpha=0.6, pad=1, lw=0))
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True), sp.set_linewidth(0.4)
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995, 0.03, 0.03)
    fig.savefig(out), plt.close(fig)
    return FAIL_TEXT[int(g["bucket"][j])]


# ------------------------------------------------------------------------------------------ main

def span(v, nd: int) -> str:
    """Range of a list as 0.36--0.41 (one value if both ends print the same)."""
    lo, hi = f"{min(v):.{nd}f}", f"{max(v):.{nd}f}"
    return lo if lo == hi else f"{lo}--{hi}"


def _pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="A", help="the 5-fold run of the final recipe")
    ap.add_argument("--out", default=str(D.REPO.parent / "report"))
    ap.add_argument("--runs-root", default=None)
    a = ap.parse_args(argv)
    out, root = Path(a.out), Path(a.runs_root) if a.runs_root else tune._runs_root()
    (out / "fig").mkdir(parents=True, exist_ok=True)
    S = json.loads((root / a.run / "tune_summary.json").read_text(encoding="utf-8"))
    records = D.load_records()
    M: dict[str, str] = {}                       # numbers.tex macros

    # split table
    fold_of = json.loads(Path(D.FOLDS_JSON).read_text(encoding="utf-8"))
    n_rec = {s: len(r) for s, r in records.items()}
    n_fil = {s: sum(map(len, r.values())) for s, r in records.items()}
    (out / "tab_split.tex").write_text(tab_split(fold_of, n_rec, n_fil), encoding="utf-8", newline="\n")

    # ablation: nested per-image tables of every step, the classical baseline, the human reference
    steps, _ = nested_tables(a.run, root, records)
    for st, ref in zip(steps, S["steps"]):
        assert abs(st["pq"] - ref["nested_pq"]) < 1e-9, f"step {ref['step']}: {st['pq']} != {ref['nested_pq']}"
    final = steps[-1]
    assert abs(final["pq"] - S["nested"]["pq"]) < 1e-9
    cl_table = tune.candidate_table(["classical"], [0, 1, 2, 3, 4], records=records, root=root)
    cl_pq, cl_per = tune.nested_pq(cl_table, records)
    cl_sum = json.loads((root / "classical" / "tune_summary.json").read_text(encoding="utf-8"))
    assert abs(cl_pq - cl_sum["nested"]["pq"]) < 1e-9
    del cl_table
    hum = human_per_image(records)
    ex = experiments_human()
    hst = boot3(hum.s, hum.tp, hum.den, n=N_BOOT)
    for k in ("pq", "sq", "rq"):
        assert abs(hst[k][0] - ex[k]) < 6e-4, f"human {k}: computed {hst[k][0]:.4f}, EXPERIMENTS.md {ex[k]}"
    labels = ["No-training baseline", "U-Net, threshold 0.5", "\\quad + peak", "\\quad + mean", "\\quad + area",
              "\\quad + limb cut (final rule)"]
    rows = [(labels[0], stat_of(cl_per))] + [(lab, stat_of(st["per"])) for lab, st in zip(labels[1:], [steps[i] for i in (0, 1, 2, 3, 5)])]
    rows.append(("Annotator vs annotator", hst))
    (out / "tab_ablation.tex").write_text(tab_ablation(rows), encoding="utf-8", newline="\n")

    # pilot
    prow, pil = pilot_rows(root, records)
    (out / "tab_pilot.tex").write_text(tab_pilot(prow), encoding="utf-8", newline="\n")
    (out / "tab_slices.tex").write_text(tab_slices(S), encoding="utf-8", newline="\n")

    # predictions under the nested rules: figures
    preds = nested_predictions(a.run, root, records, final["per"])
    stems = sorted(preds)
    res = pq.score({rid: r for s in stems for rid, r in records[s].items()}, preds)
    assert abs(res["pq"] - S["nested"]["pq"]) < 1e-3, f"pq.score {res['pq']} vs nested {S['nested']['pq']}"
    fig_iou_dice(res, out / "fig" / "iou_dice.pdf")
    fig_m2n(res, out / "fig" / "m2n.pdf")
    M["failureWhat"] = fig_examples(a.run, root, records, preds, final["per"], out / "fig" / "examples.pdf")
    split, merge = degree_shares(res)

    # macros
    cfgA = pil["cfg"]["A"]
    rule = json.loads((D.FOLDS_JSON.parent / "rule.json").read_text(encoding="utf-8"))
    assert all(abs(rule[k] - S["best_rule"][k]) < 1e-12 for k in ("t_lo", "peak", "mean", "min_area", "max_r")), "rule.json != tune summary"
    tr = [json.loads((root / a.run / f"fold{k}_train.json").read_text(encoding="utf-8")) for k in range(5)]
    pv = {r["run"]: r for r in prow}
    pqs = {r: pil["summ"][r]["nested"]["pq"] for r in pv}
    noise = abs(pqs["A"] - pqs["A2"])
    ens = [pv[r] for r in ("A+A2", "A+B", "A+C")]
    nested, (lo, hi) = S["nested"], S["nested"]["ci"]
    fb, ng = S["failure_buckets"], sum(S["failure_buckets"].values())
    miss = ng - fb["matched"]
    sz, rf = S["slices"]["size_px"], S["slices"]["r_frac"]
    w24 = {r["slice"].startswith("with a"): r["pq"] for r in S["slices"]["within_24h"]}
    gain = lambda i, j: steps[j]["pq"] - steps[i]["pq"]  # noqa: E731
    M.update(
        nBoot=str(N_BOOT), workRes=str(cfgA["work_res"]), cropPx=str(cfgA["crop"]), batchN=str(cfgA["batch"]), chosenRun="A",
        chosenDesc=f"{cfgA['work_res']} px, crop {cfgA['crop']}, batch {cfgA['batch']}",
        secPerStep=span([t["sec_per_step"] for t in tr], 2), minPerFold=span([t["wall_s"] / 60 for t in tr], 0),
        mbPerModel=f"{(root / a.run / 'fold0.pt').stat().st_size / 1e6:.0f}",
        seedNoise=f"{noise:.4f}", pilotBar=f"{max(0.010, 2 * noise):.3f}",
        gainAvsB=f"{pqs['A'] - pqs['B']:.3f}", gainCvsB=f"{pqs['C'] - pqs['B']:.3f}", gainCvsA=f"{pqs['A'] - pqs['C']:.3f}",
        shareAvsB=_pct(1 - pv["B"]["vs_a"][1]), shareCvsA=_pct(pv["C"]["vs_a"][1]),
        tLo=f"{rule['t_lo']:g}", thPeak=f"{rule['peak']:g}", thMean=f"{rule['mean']:g}", thArea=f"{rule['min_area']:g}",
        thLimb=f"{rule['max_r']:g}", tauHalf=f"{nested['pq'] / 2:.3f}",
        pqBase=f"{cl_sum['nested']['pq']:.3f}", pqBaseLo=f"{cl_sum['nested']['ci'][0]:.3f}", pqBaseHi=f"{cl_sum['nested']['ci'][1]:.3f}",
        pqHuman=f"{ex['pq']:.3f}", sqHuman=f"{ex['sq']:.3f}", rqHuman=f"{ex['rq']:.3f}", nPairs=str(ex["pairs"]),
        pqHumanLo=f"{hst['pq'][1]:.3f}", pqHumanHi=f"{hst['pq'][2]:.3f}",
        matchHuman=f"{100 * hum.tp.sum() / (hum.tp.sum() + hum.fp.sum()):.0f}",
        pqThr=f"{steps[0]['pq']:.3f}", gainPeak=f"{gain(0, 1):.3f}", gainMean=f"{gain(1, 2):.3f}", gainArea=f"{gain(2, 3):.3f}",
        pqPM=f"{steps[3]['pq']:.3f}", pqRule=f"{steps[5]['pq']:.3f}", gainLimb=f"{gain(3, 5):.4f}",
        shareLimb=_pct(tune.paired_bootstrap(steps[3]["per"], steps[5]["per"])[1], 0),
        pqOOF=f"{nested['pq']:.3f}", pqOOFlo=f"{lo:.3f}", pqOOFhi=f"{hi:.3f}", sqOOF=f"{nested['sq']:.3f}", rqOOF=f"{nested['rq']:.3f}",
        shareSplit=_pct(split), shareMerge=_pct(merge), pubNoise=f"{(hi - lo) / 2 * math.sqrt(707 / 90):.3f}",
        bucketNone=_pct(fb["never_detected"] / miss), bucketShape=_pct(fb["wrong_shape"] / miss), bucketRej=_pct(fb["discarded"] / miss),
        recSmall=_pct(sz[0]["tp"] / sz[0]["n_gt"], 0), recMid=_pct(sz[1]["tp"] / sz[1]["n_gt"], 0), recLarge=_pct(sz[2]["tp"] / sz[2]["n_gt"], 0),
        sizeLo=f"{float(NUM.findall(sz[0]['slice'])[0]):.0f}", sizeHi=f"{float(NUM.findall(sz[1]['slice'])[1]):.0f}",
        gapNear=f"{w24[True] - w24[False]:.3f}",
        ensDropLo=f"{min(-e['vs_a'][0] for e in ens):.3f}", ensDropHi=f"{max(-e['vs_a'][0] for e in ens):.3f}",
        ensShareLo=_pct(min(e["vs_a"][1] for e in ens), 0), ensShareHi=_pct(max(e["vs_a"][1] for e in ens), 0),
        repoLink="\\url{https://github.com/blowmuffin/solar-filament-segmentation-2026}")
    still = set_macros(out / "numbers.tex", M)
    print(f"wrote figures and tables to {out}; still TBD in numbers.tex: {', '.join(still) or 'none'}")


if __name__ == "__main__":
    main()
