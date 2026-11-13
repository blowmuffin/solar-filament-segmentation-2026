"""Connected-component instances from probability maps, keep rule, RLE rows, submission CSV writer."""
import argparse
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

from filseg.data import CX, CY, FOLDS_JSON, H, R, REPO, TEST_IMG, W

RUNS = (REPO / ".." / "runs").resolve()
HEADER = "filament_id,segmentation_rle"


@dataclass
class Rule:
    t_lo: float = 0.40
    peak: float = 0.70
    mean: float = 0.55
    min_area: int = 200
    max_r: float = 0.92
    tau: float | None = None


def candidates(prob: np.ndarray, t_lo: float) -> tuple[np.ndarray, list[dict]]:
    """8-connected components of prob > t_lo: int32 label map (0 = background) and one feature dict each."""
    n, lab, st, cen = cv2.connectedComponentsWithStats((prob > t_lo).astype(np.uint8), connectivity=8)
    s = prob.shape[0] / 2048
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in st[i])
        m = lab[y:y + h, x:x + w] == i
        v = prob[y:y + h, x:x + w][m]
        yy, xx = np.nonzero(m)
        ev = np.linalg.eigvalsh(np.cov(np.c_[yy, xx].T)) if area > 1 else np.zeros(2)   # ascending
        cx, cy = (float(c) for c in cen[i])
        out.append(dict(id=i, area=area, peak=float(v.max()), mean=float(v.mean()),
                        p10=float(np.percentile(v, 10)), p50=float(np.percentile(v, 50)), cy=cy, cx=cx,
                        r_frac=float(np.hypot(cx - CX * s, cy - CY * s) / (R * s)), w=w, h=h,
                        fill=area / (w * h), major=float(4 * np.sqrt(max(ev[1], 0))),
                        minor=float(4 * np.sqrt(max(ev[0], 0)))))
    return lab, out


def select(cands: list[dict], rule: Rule, booster=None) -> list[int]:
    """Ids whose peak, mean, area are >= the rule's values and r_frac <= max_r; with rule.tau set, also
    booster(cands) >= tau (booster returns one score per candidate, in order)."""
    if rule.tau is not None and booster is None:
        raise ValueError("rule.tau is set but no booster was given: the tuned rule would be applied only in part")
    score = booster(cands) if rule.tau is not None else [None] * len(cands)
    return [c["id"] for c, sc in zip(cands, score)
            if c["peak"] >= rule.peak and c["mean"] >= rule.mean and c["area"] >= rule.min_area
            and c["r_frac"] <= rule.max_r and (sc is None or sc >= rule.tau)]


def to_rles(labels: np.ndarray, ids: list[int]) -> list[dict]:
    """Column-major RLE per id (counts as str); the masks are pixel-disjoint because labels are."""
    rles = [mask_util.encode(np.asfortranarray(labels == i, dtype=np.uint8)) for i in ids]
    return [{"size": r["size"], "counts": r["counts"].decode("ascii")} for r in rles]


def write_submission(preds: dict[str, list[dict]], path) -> None:
    """Header, then `<stem>_<k>,<counts>` rows (k from 1) in stem order; `\\n` endings, no quotes."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER + "\n")
        for stem in sorted(preds):
            for k, r in enumerate(preds[stem], 1):
                c = r["counts"]
                f.write(f"{stem}_{k},{c.decode('ascii') if isinstance(c, bytes) else c}\n")


class _Slabs:
    """Reads one 2048x2048 map at a time from the (n, H, W) uint8 `prob` array of an .npz."""

    def __init__(self, path: Path, run: str):
        self.run = run
        with np.load(path) as z:
            self.stems = [str(s) for s in z["stems"]]
        self.zf = zipfile.ZipFile(path)
        self.f = self.zf.open("prob.npy")
        v = np.lib.format.read_magic(self.f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if v == (1, 0)
                                 else np.lib.format.read_array_header_2_0)(self.f)
        if dtype != np.uint8 or fortran or shape != (len(self.stems), H, W):
            raise ValueError(f"{path}: prob must be C-order uint8 ({len(self.stems)}, {H}, {W}), got {dtype} {shape}")
        self.start, self.pos = self.f.tell(), 0

    def get(self, i: int) -> np.ndarray:
        if i != self.pos:
            self.f.seek(self.start + i * H * W)
        self.pos = i + 1
        return np.frombuffer(self.f.read(H * W), np.uint8).reshape(H, W)

    def close(self):
        self.f.close()
        self.zf.close()


def _fold(path: Path) -> int:
    return int(path.stem.rsplit("fold", 1)[1])


def _fold_files(runs: list[str], split: str, root, folds: list[int] | None) -> list[tuple[str, Path]]:
    files = [(r, p) for r in runs for p in sorted((Path(root) / r).glob(f"{split}_fold*.npz"))
             if folds is None or _fold(p) in folds]
    if not files:
        raise FileNotFoundError(f"no {split}_fold*.npz under {root} for runs {runs}")
    return files


def _json_or_none(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def iter_probs(runs: list[str], split: str, root=RUNS, folds: list[int] | None = None):
    """Yield (stem, float32 prob in [0, 1]) in file order. A stem's maps are averaged over every
    <split>_fold*.npz of every run that holds it (only those of `folds` when given); a stem must come from
    every run, and for oof from exactly one fold file per run. One map per file is in memory at a time."""
    readers = [_Slabs(p, r) for r, p in _fold_files(runs, split, root, folds)]
    try:
        where: dict[str, list] = {}
        for rd in readers:
            for i, s in enumerate(rd.stems):
                where.setdefault(s, []).append((rd, i))
        for s, hits in where.items():
            got = [rd.run for rd, _ in hits]
            if set(got) != set(runs):
                raise ValueError(f"stem {s} is missing from run(s) {sorted(set(runs) - set(got))}")
            if split == "oof" and len(got) != len(runs):
                raise ValueError(f"stem {s} is in more than one oof fold file of a run: {got}")
            acc = np.zeros((H, W), np.float32)
            for rd, i in hits:
                acc += rd.get(i)
            yield s, acc / (255.0 * len(hits))
    finally:
        for rd in readers:
            rd.close()


def run(runs: list[str], split: str, rule: Rule, out, root=RUNS, stems: set[str] | None = None,
        folds: list[int] | None = None, expect_folds: int | None = None, rule_file=None) -> dict:
    """Average the maps, apply the rule, write the CSV, validate it and write the provenance sidecar
    `<out>.meta.json` (what pre-flight reads). Prints the fold files found per run; expect_folds raises when a
    run has another number of fold files for the split. Returns the preds."""
    from filseg import preflight as pf
    if stems is None:
        stems = ({p.stem for p in TEST_IMG.glob("*.jpeg")} if split == "test"
                 else set(json.loads(FOLDS_JSON.read_text(encoding="utf-8"))))
    meta_file = Path(str(out) + ".meta.json")
    meta_file.unlink(missing_ok=True)               # a failed build must not leave the previous build's record
    files = _fold_files(runs, split, root, folds)
    found = {r: [_fold(p) for q, p in files if q == r] for r in runs}
    for r, ks in found.items():
        print(f"{r}: {split} fold files {ks}")
        if expect_folds is not None and len(ks) != expect_folds:
            raise ValueError(f"{r}: {len(ks)} {split} fold files {ks}, expected {expect_folds}")
    preds = {}
    for stem, prob in iter_probs(runs, split, root, folds):
        labels, cands = candidates(prob, rule.t_lo)
        kept = select(cands, rule)
        preds[stem] = to_rles(labels, kept)
        for r, i in zip(preds[stem], kept):     # what we encode must decode to what we have in memory
            if not np.array_equal(mask_util.decode(r), labels == i):
                raise ValueError(f"{stem}: RLE of component {i} does not decode to its mask")
    if (set(preds) != stems) if split == "test" else not (set(preds) <= stems):
        raise ValueError(f"{split} stems do not match: {len(preds)} predicted, {len(stems)} expected, "
                         f"first odd ones {sorted(set(preds) ^ stems)[:3]}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_submission(preds, out)
    pf.validate_submission(out, stems)
    roundtrip = pf.roundtrip_equal(out, preds)
    if not roundtrip:
        raise ValueError(f"{out} does not read back as the masks in memory")
    status = pf._git(["status", "--porcelain"], REPO)
    meta = dict(
        csv_sha256=pf.sha256_file(out), rows=sum(map(len, preds.values())), images=len(preds), split=split, runs=runs,
        folds_requested=folds, folds_found=found, folds=sorted({k for ks in found.values() for k in ks}),
        files=[dict(run=r, file=p.name, sha256=pf.sha256_file(p)[:12], meta=_json_or_none(p.with_name(f"fold{_fold(p)}_meta.json")))
               for r, p in files],
        run_meta={r: _json_or_none(Path(root) / r / "meta.json") for r in runs},
        rule=asdict(rule), rule_file=None if rule_file is None else str(rule_file),
        rule_sha256=None if rule_file is None else pf.sha256_file(rule_file),
        commit=pf._git(["rev-parse", "HEAD"], REPO), dirty=None if status is None else status != "",
        roundtrip_equal=roundtrip)                  # the per-mask decode check above raised if it failed
    meta_file.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return preds


def _main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--split", choices=["test", "oof"], required=True)
    ap.add_argument("--rule", required=True, help="rule.json written by tune")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs-root", default=str(RUNS))
    ap.add_argument("--folds", nargs="+", type=int, help="only these <split>_fold<k>.npz files (default all)")
    ap.add_argument("--expect-folds", type=int, help="raise unless every run has exactly this many fold files")
    a = ap.parse_args(argv)
    rule = Rule(**json.loads(Path(a.rule).read_text(encoding="utf-8")))
    preds = run(a.run, a.split, rule, a.out, root=a.runs_root, folds=a.folds, expect_folds=a.expect_folds,
                rule_file=a.rule)
    n = sum(map(len, preds.values()))
    print(f"wrote {a.out}: {len(preds)} images, {n} masks ({n / len(preds):.2f} per image), validated")


if __name__ == "__main__":
    _main()
