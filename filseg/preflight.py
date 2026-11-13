"""Submission validator, PROTOCOL 7.3 pre-flight checks, preview sheet."""
import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

from filseg import pq
from filseg.data import CX, CY, FOLDS_JSON, H, R, REPO, TEST_IMG, W, load_image

HEADER = "filament_id,segmentation_rle"
RULE_JSON = REPO / "filseg" / "rule.json"
NEAR_COPY_PQ = 0.98


def _runs(counts: str) -> list[int]:
    """Run lengths of a compressed RLE string (pycocotools rleFrString); IndexError when truncated."""
    runs, p = [], 0
    while p < len(counts):
        x = k = 0
        while True:
            c = ord(counts[p]) - 48
            p += 1
            x |= (c & 0x1F) << (5 * k)
            k += 1
            if not c & 0x20:
                if c & 0x10:
                    x -= 1 << (5 * k)
                break
        runs.append(x + runs[-2] if len(runs) > 2 else x)
    return runs


def _scan(path, stems: set[str]) -> list[dict]:
    """All format checks of the file; one {stem, k, area, cy, cx} per row. ValueError on any fault."""
    try:
        text = Path(path).read_bytes().decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"{path} is not utf-8: {e}")
    if '"' in text:
        raise ValueError("quote character in file")
    if "\r" in text:
        raise ValueError("carriage return in file: line endings must be \\n")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()                                 # a single final newline is fine
    if not lines or lines[0] != HEADER:
        raise ValueError(f"header must be exactly {HEADER!r}")
    by_stem = defaultdict(list)
    for n, line in enumerate(lines[1:], 2):
        parts = line.split(",")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"line {n}: empty line or not exactly two fields")
        stem, _, k = parts[0].rpartition("_")
        if stem not in stems:
            raise ValueError(f"line {n}: stem {stem!r} is not an image of this split")
        if not (k.isascii() and k.isdigit() and str(int(k)) == k):
            raise ValueError(f"line {n}: id {parts[0]!r} does not end in _<k> with k a plain integer")
        by_stem[stem].append((int(k), parts[1], n))
    rows = []
    for stem, items in by_stem.items():
        if sorted(k for k, _, _ in items) != list(range(1, len(items) + 1)):
            raise ValueError(f"{stem}: k must be unique and run from 1 to {len(items)}")
        seen = np.zeros((H, W), bool)
        for k, counts, n in sorted(items):
            if any(not 48 <= ord(ch) <= 111 for ch in counts):
                raise ValueError(f"line {n}: counts hold characters outside an RLE string")
            try:
                runs = _runs(counts)
            except IndexError:
                raise ValueError(f"line {n}: truncated RLE string")
            if min(runs) < 0 or sum(runs) != H * W:
                raise ValueError(f"line {n}: RLE does not cover {H} x {W} pixels (covers {sum(runs)})")
            if sum(runs[1::2]) == 0:
                raise ValueError(f"line {n}: {stem}_{k} is an empty mask")
            rle = {"size": [H, W], "counts": counts}
            x, y, w, h = (int(v) for v in mask_util.toBbox(rle))        # work inside the bbox: 4 MP per mask is slow
            m = mask_util.decode(rle)[y:y + h, x:x + w].astype(bool)
            if (seen[y:y + h, x:x + w] & m).any():
                raise ValueError(f"{stem}: overlap, {stem}_{k} shares a pixel with an earlier mask")
            seen[y:y + h, x:x + w] |= m
            ys, xs = np.nonzero(m)
            rows.append(dict(stem=stem, k=k, area=len(ys), cy=float(ys.mean() + y), cx=float(xs.mean() + x)))
    return rows


def validate_submission(path, stems: set[str]) -> None:
    """ValueError on any format fault or overlap. Images of `stems` without a row are allowed."""
    _scan(path, stems)


def read_submission(path) -> dict[str, list[dict]]:
    """stem -> RLE dicts in k order (the file is assumed to have passed validate_submission)."""
    rows = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines()[1:]:
        rid, counts = line.split(",")
        stem, _, k = rid.rpartition("_")
        rows[stem].append((int(k), {"size": [H, W], "counts": counts}))
    return {s: [r for _, r in sorted(v, key=lambda t: t[0])] for s, v in rows.items()}


def roundtrip_equal(path, preds: dict[str, list[dict]]) -> bool:
    """The CSV decodes to exactly the in-memory masks (same images, same order, same pixels)."""
    back = read_submission(path)
    live = {s: r for s, r in preds.items() if r}
    return back.keys() == live.keys() and all(
        len(back[s]) == len(live[s]) and all(np.array_equal(mask_util.decode(a), mask_util.decode(b))
                                             for a, b in zip(back[s], live[s])) for s in live)


def preview_sheet(path, stems: set[str], img_dir=TEST_IMG, n: int = 12, seed: int = 0):
    """<csv stem>_preview.png next to the CSV: n random images (fixed seed) with the masks as contours.
    None when none of the stems has an image in img_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    have = [s for s in sorted(stems) if (Path(img_dir) / f"{s}.jpeg").exists()]
    if not have:
        return None
    pick = sorted(np.random.default_rng(seed).choice(len(have), min(n, len(have)), replace=False))
    preds = read_submission(path)
    colours = [(255, 60, 60), (60, 255, 60), (60, 160, 255), (255, 220, 0), (255, 60, 255), (0, 230, 230)]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, pick):
        s = have[i]
        img = cv2.cvtColor(load_image(Path(img_dir) / f"{s}.jpeg"), cv2.COLOR_GRAY2RGB)
        for k, r in enumerate(preds.get(s, [])):
            cs, _ = cv2.findContours(np.ascontiguousarray(mask_util.decode(r)), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, colours[k % len(colours)], 4)
        ax.imshow(cv2.resize(img, (768, 768), interpolation=cv2.INTER_AREA))
        ax.set_title(f"{s}  ({len(preds.get(s, []))} masks)", fontsize=9)
    out = Path(path).with_name(Path(path).stem + "_preview.png")
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out


def _git(args: list[str], repo) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, encoding="utf-8")
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def preflight(path, stems: set[str], twin=None, previous=(), twin_stems: set[str] | None = None, kernels=(),
              preds: dict | None = None, repo=REPO, rule_json=RULE_JSON, preview: bool = True,
              img_dir=TEST_IMG) -> dict:
    """Every automatable check of PROTOCOL 7.3. Values are measurements; res["checks"] maps each check to
    pass/fail and res["passed"] is their conjunction. `near_copy` is True when the file IS a near-copy.
    twin_stems: images the OOF twin covers (default: every stem of folds.json). preds: the in-memory RLEs,
    for the decoded-equals-in-memory check (filseg.instances.run does that check itself at write time)."""
    res, checks = {"path": str(path), "near_copy": None}, {}
    res["checks"] = checks
    try:
        rows = _scan(path, stems)
        checks["format"] = True
    except ValueError as e:
        checks["format"], res["format_error"], res["passed"] = False, str(e), False
        return res
    res["rows_per_image"] = len(rows) / len(stems)
    res["empty_image_frac"] = 1 - len({r["stem"] for r in rows}) / len(stems)
    res["mean_area"] = float(np.mean([r["area"] for r in rows])) if rows else 0.0
    res["off_disk"] = [f"{r['stem']}_{r['k']}" for r in rows if np.hypot(r["cx"] - CX, r["cy"] - CY) > R]
    checks["rows_per_image_3_to_12"] = 3 <= res["rows_per_image"] <= 12
    checks["empty_images_at_most_10pct"] = res["empty_image_frac"] <= 0.10
    checks["mask_centres_on_disk"] = not res["off_disk"]
    if preds is not None:
        checks["decoded_equals_in_memory"] = roundtrip_equal(path, preds)

    if twin is None:
        checks["twin_given"] = False
    else:
        ts = twin_stems if twin_stems is not None else set(json.loads(FOLDS_JSON.read_text(encoding="utf-8")))
        try:
            trows = _scan(twin, ts)
        except ValueError as e:
            checks["twin_valid"], res["twin_error"] = False, str(e)
        else:
            res["twin_rows_per_image"] = len(trows) / len(ts)
            res["twin_mean_area"] = float(np.mean([r["area"] for r in trows])) if trows else 0.0
            checks["rows_within_20pct_of_twin"] = (abs(res["rows_per_image"] - res["twin_rows_per_image"])
                                                   <= 0.2 * res["twin_rows_per_image"])
            checks["area_within_30pct_of_twin"] = (abs(res["mean_area"] - res["twin_mean_area"])
                                                   <= 0.3 * res["twin_mean_area"])

    mine = read_submission(path)
    res["near_copy_pq"] = {}
    for p in previous:
        old = read_submission(p)    # the earlier entry plays ground truth; every test image has a record
        res["near_copy_pq"][str(p)] = pq.score({f"x-{s}": old.get(s, []) for s in stems}, mine)["pq"]
    res["near_copy"] = any(v >= NEAR_COPY_PQ for v in res["near_copy_pq"].values())
    checks["not_near_copy"] = not res["near_copy"]

    status = _git(["status", "--porcelain"], repo)
    res["commit"] = _git(["rev-parse", "HEAD"], repo)
    res["rule_json_sha256"] = (hashlib.sha256(Path(rule_json).read_bytes()).hexdigest()
                               if Path(rule_json).is_file() else None)
    res["kernels"] = list(kernels)
    checks["git_tree_clean"] = status == ""
    checks["commit_rule_kernels_recorded"] = bool(res["commit"] and res["rule_json_sha256"] and res["kernels"])
    res["preview"] = str(preview_sheet(path, stems, img_dir)) if preview else None
    res["passed"] = all(checks.values())
    return res


def _main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--twin", help="the OOF twin CSV, same code on held-out folds")
    ap.add_argument("--previous", nargs="*", default=[], help="CSVs of earlier entries (near-copy check)")
    ap.add_argument("--kernels", nargs="*", default=[], help="kernel versions to record, e.g. filseg-A:v3")
    a = ap.parse_args(argv)
    stems = {p.stem for p in TEST_IMG.glob("*.jpeg")}
    if not stems:
        sys.exit(f"no test images under {TEST_IMG}")
    res = preflight(a.csv, stems, a.twin, a.previous, kernels=a.kernels)
    for name, ok in res["checks"].items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(json.dumps({k: v for k, v in res.items() if k != "checks"}, indent=1))
    side = Path(a.csv).with_name(Path(a.csv).stem + "_preflight.json")
    side.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"{'ALL CHECKS PASSED' if res['passed'] else 'PRE-FLIGHT FAILED'}; record {side}; "
          f"preview {res.get('preview')} (look at it before saying go)")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    _main()
