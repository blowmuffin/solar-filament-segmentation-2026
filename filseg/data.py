"""Paths, annotations to RLE records, soft target, normalisation, folds, training cache."""
import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

H = W = 2048
CX, CY, R = 1023.1, 1024.1, 899.7   # solar disk at 2048 px; scaled by img.shape[0] / 2048
TRAIN_JSON_SHA256 = "5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1"

TRAIN_JSON_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
REPO = Path(__file__).resolve().parent.parent
FOLDS_JSON = Path(__file__).resolve().parent / "folds.json"
KAGGLE_INPUT = Path("/kaggle/input")
ON_KAGGLE = KAGGLE_INPUT.is_dir()


def _data_root() -> Path:
    if "FILSEG_DATA" in os.environ:
        return (REPO / os.environ["FILSEG_DATA"]).resolve()     # absolute values win over REPO
    if ON_KAGGLE:
        hit = next(KAGGLE_INPUT.rglob(TRAIN_JSON_NAME), None)
        if hit is not None:
            return hit.parent.parent
    return (REPO / "../data/MAGFiLO_1.0_Kaggle_2026").resolve()


def _cache_root() -> Path:
    if "FILSEG_CACHE" in os.environ:
        return (REPO / os.environ["FILSEG_CACHE"]).resolve()
    return Path("/tmp/filseg_cache") if ON_KAGGLE else (REPO / "../cache").resolve()


DATA_ROOT = _data_root()
CACHE_ROOT = _cache_root()
TRAIN_IMG = DATA_ROOT / "train" / "train_images"
TEST_IMG = DATA_ROOT / "test" / "test_images"
TRAIN_JSON = DATA_ROOT / "train" / TRAIN_JSON_NAME


def stem_of(record_id: str) -> str:
    return record_id.split("-", 1)[1]


def load_records() -> dict[str, dict[str, list[dict]]]:
    """stem -> record id -> RLE dicts (one per filament), in file order."""
    raw = TRAIN_JSON.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TRAIN_JSON_SHA256:
        raise ValueError(f"{TRAIN_JSON} is not the competition annotation file (SHA-256 mismatch)")
    coco = json.loads(raw.decode("utf-8"))
    recs: dict[str, dict[str, list[dict]]] = {}
    for im in coco["images"]:
        recs.setdefault(stem_of(im["id"]), {})[im["id"]] = []
    for a in coco["annotations"]:
        rle = mask_util.merge(mask_util.frPyObjects(a["segmentation"], H, W))   # the host's route
        recs[stem_of(a["image_id"])][a["image_id"]].append(rle)
    return recs


def soft_target(records: dict[str, list[dict]]) -> np.ndarray:
    """float32 (H, W): mean over the records of each record's union mask."""
    t = np.zeros((H, W), np.float32)
    for rles in records.values():
        t += mask_util.decode(mask_util.merge(rles))
    return t / len(records)


def load_image(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


@lru_cache(maxsize=4)
def _disk_masks(res: int) -> tuple[np.ndarray, np.ndarray]:
    """(r < 0.98R, r > R) boolean maps at res px, from float64 radii."""
    s = res / 2048
    yy, xx = np.mgrid[:res, :res]
    r = np.hypot(xx - CX * s, yy - CY * s)
    return r < 0.98 * R * s, r > R * s


def normalize(img: np.ndarray) -> np.ndarray:
    """float32 z-score using the pixels with r < 0.98R; 0 where r > R (outside the disk)."""
    assert img.ndim == 2 and img.shape[0] == img.shape[1], img.shape
    inner, outer = _disk_masks(img.shape[0])
    v = img[inner].astype(np.float64)
    x = ((img.astype(np.float64) - v.mean()) / v.std()).astype(np.float32)
    x[outer] = 0
    return x


def make_folds(n_records: dict[str, int], k: int = 5, seed: int = 0) -> dict[str, int]:
    """Stems sorted by (record count, site, random key); fold = position mod k."""
    stems = sorted(n_records)
    key = dict(zip(stems, np.random.default_rng(seed).random(len(stems))))
    order = sorted(stems, key=lambda s: (n_records[s], s[-2:], key[s]))
    return {s: i % k for i, s in enumerate(order)}


def build_cache(out_dir, work_res: int) -> None:
    """img_<res>.u8 and soft_<res>.u8, raw uint8 (n, res, res) in stems.json order (np.memmap-ready)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = load_records()
    stems = sorted(recs)
    shape = (len(stems), work_res, work_res)
    img_mm = np.memmap(out / f"img_{work_res}.u8", np.uint8, "w+", shape=shape)
    soft_mm = np.memmap(out / f"soft_{work_res}.u8", np.uint8, "w+", shape=shape)
    for i, s in enumerate(stems):
        img, soft = load_image(TRAIN_IMG / f"{s}.jpeg"), soft_target(recs[s])
        assert img.shape == (H, W), (s, img.shape)
        if work_res != H:
            img = cv2.resize(img, (work_res, work_res), interpolation=cv2.INTER_AREA)
            soft = cv2.resize(soft, (work_res, work_res), interpolation=cv2.INTER_AREA)
        img_mm[i], soft_mm[i] = img, np.round(soft * 255).astype(np.uint8)
    img_mm.flush(), soft_mm.flush()
    (out / "stems.json").write_text(json.dumps(stems), encoding="utf-8")


def _bbox_disk(img: np.ndarray, thresh: int = 30) -> tuple[float, float, float]:
    """Centre and radius from the bounding box of the largest component of img > thresh."""
    _, _, st, _ = cv2.connectedComponentsWithStats((img > thresh).astype(np.uint8), connectivity=8)
    x, y, w, h, _ = st[1:][st[1:, cv2.CC_STAT_AREA].argmax()]
    return x + (w - 1) / 2, y + (h - 1) / 2, (w + h - 2) / 4


def fit_limb_ring(img: np.ndarray, half: int = 60) -> tuple[float, float, float]:
    """Centre and radius of the bright limb ring: per-angle radial intensity peak within +-half px of
    (CX, CY, R), then a least-squares circle with 3 px outlier rejection."""
    ang = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    rr = R + np.arange(-half, half + 1.0)
    xs, ys = CX + np.outer(np.cos(ang), rr), CY + np.outer(np.sin(ang), rr)
    pk = rr[cv2.remap(img, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR).argmax(1)]
    px, py = CX + pk * np.cos(ang), CY + pk * np.sin(ang)
    keep = np.ones(len(ang), bool)
    for _ in range(4):
        (a, b, c), *_ = np.linalg.lstsq(np.c_[2 * px[keep], 2 * py[keep], np.ones(keep.sum())],
                                        px[keep] ** 2 + py[keep] ** 2, rcond=None)
        r = np.sqrt(c + a * a + b * b)
        keep = np.abs(np.hypot(px - a, py - b) - r) < 3
    return a, b, r


def disk_check() -> dict:
    """Deviation of the estimated disk (cx, cy, r) from (CX, CY, R) over the 707 train and 180 test
    images, for both estimators: 'bbox' (largest component of img > 30) and 'ring' (limb-ring fit)."""
    paths = sorted(TRAIN_IMG.glob("*.jpeg")) + sorted(TEST_IMG.glob("*.jpeg"))
    est = {"bbox": _bbox_disk, "ring": fit_limb_ring}
    dev = {k: [] for k in est}
    for p in paths:
        img = load_image(p)
        for k, fn in est.items():
            dev[k].append(np.subtract(fn(img), (CX, CY, R)))
    out = {"n": len(paths)}
    for k, d in dev.items():
        d = np.abs(np.array(d))
        worst = d.max(1)
        out[k] = {"worst": round(float(worst.max()), 2), "worst_image": paths[int(worst.argmax())].name,
                  "worst_cx_cy_r": d.max(0).round(2).tolist(), "n_over_5px": int((worst > 5).sum())}
    return out


def _main(argv: list[str]) -> None:
    if argv == ["folds"]:
        n = {s: len(r) for s, r in load_records().items()}
        FOLDS_JSON.write_text(json.dumps(make_folds(n), indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {FOLDS_JSON}")
    elif argv == ["diskcheck"]:
        print(json.dumps(disk_check(), indent=1))
    else:
        sys.exit("usage: python -m filseg.data folds | diskcheck")


if __name__ == "__main__":
    _main(sys.argv[1:])
