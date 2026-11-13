"""No-training baseline: the dark residual of the normalised disk as a probability map, written as a run."""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from filseg import data as D
from filseg.instances import RUNS
from filseg.preflight import _git

SIGMA, SCALE, MIN_U8 = 40, 5, 8     # blur radius (px), residual that counts as p = 1 (z units), weakest kept uint8


def dark_residual(img: np.ndarray) -> np.ndarray:
    """float32 in [0, 1]: (Gaussian blur of the z-scored image, sigma 40 px) minus the image, over 5, clipped;
    0 off the disk. Dark structure on a brighter neighbourhood scores high."""
    x = D.normalize(img)
    m = np.clip((cv2.GaussianBlur(x, (0, 0), SIGMA) - x) / SCALE, 0, 1)
    m[D._disk_masks(img.shape[0])[1]] = 0
    return m


def run(root=RUNS, name="classical", folds: dict[str, int] | None = None, img_dir=D.TRAIN_IMG) -> None:
    """<root>/<name>/oof_fold<k>.npz (keys stems, prob: uint8 round(p * 255), values < 8 set to 0, the format
    of a Kaggle run) and meta.json. folds: stem -> fold, default folds.json. One fold is in memory at a time."""
    t0 = time.time()
    folds = json.loads(D.FOLDS_JSON.read_text(encoding="utf-8")) if folds is None else folds
    out = Path(root) / name
    out.mkdir(parents=True, exist_ok=True)
    for k in sorted(set(folds.values())):
        stems = sorted(s for s, f in folds.items() if f == k)
        prob = np.empty((len(stems), D.H, D.W), np.uint8)
        for i, s in enumerate(stems):
            p = np.round(dark_residual(D.load_image(Path(img_dir) / f"{s}.jpeg")) * 255).astype(np.uint8)
            p[p < MIN_U8] = 0
            prob[i] = p
        np.savez_compressed(out / f"oof_fold{k}.npz", stems=np.array(stems), prob=prob)
        print(f"fold {k}: {len(stems)} maps, {time.time() - t0:.0f} s", flush=True)
    meta = dict(commit=_git(["rev-parse", "HEAD"], D.REPO), dirty=_git(["status", "--porcelain"], D.REPO) != "",
                parameters=dict(sigma=SIGMA, scale=SCALE, min_u8=MIN_U8), seconds=round(time.time() - t0, 1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(meta))


if __name__ == "__main__":
    run()
