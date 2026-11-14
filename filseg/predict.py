"""Whole-image flip-TTA probability maps; `python -m filseg.predict --run A --fold 0` writes the fold's npz files."""
import argparse
import json
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from filseg import data as D
from filseg.config import load_cfg
from filseg.model import gpu_stats, load_model, run_dir
from filseg.train import FULL_FOLD

FLIPS = [(), (3,), (2,), (2, 3)]        # views on (N, C, H, W): as is, left-right, up-down, both


@torch.no_grad()
def predict_prob(models: list, img: np.ndarray, work_res: int | None = None, device: str = "cpu") -> np.ndarray:
    """uint8 image -> float32 map at the image's own size: mean of sigmoid over models x 4 flip views."""
    size, kind = img.shape[0], torch.device(device).type
    if work_res and work_res != size:       # the route of data.build_cache: area-resize the uint8, then normalise
        img = cv2.resize(img, (work_res, work_res), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(D.normalize(img))[None, None].to(device)
    acc = torch.zeros_like(x)
    for m in models:
        m.to(device).eval()
        if kind == "cuda": m.to(memory_format=torch.channels_last)
        for dims in FLIPS:
            with torch.autocast(kind, dtype=torch.float16, enabled=kind == "cuda"):
                logit = m(x.flip(dims))
            acc += torch.sigmoid(logit.float()).flip(dims)       # flip the view back before averaging
    acc /= len(models) * len(FLIPS)
    if acc.shape[-1] != size:
        acc = F.interpolate(acc, size=(size, size), mode="bilinear", align_corners=False)
    return acc[0, 0].cpu().numpy()


def quantize(p: np.ndarray) -> np.ndarray:
    """uint8 round(p * 255), values under 8 set to 0."""
    q = np.round(p * 255).astype(np.uint8)
    q[q < 8] = 0
    return q


def predict_fold(cfg, device: str) -> None:
    """Reload <run>/fold<k>.pt; write oof_fold<k>.npz (the fold's validation stems; none for fold 5) and test_fold<k>.npz (all test stems)."""
    out = run_dir(cfg.run)
    folds = json.loads(D.FOLDS_JSON.read_text(encoding="utf-8"))
    sets = {"oof": (sorted(s for s, k in folds.items() if k == cfg.fold), D.TRAIN_IMG),
            "test": (sorted(p.stem for p in D.TEST_IMG.glob("*.jpeg")), D.TEST_IMG)}
    if cfg.fold == FULL_FOLD: del sets["oof"]               # the full-data model has no held-out stems
    m = load_model(out / f"fold{cfg.fold}.pt", device)
    print(f"predict {cfg.run} fold {cfg.fold}: work_res {cfg.work_res}, device {device}, "
          f"{ {k: len(v[0]) for k, v in sets.items()} } images, outputs in {out}", flush=True)
    if torch.device(device).type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    t0, n = time.time(), 0
    for name, (stems, folder) in sets.items():
        if not stems: raise SystemExit(f"no {name} images (data root {D.DATA_ROOT})")
        prob = np.zeros((len(stems), D.H, D.W), np.uint8)
        for i, s in enumerate(stems):
            p = predict_prob([m], D.load_image(folder / f"{s}.jpeg"), cfg.work_res, device)
            assert p.shape == (D.H, D.W), (s, p.shape)
            prob[i] = quantize(p); n += 1
            if i % 20 == 0 or i == len(stems) - 1:
                print(f"  {name} {i + 1}/{len(stems)} {s}: {(time.time() - t0) / n:.2f} s/image", flush=True)
        np.savez_compressed(out / f"{name}_fold{cfg.fold}.npz", stems=np.array(stems), prob=prob)
        print(f"  wrote {name}_fold{cfg.fold}.npz ({(out / f'{name}_fold{cfg.fold}.npz').stat().st_size / 1e6:.0f} MB)", flush=True)
    wall = time.time() - t0
    stats = {**gpu_stats(device), "n_images": n, "sec_per_image": wall / n, "wall_s": wall}
    (out / f"fold{cfg.fold}_predict.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print("predict done", stats, flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True); ap.add_argument("--fold", type=int, required=True)
    a = ap.parse_args(argv)
    if torch.cuda.device_count() < 1: sys.exit("no CUDA device: GPU prediction runs on Kaggle, one process per GPU")
    predict_fold(load_cfg(a.run, a.fold), "cuda")


if __name__ == "__main__":
    main()
