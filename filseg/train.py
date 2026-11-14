"""One-fold training (PROTOCOL section 6): `python -m filseg.train --run A --fold 0` -> <run>/fold0.pt + fold0_log.csv."""
import argparse
import copy
import csv
import json
import math
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from filseg import data as D
from filseg.config import load_cfg
from filseg.model import build_model, gpu_stats, run_dir, save_fp16

N_VAL = 200         # fixed validation crops, for the log only


FULL_FOLD = 5       # fold index 5 = train on every stem, no validation (folds.json holds folds 0-4)


def job_seed(cfg) -> int:
    return 1000 * cfg.seed + cfg.fold      # PROTOCOL 4: 1000 x run index + fold


def open_cache(res: int):
    """(stems, img, soft): read-only np.memmaps (n, res, res) written by data.build_cache."""
    stems = json.loads((D.CACHE_ROOT / "stems.json").read_text(encoding="utf-8"))
    shape, mm = (len(stems), res, res), {}
    for name in ("img", "soft"):
        f = D.CACHE_ROOT / f"{name}_{res}.u8"
        if f.stat().st_size != np.prod(shape): raise ValueError(f"{f}: {f.stat().st_size} bytes, expected {shape}")
        mm[name] = np.memmap(f, np.uint8, "r", shape=shape)
    return stems, mm["img"], mm["soft"]


class Crops(torch.utils.data.IterableDataset):
    """Endless (image crop, soft-target crop) pairs, each (1, crop, crop) float32, from the work_res cache."""

    def __init__(self, img, soft, idx, cfg, seed):
        self.img, self.soft, self.idx, self.cfg, self.seed = img, soft, list(idx), cfg, seed
        res = img.shape[1]
        self.inner, self.outer = D._disk_masks(res)
        self.jitter = round(128 * res / 2048)                   # 128 px at 2048 scale
        rng, self.mean, self.std, self.fg = np.random.default_rng(seed), [], [], []
        for i in self.idx:                                      # once per image: z-score stats and filament pixels
            v = np.asarray(img[i])[self.inner]
            self.mean.append(float(v.mean(dtype=np.float64))); self.std.append(float(v.std(dtype=np.float64)))
            yx = np.argwhere(np.asarray(soft[i]) > 0)
            self.fg.append(yx[rng.choice(len(yx), min(len(yx), 2000), replace=False)])   # a subsample is enough

    def origin(self, rng, k):
        """Top-left corner of a crop for image k: centred on a random filament pixel (+-jitter) with probability
        fg_prob, else on a uniform point of the disk; clamped to lie inside the image."""
        res, c, fg = self.img.shape[1], self.cfg.crop, self.fg[k]
        if len(fg) and rng.random() < self.cfg.fg_prob:
            centre = fg[rng.integers(len(fg))] + rng.integers(-self.jitter, self.jitter + 1, 2)
        else:
            s, r, a = res / 2048, D.R * math.sqrt(rng.random()) * res / 2048, 2 * math.pi * rng.random()
            centre = (D.CY * s + r * math.sin(a), D.CX * s + r * math.cos(a))
        return np.clip(np.rint(centre).astype(int) - c // 2, 0, res - c)

    def sample(self, rng, aug=True):
        k = int(rng.integers(len(self.idx))); i, c = self.idx[k], self.cfg.crop
        y0, x0 = self.origin(rng, k)
        sl = (slice(y0, y0 + c), slice(x0, x0 + c))
        x = (self.img[i][sl].astype(np.float32) - self.mean[k]) / self.std[k]      # same z-score as data.normalize
        t = self.soft[i][sl].astype(np.float32) / 255                               # same coordinates as the image
        if aug:
            x = x * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
        x[self.outer[sl]] = 0                                                       # black sky stays 0, as at prediction
        if aug:                                  # one draw each, applied to image and target alike
            if rng.random() < 0.5: x, t = x[:, ::-1], t[:, ::-1]
            if rng.random() < 0.5: x, t = x[::-1], t[::-1]
            n = int(rng.integers(4)); x, t = np.rot90(x, n), np.rot90(t, n)
        return tuple(torch.from_numpy(np.ascontiguousarray(a, np.float32))[None] for a in (x, t))

    def __iter__(self):
        w = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, 0 if w is None else w.id])         # distinct stream per worker
        while True:
            yield self.sample(rng)


def loss_fn(logits, target, dice_w):
    """BCE + dice_w x soft Dice (batch-wide, +1 smoothing) against the soft target; always fp32."""
    logits = logits.float()
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + dice_w * dice


def forward_loss(model, x, t, dice_w, amp):
    with torch.autocast("cuda" if amp else "cpu", dtype=torch.float16, enabled=amp):
        logits = model(x)
    return loss_fn(logits, t, dice_w)


def lr_mult(step, warm, total):
    """Linear warm-up over `warm` steps, then cosine decay to 0 at `total`."""
    if step < warm: return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


@torch.no_grad()
def ema_update(ema, model, decay):
    """EMA of every float tensor (weights and BN running stats); integer buffers are copied."""
    live = model.state_dict()
    for k, v in ema.state_dict().items():
        if v.is_floating_point(): v.mul_(decay).add_(live[k], alpha=1 - decay)
        else: v.copy_(live[k])


@torch.no_grad()
def validate(model, vx, vt, batch, dice_w, amp, device):
    total = 0.0
    for i in range(0, len(vx), batch):
        x = vx[i:i + batch].to(device).contiguous(memory_format=torch.channels_last)
        total += forward_loss(model, x, vt[i:i + batch].to(device), dice_w, amp).item() * len(x)
    return total / len(vx)


def train(cfg, device="cuda", pretrained=True, workers=2) -> None:
    amp = torch.device(device).type == "cuda"
    out = run_dir(cfg.run); out.mkdir(parents=True, exist_ok=True)
    pt, f = out / f"fold{cfg.fold}.pt", out / f"fold{cfg.fold}"
    if pt.exists(): raise SystemExit(f"{pt} exists: run outputs are never overwritten, use a new run name")
    seed = job_seed(cfg); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = amp                        # fixed crop size
    print(f"config {json.dumps(asdict(cfg))}\nseed {seed}, device {device}, torch {torch.__version__}, "
          f"gpu {torch.cuda.get_device_name(device) if amp else 'none'}, workers {workers}, outputs in {out}", flush=True)

    stems, img, soft = open_cache(cfg.work_res)
    folds = json.loads(D.FOLDS_JSON.read_text(encoding="utf-8"))
    assert set(stems) == set(folds), "cache stems differ from folds.json"
    tr = [i for i, s in enumerate(stems) if folds[s] != cfg.fold]
    va = [i for i, s in enumerate(stems) if folds[s] == cfg.fold]
    assert tr and (va or cfg.fold == FULL_FOLD), f"fold {cfg.fold}: {len(tr)} train / {len(va)} validation images"
    print(f"cache {D.CACHE_ROOT}: img {img.shape} {img.nbytes / 1e9:.2f} GB, soft {soft.shape} {soft.nbytes / 1e9:.2f} GB; "
          f"{len(tr)} train / {len(va)} validation images", flush=True)
    t0 = time.time()
    train_ds = Crops(img, soft, tr, cfg, seed)
    if va:
        val_ds, val_rng = Crops(img, soft, va, cfg, seed), np.random.default_rng(12345)     # the same N_VAL crops every epoch
        vx, vt = (torch.stack(a) for a in zip(*(val_ds.sample(val_rng, aug=False) for _ in range(N_VAL))))
    print(f"sampler stats and {N_VAL} validation crops ready in {time.time() - t0:.0f} s", flush=True)

    model = build_model(cfg.encoder, pretrained).to(device).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch, num_workers=workers, pin_memory=amp,
                                         prefetch_factor=4 if workers else None,
                                         multiprocessing_context="fork" if workers else None)
    batches, total, step, train_s = iter(loader), cfg.epochs * cfg.steps, 0, 0.0
    warm, t_warm, val_s, steady = 10, None, 0.0, None           # steady-state s/step skips the first `warm` steps
    if amp: torch.cuda.reset_peak_memory_stats(device)
    t_all = time.time()
    with open(f"{f}_log.csv", "w", encoding="utf-8", newline="") as fh:
        log = csv.writer(fh); log.writerow(["epoch", "train_loss", "val_loss", "seconds"]); fh.flush()
        model.train()
        for epoch in range(1, cfg.epochs + 1):
            t0, run_loss = time.time(), torch.zeros((), device=device)
            for _ in range(cfg.steps):
                if step == warm:                                # clock starts after cudnn autotune and worker start-up
                    if amp: torch.cuda.synchronize(device)
                    t_warm = time.time()
                for g in opt.param_groups: g["lr"] = cfg.lr * lr_mult(step, cfg.steps, total)     # 1 warm-up epoch
                x, t = (a.to(device, non_blocking=True) for a in next(batches))
                loss = forward_loss(model, x.contiguous(memory_format=torch.channels_last), t, cfg.dice_w, amp)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                ema_update(ema, model, cfg.ema)
                run_loss += loss.detach(); step += 1
            train_loss = run_loss.item() / cfg.steps            # .item() waits for the GPU, so the clock is honest
            if not math.isfinite(train_loss): raise RuntimeError(f"epoch {epoch}: train loss is {train_loss}")
            epoch_train_s = time.time() - t0; train_s += epoch_train_s
            if t_warm is not None: steady = (time.time() - t_warm - val_s) / (step - warm)
            tv = time.time(); val_loss = validate(ema, vx, vt, cfg.batch, cfg.dice_w, amp, device) if va else float('nan'); val_s += time.time() - tv
            log.writerow([epoch, f"{train_loss:.5f}", f"{val_loss:.5f}", f"{time.time() - t0:.1f}"]); fh.flush()
            print(f"epoch {epoch}/{cfg.epochs} train_loss {train_loss:.4f} val_loss(ema) {val_loss:.4f} "
                  f"lr {opt.param_groups[0]['lr']:.2e} {epoch_train_s / cfg.steps:.3f} s/step "
                  f"epoch {time.time() - t0:.0f} s, total {time.time() - t_all:.0f} s", flush=True)
    del batches, loader
    save_fp16(ema, pt)                                          # the last epoch's averaged weights are the model
    stats = {**gpu_stats(device), "steps": total, "sec_per_step": train_s / total, "sec_per_step_steady": steady,
             "train_s": train_s, "wall_s": time.time() - t_all}
    (out / f"fold{cfg.fold}_train.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"saved {pt} ({pt.stat().st_size / 1e6:.0f} MB)\ntrain done {stats}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True); ap.add_argument("--fold", type=int, required=True)
    a = ap.parse_args(argv)
    if torch.cuda.device_count() < 1: sys.exit("no CUDA device: training runs on Kaggle, one process per GPU")
    train(load_cfg(a.run, a.fold))


if __name__ == "__main__":
    main()
