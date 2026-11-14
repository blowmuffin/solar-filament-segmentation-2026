import copy
import json

import cv2
import numpy as np
import pytest
import torch

from filseg import data as D
from filseg import model, predict, train
from filseg.config import Cfg, load_cfg


def test_fp16_roundtrip_cpu(tmp_path):
    m = model.build_model(pretrained=False).eval(); model.save_fp16(m, tmp_path / "w.pt")
    assert (tmp_path / "w.pt").stat().st_size < 60e6
    m2 = model.load_model(tmp_path / "w.pt", "cpu"); x = torch.randn(1, 1, 256, 256)
    assert next(m2.parameters()).dtype == torch.float32 and not m2.training
    for (k, v), v2 in zip(m.state_dict().items(), m2.state_dict().values()):      # exactly the fp16 rounding, nothing else
        assert torch.equal(v2, v.half().float() if v.is_floating_point() else v), k
    with torch.no_grad(): a, b = m(x), m2(x)
    # The brief's absolute 0.05 is unreachable here: an untrained net's logits are ~100 (BN stats 0/1) and it amplifies
    # a 5e-4 relative weight change to ~0.3 on its own, so the output check is relative to the logit scale.
    assert (a - b).abs().max() < 0.05 * a.abs().max()


class Times10(torch.nn.Module):
    def forward(self, x): return 10 * x


def test_tta_views_are_inverted():            # a view that is not flipped back would average mirrored copies
    img = np.random.default_rng(0).integers(0, 255, (256, 256), dtype=np.uint8)
    want = 1 / (1 + np.exp(-10 * D.normalize(img)))
    assert np.abs(predict.predict_prob([Times10()], img) - want).max() < 1e-3


def test_predict_prob_work_res_matches_training_cache():
    """Input = the cache route (INTER_AREA on uint8, then normalise); output back at the image's size."""
    class Probe(torch.nn.Module):
        seen = None
        def forward(self, x):
            if self.seen is None: self.seen = x.clone()      # first view is the unflipped one
            return x
    img = np.random.default_rng(1).integers(0, 255, (256, 256), dtype=np.uint8)
    probe = Probe(); p = predict.predict_prob([probe], img, work_res=128)
    want = D.normalize(cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA))
    assert probe.seen.shape == (1, 1, 128, 128) and np.abs(probe.seen[0, 0].numpy() - want).max() < 1e-6
    assert p.shape == (256, 256) and p.dtype == np.float32 and 0 < p.min() and p.max() < 1


def test_quantize():
    q = predict.quantize(np.array([0.0, 0.02, 0.03, 0.5, 1.0], np.float32))
    assert q.dtype == np.uint8 and q.tolist() == [0, 0, 8, 128, 255]     # 5 < 8 -> 0; 7.65 -> 8 stays


def test_lr_schedule_and_seed():
    f = lambda s: train.lr_mult(s, 10, 40)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0) and f(10) == pytest.approx(1.0)
    assert all(f(s) > f(s + 1) for s in range(10, 39)) and 0 < f(39) < 0.01
    assert train.lr_mult(29, 30, 30) == 1.0              # smoke run: one epoch = warm-up only, no division by zero
    assert train.job_seed(load_cfg("A2", 3)) == 1003 and train.job_seed(load_cfg("A", 0)) == 0


def test_loss_fn():
    rng = np.random.default_rng(0)
    z, t = rng.normal(size=(2, 1, 8, 8)).astype(np.float32), rng.random((2, 1, 8, 8)).astype(np.float32)
    p = 1 / (1 + np.exp(-z.astype(np.float64)))
    bce = -(t * np.log(p) + (1 - t) * np.log(1 - p)).mean()
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    got = train.loss_fn(torch.from_numpy(z), torch.from_numpy(t), 0.5).item()
    assert got == pytest.approx(bce + 0.5 * dice, abs=1e-5)
    hard = torch.zeros(1, 1, 8, 8); hard[..., :4] = 1
    assert train.loss_fn(40 * (2 * hard - 1), hard, 0.5) < 1e-3 < train.loss_fn(-40 * (2 * hard - 1), hard, 0.5)


def test_ema_update():
    m = model.build_model(pretrained=False); e = copy.deepcopy(m).eval()
    old = {k: v.clone() for k, v in e.state_dict().items()}
    with torch.no_grad():                                 # stand-in for training: move every float tensor by +1
        for v in m.state_dict().values():
            if v.is_floating_point(): v.add_(1.0)
            else: v.add_(5)
    train.ema_update(e, m, 0.9)
    for k, v in e.state_dict().items():
        if v.is_floating_point(): assert torch.allclose(v, 0.9 * old[k] + 0.1 * (old[k] + 1), atol=1e-5), k
        else: assert torch.equal(v, m.state_dict()[k]), k           # num_batches_tracked is copied
    assert not e.training


def fake_cache(n=2, res=256):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (n, res, res), dtype=np.uint8)
    soft = np.zeros_like(img); soft[0, 100:110, 40:200] = 128; soft[1, 30:40, 30:200] = 255
    return img, soft


def test_crops_foreground_alignment_and_augmentation():
    img, soft = fake_cache()
    # 1. fg_prob=1: every crop holds filament pixels (jitter 16 px < half crop 32 px, also at the borders)
    cfg = Cfg(run="t", fold=0, work_res=256, crop=64, fg_prob=1.0)
    ds = train.Crops(img, soft, [0, 1], cfg, seed=0); rng = np.random.default_rng(0)
    for _ in range(100):
        x, t = ds.sample(rng)
        assert x.shape == t.shape == (1, 64, 64) and x.dtype == t.dtype == torch.float32 and t.max() > 0
    # 2. image and target come from the same coordinates under the same flips/turns: soft == img  =>  x is an
    #    increasing affine function of t wherever x was not zeroed outside the disk
    ds = train.Crops(img, img, [0, 1], Cfg(run="t", fold=0, work_res=256, crop=64, fg_prob=0.5), seed=0)
    for _ in range(100):
        x, t = ds.sample(rng); keep = x != 0
        assert keep.sum() > 50 and np.corrcoef(x[keep].numpy(), t[keep].numpy())[0, 1] > 0.9999
    # 3. no augmentation, whole-image crop: exactly D.normalize and soft/255
    ds = train.Crops(img, soft, [1], Cfg(run="t", fold=0, work_res=256, crop=256), seed=0)
    x, t = ds.sample(rng, aug=False)
    assert np.abs(x[0].numpy() - D.normalize(img[1])).max() < 1e-4 and np.allclose(t[0].numpy(), soft[1] / 255, atol=1e-6)
    # 4. augmentation is not the identity and the intensity change stays inside U(0.9, 1.1), U(-0.1, 0.1)
    x2, _ = ds.sample(np.random.default_rng(5), aug=True)
    assert not torch.allclose(x, x2)
    # 5. uniform-on-disk crops stay inside the image
    ds = train.Crops(img, soft, [0, 1], Cfg(run="t", fold=0, work_res=256, crop=64, fg_prob=0.0), seed=0)
    assert all(ds.sample(rng)[0].shape == (1, 64, 64) for _ in range(300))


def test_train_pieces_cpu(tmp_path):
    """Everything the training loop does except the optimiser step: sampler, augmentation, loss, forward, backward,
    EMA on a copy, fp16 save/load, whole-image prediction. Never calls optimizer.step()."""
    img, soft = fake_cache()
    cfg = Cfg(run="t", fold=0, work_res=256, crop=64, batch=2, fg_prob=0.7)
    ds = train.Crops(img, soft, [0, 1], cfg, seed=0)
    x, t = next(iter(torch.utils.data.DataLoader(ds, batch_size=cfg.batch, num_workers=0)))
    assert x.shape == t.shape == (2, 1, 64, 64) and x.dtype == torch.float32 and 0 <= t.min() and t.max() <= 1
    m = model.build_model(pretrained=False).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(m).eval().requires_grad_(False)
    loss = train.forward_loss(m, x.contiguous(memory_format=torch.channels_last), t, cfg.dice_w, amp=False)
    assert loss.dtype == torch.float32 and loss.ndim == 0 and torch.isfinite(loss) and 0 < loss < 5
    loss.backward()
    g = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(g) == len(list(m.parameters())) and all(torch.isfinite(a).all() for a in g) and any(a.abs().max() > 0 for a in g)
    train.ema_update(ema, m, cfg.ema)
    model.save_fp16(ema, tmp_path / "e.pt"); m2 = model.load_model(tmp_path / "e.pt", "cpu")
    assert not m2.training and next(m2.parameters()).dtype == torch.float32
    p = predict.predict_prob([m2], img[0], work_res=128)
    assert p.shape == (256, 256) and p.dtype == np.float32 and 0 <= p.min() and p.max() <= 1
    assert predict.quantize(p).dtype == np.uint8


class NoStepAdamW(torch.optim.AdamW):
    def step(self, closure=None): pass                     # weights never change: no optimiser step runs locally


def test_train_loop_dry_cpu(tmp_path, monkeypatch):
    """The real train() on a tiny cache, CPU, with the optimiser step stubbed out: loop, schedule, logs, EMA, save."""
    stems = [f"s{i}" for i in range(6)]
    (tmp_path / "cache").mkdir(); runs = tmp_path / "runs"
    for name, a in zip(("img", "soft"), fake_cache(6)): a.tofile(tmp_path / "cache" / f"{name}_256.u8")
    (tmp_path / "cache" / "stems.json").write_text(json.dumps(stems), encoding="utf-8")
    folds = tmp_path / "folds.json"; folds.write_text(json.dumps({s: i // 2 for i, s in enumerate(stems)}), encoding="utf-8")
    monkeypatch.setattr(D, "CACHE_ROOT", tmp_path / "cache"); monkeypatch.setattr(D, "FOLDS_JSON", folds)
    monkeypatch.setenv("FILSEG_RUNS", str(runs)); monkeypatch.setattr(torch.optim, "AdamW", NoStepAdamW)
    monkeypatch.setattr(train, "N_VAL", 6)
    cfg = Cfg(run="t", fold=1, work_res=256, crop=64, batch=2, steps=4, epochs=3, ema=0.9)     # 12 steps > the 10 warm-up steps
    train.train(cfg, device="cpu", pretrained=False, workers=0)
    out = runs / "t"
    rows = (out / "fold1_log.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "epoch,train_loss,val_loss,seconds" and len(rows) == 4 and rows[3].startswith("3,")
    assert all(np.isfinite(float(v)) for r in rows[1:] for v in r.split(","))
    st = json.loads((out / "fold1_train.json").read_text(encoding="utf-8"))
    assert st["steps"] == 12 and st["sec_per_step"] > 0 and st["sec_per_step_steady"] > 0 and {"gpu", "peak_mem_gb"} <= set(st)
    assert next(model.load_model(out / "fold1.pt", "cpu").parameters()).dtype == torch.float32
    with pytest.raises(SystemExit): train.train(cfg, device="cpu", pretrained=False, workers=0)   # never overwrite


def test_train_full_data_fold5_cpu(tmp_path, monkeypatch, capsys):
    """Fold 5 = every stem trains, no validation set (val_loss is NaN), fold5.pt written; fold 6 still fails loudly."""
    stems = [f"s{i}" for i in range(6)]
    (tmp_path / "cache").mkdir(); runs = tmp_path / "runs"
    for name, a in zip(("img", "soft"), fake_cache(6)): a.tofile(tmp_path / "cache" / f"{name}_256.u8")
    (tmp_path / "cache" / "stems.json").write_text(json.dumps(stems), encoding="utf-8")
    folds = tmp_path / "folds.json"; folds.write_text(json.dumps({s: i // 2 for i, s in enumerate(stems)}), encoding="utf-8")
    monkeypatch.setattr(D, "CACHE_ROOT", tmp_path / "cache"); monkeypatch.setattr(D, "FOLDS_JSON", folds)
    monkeypatch.setenv("FILSEG_RUNS", str(runs)); monkeypatch.setattr(torch.optim, "AdamW", NoStepAdamW)
    kw = dict(run="t", work_res=256, crop=64, batch=2, steps=2, epochs=1, ema=0.9)
    train.train(Cfg(fold=5, **kw), device="cpu", pretrained=False, workers=0)
    assert "6 train / 0 validation images" in capsys.readouterr().out
    out = runs / "t"
    rows = (out / "fold5_log.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "epoch,train_loss,val_loss,seconds" and len(rows) == 2 and rows[1].split(",")[2] == "nan"
    assert (out / "fold5_train.json").exists() and (out / "fold5.pt").stat().st_size > 0
    with pytest.raises(AssertionError): train.train(Cfg(fold=6, **kw), device="cpu", pretrained=False, workers=0)


def test_predict_full_data_fold5_cpu(tmp_path, monkeypatch):
    """Fold 5 writes test_fold5.npz and the predict json only: no held-out stems, so no oof file."""
    (tmp_path / "test").mkdir()
    for s in ("t1", "t0"):
        cv2.imwrite(str(tmp_path / "test" / f"{s}.jpeg"), np.random.default_rng(0).integers(0, 255, (2048, 2048), dtype=np.uint8))
    folds = tmp_path / "folds.json"; folds.write_text(json.dumps({"a": 0, "b": 1}), encoding="utf-8")
    monkeypatch.setattr(D, "TEST_IMG", tmp_path / "test"); monkeypatch.setattr(D, "FOLDS_JSON", folds)
    monkeypatch.setenv("FILSEG_RUNS", str(tmp_path / "runs"))
    model.save_fp16(model.build_model(pretrained=False), tmp_path / "runs" / "t" / "fold5.pt")
    predict.predict_fold(Cfg(run="t", fold=5, work_res=256), device="cpu")
    got = sorted(p.name for p in (tmp_path / "runs" / "t").iterdir())
    assert got == ["fold5.pt", "fold5_predict.json", "test_fold5.npz"]
    assert np.load(tmp_path / "runs" / "t" / "test_fold5.npz")["stems"].tolist() == ["t0", "t1"]


def test_predict_fold_cpu(tmp_path, monkeypatch):
    """predict_fold on fake 2048 px images: file names, keys, dtypes, shapes, stems, fold selection."""
    for d in ("train", "test"):
        (tmp_path / d).mkdir()
        for s in (("a", "b") if d == "train" else ("t1", "t0")):
            cv2.imwrite(str(tmp_path / d / f"{s}.jpeg"), np.random.default_rng(0).integers(0, 255, (2048, 2048), dtype=np.uint8))
    folds = tmp_path / "folds.json"; folds.write_text(json.dumps({"a": 0, "b": 1}), encoding="utf-8")
    monkeypatch.setattr(D, "TRAIN_IMG", tmp_path / "train"); monkeypatch.setattr(D, "TEST_IMG", tmp_path / "test")
    monkeypatch.setattr(D, "FOLDS_JSON", folds); monkeypatch.setenv("FILSEG_RUNS", str(tmp_path / "runs"))
    model.save_fp16(model.build_model(pretrained=False), tmp_path / "runs" / "t" / "fold0.pt")
    predict.predict_fold(Cfg(run="t", fold=0, work_res=256), device="cpu")
    oof, test = (np.load(tmp_path / "runs" / "t" / f"{n}_fold0.npz") for n in ("oof", "test"))
    assert oof["stems"].tolist() == ["a"] and test["stems"].tolist() == ["t0", "t1"]
    for z, n in ((oof, 1), (test, 2)):
        assert z["prob"].dtype == np.uint8 and z["prob"].shape == (n, 2048, 2048)
        assert z["prob"].max() > 0 and ((z["prob"] == 0) | (z["prob"] >= 8)).all()
    st = json.loads((tmp_path / "runs" / "t" / "fold0_predict.json").read_text(encoding="utf-8"))
    assert st["n_images"] == 3 and st["sec_per_image"] > 0
