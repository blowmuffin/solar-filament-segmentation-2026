import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_util

from filseg import infer, instances, model, predict
from filseg.instances import Rule


class Affine(torch.nn.Module):
    """logit = a * (x - b) on the z-scored image: sky (x = 0) is p ~ 0, a bright blob is p = 1."""
    def __init__(self, a, b):
        super().__init__(); self.a, self.b = a, b
    def forward(self, x): return self.a * (x - self.b)


def test_ensemble_prob_equals_the_npz_route(tmp_path):
    """The notebook's map must be the one the CSV route (iter_probs over each model's quantised npz) builds,
    also when the models run at different work_res."""
    img = np.random.default_rng(0).integers(0, 255, (2048, 2048), dtype=np.uint8)
    members = []
    for run, res, seed in (("r1", 2048, 0), ("r2", 1024, 1)):
        torch.manual_seed(seed)
        m = torch.nn.Conv2d(1, 1, 3, padding=1).eval()
        members.append(({"run": run, "work_res": res}, m))
        (tmp_path / run).mkdir()
        q = predict.quantize(predict.predict_prob([m], img, res))
        np.savez(tmp_path / run / "test_fold0.npz", stems=np.array(["t0"]), prob=q[None])
    (stem, want), = instances.iter_probs(["r1", "r2"], "test", root=tmp_path)
    got = infer.ensemble_prob(members, img)
    assert got.dtype == np.float32 and np.array_equal(got, want) and 0 < got.max() < 1 and len(set(np.unique(got))) > 10


def test_segment_applies_the_rule_end_to_end():
    img = np.full((2048, 2048), 100, np.uint8)
    img[1000:1040, 1000:1100] = 255            # 4000 px, disk centre: kept
    img[1200:1210, 1200:1210] = 255            # 100 px: under min_area
    img[1000:1040, 1850:1900] = 255            # r about 0.95 R: past the limb cut
    seg = infer.segment([({"work_res": 2048}, Affine(20, 3))], img, Rule())
    assert len(seg.cands) == 3 and len(seg.kept) == 1 and len(seg.rles) == 1
    m = mask_util.decode(seg.rles[0])
    assert np.array_equal(m.astype(bool), seg.labels == seg.kept[0]) and m.sum() == 4000 and m[1000:1040, 1000:1100].all()


def _fake_run(root, run, fold, cfg=None, encoder="resnet18"):
    """A downloaded run folder: fold<k>.pt plus, with cfg, the job's fold<k>_meta.json."""
    root.joinpath(run).mkdir(parents=True, exist_ok=True)
    model.save_fp16(model.build_model(encoder, pretrained=False), root / run / f"fold{fold}.pt")
    if cfg:
        (root / run / f"fold{fold}_meta.json").write_text(json.dumps({"ok": True, "commit": "abc123", "cfg": cfg}), encoding="utf-8")


def test_manifest_roundtrip_and_checks(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "weights"
    _fake_run(runs, "t", 0, cfg={"encoder": "resnet18", "work_res": 128})
    man = infer.make_manifest(["t"], out, folds=[0], runs_root=runs)
    assert [(e["file"], e["run"], e["fold"], e["encoder"], e["work_res"], e["commit"]) for e in man["models"]] == [
        ("t_fold0.pt", "t", 0, "resnet18", 128, "abc123")]
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == man
    (entry, m), = infer.load_models(out)
    assert entry == man["models"][0] and not m.training and next(m.parameters()).dtype == torch.float32
    with open(out / "t_fold0.pt", "ab") as f: f.write(b"x")                  # a changed file is refused
    with pytest.raises(ValueError, match="SHA-256"): infer.load_models(out)
    with pytest.raises(FileNotFoundError, match="infer --run"): infer.load_models(tmp_path / "nowhere")


def test_manifest_without_meta_file_reads_runs_json(tmp_path):
    (tmp_path / "smokeB").mkdir(); (tmp_path / "smokeB" / "fold2.pt").write_bytes(b"not a model, never loaded")
    e = infer.make_manifest(["smokeB"], tmp_path / "w", folds=[2], runs_root=tmp_path)["models"][0]
    assert (e["file"], e["fold"], e["encoder"], e["work_res"], e["commit"]) == ("smokeB_fold2.pt", 2, "resnet34", 1024, None)


def test_manifest_refuses_failed_job_and_oversize(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    _fake_run(runs, "t", 0, cfg={"encoder": "resnet18", "work_res": 128})
    meta = runs / "t" / "fold0_meta.json"
    meta.write_text(json.dumps({"ok": False, "cfg": {"encoder": "resnet18", "work_res": 128}}), encoding="utf-8")
    with pytest.raises(ValueError, match="did not finish"): infer.make_manifest(["t"], tmp_path / "w", [0], runs)
    meta.write_text(json.dumps({"ok": True, "cfg": {"encoder": "resnet18", "work_res": 128}}), encoding="utf-8")
    monkeypatch.setattr(infer, "MAX_FILE_BYTES", 1000)
    with pytest.raises(ValueError, match="100 MB"): infer.make_manifest(["smokeB"], tmp_path / "w", [0], runs) if False else \
        infer.make_manifest(["t"], tmp_path / "w2", [0], runs)
    assert not (tmp_path / "w2" / "t_fold0.pt").exists()


def test_load_rule_roundtrip(tmp_path):
    rule = Rule(t_lo=0.35, peak=0.9, mean=0.6, min_area=300, max_r=0.95)
    (tmp_path / "rule.json").write_text(json.dumps(asdict(rule)), encoding="utf-8")
    assert infer.load_rule(tmp_path / "rule.json") == rule


def test_final_tags_select_one_model(tmp_path):
    """Two tagged runs in one manifest (fold 5 = all stems for AF); load_models(tag=) loads only the named file."""
    runs, out = tmp_path / "runs", tmp_path / "weights"
    cfg = {"encoder": "resnet18", "work_res": 128}
    _fake_run(runs, "AF", 5, cfg=cfg); _fake_run(runs, "A", 0, cfg=cfg)
    infer.make_manifest(["A"], out, [0], runs, tag="final2")
    man = infer.make_manifest(["AF"], out, [5], runs, tag="final1")      # adds to the manifest, keeps final2
    assert {e["file"]: e["final"] for e in man["models"]} == {"A_fold0.pt": "final2", "AF_fold5.pt": "final1"}
    (e, _), = infer.load_models(out, tag="final1")
    assert (e["run"], e["fold"]) == ("AF", 5)
    (out / "A_fold0.pt").unlink()                                        # the untagged file is not needed
    with pytest.raises(FileNotFoundError): infer.load_models(out, tag="final2")
    (e, _), = infer.load_models(out, tag="final1")
    with pytest.raises(ValueError, match="exactly one"): infer.load_models(out, tag="final3")
    infer.make_manifest(["A"], out, [0], runs, tag="final1")             # a tag moves to the new file
    assert {e["file"]: e["final"] for e in json.loads((out / "manifest.json").read_text(encoding="utf-8"))["models"]} == {
        "A_fold0.pt": "final1", "AF_fold5.pt": None}


def test_manifest_entry_rule_is_optional_and_must_exist(tmp_path):
    """`rule` (repo-relative) is stored only when given; the committed final2 entry points at a rule file that exists."""
    runs, out = tmp_path / "runs", tmp_path / "weights"
    _fake_run(runs, "t", 0, cfg={"encoder": "resnet18", "work_res": 128})
    assert "rule" not in infer.make_manifest(["t"], out, [0], runs)["models"][0]
    with pytest.raises(FileNotFoundError, match="keep rule"): infer.make_manifest(["t"], out, [0], runs, rule="nope.json")
    e = infer.make_manifest(["t"], out, [0], runs, rule="filseg/rule_fold0.json")["models"][0]
    assert e["rule"] == "filseg/rule_fold0.json" and infer.load_rule(e["rule"]).peak == 0.995
    (e, _), = infer.load_models(out)
    assert e["rule"] == "filseg/rule_fold0.json"
    final2, = [m for m in json.loads((infer.D.REPO / "weights/manifest.json").read_text(encoding="utf-8"))["models"] if m["final"] == "final2"]
    assert final2["rule"] == "filseg/rule_fold0.json" and (final2["run"], final2["fold"]) == ("A", 0)
