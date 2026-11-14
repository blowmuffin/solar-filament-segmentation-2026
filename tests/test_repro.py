import json

from tools import make_repro as mr


def test_repro_bundle(tmp_path):
    ds, k = mr.build(out=tmp_path, n_images=2, allow_dirty=True)
    assert json.loads((ds / "dataset-metadata.json").read_text(encoding="utf-8")) == {
        "title": "filseg-repo", "id": "ayushkasare/filseg-repo", "licenses": [{"name": "MIT"}]}
    assert (ds / "notebook.ipynb").exists() and list((ds / "weights").glob("*.pt"))
    meta = json.loads((k / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "ayushkasare/filseg-repro" and meta["code_file"] == "run.ipynb"
    assert meta["is_private"] and meta["enable_gpu"] and meta["enable_internet"]
    assert meta["competition_sources"] == ["filament-segmentation-2026"] and meta["dataset_sources"] == ["ayushkasare/filseg-repo"]
    nb = (k / "run.ipynb").read_text(encoding="utf-8")
    assert "jupyter nbconvert --to notebook --execute notebook.ipynb" in nb and "/kaggle/input/**/notebook.ipynb" in nb
    assert "FILSEG_N_IMAGES" in nb and "pycocotools==2.0.11" in nb


def test_repro_cpu(tmp_path):
    _, k = mr.build(out=tmp_path, cpu=True, allow_dirty=True)
    meta = json.loads((k / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert k.name == "repro-cpu" and meta["id"] == "ayushkasare/filseg-repro-cpu" and not meta["enable_gpu"]
    assert "FILSEG_N_IMAGES" not in (k / "run.ipynb").read_text(encoding="utf-8")
