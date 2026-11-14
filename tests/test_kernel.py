import json
import sys
from pathlib import Path

import pytest

from filseg import job
from tools import make_kernel as mk


def test_kernel_bundle(tmp_path):
    mk.build("t", ["A:0", "A2:0"], out=tmp_path, allow_dirty=True)
    meta = json.loads((tmp_path / "t" / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "ayushkasare/filseg-t" and meta["is_private"] and meta["enable_gpu"] and meta["enable_internet"]
    assert meta["competition_sources"] == ["filament-segmentation-2026"]
    nb = (tmp_path / "t" / "job.ipynb").read_text(encoding="utf-8")
    for f in Path("filseg").glob("*.py"): assert f"%%writefile filseg/{f.name}" in nb
    assert "python -m filseg.job A:0 A2:0" in nb and "folds.json" in nb


def test_kernel_accepts_full_data_fold(tmp_path):
    mk.build("t", ["AF:5", "AF2:5"], out=tmp_path, allow_dirty=True)
    assert "python -m filseg.job AF:5 AF2:5" in (tmp_path / "t" / "job.ipynb").read_text(encoding="utf-8")
    assert job.parse("AF:5") == ("AF", 5)


def test_kernel_cells(tmp_path):
    """Cell order and the Kaggle-notebook pitfalls: no mkdir in %%writefile, empty cell bodies are rejected."""
    mk.build("t", ["A:0"], out=tmp_path, allow_dirty=True)
    import nbformat
    cells = [c.source for c in nbformat.read(str(tmp_path / "t" / "job.ipynb"), 4).cells]
    assert cells[0] == "!nvidia-smi" and "pip install" in cells[1]
    assert "segmentation-models-pytorch==0.6.0" in cells[1] and "pycocotools==" in cells[1]
    first_write = next(i for i, c in enumerate(cells) if c.startswith("%%writefile"))
    assert "makedirs" in "".join(cells[2:first_write])                      # filseg/ and configs/ exist before writing
    writes = [c for c in cells if c.startswith("%%writefile")]
    assert all(c.split("\n", 1)[1].strip() for c in writes), "an empty %%writefile body is a UsageError in IPython"
    names = [c.split("\n", 1)[0].split()[1] for c in writes]
    assert {"filseg/folds.json", "configs/runs.json", "filseg/_commit.txt", "filseg/__init__.py"} <= set(names)
    assert cells[-1].startswith("!python -m filseg.job A:0\n") and "_exit_code" in cells[-1]


def test_kernel_embeds_commit_and_refuses_dirty(tmp_path, monkeypatch):
    git = {"rev-parse": "abc123\n", "status": " M filseg/x.py\n"}
    monkeypatch.setattr(mk, "_git", lambda *a: git[a[0]])
    with pytest.raises(RuntimeError, match="dirty"): mk.build("t", ["A:0"], out=tmp_path)
    assert not (tmp_path / "t").exists()
    git["status"] = ""
    mk.build("t", ["A:0"], out=tmp_path)
    assert "abc123" in (tmp_path / "t" / "job.ipynb").read_text(encoding="utf-8")
    assert "+dirty" not in (tmp_path / "t" / "job.ipynb").read_text(encoding="utf-8")


@pytest.mark.parametrize("name,jobs", [("Bad Name", ["A:0"]), ("t", ["Z:0"]), ("t", ["A:6"]), ("t", ["A"]), ("t", [])])
def test_kernel_rejects_bad_names_and_jobs(tmp_path, name, jobs):
    with pytest.raises((ValueError, KeyError)): mk.build(name, jobs, out=tmp_path, allow_dirty=True)


def test_job_exit_code_and_gpu_pinning(tmp_path, monkeypatch, capsys):
    """One subprocess per job, CUDA_VISIBLE_DEVICES = job index; any failing stage makes job.main non-zero."""
    code = {"smokeA": "import os; print('gpu', os.environ['CUDA_VISIBLE_DEVICES'])",
            "smokeB": "import os, sys; print('gpu', os.environ['CUDA_VISIBLE_DEVICES']); sys.exit(3)"}
    monkeypatch.setattr(job, "stage_cmds", lambda run, fold: [[sys.executable, "-c", code[run]], [sys.executable, "-c", "print('stage 2')"]])
    monkeypatch.setattr(job, "build_caches", lambda resolutions: print("caches", resolutions))
    monkeypatch.setattr(job, "nvidia_smi", lambda: None)
    monkeypatch.setenv("FILSEG_RUNS", str(tmp_path))
    assert job.main(["smokeA:0"]) == 0
    assert job.main(["smokeA:0", "smokeB:0"]) != 0                         # smokeB's first stage exits 3
    out = capsys.readouterr().out
    assert "[smokeA:0] gpu 0" in out and "[smokeB:0] gpu 1" in out and "caches [1024, 2048]" in out
    log = (tmp_path / "smokeB" / "fold0_stdout.txt").read_text(encoding="utf-8")
    assert "gpu 1" in log and "stage 2" not in log                         # a failed stage stops that job
    meta = json.loads((tmp_path / "smokeB" / "fold0_meta.json").read_text(encoding="utf-8"))
    assert meta["ok"] is False and meta["returncodes"]["train"] == 3 and meta["cfg"]["work_res"] == 1024
    meta = json.loads((tmp_path / "smokeA" / "fold0_meta.json").read_text(encoding="utf-8"))
    assert meta["ok"] is True and {"commit", "versions", "wall_s", "cfg"} <= set(meta) and meta["seed"] == 0


def test_job_rejects_unknown_run_before_any_work(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "build_caches", lambda r: pytest.fail("built caches for a bad job list"))
    monkeypatch.setenv("FILSEG_RUNS", str(tmp_path))
    for bad in (["nope:0"], ["smokeA:0", "smokeA:0"], ["smokeA"]):
        with pytest.raises((KeyError, ValueError, SystemExit)): job.main(bad)
