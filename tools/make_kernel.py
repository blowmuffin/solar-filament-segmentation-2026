"""Build the Kaggle notebook that carries the package: PY tools/make_kernel.py --job smoke --jobs smokeA:0 smokeB:0

Writes ../kaggle/<job>/job.ipynb and kernel-metadata.json (relative to the repo). Refuses a dirty git tree."""
import argparse
import json
import re
import subprocess
from pathlib import Path

import nbformat
from nbformat import v4

REPO = Path(__file__).resolve().parent.parent
OWNER, COMPETITION = "ayushkasare", "filament-segmentation-2026"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=True).stdout


def _pin(pkg: str) -> str:
    for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith(pkg + "=="): return line.strip()
    raise ValueError(f"{pkg} is not pinned in requirements.txt")


def _check(job: str, jobs: list[str]) -> None:
    """Fail here, not after the Kaggle queue: job name must slugify to itself; every RUN:FOLD must exist."""
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", job): raise ValueError(f"job name {job!r}: use lowercase letters, digits, hyphens")
    if not jobs: raise ValueError("no jobs")
    runs = json.loads((REPO / "configs" / "runs.json").read_text(encoding="utf-8"))
    for j in jobs:
        run, _, fold = j.rpartition(":")
        if run not in runs or not fold.isdigit() or int(fold) > 5: raise ValueError(f"bad job {j!r}: RUN:FOLD, RUN in runs.json, FOLD 0..5 (5 = all stems)")


def _writefile(path: str, text: str) -> str:
    # IPython rejects an empty %%writefile body, and does not create directories (see the setup cell)
    return f"%%writefile {path}\n" + (text if text.strip() else "# empty\n")


def build(job: str, jobs: list[str], out: Path = Path("../kaggle"), allow_dirty: bool = False) -> None:
    _check(job, jobs)
    dirty = _git("status", "--porcelain").strip()
    if dirty and not allow_dirty: raise RuntimeError("git tree is dirty, commit first:\n" + dirty)
    commit = _git("rev-parse", "HEAD").strip() + ("+dirty" if dirty else "")
    files = sorted((REPO / "filseg").glob("*.py")) + [REPO / "filseg" / "folds.json", REPO / "configs" / "runs.json"]
    cells = [
        "!nvidia-smi",
        f"!pip install -q {_pin('segmentation-models-pytorch')} {_pin('pycocotools')}",
        'import os\nos.chdir("/kaggle/working")\nos.makedirs("filseg", exist_ok=True)\nos.makedirs("configs", exist_ok=True)',
        *(_writefile(f.relative_to(REPO).as_posix(), f.read_text(encoding="utf-8")) for f in files),
        _writefile("filseg/_commit.txt", commit + "\n"),
        # a `!` command never raises, so assert on its exit code: a failed job then shows as an errored kernel
        f"!python -m filseg.job {' '.join(jobs)}\nassert globals().get('_exit_code', 0) == 0, f'filseg.job exited with {{_exit_code}}'",
    ]
    nb = v4.new_notebook(cells=[v4.new_code_cell(c) for c in cells])
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbformat.validate(nb)
    dest = (REPO / out / job).resolve()                  # absolute `out` wins over REPO, as in filseg.data
    dest.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(dest / "job.ipynb"))
    meta = {"id": f"{OWNER}/filseg-{job}", "title": f"filseg-{job}", "code_file": "job.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "competition_sources": [COMPETITION], "dataset_sources": [], "kernel_sources": []}
    (dest / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dest} (commit {commit}, jobs {' '.join(jobs)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", required=True); ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    build(a.job, a.jobs, allow_dirty=a.allow_dirty)
