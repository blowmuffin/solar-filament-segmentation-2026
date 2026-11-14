"""Build the Kaggle reproduction (Gate G5): PY tools/make_repro.py [--cpu] [--n-images N] [--model final1]

Writes ../kaggle/repro_dataset/ (clean `git archive HEAD` of the repo + dataset-metadata.json) and
../kaggle/repro[-cpu]/ (kernel-metadata.json + run.ipynb, which runs notebook.ipynb headless). Refuses a dirty tree."""
import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import nbformat
from nbformat import v4

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so `PY tools/make_repro.py` finds make_kernel
from make_kernel import COMPETITION, OWNER, REPO, _git, _pin  # noqa: E402

DATASET = f"{OWNER}/filseg-repo"


def build(out: Path = Path("../kaggle"), cpu: bool = False, n_images: int | None = None, model: str = "final1",
          allow_dirty: bool = False) -> tuple[Path, Path]:
    dirty = _git("status", "--porcelain").strip()
    if dirty and not allow_dirty: raise RuntimeError("git tree is dirty, commit first:\n" + dirty)
    root = (REPO / out).resolve()                        # absolute `out` wins over REPO
    ds, kdir = root / "repro_dataset", root / ("repro-cpu" if cpu else "repro")
    for d in (ds, kdir):
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    tar = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=REPO, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as t: t.extractall(ds, filter="data")
    (ds / "dataset-metadata.json").write_text(json.dumps(
        {"title": "filseg-repo", "id": DATASET, "licenses": [{"name": "MIT"}]}, indent=2) + "\n", encoding="utf-8")

    env = ["os.environ['FILSEG_OUT'] = '/kaggle/working'", f"os.environ['FILSEG_MODEL'] = {model!r}"]
    if n_images is not None: env.append(f"os.environ['FILSEG_N_IMAGES'] = {str(n_images)!r}")
    cells = [
        "!nvidia-smi || true",
        "import glob, os, shutil\n"
        "nb = glob.glob('/kaggle/input/**/notebook.ipynb', recursive=True)\n"
        "assert nb, 'dataset ayushkasare/filseg-repo is not attached'\n"
        "shutil.copytree(os.path.dirname(nb[0]), '/kaggle/working/repo', dirs_exist_ok=True)\n"
        "os.chdir('/kaggle/working/repo')",
        f"!pip install -q {_pin('segmentation-models-pytorch')} {_pin('pycocotools')}",
        "import os\n" + "\n".join(env),
        # a `!` command never raises, so assert on its exit code (see make_kernel)
        "!jupyter nbconvert --to notebook --execute notebook.ipynb --output /kaggle/working/executed.ipynb "
        "--ExecutePreprocessor.timeout=-1\n"
        "assert globals().get('_exit_code', 0) == 0, f'nbconvert exited with {_exit_code}'",
        "import hashlib, os\n"
        "p = '/kaggle/working/submission.csv'\n"
        "assert os.path.exists(p), 'submission.csv was not written'\n"
        "print(sum(1 for _ in open(p, encoding='utf-8')) - 1, 'rows, sha256', hashlib.sha256(open(p, 'rb').read()).hexdigest())",
    ]
    nb = v4.new_notebook(cells=[v4.new_code_cell(c) for c in cells])
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbformat.validate(nb)
    nbformat.write(nb, str(kdir / "run.ipynb"))
    slug = "filseg-repro-cpu" if cpu else "filseg-repro"
    meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": "run.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": not cpu, "enable_internet": True,
            "competition_sources": [COMPETITION], "dataset_sources": [DATASET], "kernel_sources": []}
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return ds, kdir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--n-images", type=int)
    ap.add_argument("--model", default="final1"); ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    ds, kdir = build(cpu=a.cpu, n_images=a.n_images, model=a.model, allow_dirty=a.allow_dirty)
    print(f"wrote {ds} and {kdir}\npush (first time):  python -m kaggle datasets create -p {ds} --dir-mode zip\n"
          f"push (later):       python -m kaggle datasets version -p {ds} -m msg --dir-mode zip\n"
          f"then:               python -m kaggle kernels push -p {kdir}")
