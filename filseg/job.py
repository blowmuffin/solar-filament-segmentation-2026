"""Run fold jobs on a Kaggle T4 x2 session: `python -m filseg.job A:0 A2:0`.

Job i gets GPU i (CUDA_VISIBLE_DEVICES=i) and runs `filseg.train`, then `filseg.predict`, as subprocesses. Their output
streams to <run>/fold<k>_stdout.txt and to this process's stdout (prefixed with the job name). Each job writes
<run>/fold<k>_meta.json; the exit status is non-zero if any job (or any stage of it) failed."""
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from importlib import metadata

from filseg import data as D
from filseg.config import load_cfg
from filseg.model import run_dir
from filseg.train import job_seed

STAGES = ("train", "predict")
_print_lock = threading.Lock()


def parse(spec: str) -> tuple[str, int]:
    run, sep, fold = spec.rpartition(":")
    if not (sep and run and fold.isdigit()): raise ValueError(f"job must look like RUN:FOLD, got {spec!r}")
    return run, int(fold)


def stage_cmds(run: str, fold: int) -> list[list[str]]:
    return [[sys.executable, "-m", f"filseg.{s}", "--run", run, "--fold", str(fold)] for s in STAGES]


def build_caches(resolutions: list[int]) -> None:
    """Training cache (img_<res>.u8, soft_<res>.u8, stems.json) in D.CACHE_ROOT, once per work_res; always rebuilt."""
    for res in resolutions:
        t0 = time.time()
        print(f"building cache at {res} px in {D.CACHE_ROOT} (data {D.DATA_ROOT})", flush=True)
        D.build_cache(D.CACHE_ROOT, res)
        sizes = {p.name: f"{p.stat().st_size / 1e9:.2f} GB" for p in sorted(D.CACHE_ROOT.glob(f"*_{res}.u8"))}
        print(f"cache {res} px ready in {time.time() - t0:.0f} s: {sizes}", flush=True)


def nvidia_smi() -> None:
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except OSError:
        print("nvidia-smi not found", flush=True)


def commit() -> str:
    f = D.REPO / "filseg" / "_commit.txt"        # written by tools/make_kernel.py
    if f.exists(): return f.read_text(encoding="utf-8").strip()
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=D.REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def versions() -> dict:
    v = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("torch", "torchvision", "segmentation-models-pytorch", "timm", "numpy", "opencv-python-headless", "pycocotools"):
        try: v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError: v[pkg] = None
    try:                                        # bookkeeping must never fail a job
        import torch
        v["cuda"], v["cudnn"] = torch.version.cuda, torch.backends.cudnn.version()
    except Exception as e:
        v["cuda"] = f"unavailable: {e!r}"
    # the Kaggle image tag lives in KAGGLE_* variables; keep them, minus anything that looks like a credential
    v["kaggle_env"] = {k: val[:200] for k, val in os.environ.items()
                       if k.startswith("KAGGLE_") and not re.search("TOKEN|KEY|SECRET|PASS|AUTH|PROXY", k)}
    return v


def say(name: str, line: str) -> None:
    with _print_lock:
        print(f"[{name}] {line}", flush=True)


def run_stage(name: str, cmd: list[str], gpu: int, log) -> int:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
    say(name, "$ " + " ".join(cmd) + f"   (CUDA_VISIBLE_DEVICES={gpu})")
    p = subprocess.Popen(cmd, cwd=D.REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in p.stdout:
        line = line.rstrip("\n")
        log.write(line + "\n"); log.flush()
        say(name, line)
    return p.wait()


def read_json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_job(gpu: int, spec: str, cfg) -> dict:
    out = run_dir(cfg.run); out.mkdir(parents=True, exist_ok=True)
    rc, wall = {}, {}
    with open(out / f"fold{cfg.fold}_stdout.txt", "w", encoding="utf-8") as log:
        for stage, cmd in zip(STAGES, stage_cmds(cfg.run, cfg.fold)):
            t0 = time.time()
            rc[stage] = run_stage(spec, cmd, gpu, log)
            wall[stage] = time.time() - t0
            say(spec, f"{stage} exited {rc[stage]} after {wall[stage]:.0f} s")
            if rc[stage] != 0: break                                  # no prediction from a failed training
    st = {s: read_json(out / f"fold{cfg.fold}_{s}.json") for s in STAGES}
    meta = {"job": spec, "ok": set(rc) == set(STAGES) and not any(rc.values()), "returncodes": rc, "cfg": asdict(cfg),
            "seed": job_seed(cfg), "commit": commit(), "versions": versions(), "gpu": st["train"].get("gpu") or st["predict"].get("gpu"),
            "gpu_index": gpu, "sec_per_step": st["train"].get("sec_per_step_steady") or st["train"].get("sec_per_step"),
            "sec_per_step_incl_startup": st["train"].get("sec_per_step"),
            "sec_per_image": st["predict"].get("sec_per_image"),
            "peak_mem_gb": {s: st[s].get("peak_mem_gb") for s in STAGES},
            "peak_reserved_gb": {s: st[s].get("peak_reserved_gb") for s in STAGES}, "wall_s": wall}
    (out / f"fold{cfg.fold}_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def main(argv=None) -> int:
    specs = sys.argv[1:] if argv is None else list(argv)
    if not specs: sys.exit("usage: python -m filseg.job RUN:FOLD [RUN:FOLD ...]")
    if len(set(specs)) != len(specs): raise ValueError(f"duplicate jobs in {specs}")
    cfgs = [load_cfg(*parse(s)) for s in specs]          # a bad run name or fold fails here, before any GPU time
    n_folds = 1 + max(json.loads(D.FOLDS_JSON.read_text(encoding="utf-8")).values())
    if any(c.fold > n_folds for c in cfgs): raise ValueError(f"fold out of range 0..{n_folds} (5 = all stems): {specs}")
    nvidia_smi()
    build_caches(sorted({c.work_res for c in cfgs}))
    with ThreadPoolExecutor(len(cfgs)) as ex:
        metas = list(ex.map(run_job, range(len(cfgs)), specs, cfgs))
    for m in metas:
        print(f"job {m['job']}: {'ok' if m['ok'] else 'FAILED ' + str(m['returncodes'])}, "
              f"{m['sec_per_step']} s/step, wall {m['wall_s']}", flush=True)
    return 0 if all(m["ok"] for m in metas) else 1


if __name__ == "__main__":
    sys.exit(main())
