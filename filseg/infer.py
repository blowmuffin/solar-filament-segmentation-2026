"""Inference from the committed weights: the weights manifest, the ensemble probability map, one image to masks.

`python -m filseg.infer --run AF --folds 5 --tag final1 --out weights` copies <run>/fold<k>.pt into weights/ and adds it
to weights/manifest.json; the notebook picks ONE entry by its `final` tag (final1 or final2), it never averages."""
import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from filseg import data as D
from filseg.config import load_cfg
from filseg.instances import Rule, candidates, select, to_rles
from filseg.model import load_model, run_dir
from filseg.predict import predict_prob, quantize
from filseg.preflight import sha256_file

MANIFEST = "manifest.json"
MAX_FILE_BYTES = 100 * 10 ** 6          # the host's limit for a weight file in the repo


class Segmentation(NamedTuple):
    prob: np.ndarray        # float32 ensemble probability map at the image's size
    labels: np.ndarray      # int32 component map of prob > rule.t_lo (0 = background)
    cands: list[dict]       # one feature dict per component
    kept: list[int]         # ids of the components the rule keeps
    rles: list[dict]        # one column-major RLE per kept component, in `kept` order


def make_manifest(runs: list[str], out_dir, folds=(0, 1, 2, 3, 4), runs_root=None, tag=None, rule=None) -> dict:
    """Copy <runs_root>/<run>/fold<k>.pt to <out_dir>/<run>_fold<k>.pt for every run and fold and add them to
    <out_dir>/manifest.json: {"models": [{file, run, fold, encoder, work_res, sha256, commit, final}]}. An existing
    manifest is kept: entries of the same file are replaced, the others stay. `rule` (repo-relative path of a keep rule,
    must exist) is stored as `rule` on the new entries and the notebook then uses it for that model. `tag` ("final1", "final2") is stored as
    `final` (null without it) and taken from any other entry, so a tag names one file. The encoder and
    work_res come from the config the job recorded next to the weights (fold<k>_meta.json), else from
    configs/runs.json. runs_root default: model.run_dir (FILSEG_RUNS, else ../runs)."""
    if rule and not (D.REPO / rule).is_file():
        raise FileNotFoundError(f"{D.REPO / rule}: the keep rule of the manifest entry does not exist")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    old = out / MANIFEST
    models = json.loads(old.read_text(encoding="utf-8"))["models"] if old.exists() else []
    for run in runs:
        src = Path(runs_root) / run if runs_root else run_dir(run)
        for k in folds:
            meta_file = src / f"fold{k}_meta.json"
            meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
            if meta and not meta.get("ok"):
                raise ValueError(f"{meta_file}: the job did not finish cleanly, its weights are not shipped")
            cfg = meta.get("cfg") or asdict(load_cfg(run, k))
            dst = out / f"{run}_fold{k}.pt"
            shutil.copyfile(src / f"fold{k}.pt", dst)
            size = dst.stat().st_size
            if size >= MAX_FILE_BYTES:
                dst.unlink()
                raise ValueError(f"{src / f'fold{k}.pt'} is {size / 1e6:.0f} MB, the limit is 100 MB")
            models = [{**m, "final": None} if tag and m.get("final") == tag else m for m in models if m["file"] != dst.name]
            models.append(dict(file=dst.name, run=run, fold=k, encoder=cfg["encoder"], work_res=cfg["work_res"],
                               sha256=sha256_file(dst), commit=meta.get("commit"), final=tag, **({"rule": rule} if rule else {})))
    manifest = {"models": models}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    return manifest


def load_models(weights_dir, device: str = "cpu", tag=None) -> list[tuple[dict, torch.nn.Module]]:
    """(manifest entry, float32 eval-mode model on `device`) for every model of <weights_dir>/manifest.json, or with
    `tag` for the one entry whose `final` is that tag (the other files need not exist). A relative weights_dir is
    taken from the repo root. Each loaded file's SHA-256 and encoder are checked against the manifest; nothing is
    downloaded."""
    wdir = D.REPO / weights_dir
    manifest = wdir / MANIFEST
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} is missing: write it with `python -m filseg.infer --run <run> --out {weights_dir}`")
    entries = json.loads(manifest.read_text(encoding="utf-8"))["models"]
    if not entries:
        raise ValueError(f"{manifest} lists no models")
    if tag is not None:
        entries = [e for e in entries if e.get("final") == tag]
        if len(entries) != 1:
            raise ValueError(f"{manifest} has {len(entries)} models tagged {tag!r}, expected exactly one")
    members = []
    for e in entries:
        path = wdir / e["file"]
        if sha256_file(path) != e["sha256"]:
            raise ValueError(f"{path} does not match the SHA-256 in {manifest}")
        model = load_model(path, device)
        if model.encoder_name != e["encoder"]:
            raise ValueError(f"{path} holds encoder {model.encoder_name!r}, the manifest says {e['encoder']!r}")
        members.append((e, model))
    return members


def load_rule(path) -> Rule:
    """The keep rule written by filseg.tune; a relative path is taken from the repo root."""
    return Rule(**json.loads((D.REPO / path).read_text(encoding="utf-8")))


def ensemble_prob(members: list[tuple[dict, torch.nn.Module]], img: np.ndarray, device: str = "cpu") -> np.ndarray:
    """float32 map at the image's size, built exactly as instances.iter_probs builds it from a run's .npz files:
    every model gives a flip-averaged map at its own work_res, quantised to uint8 (predict.quantize), and the
    maps are averaged with equal weight."""
    acc = np.zeros(img.shape, np.float32)
    for entry, model in members:
        acc += quantize(predict_prob([model], img, entry["work_res"], device))
    return acc / (255.0 * len(members))


def segment(members: list[tuple[dict, torch.nn.Module]], img: np.ndarray, rule: Rule, device: str = "cpu") -> Segmentation:
    """One image to masks: ensemble map, 8-connected components of prob > rule.t_lo, keep rule, RLEs."""
    prob = ensemble_prob(members, img, device)
    labels, cands = candidates(prob, rule.t_lo)
    kept = select(cands, rule)
    return Segmentation(prob, labels, cands, kept, to_rles(labels, kept))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--runs-root", help="folder holding <run>/fold<k>.pt (default: FILSEG_RUNS, else ../runs)")
    ap.add_argument("--tag", help="final1 or final2: mark these files in the manifest (notebook setting MODEL)")
    ap.add_argument("--rule", help="repo-relative keep rule for these entries (default: the notebook uses filseg/rule.json)")
    ap.add_argument("--out", default="weights", help="destination, relative to the repo root")
    a = ap.parse_args(argv)
    manifest = make_manifest(a.run, D.REPO / a.out, a.folds, a.runs_root, a.tag, a.rule)
    for e in manifest["models"]:
        print(f"{e['file']}: run {e['run']} fold {e['fold']} {e['encoder']} work_res {e['work_res']} final {e['final']} sha256 {e['sha256'][:12]}")
    print(f"wrote {D.REPO / a.out / MANIFEST} ({len(manifest['models'])} models)")


if __name__ == "__main__":
    main()
