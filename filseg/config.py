import json
from dataclasses import dataclass
from pathlib import Path

RUNS_JSON = Path(__file__).resolve().parent.parent / "configs" / "runs.json"


@dataclass
class Cfg:
    run: str
    fold: int
    seed: int = 0
    encoder: str = "resnet34"
    work_res: int = 2048
    crop: int = 512
    batch: int = 16
    steps: int = 250
    epochs: int = 40
    lr: float = 3e-4
    wd: float = 1e-4
    fg_prob: float = 0.7
    dice_w: float = 0.5
    ema: float = 0.999


def load_cfg(run: str, fold: int) -> Cfg:
    """Defaults overridden by configs/runs.json[run]; an unknown run or key raises."""
    runs = json.loads(RUNS_JSON.read_text(encoding="utf-8"))
    return Cfg(run=run, fold=fold, **runs[run])
