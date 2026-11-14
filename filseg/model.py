"""The network, its fp16 weight files, and where a run's files live."""
import os
from pathlib import Path

import segmentation_models_pytorch as smp
import torch

from filseg import data as D


def run_dir(run: str) -> Path:
    """FILSEG_RUNS/<run>, default /kaggle/working/<run> on Kaggle and ../runs/<run> locally."""
    if "FILSEG_RUNS" in os.environ:
        root = (D.REPO / os.environ["FILSEG_RUNS"]).resolve()       # absolute values win over REPO
    else:
        root = Path("/kaggle/working") if D.ON_KAGGLE else (D.REPO / "../runs").resolve()
    return root / run


def build_model(encoder: str = "resnet34", pretrained: bool = True) -> torch.nn.Module:
    """U-Net, 1 input channel (smp sums the ImageNet first-conv filters), 1 logit channel."""
    m = smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=1, classes=1)
    m.encoder_name = encoder        # save_fp16 stores it so load_model can rebuild the network
    return m


def save_fp16(model, path) -> None:
    """{"encoder": str, "state": float tensors halved, other tensors (BN step counters) unchanged}."""
    state = {}
    for k, v in model.state_dict().items():
        v = v.detach().cpu()
        if v.is_floating_point():
            v = v.half()
            if not torch.isfinite(v).all(): raise ValueError(f"{k} does not fit in fp16")
        state[k] = v
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": model.encoder_name, "state": state}, path)


def load_model(path, device) -> torch.nn.Module:
    """float32, eval mode, on `device`; never downloads."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    m = build_model(blob["encoder"], pretrained=False)
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in blob["state"].items()})
    return m.to(device).eval()


def gpu_stats(device) -> dict:
    """GPU name and peak memory (GB) since the last reset_peak_memory_stats, for the job's meta.json."""
    if torch.device(device).type != "cuda": return {"gpu": "cpu", "peak_mem_gb": 0.0, "peak_reserved_gb": 0.0}
    gb = 2 ** 30
    return {"gpu": torch.cuda.get_device_name(device), "peak_mem_gb": torch.cuda.max_memory_allocated(device) / gb,
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / gb}
