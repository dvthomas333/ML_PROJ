"""Shared helpers: config, seeding, environment capture, batched prediction."""
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path="configs/experiment.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_processed(cfg):
    d = np.load(cfg["data"]["processed"])
    return {k: d[k] for k in d.files}


def env_info():
    """Versions that affect numerical results. libjpeg matters: JPEG output differs across builds."""
    import PIL
    import scipy
    import sklearn
    from PIL import features

    try:
        libjpeg = features.version("jpg")
    except Exception:  # noqa: BLE001 - only informational
        libjpeg = "unknown"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "pillow": PIL.__version__,
        "libjpeg": libjpeg,
        "libjpeg_turbo": features.check_feature("libjpeg_turbo"),
    }


def to_tensor(x_uint8: np.ndarray) -> torch.Tensor:
    """N x H x W x 3 uint8 -> N x 3 x H x W float in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(x_uint8)).permute(0, 3, 1, 2).float().div_(255.0)


@torch.no_grad()
def predict(model, x_uint8: np.ndarray, device, batch_size=1024) -> np.ndarray:
    """Return P(fake) for every image."""
    model.eval()
    out = []
    for i in range(0, len(x_uint8), batch_size):
        xb = to_tensor(x_uint8[i : i + batch_size]).to(device)
        out.append(torch.sigmoid(model(xb).float()).cpu().numpy())
    return np.concatenate(out)


def load_checkpoint(path, device):
    from .models import build_model

    ckpt = torch.load(path, map_location=device, weights_only=False)
    mask = ckpt.get("mask")
    model = build_model(ckpt["model_name"], ckpt["cfg"], mask=None if mask is None else torch.as_tensor(mask))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
