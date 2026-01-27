import os
import random
from typing import Any, Tuple, TYPE_CHECKING

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None
from PIL import Image
import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None
if TYPE_CHECKING:
    import torch as torch_typing
import yaml


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        raise RuntimeError("torch_not_installed")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> Any:
    if torch is None:
        raise RuntimeError("torch_not_installed")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resize_with_pad(
    image: np.ndarray, size: int, interpolation: int | None = None
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    if cv2 is not None:
        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=interpolation if interpolation is not None else cv2.INTER_LINEAR,
        )
    else:
        pil_interp = Image.BILINEAR if interpolation is None else Image.BILINEAR
        resized = np.array(Image.fromarray(image).resize((new_w, new_h), resample=pil_interp))
    pad_top = (size - new_h) // 2
    pad_bottom = size - new_h - pad_top
    pad_left = (size - new_w) // 2
    pad_right = size - new_w - pad_left
    if cv2 is not None:
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    else:
        padded = np.pad(
            resized,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def unpad(image: np.ndarray, pads: Tuple[int, int, int, int]) -> np.ndarray:
    pad_top, pad_bottom, pad_left, pad_right = pads
    if pad_top + pad_bottom + pad_left + pad_right == 0:
        return image
    h, w = image.shape[:2]
    return image[pad_top:h - pad_bottom, pad_left:w - pad_right]


def normalize_image(image: np.ndarray) -> np.ndarray:
    return (image.astype(np.float32) / 255.0)
