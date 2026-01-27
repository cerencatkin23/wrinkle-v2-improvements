import os
from typing import Dict, Tuple

try:
    import cv2  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None
import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from wrinkle_v2.models.unet import build_unet
from wrinkle_v2.utils import get_device, normalize_image, resize_with_pad


class SegmentationModel:
    def __init__(self, cfg: Dict):
        if torch is None:
            raise RuntimeError("torch_not_installed")
        self.cfg = cfg
        self.device = get_device()
        model_cfg = cfg["model"]
        self.model = build_unet(
            encoder=model_cfg["encoder"],
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            pretrained=model_cfg["pretrained"],
        )
        self.model.to(self.device)
        self.model.eval()
        inference_cfg = cfg.get("inference", {})
        self.mask_threshold = float(inference_cfg.get("mask_threshold", 0.5))
        self.seg_input_size = int(inference_cfg.get("seg_input_size", cfg.get("input_size", 512)))
        env_size = os.environ.get("WRINKLE_SEG_INPUT_SIZE")
        if env_size:
            self.seg_input_size = int(env_size)

        ckpt = cfg.get("inference", {}).get("checkpoint_path")
        if ckpt and os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(state)

    def predict_mask(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if torch is None:
            raise RuntimeError("torch_not_installed")
        image_resized, pads = resize_with_pad(image, self.seg_input_size)
        image_norm = normalize_image(image_resized)
        tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
        mask = (probs > self.mask_threshold).astype(np.uint8)
        return mask, pads, probs
