import base64
import io
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return mask_bool
    padded = np.pad(mask_bool, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    interior = mask_bool & up & down & left & right
    return mask_bool & (~interior)


def _dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    out = mask.astype(bool)
    for _ in range(max(1, int(iterations))):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        out = (
            padded[:-2, :-2]
            | padded[:-2, 1:-1]
            | padded[:-2, 2:]
            | padded[1:-1, :-2]
            | padded[1:-1, 1:-1]
            | padded[1:-1, 2:]
            | padded[2:, :-2]
            | padded[2:, 1:-1]
            | padded[2:, 2:]
        )
    return out


def build_annotated_image_base64(
    image_rgb: np.ndarray,
    wrinkle_mask: np.ndarray,
    wrinkle_prob: Optional[np.ndarray] = None,
    region_masks: Optional[Dict[str, np.ndarray]] = None,
    mask_color: Tuple[int, int, int] = (255, 0, 0),
    mask_alpha: float = 0.35,
) -> str:
    annotated = image_rgb.copy()
    if wrinkle_mask is not None:
        mask = wrinkle_mask > 0
        if mask.any():
            fill_alpha = max(0.05, min(mask_alpha * 0.4, 0.25))
            overlay = np.zeros_like(annotated, dtype=np.uint8)
            overlay[:, :] = mask_color
            blended = (annotated.astype(np.float32) * (1.0 - fill_alpha)) + (overlay.astype(np.float32) * fill_alpha)
            annotated[mask] = blended[mask].astype(np.uint8)

            edge = _mask_boundary(mask)
            edge = _dilate_mask(edge, iterations=1)
            edge_color = np.array([255, 200, 0], dtype=np.uint8)
            edge_alpha = 0.85
            edge_blend = (
                annotated.astype(np.float32) * (1.0 - edge_alpha) + edge_color.astype(np.float32) * edge_alpha
            )
            annotated[edge] = edge_blend[edge].astype(np.uint8)

    if wrinkle_prob is not None:
        prob = np.clip(wrinkle_prob.astype(np.float32), 0.0, 1.0)
        if prob.any():
            heat = np.zeros_like(annotated, dtype=np.float32)
            heat[:, :, 0] = 255.0 * prob
            heat[:, :, 1] = 180.0 * prob
            heat[:, :, 2] = 0.0
            alpha_map = (0.25 * prob).astype(np.float32)
            annotated = (
                annotated.astype(np.float32) * (1.0 - alpha_map[..., None]) + heat * alpha_map[..., None]
            ).astype(np.uint8)

    if region_masks:
        boundary_color = np.array([0, 255, 0], dtype=np.uint8)
        for name, region in region_masks.items():
            if region is None:
                continue
            boundary = _mask_boundary(region > 0)
            if boundary.any():
                annotated[boundary] = boundary_color

        img = Image.fromarray(annotated)
        draw = ImageDraw.Draw(img)
        for name, region in region_masks.items():
            if region is None:
                continue
            ys, xs = np.where(region > 0)
            if ys.size == 0:
                continue
            cx = int(xs.mean())
            cy = int(ys.mean())
            label = name.replace("_", " ")
            draw.text((cx, cy), label, fill=(255, 255, 255))
        annotated = np.array(img)

    img = Image.fromarray(annotated)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded
