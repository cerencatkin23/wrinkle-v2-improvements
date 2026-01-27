import base64
import io

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - optional dependency
    imageio = None


def mask_to_png_base64(mask: np.ndarray) -> str:
    if mask is None:
        raise RuntimeError("mask_missing")
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    scaled = (mask > 0).astype(np.uint8) * 255

    if cv2 is not None:
        ok, encoded = cv2.imencode(".png", scaled)
        if not ok:
            raise RuntimeError("mask_encode_failed")
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    if imageio is None:
        raise RuntimeError("mask_encode_unavailable")

    buffer = io.BytesIO()
    imageio.imwrite(buffer, scaled, format="png")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def prob_to_png_base64(prob: np.ndarray, gamma: float = 1.0) -> str:
    if prob is None:
        raise RuntimeError("prob_missing")
    if prob.ndim != 2:
        raise ValueError("prob must be 2D")
    clipped = np.clip(prob, 0.0, 1.0)
    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError("gamma must be > 0")
    vis = np.power(clipped, gamma)
    scaled = (vis * 255.0).astype(np.uint8)

    if cv2 is not None:
        ok, encoded = cv2.imencode(".png", scaled)
        if not ok:
            raise RuntimeError("prob_encode_failed")
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    if imageio is None:
        raise RuntimeError("prob_encode_unavailable")

    buffer = io.BytesIO()
    imageio.imwrite(buffer, scaled, format="png")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
