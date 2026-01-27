import argparse
import os
import sys
from typing import List, Tuple

# Avoid macOS OpenMP/SHM crashes in probe runs; allow override via env.
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - optional dependency
    imageio = None

from PIL import Image, ImageDraw

def _ensure_src_on_path() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_ensure_src_on_path()

from wrinkle_v2.pipeline.inference import SegmentationModel  # noqa: E402
from wrinkle_v2.utils import load_config, unpad  # noqa: E402


def _save_png(path: str, array: np.ndarray) -> None:
    if cv2 is not None:
        ok, encoded = cv2.imencode(".png", array)
        if not ok:
            raise RuntimeError(f"encode_failed: {path}")
        with open(path, "wb") as f:
            f.write(encoded.tobytes())
        return
    if imageio is None:
        raise RuntimeError("no_png_writer_available")
    imageio.imwrite(path, array, format="png")


def _resize_prob_to_original(prob: np.ndarray, pads, out_hw: Tuple[int, int]) -> np.ndarray:
    prob_unpadded = unpad(prob, pads)
    out_h, out_w = out_hw
    if cv2 is not None:
        resized = cv2.resize(prob_unpadded, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    else:
        img = Image.fromarray(prob_unpadded.astype(np.float32), mode="F")
        resized = np.array(img.resize((out_w, out_h), resample=Image.BILINEAR))
    return np.clip(resized, 0.0, 1.0)


def _draw_lines(
    image: np.ndarray,
    stroke_mask: np.ndarray,
    lines: List[List[Tuple[int, int]]],
    line_width: int,
    delta: int,
) -> np.ndarray:
    out = image.copy()
    if cv2 is not None:
        color = (delta, delta, delta)
        for pts in lines:
            for i in range(len(pts) - 1):
                cv2.line(out, pts[i], pts[i + 1], color, thickness=line_width, lineType=cv2.LINE_AA)
                cv2.line(stroke_mask, pts[i], pts[i + 1], 255, thickness=line_width, lineType=cv2.LINE_8)
    else:
        img = Image.fromarray(out)
        draw = ImageDraw.Draw(img)
        mask_img = Image.fromarray(stroke_mask)
        mask_draw = ImageDraw.Draw(mask_img)
        for pts in lines:
            draw.line(pts, fill=(delta, delta, delta), width=line_width, joint="curve")
            mask_draw.line(pts, fill=255, width=line_width)
        out = np.array(img)
        stroke_mask[:] = np.array(mask_img)
    return out


def _build_lines(h: int, w: int, num_lines: int) -> List[List[Tuple[int, int]]]:
    lines = []
    y_forehead = int(round(h * 0.22))
    y_under_eye = int(round(h * 0.45))
    y_naso = int(round(h * 0.62))
    x_center = int(round(w * 0.5))
    span = int(round(w * 0.12))
    presets = [
        [(x_center - span, y_forehead), (x_center, y_forehead - 4), (x_center + span, y_forehead)],
        [(x_center - span, y_under_eye), (x_center - span // 2, y_under_eye + 4), (x_center, y_under_eye)],
        [(x_center, y_naso), (x_center + span // 2, y_naso + 6), (x_center + span, y_naso)],
    ]
    for i in range(min(num_lines, len(presets))):
        lines.append(presets[i])
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="sample_images_real/Ekran Resmi 2026-01-04 20.35.16.png")
    parser.add_argument("--out_dir", default="thin_probe_outputs")
    parser.add_argument("--line_strength", type=int, default=25)
    parser.add_argument("--line_width", type=int, default=1)
    parser.add_argument("--num_lines", type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    os.makedirs(args.out_dir, exist_ok=True)
    image = np.array(Image.open(args.image_path).convert("RGB"))
    h, w = image.shape[:2]
    lines = _build_lines(h, w, args.num_lines)

    cfg = load_config("configs/default.yaml")
    seg_model = SegmentationModel(cfg)

    _, pads, prob = seg_model.predict_mask(image)
    prob_resized = _resize_prob_to_original(prob, pads, (h, w))

    _save_png(os.path.join(args.out_dir, "original_prob.png"), (prob_resized * 255.0).astype(np.uint8))

    variants = [
        ("dark", -abs(args.line_strength)),
        ("light", abs(args.line_strength)),
    ]

    print("variant, mean_abs_delta, mean_abs_delta_on_strokes, max_abs_delta, verdict")
    for idx, (label, delta) in enumerate(variants, start=1):
        stroke_mask = np.zeros((h, w), dtype=np.uint8)
        variant = _draw_lines(image, stroke_mask, lines, args.line_width, delta)
        _, pads_v, prob_v = seg_model.predict_mask(variant)
        prob_v_resized = _resize_prob_to_original(prob_v, pads_v, (h, w))

        delta_map = prob_v_resized - prob_resized
        abs_delta = np.abs(delta_map)
        mean_abs = float(np.mean(abs_delta))
        stroke_bool = stroke_mask > 0
        if stroke_bool.any():
            mean_abs_stroke = float(np.mean(abs_delta[stroke_bool]))
        else:
            mean_abs_stroke = 0.0
        max_abs = float(np.max(abs_delta))
        verdict = "REACTS" if (mean_abs_stroke >= 0.02 or max_abs >= 0.10) else "NO REACTION"

        _save_png(
            os.path.join(args.out_dir, f"variant_{idx}_{label}_input.png"),
            variant.astype(np.uint8),
        )
        _save_png(
            os.path.join(args.out_dir, f"variant_{idx}_{label}_prob.png"),
            (prob_v_resized * 255.0).astype(np.uint8),
        )
        _save_png(
            os.path.join(args.out_dir, f"variant_{idx}_{label}_abs_delta.png"),
            (np.clip(abs_delta, 0.0, 1.0) * 255.0).astype(np.uint8),
        )

        print(f"{label}, {mean_abs:.4f}, {mean_abs_stroke:.4f}, {max_abs:.4f}, {verdict}")

    print(f"outputs: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
