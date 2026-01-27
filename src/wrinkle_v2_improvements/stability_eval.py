from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from wrinkle_v2.pipeline.predictor import WrinklePipeline
from wrinkle_v2.utils import load_config
from wrinkle_v2.pipeline.metrics import stability_score


@dataclass
class StabilityResult:
    image_path: str
    valid_runs: int
    stability_index: float
    std_score: float
    max_delta: float
    region_variance: Dict[str, float]


def _perturb_image(image: np.ndarray, cfg: Dict, rng: np.random.RandomState) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]

    if cfg.get("brightness_jitter", 0.0) > 0:
        delta = rng.uniform(-cfg["brightness_jitter"], cfg["brightness_jitter"]) * 255.0
        out = np.clip(out.astype(np.float32) + delta, 0, 255).astype(np.uint8)

    if cfg.get("contrast_jitter", 0.0) > 0:
        factor = 1.0 + rng.uniform(-cfg["contrast_jitter"], cfg["contrast_jitter"])
        out = np.clip((out.astype(np.float32) - 127.5) * factor + 127.5, 0, 255).astype(np.uint8)

    if cfg.get("rotation_deg", 0.0) > 0:
        angle = rng.uniform(-cfg["rotation_deg"], cfg["rotation_deg"])
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    if cfg.get("crop_pct", 0.0) > 0:
        pct = rng.uniform(0, cfg["crop_pct"])
        dx = int(round(w * pct))
        dy = int(round(h * pct))
        x0 = rng.randint(0, max(1, dx + 1))
        y0 = rng.randint(0, max(1, dy + 1))
        x1 = w - (dx - x0)
        y1 = h - (dy - y0)
        out = out[y0:y1, x0:x1]
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)

    blur_ksize = int(cfg.get("blur_ksize", 0))
    blur_prob = float(cfg.get("blur_prob", 0.0))
    if blur_ksize > 1 and rng.rand() < blur_prob:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        out = cv2.GaussianBlur(out, (k, k), 0)

    return out


def _collect_scores(results: List[Dict]) -> Tuple[List[float], Dict[str, List[float]]]:
    scores = []
    region_scores: Dict[str, List[float]] = {}
    for res in results:
        if res.get("status") != "OK":
            continue
        scores.append(float(res["global_score"]))
        for region, score in res.get("per_region_scores", {}).items():
            region_scores.setdefault(region, []).append(float(score))
    return scores, region_scores


def _variance(vals: List[float]) -> float:
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    return float(sum((v - mean) ** 2 for v in vals) / len(vals))


def evaluate_image(
    image_path: str,
    pipeline: WrinklePipeline,
    cfg: Dict,
    rng: np.random.RandomState,
) -> StabilityResult:
    bgr = cv2.imread(image_path)
    if bgr is None:
        return StabilityResult(image_path, 0, 0.0, 0.0, 0.0, {})

    perturb_cfg = cfg.get("stability_eval", {})
    n = int(perturb_cfg.get("num_perturbations", 5))

    results = []
    for _ in range(n):
        perturbed = _perturb_image(bgr, perturb_cfg, rng)
        results.append(pipeline.predict(perturbed))

    scores, region_scores = _collect_scores(results)
    if not scores:
        return StabilityResult(image_path, 0, 0.0, 0.0, 0.0, {})

    std_score = float(np.std(np.array(scores)))
    max_delta = float(max(scores) - min(scores))
    stability_index = stability_score(scores)

    region_variance = {k: _variance(v) for k, v in region_scores.items()}

    return StabilityResult(image_path, len(scores), stability_index, std_score, max_delta, region_variance)


def run_stability_eval(
    input_dir: str,
    out_dir: str,
    config_path: str,
    seed: Optional[int] = None,
    pipeline: Optional[WrinklePipeline] = None,
) -> Dict:
    cfg = load_config(config_path)
    print(
        "stability_eval config:",
        {"path": config_path, "quality_gate": cfg.get("quality_gate", {}), "stability_eval": cfg.get("stability_eval", {})},
    )
    if seed is None:
        seed = int(cfg.get("seed", 42))
    rng = np.random.RandomState(seed)

    if pipeline is None:
        pipeline = WrinklePipeline(cfg)

    os.makedirs(out_dir, exist_ok=True)

    reports = []
    for fname in sorted(os.listdir(input_dir)):
        if os.path.splitext(fname)[1].lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        path = os.path.join(input_dir, fname)
        res = evaluate_image(path, pipeline, cfg, rng)
        reports.append(res)
        out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(res), f, indent=2)

    valid = [r for r in reports if r.valid_runs > 0]
    summary = {
        "images": len(reports),
        "valid_images": len(valid),
        "mean_stability_index": float(np.mean([r.stability_index for r in valid])) if valid else 0.0,
        "mean_std_score": float(np.mean([r.std_score for r in valid])) if valid else 0.0,
        "mean_max_delta": float(np.mean([r.max_delta for r in valid])) if valid else 0.0,
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    summary = run_stability_eval(args.input_dir, args.out_dir, args.config, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
