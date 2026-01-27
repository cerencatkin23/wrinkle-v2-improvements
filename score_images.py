from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List

import cv2


def _ensure_src_on_path() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _iter_images(input_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = []
    for fname in sorted(os.listdir(input_dir)):
        if os.path.splitext(fname)[1].lower() in exts:
            paths.append(os.path.join(input_dir, fname))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="validation_report/image_scores.csv")
    args = parser.parse_args()

    _ensure_src_on_path()
    from wrinkle_v2.pipeline.predictor import WrinklePipeline
    from wrinkle_v2.utils import load_config

    cfg = load_config(args.config)
    pipeline = WrinklePipeline(cfg)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    rows = []
    for path in _iter_images(args.input_dir):
        img = cv2.imread(path)
        if img is None:
            rows.append(
                {
                    "filename": os.path.basename(path),
                    "quality_pass": False,
                    "reasons": "read_error",
                    "score": "",
                    "top_regions": "",
                }
            )
            continue
        res = pipeline.predict(img)
        quality = res.get("quality_gate", {})
        quality_pass = bool(quality.get("quality_pass", False))
        reasons = ",".join(quality.get("reasons", []))
        score = res.get("global_score") if quality_pass else ""

        top_regions = ""
        if quality_pass:
            per_region = res.get("per_region_scores", {}) or {}
            top = sorted(per_region.items(), key=lambda x: x[1], reverse=True)[:3]
            top_regions = ";".join([f"{name}:{score:.1f}" for name, score in top])

        rows.append(
            {
                "filename": os.path.basename(path),
                "quality_pass": quality_pass,
                "reasons": reasons,
                "score": score,
                "top_regions": top_regions,
            }
        )

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "quality_pass", "reasons", "score", "top_regions"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
