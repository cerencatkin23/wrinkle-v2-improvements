import json
import os
import sys

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from wrinkle_v2.pipeline.predictor import WrinklePipeline
from wrinkle_v2.utils import load_config


def find_one_test_image(root: str) -> str:
    test_dir = os.path.join(root, "test")
    for sub in ["images", ""]:
        folder = os.path.join(test_dir, sub)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if os.path.splitext(fname)[1].lower() in {".jpg", ".jpeg", ".png"}:
                return os.path.join(folder, fname)
    return ""


def main():
    os.environ.setdefault("WRINKLE_DISABLE_MEDIAPIPE", "1")
    config_path = "configs/default.yaml"
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    cfg = load_config(config_path)
    cfg["model"]["pretrained"] = False
    image_path = find_one_test_image(cfg["dataset_root"])
    if not image_path:
        print("No test image found.")
        return
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to read test image.")
        return
    pipeline = WrinklePipeline(cfg)
    result = pipeline.predict(image)
    required_keys = {"status", "quality_gate", "global_score", "per_region_scores", "measurements", "reasoning"}
    missing = required_keys - set(result.keys())
    if missing:
        print(f"Missing keys: {sorted(missing)}")
    else:
        print("Schema keys present.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
