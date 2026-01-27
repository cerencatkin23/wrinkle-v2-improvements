import argparse
import json
import os
import sys

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from wrinkle_v2.pipeline.predictor import WrinklePipeline
from wrinkle_v2.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--overlay-out", default="")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    cfg = load_config(config_path)
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Failed to read image: {args.image}")

    pipeline = WrinklePipeline(cfg)
    result = pipeline.predict(image)
    if args.overlay_out and result.get("status") == "OK":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask, pads, _ = pipeline.seg_model.predict_mask(rgb)
        from wrinkle_v2.utils import unpad

        mask_unpadded = unpad(mask, pads)
        mask_resized = cv2.resize(mask_unpadded, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = image.copy()
        overlay[mask_resized > 0] = (0, 0, 255)
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        cv2.imwrite(args.overlay_out, blended)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
