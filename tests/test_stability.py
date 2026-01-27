import json
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from wrinkle_v2_improvements.stability_eval import run_stability_eval


class DummyPipeline:
    def predict(self, image_bgr):
        mean_val = float(np.mean(image_bgr))
        score = max(0.0, min(100.0, mean_val / 255.0 * 100.0))
        return {
            "status": "OK",
            "global_score": score,
            "per_region_scores": {"forehead": score, "chin": score},
        }


class TestStabilityEval(unittest.TestCase):
    def test_stability_eval_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "images")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(input_dir, exist_ok=True)

            img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
            path = os.path.join(input_dir, "img1.jpg")
            cv2.imwrite(path, img)

            cfg_path = os.path.join(tmpdir, "config.yaml")
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write("seed: 123\n")
                f.write("stability_eval:\n")
                f.write("  num_perturbations: 3\n")
                f.write("  brightness_jitter: 0.0\n")
                f.write("  contrast_jitter: 0.0\n")
                f.write("  rotation_deg: 0.0\n")
                f.write("  crop_pct: 0.0\n")
                f.write("  blur_ksize: 0\n")
                f.write("  blur_prob: 0.0\n")

            summary = run_stability_eval(input_dir, out_dir, cfg_path, seed=123, pipeline=DummyPipeline())
            self.assertIn("mean_stability_index", summary)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "summary.json")))

            with open(os.path.join(out_dir, "summary.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["images"], 1)


if __name__ == "__main__":
    unittest.main()
