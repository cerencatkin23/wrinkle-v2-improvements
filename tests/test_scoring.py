import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from wrinkle_v2.scoring import build_reasoning, compute_scores


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.measurements = {
            "global": {
                "area_ratio": 0.03,
                "skeleton_length": 200.0,
                "thickness": 2.5,
                "density": 0.004,
            },
            "per_region": {
                "forehead": {
                    "area_ratio": 0.03,
                    "skeleton_length": 200.0,
                    "thickness": 2.5,
                    "density": 0.004,
                },
                "chin": {
                    "area_ratio": 0.01,
                    "skeleton_length": 80.0,
                    "thickness": 1.8,
                    "density": 0.002,
                },
            },
        }
        self.scoring_cfg = {
            "per_region_metric_weights": {
                "area_ratio": 0.45,
                "skeleton_length": 0.25,
                "thickness": 0.2,
                "density": 0.1,
            },
            "region_weights": {"forehead": 1.0, "chin": 1.0},
            "region_normalization": {
                "method": "robust",
                "priors": {
                    "area_ratio": {"median": 0.02, "iqr": 0.02},
                    "skeleton_length": {"median": 150.0, "iqr": 200.0},
                    "thickness": {"median": 2.0, "iqr": 1.0},
                    "density": {"median": 0.003, "iqr": 0.003},
                },
            },
            "thresholds": {
                "area_ratio": [0.02, 0.05, 0.1],
                "skeleton_length": [150.0, 400.0, 800.0],
                "thickness": [1.5, 3.0, 5.0],
                "density": [0.002, 0.006, 0.012],
            },
        }

    def test_no_confidence_usage(self):
        scores1 = compute_scores(self.measurements, self.scoring_cfg)
        measurements2 = dict(self.measurements)
        measurements2["global"] = dict(self.measurements["global"])
        measurements2["global"]["confidence"] = 0.99
        scores2 = compute_scores(measurements2, self.scoring_cfg)
        self.assertEqual(scores1, scores2)

    def test_reasoning_deterministic(self):
        scores = compute_scores(self.measurements, self.scoring_cfg)
        quality = {"quality_pass": True, "reasons": [], "flags": {}}
        r1 = build_reasoning(quality, self.measurements, scores, self.scoring_cfg)
        r2 = build_reasoning(quality, self.measurements, scores, self.scoring_cfg)
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
