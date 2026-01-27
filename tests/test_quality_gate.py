import unittest

import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from wrinkle_v2.quality.global_quality_gate import GlobalQualityGate


class TestGlobalQualityGate(unittest.TestCase):
    def _gate(self, cfg=None):
        return GlobalQualityGate(cfg=cfg)

    def _landmarks(self, points):
        class _LM:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        return [_LM(x, y) for x, y in points]

    def test_blank_image_no_face(self):
        cfg = {"require_face_detection": True}
        gate = self._gate(cfg)
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        res = gate.evaluate(img)
        self.assertFalse(res.quality_pass)
        self.assertIn("no_face_detected", res.reasons)

    def test_dark_image_brightness_fail(self):
        cfg = {"require_face_detection": False, "min_brightness": 60.0}
        gate = self._gate(cfg)
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        res = gate.evaluate(img)
        self.assertFalse(res.quality_pass)
        self.assertIn("image_too_dark", res.reasons)

    def test_bright_image_brightness_fail(self):
        cfg = {"require_face_detection": False, "max_brightness": 200.0}
        gate = self._gate(cfg)
        img = np.full((256, 256, 3), 255, dtype=np.uint8)
        res = gate.evaluate(img)
        self.assertFalse(res.quality_pass)
        self.assertIn("image_too_bright", res.reasons)

    def test_blurred_image_fails_blur(self):
        cfg = {"require_face_detection": False, "blur_laplacian_var": 120.0}
        gate = self._gate(cfg)
        img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        res = gate.evaluate(blurred)
        self.assertFalse(res.quality_pass)
        self.assertIn("image_blurry", res.reasons)

    def test_random_image_passes_without_face_requirement(self):
        cfg = {"require_face_detection": False}
        gate = self._gate(cfg)
        img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        res = gate.evaluate(img)
        self.assertTrue(res.quality_pass)

    def test_landmark_fallback_passes_when_face_missing(self):
        cfg = {
            "require_face_detection": True,
            "allow_landmark_face_fallback": True,
            "landmark_face_area_ratio_min": 0.04,
            "blur_laplacian_var": 0.0,
            "min_brightness": 0.0,
            "max_brightness": 255.0,
        }
        gate = self._gate(cfg)
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        landmarks = self._landmarks([(0.3, 0.3), (0.7, 0.7), (0.3, 0.7), (0.7, 0.3)])
        res = gate.evaluate(img, landmarks=landmarks)
        self.assertTrue(res.quality_pass)
        self.assertTrue(res.flags.get("face_detected_via_landmarks"))
        self.assertGreater(res.flags.get("face_area_ratio", 0.0), 0.0)

    def test_landmark_fallback_disabled_keeps_no_face(self):
        cfg = {
            "require_face_detection": True,
            "allow_landmark_face_fallback": True,
            "landmark_face_area_ratio_min": 0.04,
            "blur_laplacian_var": 0.0,
            "min_brightness": 0.0,
            "max_brightness": 255.0,
        }
        gate = self._gate(cfg)
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        res = gate.evaluate(img, landmarks=None)
        self.assertFalse(res.quality_pass)
        self.assertIn("no_face_detected", res.reasons)


if __name__ == "__main__":
    unittest.main()
