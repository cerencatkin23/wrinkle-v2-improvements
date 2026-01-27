from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None
import numpy as np

DEFAULT_QUALITY_CFG = {
    "require_face_detection": True,
    "min_face_area_ratio": 0.06,
    "allow_landmark_face_fallback": False,
    "landmark_face_area_ratio_min": 0.06,
    "blur_laplacian_var": 70.0,
    "min_brightness": 50.0,
    "max_brightness": 210.0,
}


@dataclass
class QualityResult:
    quality_pass: bool
    reasons: List[str]
    flags: Dict[str, float | bool]


class GlobalQualityGate:
    def __init__(self, cfg: Dict | None = None):
        self.cfg = DEFAULT_QUALITY_CFG.copy()
        if cfg:
            self.cfg.update(cfg)
        if cv2 is not None:
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        else:
            self.haar = None

    def evaluate(self, image: np.ndarray, landmarks=None) -> QualityResult:
        if cv2 is None:
            return QualityResult(
                quality_pass=False,
                reasons=["opencv_not_installed"],
                flags={
                    "opencv_available": False,
                    "landmarks_available": False,
                    "face_detected_via_landmarks": False,
                    "face_area_ratio": 0.0,
                    "landmark_face_area_ratio": 0.0,
                },
            )
        reasons: List[str] = []
        flags: Dict[str, float | bool] = {}
        flags["landmarks_available"] = landmarks is not None

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        blur_value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        flags["blur_value"] = blur_value
        blur_ok = blur_value >= float(self.cfg["blur_laplacian_var"])
        flags["blur_ok"] = blur_ok
        if not blur_ok:
            reasons.append("image_blurry")

        mean_brightness = float(np.mean(gray))
        flags["mean_brightness"] = mean_brightness
        min_bright = float(self.cfg["min_brightness"])
        max_bright = float(self.cfg["max_brightness"])
        brightness_ok = min_bright <= mean_brightness <= max_bright
        flags["brightness_ok"] = brightness_ok
        if mean_brightness < min_bright:
            reasons.append("image_too_dark")
        if mean_brightness > max_bright:
            reasons.append("image_too_bright")

        face_detected, face_area_ratio = self._face_area_ratio(gray, w, h)
        flags["face_detected"] = face_detected
        flags["face_area_ratio"] = face_area_ratio
        flags["face_detected_via_landmarks"] = False
        flags["landmark_face_area_ratio"] = 0.0

        if landmarks is not None:
            landmark_ratio = self._landmark_face_area_ratio(landmarks, w, h)
            flags["landmark_face_area_ratio"] = landmark_ratio
            if not face_detected and bool(self.cfg.get("allow_landmark_face_fallback", False)):
                if landmark_ratio > 0.0:
                    face_detected = True
                    face_area_ratio = landmark_ratio
                    flags["face_detected"] = True
                    flags["face_area_ratio"] = face_area_ratio
                    flags["face_detected_via_landmarks"] = True

        require_face = bool(self.cfg["require_face_detection"])
        if require_face and not face_detected:
            reasons.append("no_face_detected")
        if face_detected:
            min_ratio = float(self.cfg["min_face_area_ratio"])
            if flags.get("face_detected_via_landmarks"):
                min_ratio = float(self.cfg.get("landmark_face_area_ratio_min", min_ratio))
            face_ok = face_area_ratio >= min_ratio
            flags["face_size_ok"] = face_ok
            if not face_ok:
                reasons.append("face_too_small")

        quality_pass = len(reasons) == 0
        return QualityResult(quality_pass=quality_pass, reasons=reasons, flags=flags)

    def _face_area_ratio(self, gray: np.ndarray, w: int, h: int) -> tuple[bool, float]:
        if self.haar is None:
            return False, 0.0
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return False, 0.0
        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        face_area_ratio = float((fw * fh) / max(1.0, float(w * h)))
        return True, face_area_ratio

    def _landmark_face_area_ratio(self, landmarks, w: int, h: int) -> float:
        xs = []
        ys = []
        for lm in landmarks:
            x = getattr(lm, "x", None)
            y = getattr(lm, "y", None)
            if x is None or y is None:
                continue
            xs.append(float(x))
            ys.append(float(y))
        if not xs or not ys:
            return 0.0
        max_x = max(xs)
        max_y = max(ys)
        if max_x > 1.5 or max_y > 1.5:
            min_x = max(0.0, min(xs))
            max_x = min(float(w), max_x)
            min_y = max(0.0, min(ys))
            max_y = min(float(h), max_y)
            face_w = max(0.0, max_x - min_x)
            face_h = max(0.0, max_y - min_y)
        else:
            min_x = max(0.0, min(xs))
            max_x = min(1.0, max_x)
            min_y = max(0.0, min(ys))
            max_y = min(1.0, max_y)
            face_w = max(0.0, (max_x - min_x) * float(w))
            face_h = max(0.0, (max_y - min_y) * float(h))
        if face_w <= 0.0 or face_h <= 0.0:
            return 0.0
        return float((face_w * face_h) / max(1.0, float(w * h)))
