import os
import time
from typing import Dict, List, Tuple

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None
import numpy as np


class QualityGate:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        disable_mp = bool(int(os.environ.get("WRINKLE_DISABLE_MEDIAPIPE", "0")))
        try:
            os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
            import mediapipe as mp  # pylint: disable=import-error

            if disable_mp:
                self.face_mesh = None
            else:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.2,
                    min_tracking_confidence=0.2,
                )
                self.face_mesh.process(np.zeros((64, 64, 3), dtype=np.uint8))
        except Exception:
            self.face_mesh = None
        if cv2 is not None:
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        else:
            self.haar = None

    def evaluate(self, image: np.ndarray) -> Dict:
        if cv2 is None:
            return {
                "quality_pass": False,
                "reasons": ["opencv_not_installed"],
                "measurements": {},
            }
        reasons: List[str] = []
        measurements: Dict[str, float] = {}

        h, w = image.shape[:2]
        min_dim = min(h, w)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        measurements["blur_laplacian_var"] = float(blur_var)
        if blur_var < self.cfg["blur_laplacian_var"]:
            reasons.append("image_blurry")

        mean_brightness = float(np.mean(gray))
        measurements["mean_brightness"] = mean_brightness
        if mean_brightness < self.cfg["min_brightness"]:
            reasons.append("image_too_dark")
        if mean_brightness > self.cfg["max_brightness"]:
            reasons.append("image_too_bright")

        landmarks = None
        if self.face_mesh is not None:
            mp_image = self._resize_for_mediapipe(image)
            for _ in range(5):
                results = self.face_mesh.process(mp_image)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    break
                time.sleep(0.1)

        if landmarks is None:
            face_box = self._haar_face_box(gray)
            if face_box is None:
                return {
                    "quality_pass": False,
                    "reasons": ["no_face_detected"],
                    "measurements": measurements,
                }
            x, y, fw, fh = face_box
            face_size = min(fw, fh) / float(min_dim)
            measurements["face_size_ratio"] = float(face_size)
            if face_size < self.cfg["min_face_size"]:
                reasons.append("face_too_small")
            if not self.cfg.get("allow_haar_fallback", False):
                reasons.append("landmarks_missing")
                return {
                    "quality_pass": False,
                    "reasons": reasons,
                    "measurements": measurements,
                }
            measurements["landmark_fallback_used"] = 1.0
            quality_pass = len(reasons) == 0
            return {
                "quality_pass": quality_pass,
                "reasons": reasons,
                "measurements": measurements,
            }
        missing = 0
        xs = []
        ys = []
        for lm in landmarks:
            if lm.x < 0 or lm.x > 1 or lm.y < 0 or lm.y > 1:
                missing += 1
            xs.append(lm.x)
            ys.append(lm.y)
        measurements["missing_landmarks"] = float(missing)
        if missing > self.cfg["max_missing_landmarks"]:
            reasons.append("landmarks_missing")

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        face_w = (max_x - min_x) * w
        face_h = (max_y - min_y) * h
        face_size = min(face_w, face_h) / float(min_dim)
        measurements["face_size_ratio"] = float(face_size)
        if face_size < self.cfg["min_face_size"]:
            reasons.append("face_too_small")

        yaw, pitch, roll = self._estimate_pose(landmarks)
        measurements["yaw"] = float(yaw)
        measurements["pitch"] = float(pitch)
        measurements["roll"] = float(roll)
        if abs(yaw) > self.cfg["max_abs_yaw"]:
            reasons.append("yaw_too_large")
        if abs(pitch) > self.cfg["max_abs_pitch"]:
            reasons.append("pitch_too_large")
        if abs(roll) > self.cfg["max_abs_roll"]:
            reasons.append("roll_too_large")

        quality_pass = len(reasons) == 0
        return {
            "quality_pass": quality_pass,
            "reasons": reasons,
            "measurements": measurements,
        }

    def _estimate_pose(self, landmarks) -> Tuple[float, float, float]:
        left_eye_outer = landmarks[33]
        right_eye_outer = landmarks[263]
        nose_tip = landmarks[1]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]

        eye_dx = right_eye_outer.x - left_eye_outer.x
        eye_dy = right_eye_outer.y - left_eye_outer.y
        roll = np.degrees(np.arctan2(eye_dy, eye_dx))

        eye_center_x = (left_eye_outer.x + right_eye_outer.x) / 2.0
        mouth_center_x = (left_mouth.x + right_mouth.x) / 2.0
        yaw = np.degrees(np.arctan2(nose_tip.x - eye_center_x, eye_dx)) * 1.5

        eye_center_y = (left_eye_outer.y + right_eye_outer.y) / 2.0
        mouth_center_y = (left_mouth.y + right_mouth.y) / 2.0
        pitch = np.degrees(np.arctan2(nose_tip.y - eye_center_y, mouth_center_y - eye_center_y)) * 1.2

        return float(yaw), float(pitch), float(roll)

    def _haar_face_box(self, gray: np.ndarray):
        if self.haar is None:
            return None
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        return max(faces, key=lambda b: b[2] * b[3])

    def _resize_for_mediapipe(self, image: np.ndarray, max_size: int = 512) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) <= max_size:
            return image
        scale = max_size / float(max(h, w))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        if cv2 is None:
            from PIL import Image

            return np.array(Image.fromarray(image).resize((new_w, new_h), resample=Image.BILINEAR))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
