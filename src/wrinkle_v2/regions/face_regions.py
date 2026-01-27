import os
import time
from typing import Dict, List, Tuple

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None
import numpy as np
try:
    from skimage.morphology import skeletonize
except ImportError:  # pragma: no cover - optional dependency
    def skeletonize(*args, **kwargs):
        raise RuntimeError("skimage_not_installed")
from PIL import Image, ImageDraw


FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]

MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]

REGION_INDICES = {
    "forehead": {"top": FACE_OVAL[:18], "brow": LEFT_BROW + RIGHT_BROW[::-1]},
    "glabella": [9, 10, 151, 337, 336, 107, 66, 105, 63, 70],
    "bunny_lines": [6, 197, 195, 5, 4, 1, 168],
    "crows_feet_left": [33, 246, 161, 160, 159, 158, 157, 173, 133, 130, 226, 247],
    "crows_feet_right": [263, 466, 388, 387, 386, 385, 384, 398, 362, 359, 446, 467],
    "under_eye_left": [144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 130],
    "under_eye_right": [374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 359],
    "nasolabial_left": [205, 50, 187, 93, 61, 78, 191],
    "nasolabial_right": [425, 280, 411, 323, 291, 308, 415],
    "smoker_lines": MOUTH_OUTER,
    "chin": [84, 17, 314, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58],
}


class RegionMapper:
    def __init__(self):
        disable_mp = bool(int(os.environ.get("WRINKLE_DISABLE_MEDIAPIPE", "0")))
        if disable_mp:
            self.face_mesh = None
        else:
            try:
                os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
                import mediapipe as mp  # pylint: disable=import-error

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

    def get_landmarks(self, image: np.ndarray):
        if self.face_mesh is None:
            return None
        mp_image = self._resize_for_mediapipe(image)
        for _ in range(5):
            results = self.face_mesh.process(mp_image)
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0].landmark
            time.sleep(0.1)
        return None

    def face_mask(self, image_shape: Tuple[int, int], landmarks) -> np.ndarray:
        return self._polygon_mask(image_shape, self._landmarks_to_points(landmarks, FACE_OVAL, image_shape))

    def region_masks(self, image_shape: Tuple[int, int], landmarks) -> Dict[str, np.ndarray]:
        masks = {}
        for name, indices in REGION_INDICES.items():
            if name == "forehead":
                top = self._landmarks_to_points(landmarks, indices["top"], image_shape)
                brow = self._landmarks_to_points(landmarks, indices["brow"], image_shape)
                poly = top + brow[::-1]
            else:
                poly = self._landmarks_to_points(landmarks, indices, image_shape)
            masks[name] = self._polygon_mask(image_shape, poly)
        return masks

    def fallback_regions_from_haar(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if cv2 is None or self.haar is None:
            return None, None
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return None, None
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        h_img, w_img = image.shape[:2]

        face_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        center = (x + w // 2, y + h // 2)
        axes = (max(1, w // 2), max(1, h // 2))
        self._ellipse_mask(face_mask, center, axes)

        def rect_mask(x0, y0, x1, y1):
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            self._rect_mask(mask, x0, y0, x1, y1)
            return mask

        def clamp(val, lo, hi):
            return max(lo, min(hi, val))

        def r(xr0, yr0, xr1, yr1):
            x0 = clamp(int(round(x + xr0 * w)), 0, w_img - 1)
            x1 = clamp(int(round(x + xr1 * w)), 0, w_img - 1)
            y0 = clamp(int(round(y + yr0 * h)), 0, h_img - 1)
            y1 = clamp(int(round(y + yr1 * h)), 0, h_img - 1)
            return rect_mask(x0, y0, x1, y1)

        region_masks = {
            "forehead": r(0.15, 0.0, 0.85, 0.25),
            "glabella": r(0.4, 0.25, 0.6, 0.4),
            "bunny_lines": r(0.4, 0.35, 0.6, 0.55),
            "crows_feet_left": r(0.0, 0.25, 0.2, 0.5),
            "crows_feet_right": r(0.8, 0.25, 1.0, 0.5),
            "under_eye_left": r(0.2, 0.35, 0.4, 0.55),
            "under_eye_right": r(0.6, 0.35, 0.8, 0.55),
            "nasolabial_left": r(0.2, 0.55, 0.4, 0.75),
            "nasolabial_right": r(0.6, 0.55, 0.8, 0.75),
            "smoker_lines": r(0.35, 0.65, 0.65, 0.82),
            "chin": r(0.25, 0.8, 0.75, 1.0),
        }

        return face_mask, region_masks

    def fallback_regions_full_image(self, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        h_img, w_img = image_shape
        face_mask = np.ones((h_img, w_img), dtype=np.uint8)

        def rect_mask(x0, y0, x1, y1):
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            self._rect_mask(mask, x0, y0, x1, y1)
            return mask

        def clamp(val, lo, hi):
            return max(lo, min(hi, val))

        def r(xr0, yr0, xr1, yr1):
            x0 = clamp(int(round(xr0 * w_img)), 0, w_img - 1)
            x1 = clamp(int(round(xr1 * w_img)), 0, w_img - 1)
            y0 = clamp(int(round(yr0 * h_img)), 0, h_img - 1)
            y1 = clamp(int(round(yr1 * h_img)), 0, h_img - 1)
            return rect_mask(x0, y0, x1, y1)

        region_masks = {
            "forehead": r(0.15, 0.0, 0.85, 0.25),
            "glabella": r(0.4, 0.25, 0.6, 0.4),
            "bunny_lines": r(0.4, 0.35, 0.6, 0.55),
            "crows_feet_left": r(0.0, 0.25, 0.2, 0.5),
            "crows_feet_right": r(0.8, 0.25, 1.0, 0.5),
            "under_eye_left": r(0.2, 0.35, 0.4, 0.55),
            "under_eye_right": r(0.6, 0.35, 0.8, 0.55),
            "nasolabial_left": r(0.2, 0.55, 0.4, 0.75),
            "nasolabial_right": r(0.6, 0.55, 0.8, 0.75),
            "smoker_lines": r(0.35, 0.65, 0.65, 0.82),
            "chin": r(0.25, 0.8, 0.75, 1.0),
        }

        return face_mask, region_masks

    def compute_measurements(self, wrinkle_mask: np.ndarray, face_mask: np.ndarray, region_masks: Dict[str, np.ndarray]) -> Dict:
        if cv2 is None:
            raise RuntimeError("opencv_not_installed")
        measurements = {}
        face_area = max(1, int(np.sum(face_mask > 0)))
        wrinkle_area = int(np.sum((wrinkle_mask > 0) & (face_mask > 0)))
        area_ratio = wrinkle_area / float(face_area)

        skeleton = skeletonize(wrinkle_mask > 0)
        skeleton_length = int(np.sum(skeleton))

        dist = cv2.distanceTransform((wrinkle_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        thickness = float(2.0 * np.mean(dist[wrinkle_mask > 0])) if wrinkle_area > 0 else 0.0
        density = self._component_density(wrinkle_mask, face_mask)

        measurements["global"] = {
            "area_ratio": float(area_ratio),
            "coverage_pct": float(area_ratio * 100.0),
            "skeleton_length": float(skeleton_length),
            "thickness": float(thickness),
            "density": float(density),
            "face_area": float(face_area),
        }

        per_region = {}
        for name, mask in region_masks.items():
            region_area = max(1, int(np.sum(mask > 0)))
            region_wrinkle = int(np.sum((wrinkle_mask > 0) & (mask > 0)))
            region_ratio = region_wrinkle / float(region_area)
            region_skel = int(np.sum(skeleton & (mask > 0)))
            region_thickness = float(2.0 * np.mean(dist[(wrinkle_mask > 0) & (mask > 0)])) if region_wrinkle > 0 else 0.0
            region_density = self._component_density(wrinkle_mask, mask)
            per_region[name] = {
                "area_ratio": float(region_ratio),
                "coverage_pct": float(region_ratio * 100.0),
                "skeleton_length": float(region_skel),
                "skeleton_length_density": float(region_skel / float(region_area)),
                "thickness": float(region_thickness),
                "density": float(region_density),
                "region_area": float(region_area),
            }
        measurements["per_region"] = per_region
        return measurements

    def _landmarks_to_points(self, landmarks, indices: List[int], image_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        h, w = image_shape
        points = []
        for idx in indices:
            lm = landmarks[idx]
            x = int(round(lm.x * w))
            y = int(round(lm.y * h))
            points.append((x, y))
        return points

    def _polygon_mask(self, image_shape: Tuple[int, int], points: List[Tuple[int, int]]) -> np.ndarray:
        h, w = image_shape
        if not points:
            return np.zeros((h, w), dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        if cv2 is not None:
            cv2.fillPoly(mask, [pts], 1)
        else:
            img = Image.fromarray(mask)
            draw = ImageDraw.Draw(img)
            draw.polygon([tuple(p) for p in pts], outline=1, fill=1)
            mask = np.array(img, dtype=np.uint8)
        return mask

    def _component_density(self, wrinkle_mask: np.ndarray, region_mask: np.ndarray) -> float:
        if cv2 is None:
            raise RuntimeError("opencv_not_installed")
        region_area = max(1, int(np.sum(region_mask > 0)))
        masked = ((wrinkle_mask > 0) & (region_mask > 0)).astype(np.uint8)
        num, _ = cv2.connectedComponents(masked)
        comps = max(0, num - 1)
        return float(comps / region_area)

    def _resize_for_mediapipe(self, image: np.ndarray, max_size: int = 512) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) <= max_size:
            return image
        scale = max_size / float(max(h, w))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        if cv2 is None:
            return np.array(Image.fromarray(image).resize((new_w, new_h), resample=Image.BILINEAR))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _rect_mask(self, mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
        if cv2 is not None:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 1, -1)
            return
        img = Image.fromarray(mask)
        draw = ImageDraw.Draw(img)
        draw.rectangle([x0, y0, x1, y1], outline=1, fill=1)
        mask[:] = np.array(img, dtype=np.uint8)

    def _ellipse_mask(self, mask: np.ndarray, center: Tuple[int, int], axes: Tuple[int, int]) -> None:
        if cv2 is not None:
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
            return
        cx, cy = center
        ax, ay = axes
        bbox = [cx - ax, cy - ay, cx + ax, cy + ay]
        img = Image.fromarray(mask)
        draw = ImageDraw.Draw(img)
        draw.ellipse(bbox, outline=1, fill=1)
        mask[:] = np.array(img, dtype=np.uint8)
