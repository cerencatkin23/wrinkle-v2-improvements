from typing import Dict

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None
from PIL import Image
import numpy as np

from wrinkle_v2.calibration import ScoreCalibrator
from wrinkle_v2.quality.global_quality_gate import GlobalQualityGate
from wrinkle_v2.regions.face_regions import RegionMapper
from wrinkle_v2.scoring import build_reasoning, compute_scores
from wrinkle_v2.utils import unpad


class WrinklePipeline:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.quality = GlobalQualityGate(cfg.get("quality_gate", {}))
        self.region_mapper = RegionMapper()
        self.seg_model = None
        self.calibrator = ScoreCalibrator(cfg.get("calibration", {}))

    def predict(self, image_bgr: np.ndarray) -> Dict:
        if cv2 is not None:
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            image = image_bgr[:, :, ::-1].copy()
        landmarks = None
        allow_landmark_fallback = bool(self.cfg.get("quality_gate", {}).get("allow_landmark_face_fallback", False))
        if allow_landmark_fallback:
            landmarks = self.region_mapper.get_landmarks(image)
        quality = self.quality.evaluate(image, landmarks=landmarks)
        quality_dict = {
            "quality_pass": quality.quality_pass,
            "reasons": quality.reasons,
            "flags": quality.flags,
        }
        if not quality.quality_pass:
            return {
                "status": "NO_SCORE",
                "quality_gate": quality_dict,
                "global_score": None,
                "per_region_scores": None,
                "measurements": None,
                "reasoning": build_reasoning(quality_dict, {}, {}, self.cfg["scoring"]),
            }

        if landmarks is None:
            landmarks = self.region_mapper.get_landmarks(image)
        face_mask = None
        region_masks = None
        if landmarks is None:
            face_mask, region_masks = self.region_mapper.fallback_regions_from_haar(image)
            if face_mask is None or region_masks is None:
                if not self.cfg.get("quality_gate", {}).get("require_face_detection", True):
                    face_mask, region_masks = self.region_mapper.fallback_regions_full_image(image.shape[:2])
                else:
                    quality_dict["quality_pass"] = False
                    quality_dict["reasons"].append("no_face_landmarks")
                    return {
                        "status": "NO_SCORE",
                        "quality_gate": quality_dict,
                        "global_score": None,
                        "per_region_scores": None,
                        "measurements": None,
                        "reasoning": build_reasoning(quality_dict, {}, {}, self.cfg["scoring"]),
                    }

        if self.seg_model is None:
            from wrinkle_v2.pipeline.inference import SegmentationModel

            self.seg_model = SegmentationModel(self.cfg)
        mask, pads, probs = self.seg_model.predict_mask(image)
        mask_unpadded = unpad(mask, pads)
        prob_unpadded = unpad(probs, pads)
        if cv2 is not None:
            mask_resized = cv2.resize(
                mask_unpadded, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            prob_resized = cv2.resize(
                prob_unpadded, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        else:
            mask_resized = np.array(
                Image.fromarray(mask_unpadded).resize((image.shape[1], image.shape[0]), resample=Image.NEAREST)
            )
            prob_img = Image.fromarray(prob_unpadded.astype(np.float32), mode="F")
            prob_resized = np.array(prob_img.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR))
        prob_resized = np.clip(prob_resized, 0.0, 1.0)

        if face_mask is None or region_masks is None:
            face_mask = self.region_mapper.face_mask(image.shape[:2], landmarks)
            region_masks = self.region_mapper.region_masks(image.shape[:2], landmarks)
        enabled = set(self.cfg.get("regions", {}).get("enabled", []))
        if enabled:
            region_masks = {k: v for k, v in region_masks.items() if k in enabled}
        measurements = self.region_mapper.compute_measurements(mask_resized, face_mask, region_masks)
        measurements = self._apply_crows_feet_smoothing(mask_resized, region_masks, measurements)

        scores = compute_scores(measurements, self.cfg["scoring"])
        scores = self._apply_quality_penalty(scores, quality_dict, self.cfg["quality_gate"])
        calib_cfg = self.cfg.get("calibration", {})
        scores = self.calibrator.apply_scores(
            scores, age=calib_cfg.get("age"), skin_type=calib_cfg.get("skin_type")
        )
        reasoning = build_reasoning(quality_dict, measurements, scores, self.cfg["scoring"])

        return {
            "status": "OK",
            "quality_gate": quality_dict,
            "global_score": scores["global_score"],
            "per_region_scores": scores["per_region_scores"],
            "measurements": measurements,
            "reasoning": reasoning,
            "wrinkle_mask": mask_resized,
            "region_masks": region_masks,
            "wrinkle_prob": prob_resized,
        }

    def _apply_crows_feet_smoothing(self, wrinkle_mask: np.ndarray, region_masks: Dict, measurements: Dict) -> Dict:
        if cv2 is None:
            return measurements
        feat_cfg = self.cfg.get("scoring", {}).get("features", {}).get("crows_feet_invariant", {})
        if not feat_cfg.get("enabled", False):
            return measurements
        if not feat_cfg.get("smooth_mask", False):
            return measurements

        ksize = int(feat_cfg.get("smooth_ksize", 3))
        if ksize < 2:
            return measurements
        if ksize % 2 == 0:
            ksize += 1

        kernel = np.ones((ksize, ksize), dtype=np.uint8)
        for region in ("crows_feet_left", "crows_feet_right"):
            mask = region_masks.get(region)
            if mask is None:
                continue
            region_area = max(1, int(np.sum(mask > 0)))
            region_wrinkle = ((wrinkle_mask > 0) & (mask > 0)).astype(np.uint8)
            smoothed = cv2.morphologyEx(region_wrinkle, cv2.MORPH_OPEN, kernel)
            smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
            region_ratio = float(np.sum(smoothed > 0) / float(region_area))
            skel = skeletonize(smoothed > 0)
            skel_len = int(np.sum(skel))
            skel_density = float(skel_len / float(region_area))

            vals = dict(measurements["per_region"].get(region, {}))
            vals["area_ratio"] = region_ratio
            vals["density"] = region_ratio
            vals["skeleton_length"] = float(skel_len)
            vals["skeleton_length_density"] = skel_density
            measurements["per_region"][region] = vals

        return measurements

    def _apply_quality_penalty(self, scores: Dict, quality: Dict, quality_cfg: Dict) -> Dict:
        if not quality.get("quality_pass", False):
            return scores

        flags = quality.get("flags", {})
        blur_value = float(flags.get("blur_value", 0.0))
        blur_thr = float(quality_cfg.get("blur_laplacian_var", 80.0))
        mean_brightness = float(flags.get("mean_brightness", 0.0))
        min_bright = float(quality_cfg.get("min_brightness", 60.0))
        max_bright = float(quality_cfg.get("max_brightness", 200.0))

        penalty = 0.0
        if blur_value < blur_thr * 1.2:
            penalty += max(0.0, (blur_thr * 1.2 - blur_value) / max(1.0, blur_thr * 1.2)) * 5.0
        if mean_brightness < min_bright * 1.1:
            penalty += max(0.0, (min_bright * 1.1 - mean_brightness) / max(1.0, min_bright * 1.1)) * 3.0
        if mean_brightness > max_bright * 0.9:
            penalty += max(0.0, (mean_brightness - max_bright * 0.9) / max(1.0, max_bright * 0.1)) * 3.0

        penalty = min(5.0, penalty)
        if penalty <= 0.0:
            return scores

        global_score = max(0.0, scores["global_score"] - penalty)
        if scores["global_score"] > 0:
            ratio = global_score / scores["global_score"]
        else:
            ratio = 1.0

        per_region_scores = {k: max(0.0, v * ratio) for k, v in scores["per_region_scores"].items()}

        flags.setdefault("quality_penalty", penalty)
        quality["flags"] = flags
        scores["global_score"] = global_score
        scores["per_region_scores"] = per_region_scores
        return scores
