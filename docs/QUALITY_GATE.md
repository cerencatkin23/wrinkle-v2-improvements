GlobalQualityGate
=================

Overview
- GlobalQualityGate is concern-agnostic and runs before any skin pipeline.
- It validates general photo quality (face detected, face size, blur, brightness).

Checks and thresholds (DEFAULT_QUALITY_CFG)
- require_face_detection: if true, no face -> NO_SCORE
- min_face_area_ratio: minimum face area / image area
- blur_laplacian_var: variance of Laplacian threshold
- min_brightness / max_brightness: mean brightness bounds

NO_SCORE triggers
- no_face_detected (when require_face_detection is true)
- face_too_small (face area below min_face_area_ratio)
- image_blurry (blur_value < blur_laplacian_var)
- image_too_dark / image_too_bright

Flags (soft signals)
- blur_value
- mean_brightness
- face_area_ratio
- blur_ok / brightness_ok / face_detected

Wrinkle scoring soft influence
- If the global gate passes, wrinkle scoring applies a small penalty (0–5 points)
  based on blur/brightness proximity to thresholds.
- Per-region scores are scaled proportionally with the global score.

Configuration
- Use configs/default.yaml for production defaults.
- Override in configs/local.yaml for experiments.

Usage
- GlobalQualityGate is used in src/wrinkle_v2/pipeline/predictor.py.
