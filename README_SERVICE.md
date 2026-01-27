# Wrinkle V2 Service

This service provides **measurement-based wrinkle scoring** from a single face image. It does **not** predict age and makes no medical claims. The output is derived from segmentation mask geometry and region-specific measurements (area ratio, skeleton length, thickness, density), plus a deterministic reasoning string.

## High-Level Architecture
1) **Quality gate** (blur, brightness, face size/detection)\n
2) **Segmentation** (UNet) -> binary wrinkle mask\n
3) **Region mapping** (face landmarks)\n
4) **Measurement-based scoring** (no model confidence used)\n
5) **Reasoning** (deterministic, rule-based)\n

## API Contract

### GET /health
Response:
```json
{"ok": true}
```

### GET /ready
Response:
```json
{"ready": true}
```

### GET /metrics
Prometheus metrics endpoint (counters, histograms). Response is text/plain.

### POST /analyze
Multipart form upload with field `file`.

#### OK Response Schema
```json
{
  "status": "OK",
  "score": 42.7,
  "reasons": null,
  "top_regions": ["forehead:78.2", "nasolabial_left:65.6", "under_eye_right:62.1"],
  "reasoning": "Quality gate passed. ...",
  "annotated_image_base64": "<base64 png>",
  "quality": {
    "quality_pass": true,
    "reasons": [],
    "flags": {
      "blur_value": 85.1,
      "blur_ok": true,
      "mean_brightness": 122.3,
      "brightness_ok": true,
      "face_detected": true,
      "face_area_ratio": 0.14
    }
  },
  "request_id": "a7e2c7b1-acde-4a65-9b1d-4b0e2d4c51e4",
  "latency_ms": 182.4
}
```
`annotated_image_base64` is optional and only returned when `service.return_annotated_image` is enabled.
It is intended for QA/debug/review overlays (mask + region contours).
Note: local macOS runs may fail due to a PyTorch/OpenMP SHM2 runtime issue; containerized Linux execution is the supported path.
For QA/debug, set `WRINKLE_RETURN_DEBUG_MASKS=true` to include raw `wrinkle_mask_base64`,
`wrinkle_prob_base64`, `wrinkle_prob_gamma_base64`, and `wrinkle_thresh_masks_base64` (plus
optionally `region_masks_base64`) in OK responses. The prob/gamma maps help inspect thin-wrinkle
signal that can be lost in binary thresholding, and the threshold sweep shows sensitivity.

#### NO_SCORE Response Schema
```json
{
  "status": "NO_SCORE",
  "score": null,
  "reasons": ["no_face_detected"],
  "top_regions": null,
  "reasoning": "No score due to quality gate: no_face_detected.",
  "quality": {
    "quality_pass": false,
    "reasons": ["no_face_detected"],
    "flags": {
      "blur_value": 64.2,
      "blur_ok": true,
      "mean_brightness": 110.7,
      "brightness_ok": true,
      "face_detected": false,
      "face_area_ratio": 0.0
    }
  },
  "request_id": "5c8a8ef5-5146-4d19-98d1-8c0f4cf63d3b",
  "latency_ms": 120.1
}
```

#### ERROR Response Schema
```json
{
  "status": "ERROR",
  "score": null,
  "reasons": ["pipeline_error"],
  "top_regions": null,
  "reasoning": null,
  "quality": null,
  "request_id": "dbbb7f29-98b7-4e5d-9e55-60a36e04efaa",
  "latency_ms": 40.0
}
```

## OpenAPI / Swagger
- Swagger UI: `GET /docs`
- Example analyze:
  ```bash
  curl -F "file=@sample_images_real/Ekran Resmi 2026-01-04 20.35.16.png" http://localhost:8000/analyze
  ```
- Metrics:
  ```bash
  curl http://localhost:8000/metrics
  ```

## Docker
Build:
```bash
docker build -f Dockerfile.service -t wrinkle-v2-service .
```
Run:
```bash
docker run --rm -p 8000:8000 wrinkle-v2-service
```

## Environment Variables
- `WRINKLE_CFG` (default: `configs/default.yaml`)
- `WRINKLE_SEG_INPUT_SIZE` (optional, e.g. `768`): overrides segmentation input size; higher values increase latency but can surface thin-wrinkle signal in debug masks.

## Production Notes
- Stateless service; scale horizontally.\n
- Model and config loaded once at startup.\n
- Observability included: structured logs with `request_id`, Prometheus metrics.\n
- Scoring is measurement-based only (no model confidence).\n
