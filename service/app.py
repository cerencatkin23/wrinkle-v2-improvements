from __future__ import annotations

import io
import logging
import os
import time
import uuid
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi import Request
from fastapi.responses import JSONResponse, Response
from PIL import Image
try:
    import prometheus_client  # noqa: F401
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    PROM_ENABLED = True
except ImportError:  # pragma: no cover - fallback for optional dependency
    PROM_ENABLED = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    def generate_latest() -> bytes:
        return b""

try:
    from pythonjsonlogger import jsonlogger

    JSONLOG_ENABLED = True
except ImportError:  # pragma: no cover - fallback for optional dependency
    jsonlogger = None
    JSONLOG_ENABLED = False

import sys


def _ensure_src_on_path() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_ensure_src_on_path()

from wrinkle_v2.pipeline.predictor import WrinklePipeline  # noqa: E402
from wrinkle_v2.annotate_image import build_annotated_image_base64  # noqa: E402
from wrinkle_v2.mask_encode import mask_to_png_base64, prob_to_png_base64  # noqa: E402
from wrinkle_v2.utils import load_config  # noqa: E402
from service.metrics import (  # noqa: E402
    LATENCY_MS,
    NO_SCORE_REASONS,
    PROM_ENABLED as METRICS_ENABLED,
    REQUEST_COUNT,
    RESULT_COUNT,
)

app = FastAPI()

PIPELINE: Optional[WrinklePipeline] = None
READY: bool = False
READY_ERROR: Optional[str] = None
RETURN_ANNOTATED_IMAGE: bool = False
RETURN_DEBUG_MASKS: bool = False

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES = 10 * 1024 * 1024


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("wrinkle_service")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    if JSONLOG_ENABLED and jsonlogger is not None:
        formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(message)s %(request_id)s")
    else:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s %(request_id)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


LOGGER = _configure_logger()


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@app.on_event("startup")
def _startup() -> None:
    global PIPELINE, READY, READY_ERROR, RETURN_ANNOTATED_IMAGE, RETURN_DEBUG_MASKS
    cfg_path = os.environ.get("WRINKLE_CFG", "configs/default.yaml")
    try:
        cfg = load_config(cfg_path)
        PIPELINE = WrinklePipeline(cfg)
        service_cfg = cfg.get("service", {})
        RETURN_ANNOTATED_IMAGE = bool(service_cfg.get("return_annotated_image", False))
        RETURN_DEBUG_MASKS = bool(service_cfg.get("return_debug_masks", False))
        env_flag = os.environ.get("WRINKLE_RETURN_ANNOTATED_IMAGE")
        if env_flag is not None:
            RETURN_ANNOTATED_IMAGE = _parse_bool(env_flag)
        env_masks_flag = os.environ.get("WRINKLE_RETURN_DEBUG_MASKS")
        if env_masks_flag is not None:
            RETURN_DEBUG_MASKS = _parse_bool(env_masks_flag)
        READY = True
        READY_ERROR = None
    except Exception:
        PIPELINE = None
        READY = False
        READY_ERROR = repr(sys.exc_info()[1])
        LOGGER.exception("startup_failed", extra={"cfg_path": cfg_path})


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    request.state.request_id = request_id
    content_length = request.headers.get("content-length")
    LOGGER.info(
        "request_start",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else None,
            "content_length": content_length,
        },
    )
    start = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start) * 1000.0
    if METRICS_ENABLED:
        REQUEST_COUNT.labels(endpoint=request.url.path, http_status=str(response.status_code)).inc()
        LATENCY_MS.labels(endpoint=request.url.path).observe(latency_ms)
    LOGGER.info(
        "request_end",
        extra={
            "request_id": request_id,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 3),
            "result_status": getattr(request.state, "result_status", None),
        },
    )
    return response


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/ready")
def ready() -> dict:
    return {"ready": READY, "error": READY_ERROR}


@app.get("/metrics")
def metrics() -> Response:
    if not PROM_ENABLED:
        return JSONResponse(status_code=501, content={"available": False, "reason": "prometheus_client not installed"})
    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="unsupported_content_type")

    payload = await file.read()
    if len(payload) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="file_too_large")

    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="pipeline_not_ready")

    start = time.time()
    try:
        img = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="decode_failed")

    rgb = np.array(img)
    bgr = rgb[:, :, ::-1].copy()

    try:
        res = PIPELINE.predict(bgr)
    except Exception:
        LOGGER.exception("pipeline_crash", extra={"request_id": request.state.request_id})
        latency = (time.time() - start) * 1000.0
        if METRICS_ENABLED:
            RESULT_COUNT.labels(result_status="NO_SCORE").inc()
            NO_SCORE_REASONS.labels(reason="pipeline_error").inc()
        request.state.result_status = "NO_SCORE"
        return JSONResponse(
            status_code=200,
            content={
                "status": "NO_SCORE",
                "score": None,
                "reasons": ["pipeline_error"],
                "top_regions": None,
                "reasoning": "No score due to pipeline_error.",
                "quality": {"quality_pass": False, "reasons": ["pipeline_error"], "flags": {}},
                "request_id": request.state.request_id,
                "latency_ms": latency,
            },
        )

    quality = res.get("quality_gate", {})
    quality_pass = bool(quality.get("quality_pass", False))
    reasons = quality.get("reasons", [])
    reasoning = res.get("reasoning")
    per_region = res.get("per_region_scores", {}) or {}
    top_regions: Optional[List[str]] = None
    if quality_pass:
        top = sorted(per_region.items(), key=lambda x: x[1], reverse=True)[:3]
        top_regions = [f"{name}:{score:.1f}" for name, score in top]

    annotated_image_base64 = None
    if quality_pass and RETURN_ANNOTATED_IMAGE:
        try:
            mask = res.get("wrinkle_mask")
            region_masks = res.get("region_masks")
            prob = res.get("wrinkle_prob")
            if mask is not None and region_masks is not None:
                annotated_image_base64 = build_annotated_image_base64(
                    rgb, mask, wrinkle_prob=prob, region_masks=region_masks
                )
        except Exception:
            LOGGER.warning("annotation_failed", extra={"request_id": request.state.request_id})

    wrinkle_mask_base64 = None
    wrinkle_prob_base64 = None
    wrinkle_prob_gamma_base64 = None
    wrinkle_prob_gamma2_base64 = None
    wrinkle_thresh_masks_base64 = None
    region_masks_base64 = None
    if quality_pass and RETURN_DEBUG_MASKS:
        try:
            mask = res.get("wrinkle_mask")
            if mask is not None:
                wrinkle_mask_base64 = mask_to_png_base64(mask)
            prob = res.get("wrinkle_prob")
            if prob is not None:
                wrinkle_prob_base64 = prob_to_png_base64(prob, gamma=1.0)
                wrinkle_prob_gamma_base64 = prob_to_png_base64(prob, gamma=0.5)
                wrinkle_prob_gamma2_base64 = prob_to_png_base64(prob, gamma=0.33)
                thresholds = [0.15, 0.2, 0.3, 0.4]
                wrinkle_thresh_masks_base64 = {
                    f"t{int(t * 100):03d}": mask_to_png_base64((prob >= t).astype(np.uint8))
                    for t in thresholds
                }
            region_masks = res.get("region_masks")
            if region_masks:
                region_masks_base64 = {name: mask_to_png_base64(val) for name, val in region_masks.items()}
        except Exception:
            LOGGER.exception("debug_mask_failed", extra={"request_id": request.state.request_id})
            raise

    latency = (time.time() - start) * 1000.0
    result_status = "OK" if quality_pass else "NO_SCORE"
    request.state.result_status = result_status
    if METRICS_ENABLED:
        RESULT_COUNT.labels(result_status=result_status).inc()
    if not quality_pass and reasons:
        if METRICS_ENABLED:
            for reason in reasons:
                NO_SCORE_REASONS.labels(reason=reason).inc()
    response_payload = {
        "status": result_status,
        "score": res.get("global_score") if quality_pass else None,
        "reasons": reasons if reasons else None,
        "top_regions": top_regions,
        "reasoning": reasoning,
        "quality": quality if quality else None,
        "request_id": request.state.request_id,
        "latency_ms": latency,
    }
    if quality_pass and RETURN_ANNOTATED_IMAGE and annotated_image_base64:
        response_payload["annotated_image_base64"] = annotated_image_base64
    if quality_pass and RETURN_DEBUG_MASKS and wrinkle_mask_base64:
        response_payload["wrinkle_mask_base64"] = wrinkle_mask_base64
        if wrinkle_prob_base64:
            response_payload["wrinkle_prob_base64"] = wrinkle_prob_base64
        if wrinkle_prob_gamma_base64:
            response_payload["wrinkle_prob_gamma_base64"] = wrinkle_prob_gamma_base64
        if wrinkle_prob_gamma2_base64:
            response_payload["wrinkle_prob_gamma2_base64"] = wrinkle_prob_gamma2_base64
        if wrinkle_thresh_masks_base64:
            response_payload["wrinkle_thresh_masks_base64"] = wrinkle_thresh_masks_base64
        if region_masks_base64:
            response_payload["region_masks_base64"] = region_masks_base64
    return JSONResponse(content=response_payload)
