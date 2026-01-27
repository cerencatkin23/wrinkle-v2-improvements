import json
import os
import sys

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

sys.path.append(os.path.join(os.path.dirname(__file__), \"..\", \"src\"))

from wrinkle_v2.pipeline.predictor import WrinklePipeline
from wrinkle_v2.utils import load_config

app = FastAPI(title="Wrinkle V2")

_config_path = "configs/default.yaml"
if not os.path.exists(_config_path):
    _config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
cfg = load_config(_config_path)
pipeline = WrinklePipeline(cfg)


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    content = await image.read()
    data = np.frombuffer(content, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse(status_code=400, content={"error": "invalid_image"})
    result = pipeline.predict(bgr)
    return JSONResponse(content=json.loads(json.dumps(result)))
