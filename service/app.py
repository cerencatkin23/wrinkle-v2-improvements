import os
from pathlib import Path
import io
import requests
import numpy as np
from PIL import Image

import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response

# ----------------------------
# Config
# ----------------------------
DRIVE_FILE_ID = os.environ.get("WRINKLE_TS_DRIVE_ID", "1zoski3gZwCH5dCa83wNJFe4kGptfaceQ")
ASSETS_DIR = Path(os.environ.get("WRINKLE_ASSETS_DIR", "service/assets"))
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ASSETS_DIR / "wrinkle_unet_best.ts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = int(os.environ.get("WRINKLE_IMG_SIZE", "512"))

# ----------------------------
# Download from Google Drive
# ----------------------------
def _download_gdrive(file_id: str, out_path: Path, chunk_size: int = 1 << 20):
    if out_path.exists() and out_path.stat().st_size > 5_000_000:
        print(f"✅ Model cached: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
        return

    url = "https://drive.google.com/uc?export=download"
    s = requests.Session()

    r = s.get(url, params={"id": file_id}, stream=True, timeout=60)
    r.raise_for_status()

    token = None
    for k, v in r.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    params = {"id": file_id, "confirm": token} if token else {"id": file_id}
    r = s.get(url, params=params, stream=True, timeout=60)
    r.raise_for_status()

    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    tmp.replace(out_path)
    print(f"✅ Downloaded: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

# ----------------------------
# Pre/post
# ----------------------------
def preprocess(pil: Image.Image, size: int):
    pil = pil.convert("RGB").resize((size, size))
    x = np.asarray(pil).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = torch.from_numpy(x).unsqueeze(0)  # 1CHW
    return x, pil

def overlay_mask(pil: Image.Image, mask: np.ndarray, alpha=0.45):
    img = np.asarray(pil).astype(np.float32)
    red = np.zeros_like(img)
    red[..., 0] = 255

    m = (mask > 0.5).astype(np.float32)[..., None]  # H W 1
    out = img * (1 - m * alpha) + red * (m * alpha)
    return Image.fromarray(out.astype(np.uint8))

# ----------------------------
# App + model load
# ----------------------------
app = FastAPI()

_download_gdrive(DRIVE_FILE_ID, MODEL_PATH)

model = torch.jit.load(str(MODEL_PATH), map_location=DEVICE)
model.eval()
print("✅ TorchScript model loaded on", DEVICE)

@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    pil = Image.open(io.BytesIO(data))

    x, pil_resized = preprocess(pil, IMG_SIZE)
    x = x.to(DEVICE)

    with torch.no_grad():
        logits = model(x)  # expected [1,1,H,W]
        probs = torch.sigmoid(logits)
        mask = probs[0, 0].detach().cpu().numpy()

    out = overlay_mask(pil_resized, mask)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
