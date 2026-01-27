import os
from typing import List, Optional, Tuple

import gradio as gr
import requests
from fastapi import FastAPI

from service.app import app as api_app

# Mount FastAPI backend at /api and Gradio UI at /
app = FastAPI()
app.mount("/api", api_app)

PORT = os.environ.get("PORT", "7860")
ANALYZE_URL = f"http://127.0.0.1:{PORT}/api/analyze"
TIMEOUT_S = int(os.environ.get("WRINKLE_SERVICE_TIMEOUT_S", "120"))


def _format_top_regions(top_regions: Optional[List[str]]) -> str:
    if not top_regions:
        return "None"
    return "\n".join(f"- {r}" for r in top_regions)


def analyze_image(image) -> Tuple[str, str, str]:
    """
    Send the uploaded image to the local FastAPI /api/analyze endpoint.
    Map JSON response -> UI fields.
    """
    if image is None:
        return "No image provided.", "N/A", "None"

    try:
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        files = {"file": ("upload.png", buf.getvalue(), "image/png")}
        resp = requests.post(ANALYZE_URL, files=files, timeout=TIMEOUT_S)
    except Exception:
        return "Failed to reach the local service.", "N/A", "None"

    if resp.status_code >= 400:
        return f"Service error (HTTP {resp.status_code}).", "N/A", "None"

    try:
        data = resp.json()
    except Exception:
        return "Invalid response from service.", "N/A", "None"

    status = data.get("status")
    if status != "OK":
        reasons = data.get("reasons") or []
        reason_text = ", ".join(reasons) if isinstance(reasons, list) else str(reasons)
        msg = "No score available."
        if reason_text:
            msg += f" Reasons: {reason_text}"
        return msg, "N/A", "None"

    score = data.get("score")
    score_text = f"{float(score):.2f}" if isinstance(score, (int, float)) else "N/A"
    top_regions = data.get("top_regions")
    top_regions_text = _format_top_regions(top_regions)
    return "OK", score_text, top_regions_text


with gr.Blocks(title="Wrinkle Analysis Demo") as demo:
    gr.Markdown(
        "# Wrinkle Analysis Demo\n"
        "Upload a face image and click Analyze to receive a wrinkle score.\n"
        "This Space runs the FastAPI backend at `/api` and the UI at `/`."
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")

    analyze_btn = gr.Button("Analyze")
    status_out = gr.Textbox(label="Status", interactive=False)
    score_out = gr.Textbox(label="Overall Score", interactive=False)
    regions_out = gr.Markdown(label="Top Regions")

    analyze_btn.click(
        fn=analyze_image,
        inputs=[image_input],
        outputs=[status_out, score_out, regions_out],
    )


app = gr.mount_gradio_app(app, demo, path="/")
