Handover Notes — Wrinkle V2 Service
==================================

Audience: Software/Backend team

## How to call the service
- Endpoint: `POST /analyze`
- Send multipart form with field `file` containing the image (jpeg/png/webp).
- Example:
  ```bash
  curl -F "file=@/path/to/image.jpg" http://<host>:8000/analyze
  ```

## Expected latency
- Current baseline (CPU) is ~150–400 ms per request depending on image size and system load.\n
- Latency is returned in the response field `latency_ms`.

## Status meanings
- `OK`: quality gate passed and score returned.
- `NO_SCORE`: quality gate failed (e.g., no face detected, too dark/blurred). No score is provided.
- `ERROR`: internal failure (pipeline error). Retry or fall back to a user-facing message.

## What NO_SCORE means for UX
- Do not show a numeric score.\n
- Show a short explanation or prompt: “Please use a clearer, front-facing photo.”\n
- Reasons are provided in `reasons` list.

## What the model does NOT do
- Does **not** estimate age.\n
- Does **not** make medical claims or diagnostics.\n
- Uses measurement-based scoring from a wrinkle mask only.

## Ownership boundaries
- ML team owns model behavior, configs, and scoring logic.\n
- Backend team owns deployment, scaling, gateway, timeouts, and monitoring.\n

