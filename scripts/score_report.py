import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

import urllib.error

READY_URL = os.environ.get("WRINKLE_READY_URL", "http://localhost:8000/ready")
READY_TIMEOUT_S = int(os.environ.get("WRINKLE_READY_TIMEOUT_S", "120"))
SERVICE_LOG_FILE = os.environ.get("WRINKLE_SERVICE_LOG", "")


def wait_ready(timeout_s: int = READY_TIMEOUT_S) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(READY_URL, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    body = resp.read().decode("utf-8", errors="replace")
                    data = json.loads(body)
                    if isinstance(data, dict) and data.get("ready") is True:
                        return
        except Exception:
            pass
        time.sleep(1)
    print(f"ERROR: Service not ready at {READY_URL} after {timeout_s}s")
    if SERVICE_LOG_FILE and Path(SERVICE_LOG_FILE).exists():
        print(f"--- Last 80 lines of {SERVICE_LOG_FILE} ---")
        try:
            lines = Path(SERVICE_LOG_FILE).read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines[-80:]:
                print(line)
        except Exception as exc:
            print(f"Failed to read log file: {exc}")
    raise SystemExit(2)


SERVICE_URL = os.environ.get("WRINKLE_SERVICE_URL", "http://localhost:8000/analyze")
INPUT_DIR = os.environ.get("WRINKLE_INPUT_DIR", "sample_images_real")

INCLUDE_WEBP = os.environ.get("WRINKLE_INCLUDE_WEBP", "0") in {"1", "true", "TRUE", "yes", "on"}
INCLUDE_AVIF = os.environ.get("WRINKLE_INCLUDE_AVIF", "0") in {"1", "true", "TRUE", "yes", "on"}
ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}
if INCLUDE_WEBP:
    ALLOWED_EXTS.add(".webp")
if INCLUDE_AVIF:
    ALLOWED_EXTS.add(".avif")

def post_multipart_image(url: str, file_path: Path, timeout: int = 60) -> Dict[str, Any]:
    """
    Send multipart/form-data request with a single field named 'file'.
    Uses only stdlib (urllib) to avoid dependency issues.
    """
    boundary = f"----Boundary{int(time.time() * 1000)}"
    crlf = "\r\n"
    filename = file_path.name

    content_type = "application/octet-stream"
    if file_path.suffix.lower() in [".jpg", ".jpeg"]:
        content_type = "image/jpeg"
    elif file_path.suffix.lower() == ".png":
        content_type = "image/png"
    elif file_path.suffix.lower() == ".webp":
        content_type = "image/webp"
    elif file_path.suffix.lower() == ".avif":
        content_type = "image/avif"

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    parts = []
    parts.append(f"--{boundary}{crlf}".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"{crlf}'.encode()
    )
    parts.append(f"Content-Type: {content_type}{crlf}{crlf}".encode())
    parts.append(file_bytes)
    parts.append(crlf.encode())
    parts.append(f"--{boundary}--{crlf}".encode())

    body = b"".join(parts)

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp_body = resp.read().decode("utf-8", errors="replace")
        return json.loads(resp_body)

def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def main():
    wait_ready()

    root = Path(INPUT_DIR)
    if not root.exists():
        raise SystemExit(f"Input dir not found: {root.resolve()}")

    files = [p for p in sorted(root.iterdir()) if p.is_file()]
    if not files:
        raise SystemExit(f"No images found in {root.resolve()} (exts: {sorted(ALLOWED_EXTS)})")

    rows: List[Dict[str, Any]] = []
    no_score_reasons = Counter()
    region_freq = Counter()
    ok_scores: List[float] = []

    for p in files:
        t0 = time.time()
        status = "ERROR"
        score = None
        reasons_str = ""
        quality_pass = None
        latency_ms = None
        top_regions_str = ""
        err = ""

        if p.suffix.lower() not in ALLOWED_EXTS:
            status = "SKIPPED"
            reasons_str = "unsupported_format"
            err = "skipped_unsupported_ext"
        else:
            try:
                data = post_multipart_image(SERVICE_URL, p, timeout=120)
                latency_ms = data.get("latency_ms")
                status = data.get("status") or "ERROR"
                score = data.get("score")
                reasons = data.get("reasons") or []
                if isinstance(reasons, list):
                    reasons_str = ";".join(str(r) for r in reasons)
                else:
                    reasons_str = str(reasons)

                quality_pass = safe_get(data, ["quality", "quality_pass"], None)
                top_regions = data.get("top_regions") or []
                if isinstance(top_regions, list):
                    top_regions_str = ";".join(top_regions)
                    # count region names only (split "name:val")
                    for tr in top_regions:
                        name = str(tr).split(":")[0].strip()
                        if name:
                            region_freq[name] += 1
                else:
                    top_regions_str = str(top_regions)

                if status == "OK" and isinstance(score, (int, float)):
                    ok_scores.append(float(score))
                elif status == "NO_SCORE":
                    for r in (data.get("reasons") or []):
                        no_score_reasons[str(r)] += 1

            except Exception as e:
                err = repr(e)
                # no_score_reasons not updated on hard errors

        dt_ms = (time.time() - t0) * 1000.0

        rows.append({
            "filename": p.name,
            "ext": p.suffix.lower(),
            "status": status,
            "score": score if isinstance(score, (int, float)) else "",
            "reasons": reasons_str,
            "quality_pass": quality_pass if isinstance(quality_pass, bool) else "",
            "latency_ms": latency_ms if isinstance(latency_ms, (int, float)) else "",
            "client_dt_ms": round(dt_ms, 2),
            "top_regions": top_regions_str,
            "error": err,
        })

    # Write CSV
    csv_path = Path("scores_report.csv")
    cols = ["filename","ext","status","score","reasons","quality_pass","latency_ms","client_dt_ms","top_regions","error"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            def esc(x: Any) -> str:
                s = "" if x is None else str(x)
                if any(c in s for c in [",", '"', "\n", "\r"]):
                    s = '"' + s.replace('"', '""') + '"'
                return s
            f.write(",".join(esc(r.get(c, "")) for c in cols) + "\n")

    total = len(rows)
    ok_count = sum(1 for r in rows if r["status"] == "OK")
    no_score_count = sum(1 for r in rows if r["status"] == "NO_SCORE")
    skipped_count = sum(1 for r in rows if r["status"] == "SKIPPED")
    error_count = total - ok_count - no_score_count - skipped_count

    summary: Dict[str, Any] = {
        "service_url": SERVICE_URL,
        "input_dir": str(root),
        "total_images": total,
        "ok_count": ok_count,
        "no_score_count": no_score_count,
        "error_count": error_count,
        "skipped_count": skipped_count,
        "ok_rate": ok_count / total,
        "no_score_rate": no_score_count / total,
        "skipped_rate": skipped_count / total,
        "score_stats": {},
        "histogram": {},
        "top_no_score_reasons": no_score_reasons.most_common(10),
        "top_regions_frequency": region_freq.most_common(15),
    }

    if ok_scores:
        ok_scores_sorted = sorted(ok_scores)
        summary["score_stats"] = {
            "min": min(ok_scores_sorted),
            "max": max(ok_scores_sorted),
            "mean": sum(ok_scores_sorted) / len(ok_scores_sorted),
            "median": statistics.median(ok_scores_sorted),
            "std": statistics.pstdev(ok_scores_sorted) if len(ok_scores_sorted) > 1 else 0.0,
            "n": len(ok_scores_sorted),
        }

        # histogram buckets
        buckets = [(0,20),(20,40),(40,60),(60,80),(80,100)]
        hist = {}
        for lo, hi in buckets:
            key = f"{lo}-{hi}"
            hist[key] = sum(1 for s in ok_scores_sorted if (s >= lo and (s < hi or (hi==100 and s<=100))))
        summary["histogram"] = hist

    json_path = Path("scores_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("OK/NO_SCORE/ERROR/SKIPPED:", ok_count, no_score_count, error_count, skipped_count, f"(total={total})")
    print(
        "Rates:",
        f"OK={ok_count/total:.2%}",
        f"NO_SCORE={no_score_count/total:.2%}",
        f"SKIPPED={skipped_count/total:.2%}",
    )
    if summary["score_stats"]:
        ss = summary["score_stats"]
        print("Score stats (OK only):",
              f"min={ss['min']:.2f}", f"max={ss['max']:.2f}",
              f"mean={ss['mean']:.2f}", f"median={ss['median']:.2f}", f"std={ss['std']:.2f}", f"n={ss['n']}")
        print("Histogram:", summary["histogram"])
    if no_score_reasons:
        print("Top NO_SCORE reasons:", no_score_reasons.most_common(5))
    if region_freq:
        print("Top regions frequency:", region_freq.most_common(10))

    print("\nSCORING REPORT GENERATED")
    print("CSV:", str(csv_path.resolve()))
    print("JSON:", str(json_path.resolve()))

if __name__ == "__main__":
    main()
