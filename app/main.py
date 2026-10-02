"""
Smoking Detection API — petrol factory deployment (zero-tolerance).

Two-stage pipeline per video clip:
  1. Behaviour model (ResNet18+LSTM, trained on HMDB51 smoke/not-smoke clips)
     watches the full 16-frame clip for the smoking ACTION. This is the
     primary trigger: tuned for high recall, since a missed real event in a
     petrol facility is far more costly than a false alarm a human reviews
     in seconds.
  2. Only when the behaviour model says "Smoking", the YOLO cigarette
     detector runs on the clip's middle frame to localize the cigarette in
     a bounding box, giving a concrete evidence image for the alert rather
     than a bare "smoking" flag.

Every triggered alert is logged to a local SQLite database for a permanent,
reviewable history.
"""
import os
import sqlite3
import tempfile
import time
from contextlib import asynccontextmanager

import cv2
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from ultralytics import YOLO

from .behaviour_model import load_behaviour_model
from .video_utils import extract_clip_tensor, middle_frame

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DETECTOR_MODEL_PATH = os.getenv("DETECTOR_MODEL_PATH", "models/best_v3.pt")
BEHAVIOUR_MODEL_PATH = os.getenv("BEHAVIOUR_MODEL_PATH", "models/best_behavior_model.pth")
DETECTOR_CONF_THRESHOLD = float(os.getenv("DETECTOR_CONF_THRESHOLD", "0.3"))
ALERT_DEDUP_SECONDS = float(os.getenv("ALERT_DEDUP_SECONDS", "30"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
DB_PATH = os.getenv("DB_PATH", "alerts.db")
MEDIA_DIR = os.getenv("MEDIA_DIR", "media")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
os.makedirs(MEDIA_DIR, exist_ok=True)

state = {}


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            behaviour_confidence REAL NOT NULL,
            detector_confidence REAL,
            detector_bbox TEXT
        )
    """)
    conn.commit()
    conn.close()


def _log_alert(zone_id: str, now: float, behaviour_conf: float,
               detector_conf: float | None, detector_bbox: list | None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO alerts (zone_id, timestamp, behaviour_confidence, "
        "detector_confidence, detector_bbox) VALUES (?, ?, ?, ?, ?)",
        (zone_id, now, behaviour_conf, detector_conf,
         str(detector_bbox) if detector_bbox else None),
    )
    conn.commit()
    conn.close()


def _should_log_alert(zone_id: str, now: float) -> bool:
    conn = sqlite3.connect(DB_PATH)

    try:
        row = conn.execute(
            "SELECT MAX(timestamp) FROM alerts WHERE zone_id = ?",
            (zone_id,),
        ).fetchone()
    finally:
        conn.close()

    last_timestamp = row[0] if row else None

    return (
        last_timestamp is None
        or (now - float(last_timestamp)) >= ALERT_DEDUP_SECONDS
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(DETECTOR_MODEL_PATH):
        raise RuntimeError(f"Detector model not found at '{DETECTOR_MODEL_PATH}'")
    if not os.path.exists(BEHAVIOUR_MODEL_PATH):
        raise RuntimeError(f"Behaviour model not found at '{BEHAVIOUR_MODEL_PATH}'")

    state["detector"] = YOLO(DETECTOR_MODEL_PATH)
    state["behaviour"] = load_behaviour_model(BEHAVIOUR_MODEL_PATH)
    _init_db()
    yield
    state.clear()


app = FastAPI(
    title="Smoking Detection API — Petrol Factory (Zero Tolerance)",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_upload(file: UploadFile) -> bytes:
    data = file.file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
    return data


def _run_behaviour_model(video_path: str):
    clip = extract_clip_tensor(video_path)  # [1, 16, 3, 224, 224]
    with torch.no_grad():
        logits = state["behaviour"](clip)
        probs = torch.softmax(logits, dim=1)[0]
    not_smoking_conf = float(probs[0])
    smoking_conf = float(probs[1])
    is_smoking = smoking_conf > not_smoking_conf
    return is_smoking, smoking_conf


def _run_detector_on_frame(frame):
    result = state["detector"].predict(frame, conf=DETECTOR_CONF_THRESHOLD, verbose=False)[0]
    detections = []
    for box in result.boxes:
        detections.append({
            "label": result.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 3),
            "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
        })
    return result, detections


# ---------------------------------------------------------------------------

def _save_latest_media(video_path: str) -> dict:
    """
    Save the latest alert evidence:
      - latest_evidence.jpg
      - latest_annotated.mp4

    The evidence image uses the middle frame.
    The annotated video runs the cigarette detector on every frame,
    then converts the result to browser-friendly H.264 MP4.
    """
    evidence_path = os.path.join(MEDIA_DIR, "latest_evidence.jpg")
    annotated_raw_path = os.path.join(
        MEDIA_DIR,
        "latest_annotated_raw.mp4",
    )
    annotated_path = os.path.join(
        MEDIA_DIR,
        "latest_annotated.mp4",
    )

    # ---------------------------------------------------------
    # Evidence frame
    # ---------------------------------------------------------
    frame = middle_frame(video_path)
    result, _ = _run_detector_on_frame(frame)

    plotted = result.plot()

    if not cv2.imwrite(evidence_path, plotted):
        raise RuntimeError("Could not save evidence image")

    # ---------------------------------------------------------
    # Annotated raw video
    # ---------------------------------------------------------
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video for annotation: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0 or fps > 120:
        fps = 15.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video dimensions")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        annotated_raw_path,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not open raw video writer")

    try:
        while True:
            ok, current_frame = cap.read()

            if not ok:
                break

            current_result, _ = _run_detector_on_frame(current_frame)
            annotated_frame = current_result.plot()

            writer.write(annotated_frame)

    finally:
        writer.release()
        cap.release()

    if not os.path.exists(annotated_raw_path):
        raise RuntimeError(
            "Raw annotated video was not created"
        )

    # ---------------------------------------------------------
    # Convert to browser-friendly H.264 MP4
    # ---------------------------------------------------------
    import subprocess

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        annotated_raw_path,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        annotated_path,
    ]

    completed = subprocess.run(
        ffmpeg_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg conversion failed:\n"
            + completed.stderr[-3000:]
        )

    try:
        os.remove(annotated_raw_path)
    except OSError:
        pass

    if not os.path.exists(annotated_path):
        raise RuntimeError(
            "Browser-compatible annotated video was not created"
        )

    base = PUBLIC_BASE_URL.rstrip("/")

    return {
        "evidence_url": (
            base + "/alerts/latest/evidence"
        ),
        "annotated_video_url": (
            base + "/alerts/latest/video"
        ),
    }

# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "detector_model": DETECTOR_MODEL_PATH,
        "behaviour_model": BEHAVIOUR_MODEL_PATH,
        "detector_classes": state["detector"].names,
        "detector_conf_threshold": DETECTOR_CONF_THRESHOLD,
    }


@app.post("/detect/clip")
def detect_clip(zone_id: str, file: UploadFile = File(...)):
    """
    Submit a short video clip (a few seconds) for one zone.

    Stage 1: the behaviour model classifies the full clip as Smoking / Not
    Smoking (primary trigger, tuned for high recall).
    Stage 2: only if Smoking, the cigarette detector runs on the clip's
    middle frame to localize the cigarette for evidence.

    Any "Smoking" result is treated as an alert and logged immediately —
    no multi-frame confirmation delay, given the zero-tolerance requirement.
    """
    data = _read_upload(file)
    suffix = os.path.splitext(file.filename or "clip.mp4")[1] or ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        video_path = tmp.name

    try:
        is_smoking, behaviour_conf = _run_behaviour_model(video_path)

        response = {
            "zone_id": zone_id,
            "behaviour": {
                "smoking_detected": is_smoking,
                "confidence": round(behaviour_conf, 3),
            },
            "detector": None,
            "alert": False,
            "alert_logged": False,
            "evidence_url": None,
            "annotated_video_url": None,
        }

        if is_smoking:
            frame = middle_frame(video_path)
            _, detections = _run_detector_on_frame(frame)
            cig_hits = [d for d in detections if d["label"].lower() == "cigarette"]
            detector_conf = max((d["confidence"] for d in cig_hits), default=None)
            detector_bbox = cig_hits[0]["bbox"] if cig_hits else None

            response["detector"] = {
                "detections": detections,
                "localized": bool(cig_hits),
            }
            response["alert"] = True

            now = time.time()
            should_log = _should_log_alert(zone_id, now)

            if should_log:
                _log_alert(
                    zone_id,
                    now,
                    behaviour_conf,
                    detector_conf,
                    detector_bbox,
                )

                try:
                    media = _save_latest_media(video_path)

                    response["evidence_url"] = media["evidence_url"]
                    response["annotated_video_url"] = media["annotated_video_url"]

                except Exception as media_exc:
                    print(
                        f"WARNING: Could not generate alert media: {media_exc}",
                        flush=True,
                    )

            response["alert_logged"] = should_log

        return response
    finally:
        os.remove(video_path)


@app.post("/detect/clip/evidence_frame")
def detect_clip_evidence_frame(file: UploadFile = File(...)):
    """
    Debugging/demo helper: given a clip, returns the middle frame as a JPEG
    with the cigarette detector's boxes drawn on it (if any), regardless of
    what the behaviour model says. Useful for visually sanity-checking the
    detector stage on its own.
    """
    data = _read_upload(file)
    suffix = os.path.splitext(file.filename or "clip.mp4")[1] or ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        video_path = tmp.name

    try:
        frame = middle_frame(video_path)
        result, _ = _run_detector_on_frame(frame)
        ok, buf = cv2.imencode(".jpg", result.plot())
        return Response(content=buf.tobytes(), media_type="image/jpeg")
    finally:
        os.remove(video_path)



@app.get("/alerts/latest/evidence")
def latest_evidence():
    path = os.path.join(MEDIA_DIR, "latest_evidence.jpg")

    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="No evidence available yet",
        )

    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )


@app.get("/alerts/latest/video")
def latest_video():
    path = os.path.join(MEDIA_DIR, "latest_annotated.mp4")

    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="No annotated video available yet",
        )

    return FileResponse(
        path,
        media_type="video/mp4",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )




@app.get("/alerts/stats")
def alerts_stats():
    conn = sqlite3.connect(DB_PATH)

    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM alerts"
        ).fetchone()[0]

        localized = conn.execute(
            """
            SELECT COUNT(*)
            FROM alerts
            WHERE detector_bbox IS NOT NULL
            """
        ).fetchone()[0]

        avg_behaviour = conn.execute(
            "SELECT AVG(behaviour_confidence) FROM alerts"
        ).fetchone()[0]

        avg_detector = conn.execute(
            """
            SELECT AVG(detector_confidence)
            FROM alerts
            WHERE detector_confidence IS NOT NULL
            """
        ).fetchone()[0]

        zones = conn.execute(
            "SELECT COUNT(DISTINCT zone_id) FROM alerts"
        ).fetchone()[0]

        last_alert = conn.execute(
            "SELECT MAX(timestamp) FROM alerts"
        ).fetchone()[0]

    finally:
        conn.close()

    return {
        "total_alerts": int(total),
        "localized_alerts": int(localized),
        "not_localized_alerts": int(total - localized),
        "average_behaviour_confidence": (
            float(avg_behaviour) if avg_behaviour is not None else None
        ),
        "average_detector_confidence": (
            float(avg_detector) if avg_detector is not None else None
        ),
        "zones_with_alerts": int(zones),
        "last_alert_timestamp": (
            float(last_alert) if last_alert is not None else None
        ),
    }


@app.get("/alerts/stats/by-zone")
def alerts_by_zone():
    conn = sqlite3.connect(DB_PATH)

    try:
        rows = conn.execute(
            """
            SELECT zone_id, COUNT(*) AS alert_count
            FROM alerts
            GROUP BY zone_id
            ORDER BY alert_count DESC
            """
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "zone_id": row[0],
            "alert_count": int(row[1]),
        }
        for row in rows
    ]


@app.get("/alerts/history/dashboard")
def alerts_history_dashboard(limit: int = 50):
    conn = sqlite3.connect(DB_PATH)

    try:
        rows = conn.execute(
            """
            SELECT
                id,
                zone_id,
                timestamp,
                behaviour_confidence,
                detector_confidence,
                detector_bbox
            FROM alerts
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    from datetime import datetime, timezone

    return [
        {
            "id": int(row[0]),
            "zone_id": row[1],
            "timestamp": datetime.fromtimestamp(
                float(row[2]),
                tz=timezone.utc,
            ).isoformat(),
            "behaviour_confidence": float(row[3]),
            "detector_confidence": (
                float(row[4]) if row[4] is not None else None
            ),
            "localized": row[5] is not None,
            "detector_bbox": row[5],
        }
        for row in rows
    ]


@app.get("/alerts/latest/media")
def latest_media():
    base = PUBLIC_BASE_URL.rstrip("/")

    evidence_path = os.path.join(
        MEDIA_DIR,
        "latest_evidence.jpg",
    )

    video_path = os.path.join(
        MEDIA_DIR,
        "latest_annotated.mp4",
    )

    return {
        "evidence_url": (
            base + "/alerts/latest/evidence"
            if os.path.exists(evidence_path)
            else None
        ),
        "annotated_video_url": (
            base + "/alerts/latest/video"
            if os.path.exists(video_path)
            else None
        ),
    }

@app.get("/alerts/history")
def alerts_history(zone_id: str | None = None, limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if zone_id:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE zone_id = ? ORDER BY timestamp DESC LIMIT ?",
            (zone_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return {"count": len(rows), "alerts": [dict(r) for r in rows]}

