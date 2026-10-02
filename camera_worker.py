import os
import time
import cv2
import tempfile
import urllib.request
import urllib.parse
import uuid


API_URL = os.getenv("API_URL", "http://api:8000")
VIDEO_PATH = os.getenv("VIDEO_PATH", "")
ZONE_ID = os.getenv("ZONE_ID", "zone_1")

CLIP_SECONDS = float(os.getenv("CLIP_SECONDS", "2"))
FALLBACK_FPS = float(os.getenv("FALLBACK_FPS", "15"))
RESTART_DELAY = float(os.getenv("RESTART_DELAY", "1"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "120"))


def post_clip(video_path: str) -> None:
    boundary = "----SmokingDetectionBoundary" + uuid.uuid4().hex

    with open(video_path, "rb") as f:
        file_data = f.read()

    filename = os.path.basename(video_path)

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: video/x-msvideo\r\n"
        f"\r\n"
    ).encode("utf-8") + file_data + (
        f"\r\n--{boundary}--\r\n"
    ).encode("utf-8")

    url = (
        f"{API_URL}/detect/clip"
        f"?zone_id={urllib.parse.quote(ZONE_ID)}"
    )

    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=HTTP_TIMEOUT
    ) as response:
        result = response.read().decode("utf-8")

        print(
            f"[camera_worker] zone={ZONE_ID} "
            f"status={response.status} result={result}",
            flush=True,
        )


def capture_clip(cap) -> str | None:
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0 or fps > 120:
        fps = FALLBACK_FPS

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        return None

    frame_count_target = max(
        1,
        int(round(fps * CLIP_SECONDS))
    )

    frames = []

    for _ in range(frame_count_target):
        ok, frame = cap.read()

        if not ok:
            break

        frames.append(frame)

    if not frames:
        return None

    tmp = tempfile.NamedTemporaryFile(
        suffix=".avi",
        delete=False,
    )

    tmp_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    writer = cv2.VideoWriter(
        tmp_path,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        raise RuntimeError("Could not open VideoWriter")

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    return tmp_path


def run():
    if not VIDEO_PATH:
        raise RuntimeError(
            "VIDEO_PATH is not set."
        )

    if not os.path.exists(VIDEO_PATH):
        raise RuntimeError(
            f"Video file not found: {VIDEO_PATH}"
        )

    print(
        f"[camera_worker] Starting simulated camera",
        flush=True,
    )

    print(
        f"[camera_worker] zone={ZONE_ID}",
        flush=True,
    )

    print(
        f"[camera_worker] video={VIDEO_PATH}",
        flush=True,
    )

    print(
        f"[camera_worker] API={API_URL}",
        flush=True,
    )

    while True:
        cap = cv2.VideoCapture(VIDEO_PATH)

        if not cap.isOpened():
            print(
                "[camera_worker] Could not open video. "
                f"Retrying in {RESTART_DELAY}s...",
                flush=True,
            )

            cap.release()
            time.sleep(RESTART_DELAY)
            continue

        print(
            "[camera_worker] Simulated camera connected.",
            flush=True,
        )

        try:
            while True:
                clip_path = capture_clip(cap)

                if clip_path is None:
                    print(
                        "[camera_worker] End of video. "
                        "Restarting from beginning...",
                        flush=True,
                    )
                    break

                try:
                    post_clip(clip_path)

                except Exception as exc:
                    print(
                        f"[camera_worker] Inference request failed: {exc}",
                        flush=True,
                    )

                finally:
                    try:
                        os.remove(clip_path)
                    except OSError:
                        pass

        finally:
            cap.release()

        time.sleep(RESTART_DELAY)


if __name__ == "__main__":
    run()
