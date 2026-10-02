"""
Video preprocessing for the behaviour model: extract 16 uniformly spaced
frames from a clip, resize to 224x224, convert to RGB, ImageNet-normalize,
and stack into the [1, 16, 3, 224, 224] tensor the model expects.
"""
import cv2
import numpy as np
import torch

from .behaviour_model import IMAGENET_MEAN, IMAGENET_STD

NUM_FRAMES = 16
FRAME_SIZE = 224


def extract_clip_tensor(video_path: str) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video: {video_path}")

    frames = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from: {video_path}")

    total_frames = len(frames)

    indices = np.linspace(
        0, total_frames - 1, NUM_FRAMES
    ).astype(int)

    selected = [frames[i] for i in indices]

    if len(selected) != NUM_FRAMES:
        raise ValueError(
            f"Expected {NUM_FRAMES} frames, got {len(selected)} from: {video_path}"
        )

    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    processed = []

    for frame in selected:
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0

        frame = (frame - mean) / std
        frame = np.transpose(frame, (2, 0, 1))
        processed.append(frame)

    clip = np.stack(processed, axis=0)
    tensor = torch.from_numpy(clip).unsqueeze(0).float()

    return tensor

def middle_frame(video_path: str) -> np.ndarray:
    """Returns the middle frame of the clip as a BGR numpy array,
    used to run the YOLO cigarette detector for a localized evidence box.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video: {video_path}")

    frames = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from: {video_path}")

    return frames[len(frames) // 2]