"""Video metadata, deterministic sampling, and RGB frame loading."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    """Basic properties needed to connect video frames to annotations."""

    path: Path
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class SampledVideo:
    """Sampled RGB frames and their original video-frame indices."""

    frames: np.ndarray
    frame_indices: np.ndarray
    metadata: VideoMetadata


def read_video_at_indices(path: str | Path, frame_indices: np.ndarray) -> tuple[list[np.ndarray], VideoMetadata]:
    """Decode only requested sorted frame indices as RGB images."""
    if frame_indices.ndim != 1 or frame_indices.dtype.kind not in "iu":
        raise ValueError("frame_indices must be a one-dimensional integer array.")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    metadata = VideoMetadata(
        path=Path(path),
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    if np.any(frame_indices < 0) or np.any(frame_indices >= metadata.frame_count):
        capture.release()
        raise ValueError("Requested frame index is outside the video.")
    frames: list[np.ndarray] = []
    for index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = capture.read()
        if not success:
            capture.release()
            raise ValueError(f"Could not decode frame {index} from {path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames, metadata


def sample_frame_indices(frame_count: int, stride: int, start: int = 0) -> np.ndarray:
    """Return valid frame numbers ``start, start + stride, ...``."""
    if frame_count <= 0 or stride <= 0 or start < 0 or start >= frame_count:
        raise ValueError("frame_count must be positive; stride and start must be valid.")
    return np.arange(start, frame_count, stride, dtype=np.int64)


def read_video(path: str | Path, stride: int = 1, start: int = 0) -> SampledVideo:
    """Read selected frames as RGB ``uint8`` images.

    OpenCV decodes images as BGR; conversion to RGB happens before returning.
    """
    video_path = Path(path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata = VideoMetadata(
        path=video_path,
        frame_count=frame_count,
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    indices = sample_frame_indices(frame_count, stride, start)
    frames: list[np.ndarray] = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()
        if not success:
            capture.release()
            raise ValueError(f"Could not decode frame {frame_index} from {video_path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return SampledVideo(np.stack(frames), indices, metadata)
