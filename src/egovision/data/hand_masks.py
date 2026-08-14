"""Parse sparse GTEA hand polygons and create restricted RGB crops."""

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class HandMask:
    """Combined hand polygon box for one original video frame."""

    frame_index: int
    box: tuple[int, int, int, int]


def parse_hand_mask(path: str | Path, margin: int = 16) -> HandMask:
    """Parse all hand polygons and return one clipped bounding box."""
    root = ET.parse(path).getroot()
    points = [
        (int(point.findtext("x")), int(point.findtext("y")))
        for point in root.findall(".//object/polygon/pt")
    ]
    if not points:
        raise ValueError(f"No hand polygon points found in {path}")
    width = int(root.findtext("imagesize/ncols"))
    height = int(root.findtext("imagesize/nrows"))
    x_values, y_values = zip(*points)
    x0 = max(0, min(x_values) - margin)
    y0 = max(0, min(y_values) - margin)
    x1 = min(width, max(x_values) + margin + 1)
    y1 = min(height, max(y_values) + margin + 1)
    match = re.search(r"_(\d+)\.xml$", Path(path).name)
    if match is None:
        raise ValueError(f"Could not read frame index from {path}")
    return HandMask(int(match.group(1)), (x0, y0, x1, y1))


def index_hand_masks(annotation_dir: str | Path, margin: int = 16) -> dict[str, dict[int, HandMask]]:
    """Index masks by normalized video stem and original frame number."""
    index: dict[str, dict[int, HandMask]] = {}
    for path in Path(annotation_dir).glob("*.xml"):
        name = path.stem.lower()
        video_key, _, frame_text = name.rpartition("_")
        mask = parse_hand_mask(path, margin=margin)
        index.setdefault(video_key, {})[int(frame_text)] = mask
    return index


def crop_frame(frame: np.ndarray, mask: HandMask) -> np.ndarray:
    """Crop an RGB ``[H, W, 3]`` frame using a mask box."""
    x0, y0, x1, y1 = mask.box
    height, width = frame.shape[:2]
    return frame[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
