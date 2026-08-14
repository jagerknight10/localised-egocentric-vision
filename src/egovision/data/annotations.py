"""Temporal annotation types and frame-to-label alignment."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

import torch


@dataclass(frozen=True)
class ActionSegment:
    """An action occupying inclusive video-frame indices."""

    start_frame: int
    end_frame: int
    label: int

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("Action segment must have valid inclusive frame bounds.")
        if self.label < 0:
            raise ValueError("Action labels must be non-negative.")


@dataclass(frozen=True)
class NamedActionSegment:
    """An annotated action before conversion to a numeric class ID."""

    name: str
    start_frame: int
    end_frame: int


_ACTION_PATTERN = re.compile(r"^<([^>]+)><([^>]*)> \((\d+)-(\d+)\) \[(?:0|1)\]$")


def parse_action_annotations(path: str | Path) -> tuple[NamedActionSegment, ...]:
    """Parse action lines and ignore object-presence lines."""
    segments: list[NamedActionSegment] = []
    for raw_line in Path(path).read_text().splitlines():
        match = _ACTION_PATTERN.match(raw_line.strip())
        if match is None:
            continue
        verb, nouns, start, end = match.groups()
        name = " ".join([verb, *nouns.replace(",", " ").split()]).strip()
        segments.append(NamedActionSegment(name, int(start), int(end)))
    if not segments:
        raise ValueError(f"No action annotations found in {path}")
    return tuple(segments)


def encode_action_segments(
    segments: tuple[NamedActionSegment, ...],
    label_map: dict[str, int] | None = None,
) -> tuple[tuple[ActionSegment, ...], dict[str, int]]:
    """Encode segments using a supplied map or a local deterministic map."""
    if label_map is None:
        names = sorted({segment.name for segment in segments})
        label_map = {name: index for index, name in enumerate(names)}
    missing = {segment.name for segment in segments} - label_map.keys()
    if missing:
        raise ValueError(f"Label map is missing actions: {sorted(missing)}")
    encoded = tuple(
        ActionSegment(segment.start_frame, segment.end_frame, label_map[segment.name])
        for segment in segments
    )
    return encoded, label_map


def build_global_label_map(
    annotation_paths: tuple[str | Path, ...],
) -> dict[str, int]:
    """Build one stable action-name-to-ID map from all annotation files."""
    names = {
        segment.name
        for path in annotation_paths
        for segment in parse_action_annotations(path)
    }
    return {name: index for index, name in enumerate(sorted(names))}


def align_frame_labels(
    frame_indices: torch.Tensor,
    segments: tuple[ActionSegment, ...],
    background_label: int = -1,
    overlap_policy: Literal["error", "later"] = "error",
) -> torch.Tensor:
    """Assign one label to each sampled frame index.

    Unannotated frames receive ``background_label``. With ``overlap_policy``
    set to ``"later"``, a later-listed action wins at a shared boundary.
    """
    if overlap_policy not in {"error", "later"}:
        raise ValueError("overlap_policy must be 'error' or 'later'.")
    if frame_indices.ndim != 1 or frame_indices.dtype not in (torch.int32, torch.int64):
        print(frame_indices.ndim, frame_indices.dtype)
        raise ValueError("frame_indices must be a one-dimensional integer tensor.")
    labels = torch.full_like(frame_indices, background_label)
    for segment in segments:
        selected = (frame_indices >= segment.start_frame) & (
            frame_indices <= segment.end_frame
        )
        if overlap_policy == "error" and torch.any((labels != background_label) & selected):
            raise ValueError("Action segments overlap on a sampled frame.")
        labels[selected] = segment.label
    return labels
