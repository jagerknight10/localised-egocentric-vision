"""Temporal annotation types and frame-to-label alignment."""

from dataclasses import dataclass

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


def align_frame_labels(
    frame_indices: torch.Tensor,
    segments: tuple[ActionSegment, ...],
    background_label: int = -1,
) -> torch.Tensor:
    """Assign one label to each sampled frame index.

    Unannotated frames receive ``background_label``. Overlapping segments are
    rejected because ambiguous labels would hide annotation errors.
    """
    if frame_indices.ndim != 1 or frame_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("frame_indices must be a one-dimensional integer tensor.")
    labels = torch.full_like(frame_indices, background_label)
    for segment in segments:
        selected = (frame_indices >= segment.start_frame) & (
            frame_indices <= segment.end_frame
        )
        if torch.any((labels != background_label) & selected):
            raise ValueError("Action segments overlap on a sampled frame.")
        labels[selected] = segment.label
    return labels
