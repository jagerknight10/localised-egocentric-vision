import pytest
import torch

from egovision.data.annotations import ActionSegment, align_frame_labels


def test_alignment_uses_inclusive_segment_bounds() -> None:
    frames = torch.tensor([0, 2, 4, 6, 8])
    segments = (ActionSegment(2, 4, 7), ActionSegment(6, 6, 3))
    assert align_frame_labels(frames, segments).tolist() == [-1, 7, 7, 3, -1]


def test_overlapping_segments_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        align_frame_labels(
            torch.tensor([3]),
            (ActionSegment(1, 3, 1), ActionSegment(3, 5, 2)),
        )
