import numpy as np
import pytest

from egovision.data.video import sample_frame_indices


def test_sampling_preserves_annotation_frame_numbers() -> None:
    np.testing.assert_array_equal(sample_frame_indices(10, stride=3), [0, 3, 6, 9])
    np.testing.assert_array_equal(sample_frame_indices(10, stride=3, start=1), [1, 4, 7])


def test_invalid_sampling_is_rejected() -> None:
    with pytest.raises(ValueError):
        sample_frame_indices(frame_count=0, stride=1)
    with pytest.raises(ValueError):
        sample_frame_indices(frame_count=10, stride=0)
