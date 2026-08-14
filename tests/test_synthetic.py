import torch

from egovision.data.synthetic import SyntheticSequenceDataset


def test_synthetic_sample_shapes_and_alignment() -> None:
    sample = SyntheticSequenceDataset(
        num_sequences=1, sequence_length=5, feature_dim=4, num_classes=2
    )[0]

    assert sample.features.shape == (5, 4)
    assert sample.labels.shape == (5,)
    assert sample.frame_indices.tolist() == [0, 1, 2, 3, 4]
    assert sample.labels.tolist() == [0, 1, 0, 1, 0]


def test_synthetic_dataset_is_reproducible() -> None:
    first = SyntheticSequenceDataset(seed=7)[0]
    second = SyntheticSequenceDataset(seed=7)[0]
    assert torch.equal(first.features, second.features)
