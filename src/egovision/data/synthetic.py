"""Small deterministic sequence dataset for pipeline development."""

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SequenceSample:
    """One temporally aligned sequence."""

    features: torch.Tensor
    labels: torch.Tensor
    frame_indices: torch.Tensor


class SyntheticSequenceDataset(Dataset[SequenceSample]):
    """Generate fixed-shape feature sequences without downloading data."""

    def __init__(
        self,
        num_sequences: int = 4,
        sequence_length: int = 16,
        feature_dim: int = 8,
        num_classes: int = 3,
        seed: int = 0,
    ) -> None:
        if min(num_sequences, sequence_length, feature_dim, num_classes) <= 0:
            raise ValueError("Dataset dimensions must be positive.")

        generator = torch.Generator().manual_seed(seed)
        self._samples = tuple(
            SequenceSample(
                features=torch.randn(sequence_length, feature_dim, generator=generator),
                labels=torch.arange(sequence_length) % num_classes,
                frame_indices=torch.arange(sequence_length),
            )
            for _ in range(num_sequences)
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> SequenceSample:
        return self._samples[index]
