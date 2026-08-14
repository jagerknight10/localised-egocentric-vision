"""Datasets and subject splits for cached frame features."""

from pathlib import Path
import re

import torch
from torch.utils.data import Dataset

from egovision.features.cache import load_feature_cache


class CachedFrameDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Flatten labeled cached sequences into independent frame examples."""

    def __init__(self, cache_paths: tuple[str | Path, ...], background_label: int = -1) -> None:
        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for path in cache_paths:
            cache = load_feature_cache(path)
            valid = cache["labels"] != background_label
            features.append(cache["features"][valid].float())
            labels.append(cache["labels"][valid].long())
        if not features:
            raise ValueError("No cache files were provided.")
        self.features = torch.cat(features)
        self.labels = torch.cat(labels)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class CachedWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Return chronological windows from cached feature sequences."""

    def __init__(
        self,
        cache_paths: tuple[str | Path, ...],
        context: int = 128,
        background_label: int = -1,
    ) -> None:
        if context <= 0:
            raise ValueError("context must be positive.")
        self.windows: list[tuple[torch.Tensor, torch.Tensor]] = []
        for path in cache_paths:
            cache = load_feature_cache(path)
            features = cache["features"].float()
            labels = cache["labels"].long()
            if features.shape[0] != labels.shape[0]:
                raise ValueError(f"Unaligned cache: {path}")
            for start in range(0, features.shape[0], context):
                end = min(start + context, features.shape[0])
                window_features = features[start:end]
                window_labels = labels[start:end]
                if torch.any(window_labels != background_label):
                    self.windows.append((window_features, window_labels))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[index]


def subject_id_from_cache(path: str | Path) -> str:
    """Extract a subject such as ``S1`` from a cache filename."""
    match = re.match(r"(S\d+)_", Path(path).stem)
    if match is None:
        raise ValueError(f"Could not extract subject ID from {path}")
    return match.group(1)


def split_by_subject(
    cache_paths: tuple[str | Path, ...], held_out_subject: str
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Return train/test caches without splitting frames from one subject."""
    train = tuple(Path(path) for path in cache_paths if subject_id_from_cache(path) != held_out_subject)
    test = tuple(Path(path) for path in cache_paths if subject_id_from_cache(path) == held_out_subject)
    if not test:
        raise ValueError(f"No caches found for held-out subject {held_out_subject}")
    return train, test
