"""Portable storage for extracted visual features."""

from pathlib import Path
from typing import Any

import torch


def save_feature_cache(
    path: str | Path,
    features: torch.Tensor,
    labels: torch.Tensor,
    frame_indices: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    """Save tensors and provenance metadata as one CPU-side artifact."""
    if features.ndim != 2 or labels.ndim != 1 or frame_indices.ndim != 1:
        raise ValueError("Expected features [T, D], labels [T], and frame_indices [T].")
    if features.shape[0] != labels.shape[0] or labels.shape != frame_indices.shape:
        raise ValueError("Features, labels, and frame_indices must share T.")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features.detach().cpu(),
            "labels": labels.detach().cpu(),
            "frame_indices": frame_indices.detach().cpu(),
            "metadata": dict(metadata),
        },
        output,
    )


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    """Load and validate a feature cache created by ``save_feature_cache``."""
    cache = torch.load(path, map_location="cpu", weights_only=True)
    required = {"features", "labels", "frame_indices", "metadata"}
    if set(cache) != required:
        raise ValueError(f"Cache must contain exactly {sorted(required)}")
    if cache["features"].shape[0] != cache["labels"].shape[0]:
        raise ValueError("Cached features and labels are not aligned.")
    return cache
