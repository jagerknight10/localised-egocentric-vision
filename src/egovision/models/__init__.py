"""Trainable models for action understanding."""

from .causal_transformer import CausalTemporalTransformer
from .spatial_allocator import (
    SpatialImportanceAllocator,
    box_to_importance_map,
    importance_kl_loss,
)

__all__ = [
    "CausalTemporalTransformer",
    "SpatialImportanceAllocator",
    "box_to_importance_map",
    "importance_kl_loss",
]
