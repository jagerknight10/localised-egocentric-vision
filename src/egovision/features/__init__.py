"""Frozen visual feature extraction utilities."""

from .dinov2 import DinoV2FeatureExtractor
from .cache import load_feature_cache, save_feature_cache

__all__ = ["DinoV2FeatureExtractor", "load_feature_cache", "save_feature_cache"]
