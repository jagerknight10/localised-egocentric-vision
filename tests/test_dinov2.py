import numpy as np
import pytest
import torch

from egovision.features.dinov2 import DinoV2FeatureExtractor


def test_extractor_requires_explicit_weight_loading() -> None:
    extractor = DinoV2FeatureExtractor()
    with pytest.raises(RuntimeError, match="load"):
        extractor.extract(np.zeros((2, 32, 32, 3), dtype=np.uint8))


def test_extractor_records_model_and_device() -> None:
    extractor = DinoV2FeatureExtractor(device="cpu")
    assert extractor.model_id == "facebook/dinov2-small"
    assert extractor.device == torch.device("cpu")
