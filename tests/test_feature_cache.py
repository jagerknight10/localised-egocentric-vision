import torch

from egovision.features.cache import load_feature_cache, save_feature_cache


def test_feature_cache_round_trip(tmp_path) -> None:
    path = tmp_path / "sample.pt"
    features = torch.randn(3, 4)
    labels = torch.tensor([1, 2, 1])
    indices = torch.tensor([0, 15, 30])
    save_feature_cache(path, features, labels, indices, {"stride": 15})

    cache = load_feature_cache(path)
    assert torch.equal(cache["features"], features)
    assert torch.equal(cache["labels"], labels)
    assert torch.equal(cache["frame_indices"], indices)
    assert cache["metadata"]["stride"] == 15
