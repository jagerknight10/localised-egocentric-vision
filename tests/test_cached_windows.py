import torch

from egovision.data.cached import CachedWindowDataset
from egovision.features.cache import save_feature_cache


def test_cached_windows_preserve_sequence_order(tmp_path) -> None:
    path = tmp_path / "S1_Tea_C1.pt"
    save_feature_cache(
        path,
        torch.arange(10, dtype=torch.float32).reshape(5, 2),
        torch.tensor([0, 1, -1, 2, 3]),
        torch.arange(5),
        {},
    )
    dataset = CachedWindowDataset((path,), context=3)
    assert len(dataset) == 2
    assert dataset[0][0][:, 0].tolist() == [0.0, 2.0, 4.0]
    assert dataset[1][0][:, 0].tolist() == [6.0, 8.0]
