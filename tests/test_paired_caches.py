import torch

from egovision.features.cache import save_feature_cache
from scripts.make_paired_full_caches import run


def test_paired_cache_uses_same_indices_and_labels(tmp_path) -> None:
    full_dir, restricted_dir, output_dir = (tmp_path / name for name in ("full", "restricted", "out"))
    full_dir.mkdir()
    restricted_dir.mkdir()
    save_feature_cache(full_dir / "S1_Tea_C1.pt", torch.randn(4, 2), torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]), {})
    save_feature_cache(restricted_dir / "S1_Tea_C1.pt", torch.randn(2, 2), torch.tensor([1, 3]), torch.tensor([1, 3]), {})
    run(full_dir, restricted_dir, output_dir)
    cache = torch.load(output_dir / "S1_Tea_C1.pt", weights_only=True)
    assert cache["features"].shape == (2, 2)
    assert cache["labels"].tolist() == [1, 3]
