import torch

from egovision.data.cached import CachedFrameDataset, split_by_subject
from egovision.features.cache import save_feature_cache


def test_cached_dataset_filters_background_and_splits_subjects(tmp_path) -> None:
    paths = []
    for subject in ("S1", "S2"):
        path = tmp_path / f"{subject}_Tea_C1.pt"
        save_feature_cache(
            path,
            torch.randn(3, 4),
            torch.tensor([0, -1, 1]),
            torch.tensor([0, 1, 2]),
            {"video_id": path.stem},
        )
        paths.append(path)

    train, test = split_by_subject(tuple(paths), "S2")
    dataset = CachedFrameDataset(train)
    assert len(dataset) == 2
    assert len(test) == 1
