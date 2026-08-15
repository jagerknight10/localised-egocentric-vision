"""Create full-frame caches aligned to restricted-mask cache frame indices."""

from pathlib import Path
import torch

from egovision.features.cache import load_feature_cache, save_feature_cache


def run(
    full_dir: Path = Path("data/features"),
    restricted_dir: Path = Path("data/features_restricted"),
    output_dir: Path = Path("data/features_paired_full"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for restricted_path in sorted(restricted_dir.glob("*.pt")):
        full_path = full_dir / restricted_path.name
        if not full_path.exists():
            raise FileNotFoundError(f"Missing full cache: {full_path}")
        restricted = load_feature_cache(restricted_path)
        full = load_feature_cache(full_path)
        lookup = {int(index): position for position, index in enumerate(full["frame_indices"])}
        positions = [lookup[int(index)] for index in restricted["frame_indices"]]
        features = full["features"][positions]
        labels = full["labels"][positions]
        if not torch.equal(labels, restricted["labels"]):
            raise ValueError(f"Label mismatch for {restricted_path.name}")
        save_feature_cache(
            output_dir / restricted_path.name,
            features,
            labels,
            restricted["frame_indices"],
            {**full["metadata"], "paired_with": "oracle_hand_mask_crop"},
        )
        print(f"saved {output_dir / restricted_path.name} shape={tuple(features.shape)}")


if __name__ == "__main__":
    run()
