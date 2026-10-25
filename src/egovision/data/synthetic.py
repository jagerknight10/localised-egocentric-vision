"""Small deterministic sequence dataset for pipeline development."""

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SequenceSample:
    """One temporally aligned sequence."""

    features: torch.Tensor
    labels: torch.Tensor
    frame_indices: torch.Tensor


@dataclass(frozen=True)
class PatchSequenceSample:
    """One sequence of per-frame DINOv2-style patch tokens.

    Patches are 16x16 grid cells (row-major); a 4x4-cell box carries the
    hand-object signal and moves over time, while the rest of the grid
    is noise. The action label changes at segment boundaries, so a good
    allocator must track the moving box from content and the recognizer
    must use the localized evidence.
    """

    patches: torch.Tensor                     # [time, 256, patch_dim]
    labels: torch.Tensor                      # [time]
    boxes: list[tuple[int, int, int, int]]    # (x0, y0, x1, y1) in pixels
    frame_indices: torch.Tensor               # [time]


class SyntheticSequenceDataset(Dataset[SequenceSample]):
    """Generate fixed-shape feature sequences without downloading data."""

    def __init__(
        self,
        num_sequences: int = 4,
        sequence_length: int = 16,
        feature_dim: int = 8,
        num_classes: int = 3,
        seed: int = 0,
    ) -> None:
        if min(num_sequences, sequence_length, feature_dim, num_classes) <= 0:
            raise ValueError("Dataset dimensions must be positive.")

        generator = torch.Generator().manual_seed(seed)
        self._samples = tuple(
            SequenceSample(
                features=torch.randn(sequence_length, feature_dim, generator=generator),
                labels=torch.arange(sequence_length) % num_classes,
                frame_indices=torch.arange(sequence_length),
            )
            for _ in range(num_sequences)
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> SequenceSample:
        return self._samples[index]


def make_signal_patch_sequence(
    length: int = 48,
    grid: int = 16,
    patch_dim: int = 384,
    num_classes: int = 4,
    segment_length: int = 12,
    signal_std: float = 4.0,
    noise_std: float = 1.0,
    seed: int = 0,
) -> PatchSequenceSample:
    """Build a synthetic patch-token sequence with a moving signal box.

    The sequence is split into ``length // segment_length`` segments.
    Segment ``k`` has action label ``k % num_classes``; inside its box,
    patch tokens are noise + class signal + hand signal. Outside the
    box, tokens are pure noise. The box position is fixed per segment
    and jitters by one cell every other step within the segment, so the
    allocator must follow the hand from content, not from a fixed
    position prior.

    Args:
        length: timesteps; must be a multiple of segment_length.
        grid: patch grid size per side (DINOv2-small at 224px: 16).
        patch_dim: dimension per patch token (DINOv2-small: 384).
        num_classes: number of action classes.
        segment_length: timesteps per action segment.
        signal_std: scale of the in-box class/hand signal.
        noise_std: scale of the background noise.
        seed: RNG seed for reproducible generation.

    Returns:
        PatchSequenceSample with patches shaped [length, grid^2, patch_dim].
    """
    if length <= 0 or length % segment_length != 0:
        raise ValueError("length must be a positive multiple of segment_length.")
    if grid <= 0 or patch_dim <= 0 or num_classes <= 0:
        raise ValueError("grid, patch_dim, and num_classes must be positive.")

    generator = torch.Generator().manual_seed(seed)
    num_segments = length // segment_length
    num_patches = grid * grid

    # Fixed per-class signal directions, drawn before the noise stream so
    # the same class always carries the same content signature.
    class_signals = torch.randn(num_classes, patch_dim, generator=generator) * signal_std
    hand_signal = torch.randn(patch_dim, generator=generator) * signal_std

    patches = torch.zeros(length, num_patches, patch_dim)
    labels = torch.zeros(length, dtype=torch.long)
    boxes: list[tuple[int, int, int, int]] = []

    # Each cell spans px x px pixels in a 256x256 image (DINOv2 input).
    px = 256 // grid

    for segment in range(num_segments):
        label = segment % num_classes
        # 0..grid-5; the +1 jitter keeps r0+c0 within [0, grid-4] so the
        # 4x4 box (rows r0..r0+3) never leaves the grid.
        offset_row = int(torch.randint(0, grid - 4, (1,), generator=generator).item())
        offset_col = int(torch.randint(0, grid - 4, (1,), generator=generator).item())
        start = segment * segment_length
        for step in range(segment_length):
            t = start + step
            # Jitter the box edge-inward by one cell on alternating steps,
            # mimicking a hand shifting within the scene.
            jitter = step % 2
            r0 = offset_row + jitter
            c0 = offset_col + jitter
            r1, c1 = r0 + 4, c0 + 4
            labels[t] = label
            boxes.append((c0 * px, r0 * px, c1 * px, r1 * px))
            for r in range(r0, r1):
                for c in range(c0, c1):
                    patches[t, r * grid + c] = (
                        torch.randn(patch_dim, generator=generator) * noise_std
                        + class_signals[label]
                        + hand_signal
                    )

    return PatchSequenceSample(
        patches=patches,
        labels=labels,
        boxes=boxes,
        frame_indices=torch.arange(length),
    )
