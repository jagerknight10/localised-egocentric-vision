"""Learned spatial-importance allocator (soft crop) for egocentric frames.

The allocator converts one frame's frozen DINOv2 patch-token sequence into
(a) a 16x16 spatial importance map and (b) one localized visual feature.
Both come from a single mechanism: a learned query attends over all 256
patch tokens. The softmax attention weights are the importance map; the
weighted sum of patch tokens is the feature.

This is the "learned spatial allocation" block in the EgoFocus design:
instead of encoding the whole frame (CLS token) or using an oracle
hand-box crop, the model learns which grid cells carry task-relevant
evidence. Training is joint with the recognition objective; the KL term
below supervises the map from ground-truth box annotations when available.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpatialImportanceAllocator(nn.Module):
    """Single-query attention pooling over a 16x16 grid of patch tokens.

    Scores for patch ``p`` are ``scale * (query . token_p) + bias_p``.
    The content term lets the allocator follow whatever region currently
    carries task-relevant signal (the hand-object region moves); the bias
    term is a learned position prior over the grid.
    """

    def __init__(self, feature_dim: int = 384, grid: int = 16) -> None:
        super().__init__()
        if feature_dim <= 0 or grid <= 0:
            raise ValueError("feature_dim and grid must be positive.")
        self.grid = grid
        self.num_patches = grid * grid
        # [D]: shared content query.
        self.query = nn.Parameter(torch.empty(feature_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        # [num_patches]: per-patch position prior, zero at init.
        self.position_bias = nn.Parameter(torch.zeros(self.num_patches))
        self.scale = feature_dim ** -0.5

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate patch tokens into an importance map and a feature.

        Args:
            patches: [batch, time, num_patches, feature_dim] patch tokens
                in row-major grid order (patch 0 = top-left cell).

        Returns:
            weights: [batch, time, num_patches]; each row is a
                distribution over the grid cells (sums to 1).
            feature: [batch, time, feature_dim]; weighted patch aggregate,
                compatible with the downstream causal Transformer.
        """
        if patches.ndim != 4 or patches.shape[2] != self.num_patches:
            raise ValueError(
                f"patches must have shape [B, T, {self.num_patches}, D]; "
                f"got {tuple(patches.shape)}"
            )
        scores = torch.einsum("btpd,d->btp", patches, self.query) * self.scale
        scores = scores + self.position_bias
        weights = torch.softmax(scores, dim=-1)
        feature = torch.einsum("btp,btpd->btd", weights, patches)
        return weights, feature


def box_to_importance_map(
    box: tuple[int, int, int, int],
    grid: int,
    width: int,
    height: int,
    margin_px: int = 0,
) -> torch.Tensor:
    """Build a normalized [grid, grid] target map from a pixel box.

    Each grid cell receives a weight proportional to its overlap area
    with the (optionally dilated) box, then the map is normalized to
    sum to 1 so it is a valid distribution.

    Args:
        box: (x0, y0, x1, y1) in pixels, x1/y1 exclusive.
        grid: number of cells per side.
        width: image width in pixels.
        height: image height in pixels.
        margin_px: dilate the box by this many pixels first.

    Returns:
        FloatTensor [grid, grid] summing to 1.
    """
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        raise ValueError("box must have positive area (x1 > x0 and y1 > y0).")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")
    x0 = max(0, int(x0) - int(margin_px))
    y0 = max(0, int(y0) - int(margin_px))
    x1 = min(width, int(x1) + int(margin_px))
    y1 = min(height, int(y1) + int(margin_px))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Dilated box has no area inside the image.")
    grid_map = torch.zeros(grid, grid)
    for row in range(grid):
        cy0, cy1 = row * height / grid, (row + 1) * height / grid
        for col in range(grid):
            cx0, cx1 = col * width / grid, (col + 1) * width / grid
            overlap_x = max(0.0, min(cx1, x1) - max(cx0, x0))
            overlap_y = max(0.0, min(cy1, y1) - max(cy0, y0))
            grid_map[row, col] = overlap_x * overlap_y
    total = grid_map.sum()
    if total == 0:
        raise ValueError("Box does not overlap the grid.")
    return grid_map / total


def importance_kl_loss(weights: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL(target || predicted) between importance distributions.

    Args:
        weights: [*, num_patches] predicted probabilities (softmax output).
        target: [*, num_patches] target distribution, same leading shape
            (usually from box_to_importance_map).

    Returns:
        Scalar loss; ~0 when the distributions match.
    """
    if weights.shape != target.shape:
        raise ValueError(
            f"weights and target must share shape; got {tuple(weights.shape)} "
            f"vs {tuple(target.shape)}"
        )
    return F.kl_div(weights.clamp_min(1e-12).log(), target, reduction="batchmean")
