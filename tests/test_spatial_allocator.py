"""Tests for the learned spatial-importance allocator.

Coverage: output contracts (shapes, distributions), the identity that the
feature is exactly the weighted aggregate of patch tokens, gradient flow
through both outputs, near-uniform initialization, box-to-map geometry,
the KL loss, localization convergence on synthetic data, and a joint
allocator + causal Transformer smoke test with the real 384-d contract.
"""

import pytest
import torch

from egovision.data.synthetic import make_signal_patch_sequence
from egovision.models import (
    CausalTemporalTransformer,
    SpatialImportanceAllocator,
    box_to_importance_map,
    importance_kl_loss,
)


def test_allocator_output_contract() -> None:
    allocator = SpatialImportanceAllocator(feature_dim=384, grid=16)
    patches = torch.randn(2, 5, 256, 384)
    weights, feature = allocator(patches)

    assert weights.shape == (2, 5, 256)
    assert feature.shape == (2, 5, 384)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)
    assert torch.all(weights > 0)


def test_feature_is_weighted_patch_aggregate() -> None:
    allocator = SpatialImportanceAllocator(feature_dim=16, grid=4)
    patches = torch.randn(1, 3, 16, 16)
    weights, feature = allocator(patches)
    expected = torch.einsum("btp,btpd->btd", weights, patches)
    assert torch.allclose(feature, expected, atol=1e-6)


def test_allocator_has_gradients_on_both_outputs() -> None:
    allocator = SpatialImportanceAllocator(feature_dim=8, grid=4)
    patches = torch.randn(1, 2, 16, 8)
    weights, feature = allocator(patches)
    loss = weights.pow(2).sum() + feature.pow(2).sum()
    loss.backward()
    assert allocator.query.grad is not None and allocator.query.grad.abs().sum() > 0
    assert allocator.position_bias.grad is not None
    assert allocator.position_bias.grad.abs().sum() > 0


def test_initialization_is_near_uniform() -> None:
    torch.manual_seed(0)
    allocator = SpatialImportanceAllocator(feature_dim=64, grid=16)
    patches = torch.randn(4, 10, 256, 64)
    with torch.no_grad():
        weights, _ = allocator(patches)
    uniform = torch.full_like(weights, 1.0 / 256)
    # Trunc-normal query at std 0.02 and zero bias: scores are tiny, so
    # each cell stays within a few percent of uniform probability.
    assert (weights - uniform).abs().max() < 0.01


def test_box_to_importance_map_geometry() -> None:
    # Full-frame box: uniform map.
    full = box_to_importance_map((0, 0, 256, 256), grid=16, width=256, height=256)
    assert torch.allclose(full, torch.full((16, 16), 1.0 / 256), atol=1e-6)

    # A 4x4-cell box at cells (2,3): exactly 16 cells, equal weight.
    box = box_to_importance_map((48, 32, 112, 96), grid=16, width=256, height=256)
    assert abs(box.sum() - 1.0) < 1e-6
    assert box[2:6, 3:7].min() > 0
    assert box[2:6, 3:7].sum() == 1.0
    assert box.sum() - box[2:6, 3:7].sum() < 1e-9

    # Margin dilates the box.
    dilated = box_to_importance_map((48, 32, 112, 96), grid=16, width=256, height=256, margin_px=16)
    assert dilated[0:2, 1:9].sum() > 0


def test_box_to_importance_map_validation() -> None:
    with pytest.raises(ValueError, match="positive area"):
        box_to_importance_map((10, 10, 10, 20), grid=16, width=256, height=256)
    with pytest.raises(ValueError, match="positive"):
        box_to_importance_map((0, 0, 10, 10), grid=16, width=0, height=256)


def test_importance_kl_loss() -> None:
    target = torch.full((2, 256), 1.0 / 256)
    predicted = torch.full((2, 256), 1.0 / 256)
    assert importance_kl_loss(predicted, target).item() < 1e-6

    peak = torch.full((2, 256), 1.0 / 256)
    peak[:, 5] = 1.0
    loss = importance_kl_loss(predicted, peak)
    assert loss.item() > 0
    # Matching the peak should reduce the loss.
    assert importance_kl_loss(peak, peak).item() < loss.item()


def test_importance_kl_loss_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share shape"):
        importance_kl_loss(torch.rand(2, 4), torch.rand(3, 256))


def _train_until_localized(max_steps: int = 400, seed: int = 0) -> torch.Tensor:
    """Train the allocator on one synthetic clip; return mean test weights."""
    torch.manual_seed(seed)
    sample = make_signal_patch_sequence(length=48, grid=16, patch_dim=64, num_classes=4, seed=seed)
    patches = sample.patches.unsqueeze(0)  # [1, T, 256, D]
    targets = torch.stack([
        box_to_importance_map(box, grid=16, width=256, height=256)
        for box in sample.boxes
    ]).reshape(-1, 256)

    allocator = SpatialImportanceAllocator(feature_dim=64, grid=16)
    optimizer = torch.optim.Adam(allocator.parameters(), lr=1e-2)
    for _ in range(max_steps):
        weights, _ = allocator(patches)
        loss = importance_kl_loss(weights.reshape(-1, 256), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        weights, _ = allocator(patches)
    return weights


def test_allocator_localizes_the_signal_box() -> None:
    weights = _train_until_localized(seed=0)  # [1, 48, 256]
    # For each frame, the predicted map must put most of its mass inside
    # the ground-truth 4x4 box (plus a one-cell halo).
    sample = make_signal_patch_sequence(length=48, grid=16, patch_dim=64, num_classes=4, seed=0)
    inside_mass = []
    for t, (x0, y0, x1, y1) in enumerate(sample.boxes):
        c0, c1 = max(0, x0 // 16 - 1), min(16, (x1 + 15) // 16 + 1)
        r0, r1 = max(0, y0 // 16 - 1), min(16, (y1 + 15) // 16 + 1)
        inside_mass.append(float(weights[0, t, :].reshape(16, 16)[r0:r1, c0:c1].sum()))
    assert sum(inside_mass) / len(inside_mass) > 0.7


def test_joint_allocator_and_causal_transformer_smoke() -> None:
    """End-to-end shape contract at real DINOv2-small dimensions."""
    torch.manual_seed(0)
    sample = make_signal_patch_sequence(length=36, grid=16, patch_dim=384, num_classes=4, seed=1)
    allocator = SpatialImportanceAllocator(feature_dim=384, grid=16)
    transformer = CausalTemporalTransformer(feature_dim=384, num_classes=4, max_context=36)

    patches = sample.patches.unsqueeze(0)  # [1, 36, 256, 384]
    weights, features = allocator(patches)
    logits = transformer(features)  # [1, 36, 4]
    assert weights.shape == (1, 36, 256)
    assert features.shape == (1, 36, 384)
    assert logits.shape == (1, 36, 4)

    # Joint loss: action cross-entropy + KL localization, both backprop.
    targets = torch.stack([
        box_to_importance_map(box, grid=16, width=256, height=256) for box in sample.boxes
    ]).reshape(1, 36, 256)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 4), sample.labels.unsqueeze(0).reshape(-1)
    ) + 0.1 * importance_kl_loss(weights.reshape(-1, 256), targets.reshape(-1, 256))
    loss.backward()
    assert allocator.query.grad is not None
    assert transformer.classifier.weight.grad is not None
