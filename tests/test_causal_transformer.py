import torch

from egovision.models.causal_transformer import CausalTemporalTransformer


def test_causal_transformer_shape_and_parameter_budget() -> None:
    model = CausalTemporalTransformer(feature_dim=384, num_classes=73)
    logits = model(torch.randn(2, 16, 384))
    assert logits.shape == (2, 16, 73)
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000


def test_causal_transformer_cannot_use_future_features() -> None:
    model = CausalTemporalTransformer(feature_dim=8, num_classes=3, dropout=0.0)
    model.eval()
    original = torch.randn(1, 6, 8)
    changed_future = original.clone()
    changed_future[:, 4:] += 100.0

    with torch.no_grad():
        first_logits = model(original)
        changed_logits = model(changed_future)

    assert torch.allclose(first_logits[:, :4], changed_logits[:, :4], atol=1e-6)


def test_context_limit_is_enforced() -> None:
    model = CausalTemporalTransformer(feature_dim=8, num_classes=3, max_context=4)
    with torch.no_grad():
        try:
            model(torch.randn(1, 5, 8))
        except ValueError as error:
            assert "max_context" in str(error)
        else:
            raise AssertionError("Expected context limit error")
