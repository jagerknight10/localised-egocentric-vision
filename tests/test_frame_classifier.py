import torch

from egovision.models.frame_classifier import FrameLinearClassifier


def test_frame_classifier_output_shape() -> None:
    model = FrameLinearClassifier(feature_dim=8, num_classes=3)
    logits = model(torch.randn(2, 5, 8))
    assert logits.shape == (2, 5, 3)


def test_frame_classifier_is_frame_independent() -> None:
    model = FrameLinearClassifier(feature_dim=4, num_classes=2)
    first = torch.randn(1, 3, 4)
    changed_future = first.clone()
    changed_future[:, 2] += 100

    original_logits = model(first)
    changed_logits = model(changed_future)
    assert torch.equal(original_logits[:, :2], changed_logits[:, :2])
