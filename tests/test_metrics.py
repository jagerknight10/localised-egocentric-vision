import torch

from egovision.metrics import balanced_frame_accuracy, majority_class_accuracy, masked_frame_accuracy


def test_masked_accuracy_ignores_background() -> None:
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0], [0.0, 5.0]])
    labels = torch.tensor([0, 1, -1])
    assert masked_frame_accuracy(logits, labels) == 1.0


def test_majority_accuracy_and_balanced_accuracy() -> None:
    logits = torch.tensor([[5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    labels = torch.tensor([0, 0, 0, 1])
    assert majority_class_accuracy(labels) == 0.75
    assert balanced_frame_accuracy(logits, labels) == 1.0
