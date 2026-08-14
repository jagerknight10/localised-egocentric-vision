"""Metrics for frame-level action predictions."""

import torch


def masked_frame_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -1
) -> float:
    """Compute accuracy while excluding unlabeled frames."""
    mask = labels != ignore_index
    if not torch.any(mask):
        raise ValueError("No labeled frames available for accuracy.")
    predictions = logits.argmax(dim=-1)
    return float((predictions[mask] == labels[mask]).float().mean())


def majority_class_accuracy(labels: torch.Tensor) -> float:
    """Return accuracy from always predicting the most frequent label."""
    if labels.numel() == 0:
        raise ValueError("No labels available.")
    majority = torch.bincount(labels).argmax()
    return float((labels == majority).float().mean())


def balanced_frame_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Average per-class recall, giving each observed class equal weight."""
    predictions = logits.argmax(dim=-1)
    recalls = []
    for label in torch.unique(labels):
        class_mask = labels == label
        recalls.append((predictions[class_mask] == label).float().mean())
    return float(torch.stack(recalls).mean())


def confusion_matrix(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Return rows=actual labels, columns=predicted labels."""
    predictions = logits.argmax(dim=-1)
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    matrix.index_put_((labels, predictions), torch.ones_like(labels, dtype=torch.int64), accumulate=True)
    return matrix
