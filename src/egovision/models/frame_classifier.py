"""Frame-independent action classification baseline."""

import torch
from torch import nn


class FrameLinearClassifier(nn.Module):
    """Predict an action independently from each frame feature."""

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        if feature_dim <= 0 or num_classes <= 0:
            raise ValueError("feature_dim and num_classes must be positive.")
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits for features shaped ``[batch, time, feature_dim]``."""
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, feature_dim].")
        return self.classifier(features)
