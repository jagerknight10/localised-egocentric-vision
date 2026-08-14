"""Small causal Transformer for streaming action prediction."""

import torch
from torch import nn


class CausalTemporalTransformer(nn.Module):
    """Predict an action at each timestep using only current/past features."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        max_context: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(feature_dim, num_classes, d_model, nhead, num_layers, max_context) <= 0:
            raise ValueError("Model dimensions must be positive.")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.max_context = max_context
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_context, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits for features shaped ``[batch, time, feature_dim]``."""
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, feature_dim].")
        _, time, _ = features.shape
        if time > self.max_context:
            raise ValueError(f"Sequence length {time} exceeds max_context={self.max_context}.")
        hidden = self.input_projection(features) + self.position_embedding[:, :time]
        causal_mask = torch.triu(
            torch.ones(time, time, dtype=torch.bool, device=features.device), diagonal=1
        )
        hidden = self.encoder(hidden, mask=causal_mask)
        return self.classifier(hidden)
