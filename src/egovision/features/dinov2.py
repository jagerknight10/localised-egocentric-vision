"""DINOv2 feature extraction with an explicit weight-loading boundary."""

from typing import Any

import numpy as np
import torch


class DinoV2FeatureExtractor:
    """Load a frozen DINOv2 image encoder and return CLS embeddings."""

    def __init__(
        self,
        model_id: str = "facebook/dinov2-small",
        device: torch.device | str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device)
        self.processor: Any | None = None
        self.model: Any | None = None

    def load(self) -> None:
        """Download/load processor and weights, then freeze the encoder."""
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def extract(self, frames: np.ndarray) -> torch.Tensor:
        """Return CLS features for RGB frames shaped ``[T, H, W, 3]``."""
        if self.processor is None or self.model is None:
            raise RuntimeError("Call load() before extract().")
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError("frames must be uint8 RGB data shaped [T, H, W, 3].")
        inputs = self.processor(images=list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0, :].cpu()
