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
    def extract(self, frames: np.ndarray | list[np.ndarray], batch_size: int | None = None) -> torch.Tensor:
        """Return CLS features for RGB frames or variable-size RGB crops."""
        if self.processor is None or self.model is None:
            raise RuntimeError("Call load() before extract().")
        frame_list = list(frames)
        if not frame_list or any(
            frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8
            for frame in frame_list
        ):
            raise ValueError("frames must be non-empty uint8 RGB images shaped [H, W, 3].")
        if batch_size is None or batch_size >= len(frame_list):
            batches = [frame_list]
        else:
            batches = [frame_list[start:start + batch_size] for start in range(0, len(frame_list), batch_size)]
        embeddings = []
        for batch in batches:
            inputs = self.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            outputs = self.model(pixel_values=pixel_values)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())
        return torch.cat(embeddings)
