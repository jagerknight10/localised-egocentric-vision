"""Runtime device selection."""

from typing import Literal

import torch

DeviceName = Literal["auto", "cpu", "cuda", "mps"]


def get_device(preference: DeviceName = "auto") -> torch.device:
    """Return a requested device, or the best available device for ``auto``."""
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
