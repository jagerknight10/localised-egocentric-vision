import pytest

torch = pytest.importorskip("torch")

from egovision.device import get_device


def test_cpu_device_is_always_available() -> None:
    device = get_device("cpu")
    assert device.type == "cpu"
    assert torch.zeros(2, device=device).shape == (2,)


def test_auto_device_is_supported() -> None:
    assert get_device().type in {"cpu", "cuda", "mps"}
