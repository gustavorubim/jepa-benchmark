from __future__ import annotations

import pytest
import torch

from jepa_robotics.utils.device import get_torch_device


def test_cpu_device() -> None:
    assert get_torch_device("cpu") == torch.device("cpu")


def test_auto_returns_available_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert get_torch_device("auto") == torch.device("cpu")


def test_explicit_unavailable_cuda_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        get_torch_device("cuda")


def test_invalid_device_fails() -> None:
    with pytest.raises(ValueError, match="preferred"):
        get_torch_device("tpu")
