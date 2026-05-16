"""Central PyTorch device selection."""

from __future__ import annotations

import torch


def get_torch_device(preferred: str = "auto") -> torch.device:
    """Return a torch device, failing clearly for unavailable explicit devices."""
    normalized = preferred.lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if normalized == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false.")
        return torch.device("mps")
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    msg = "preferred must be one of: auto, cpu, cuda, mps"
    raise ValueError(msg)
