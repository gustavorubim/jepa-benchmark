from __future__ import annotations

from pathlib import Path

import pytest

from jepa_robotics.config.schema import BenchmarkConfig
from jepa_robotics.rl.sb3_train import _policy_kwargs, train_rl_baseline


def test_policy_kwargs_requires_jepa_checkpoint() -> None:
    config = BenchmarkConfig().smoke_copy()
    config.rl.policy_net_arch = [32, 32]
    config.rl.n_critics = 2
    base_kwargs = _policy_kwargs(config, None, None, freeze_encoder=True)
    assert base_kwargs["net_arch"] == [32, 32]
    assert base_kwargs["n_critics"] == 2
    with pytest.raises(ValueError, match="encoder-checkpoint"):
        _policy_kwargs(config, "jepa", None, freeze_encoder=True)
    kwargs = _policy_kwargs(config, "jepa", Path("encoder.pt"), freeze_encoder=False)
    assert kwargs["features_extractor_kwargs"]["freeze_encoder"] is False
    with pytest.raises(ValueError, match="Unsupported feature"):
        _policy_kwargs(config, "unknown", Path("encoder.pt"), freeze_encoder=True)


def test_train_rl_skip_existing(tmp_path: Path) -> None:
    config = BenchmarkConfig().smoke_copy()
    out = tmp_path / "rl"
    out.mkdir()
    for name in ["model.zip", "metrics.csv", "config_resolved.yaml"]:
        (out / name).write_text("done", encoding="utf-8")
    result = train_rl_baseline(config, method="sac", seed=0, output_dir=out)
    assert result == out
