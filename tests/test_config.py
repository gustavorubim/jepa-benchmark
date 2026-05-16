from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jepa_robotics.config.load import dump_config, load_config
from jepa_robotics.config.schema import BenchmarkConfig


def test_load_config_and_smoke_copy(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "x", "output_dir": str(tmp_path / "out"), "seeds": [7]},
                "env": {"id": "FetchReachDense-v3"},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.experiment.seeds == [7]
    smoke = config.smoke_copy()
    assert smoke.env.id == "ToyGoal-v0"
    assert smoke.device.preferred == "cpu"


def test_dump_config_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "resolved.yaml"
    dump_config(BenchmarkConfig(), path)
    assert load_config(path).env.id == "FetchReachDense-v3"


def test_invalid_yaml_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
