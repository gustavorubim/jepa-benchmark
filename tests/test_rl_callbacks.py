from __future__ import annotations

from jepa_robotics.rl.callbacks import EarlyStopState


def test_early_stop_state_tracks_patience() -> None:
    state = EarlyStopState(threshold=0.8, patience=2)
    assert not state.observe(0.7)
    assert not state.observe(0.9)
    assert state.consecutive_successes == 1
    assert state.observe(0.85)
    assert state.should_stop
    assert state.observe(0.1) is False
    assert state.consecutive_successes == 0


def test_early_stop_disabled_without_threshold() -> None:
    state = EarlyStopState(threshold=None, patience=1)
    assert not state.observe(1.0)
    assert not state.should_stop
