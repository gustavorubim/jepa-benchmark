"""HER configuration helpers."""

from __future__ import annotations


def default_her_kwargs() -> dict[str, object]:
    return {"n_sampled_goal": 4, "goal_selection_strategy": "future", "copy_info_dict": True}
