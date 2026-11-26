"""Privileged reward shaping for Role 2 PPO training (`REVISED_PLAN.md §2`).

Uses `env.privileged_state()` via `PrivilegedEventTracker` so training rewards match
the exact events logged by Role 4's evaluator without altering the agent's official inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from training.interfaces import PrivilegedEventTracker, StepEvents


@dataclass(frozen=True)
class RewardConfig:
    """Hierarchical reward weights keeping exit completion dominant over combat/exploration."""

    exit_reward: float = 10.0
    efficiency_bonus: float = 2.0
    key_pickup_reward: float = 3.0
    locked_door_unlock_reward: float = 4.0
    door_open_reward: float = 1.5
    new_cell_reward: float = 0.15
    max_exploration_cells: int = 40
    kill_reward: float = 0.5
    damage_penalty: float = -0.01
    death_penalty: float = -2.0
    step_penalty: float = -0.002
    cell_size: float = 64.0
    open_threshold: float = 8.0


class RewardShaper:
    """Computes dense, bounded per-step training rewards from `env.privileged_state()`."""

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        self.tracker = PrivilegedEventTracker(
            open_threshold=self.config.open_threshold,
            cell_size=self.config.cell_size,
        )

    @property
    def unique_cells_visited(self) -> int:
        return self.tracker.unique_cells_visited

    def reset(self, priv_state: dict[str, Any]) -> None:
        self.tracker.reset(priv_state)

    def step(self, priv_state: dict[str, Any]) -> tuple[float, StepEvents, dict[str, float]]:
        cfg = self.config
        events, entered_new_cell = self.tracker.update(priv_state)

        explore = 0.0
        if entered_new_cell and self.tracker.unique_cells_visited - 1 <= cfg.max_exploration_cells:
            explore = cfg.new_cell_reward

        efficiency = 0.0
        if events.exit_reached:
            ep_ticks = max(1, int(priv_state.get("episode_ticks", 1)))
            remaining = max(0.0, 1.0 - float(priv_state.get("tic", ep_ticks)) / float(ep_ticks))
            efficiency = cfg.efficiency_bonus * remaining

        components = {
            "exit": cfg.exit_reward if events.exit_reached else 0.0,
            "efficiency": efficiency,
            "key": cfg.key_pickup_reward * events.keys_picked_up,
            "unlock": cfg.locked_door_unlock_reward * events.locked_doors_unlocked,
            "door": cfg.door_open_reward * events.doors_opened,
            "explore": explore,
            "kill": cfg.kill_reward * events.kills,
            "damage": cfg.damage_penalty * events.damage_taken,
            "death": cfg.death_penalty if events.dead else 0.0,
            "time": cfg.step_penalty,
        }
        return float(sum(components.values())), events, components
