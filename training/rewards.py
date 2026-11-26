"""Privileged reward shaping for Role 2 PPO training (`REVISED_PLAN.md §2`).

Uses `env.privileged_state()` via `PrivilegedEventTracker` so training rewards match
the exact events logged by Role 4's evaluator without altering the agent's official inputs.

Hierarchy invariant:
  exit (10.0 + up to 3.0 time/health efficiency)
    > bounded total exploration (max 6.0 pre-key + max 1.8 post-key)
    > locked-door unlock (4.0) > key pickup (3.0) > door open (1.5)
    > combat (0.75/kill, -0.02/HP damage, strictly smaller heal bonus <= 0.01/HP)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from training.interfaces import PrivilegedEventTracker, StepEvents


@dataclass(frozen=True)
class RewardConfig:
    """Hierarchical reward weights keeping exit completion dominant over combat/exploration."""

    exit_reward: float = 10.0
    efficiency_bonus: float = 1.5
    health_exit_bonus: float = 1.5
    key_pickup_reward: float = 3.0
    locked_door_unlock_reward: float = 4.0
    door_open_reward: float = 1.5
    new_cell_reward: float = 0.04
    max_exploration_cells: int = 150
    post_key_cell_reward: float = 0.03
    max_post_key_cells: int = 60
    kill_reward: float = 0.75
    damage_penalty: float = -0.02
    death_penalty: float = -2.0
    step_penalty: float = -0.002
    door_bounce_penalty: float = -0.05
    wasted_ammo_penalty: float = -0.005
    health_pickup_reward_per_hp: float = 0.01
    health_pickup_threshold: float = 80.0
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
        self._post_key_active: bool = False
        self._post_key_cells: set[tuple[int, int]] = set()
        self._post_key_rewarded: int = 0
        self._prev_health: float | None = None
        self._prev_ceil: np.ndarray | None = None
        self._rising_doors: set[int] = set()
        self._prev_selected_ammo: float | None = None

    @property
    def unique_cells_visited(self) -> int:
        return self.tracker.unique_cells_visited

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        cs = self.config.cell_size
        return (int(np.floor(x / cs)), int(np.floor(y / cs)))

    def reset(self, priv_state: dict[str, Any], obs: dict[str, Any] | None = None) -> None:
        self.tracker.reset(priv_state)
        self._post_key_active = False
        self._post_key_cells.clear()
        self._post_key_rewarded = 0
        self._prev_health = float(priv_state.get("health", 100.0))
        ceil = priv_state.get("sector_ceiling")
        self._prev_ceil = None if ceil is None else np.array(ceil, dtype=np.float64, copy=True)
        self._rising_doors.clear()
        if obs is not None and "vitals" in obs:
            self._prev_selected_ammo = float(obs["vitals"][3])
        else:
            self._prev_selected_ammo = None

    def step(
        self,
        priv_state: dict[str, Any],
        *,
        action: np.ndarray | None = None,
        prev_action: np.ndarray | None = None,
        obs: dict[str, Any] | None = None,
    ) -> tuple[float, StepEvents, dict[str, float]]:
        cfg = self.config
        events, entered_new_cell = self.tracker.update(priv_state)
        cell = self._cell(float(priv_state["x"]), float(priv_state["y"]))

        # Primary bounded first-visit exploration
        explore = 0.0
        if entered_new_cell and self.tracker.unique_cells_visited - 1 <= cfg.max_exploration_cells:
            explore = cfg.new_cell_reward

        # Secondary bounded post-key exploration so backtracking to a locked door earns signal
        post_key_explore = 0.0
        if events.keys_picked_up > 0:
            self._post_key_active = True
            self._post_key_cells = {cell}
            self._post_key_rewarded = 0
        elif self._post_key_active and cell not in self._post_key_cells:
            self._post_key_cells.add(cell)
            if self._post_key_rewarded < cfg.max_post_key_cells:
                post_key_explore = cfg.post_key_cell_reward
                self._post_key_rewarded += 1

        # Exit time + remaining-health efficiency bonus (docs/EVALUATION.md)
        efficiency = 0.0
        health_exit = 0.0
        curr_health = float(priv_state.get("health", 0.0))
        if events.exit_reached:
            ep_ticks = max(1, int(priv_state.get("episode_ticks", 1)))
            remaining = max(0.0, 1.0 - float(priv_state.get("tic", ep_ticks)) / float(ep_ticks))
            efficiency = cfg.efficiency_bonus * remaining
            health_exit = cfg.health_exit_bonus * min(1.0, max(0.0, curr_health / 100.0))

        # Bounded health-pickup bonus when hurt (strictly < |damage_penalty| to forbid damage farming)
        health_pickup = 0.0
        if self._prev_health is not None:
            hp_gain = curr_health - self._prev_health
            if hp_gain > 0.0 and self._prev_health < cfg.health_pickup_threshold:
                capped_gain = min(hp_gain, cfg.health_pickup_threshold - self._prev_health)
                health_pickup = cfg.health_pickup_reward_per_hp * capped_gain
        self._prev_health = curr_health

        # Exact door-reversal penalty: fires only when USE pulsed 0->1 and a rising door dropped
        door_bounce = 0.0
        ceil_raw = priv_state.get("sector_ceiling")
        locks_raw = priv_state.get("door_lock")
        if ceil_raw is not None and self._prev_ceil is not None and len(ceil_raw) == len(self._prev_ceil):
            curr_ceil = np.asarray(ceil_raw, dtype=np.float64)
            delta_ceil = curr_ceil - self._prev_ceil
            use_pulsed = (
                action is not None
                and len(action) > 7
                and int(action[7]) == 1
                and (prev_action is None or len(prev_action) <= 7 or int(prev_action[7]) == 0)
            )
            new_rising: set[int] = set()
            for idx in range(len(curr_ceil)):
                is_door = locks_raw is not None and idx < len(locks_raw) and int(locks_raw[idx]) >= 0
                if not is_door:
                    continue
                if delta_ceil[idx] > 0.5:
                    new_rising.add(idx)
                elif delta_ceil[idx] < -0.5 and idx in self._rising_doors and use_pulsed:
                    door_bounce += cfg.door_bounce_penalty
            self._rising_doors = new_rising
            self._prev_ceil = curr_ceil.copy()

        # Small wasted-ammo penalty when ammo drops with zero visible enemies in FOV
        wasted_ammo = 0.0
        if obs is not None and "vitals" in obs and "visible_entities" in obs:
            curr_ammo = float(obs["vitals"][3])
            if self._prev_selected_ammo is not None and curr_ammo < self._prev_selected_ammo - 1e-6:
                ents = obs["visible_entities"]
                mask = np.asarray(ents["mask"], dtype=bool)
                etypes = np.asarray(ents["entity_type"], dtype=np.int64)
                has_visible_enemy = bool(np.any(mask & (etypes >= 1) & (etypes <= 3)))
                if not has_visible_enemy:
                    wasted_ammo = cfg.wasted_ammo_penalty
            self._prev_selected_ammo = curr_ammo

        components = {
            "exit": cfg.exit_reward if events.exit_reached else 0.0,
            "efficiency": efficiency,
            "health_exit": health_exit,
            "key": cfg.key_pickup_reward * events.keys_picked_up,
            "unlock": cfg.locked_door_unlock_reward * events.locked_doors_unlocked,
            "door": cfg.door_open_reward * events.doors_opened,
            "explore": explore,
            "post_key_explore": post_key_explore,
            "kill": cfg.kill_reward * events.kills,
            "damage": cfg.damage_penalty * events.damage_taken,
            "death": cfg.death_penalty if events.dead else 0.0,
            "health_pickup": health_pickup,
            "door_bounce": door_bounce,
            "wasted_ammo": wasted_ammo,
            "time": cfg.step_penalty,
        }
        return float(sum(components.values())), events, components
