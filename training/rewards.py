"""Privileged reward shaping for Role 2 PPO training (`REVISED_PLAN.md §2` + `TRACK1_IMPROVEMENTS.md`).

Uses `env.privileged_state()` via `PrivilegedEventTracker` plus action/observation signals so
training rewards match all four scoring categories without altering official evaluation inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from training.interfaces import PrivilegedEventTracker, StepEvents

_ATTACK_IDX = 6
_USE_IDX = 7
_WEAPON_SWITCH_IDX = 8
_ENEMY_ENTITY_TYPES = (1, 2, 3)


@dataclass(frozen=True)
class RewardConfig:
    """Hierarchical reward weights keeping exit completion dominant over exploration & combat."""

    exit_reward: float = 12.0
    efficiency_bonus: float = 2.5
    health_efficiency_bonus: float = 1.5
    key_pickup_reward: float = 4.0
    locked_door_unlock_reward: float = 5.0
    door_open_reward: float = 2.0
    new_cell_reward: float = 0.10
    max_exploration_cells: int = 250
    explore_decay_cells: float = 60.0
    post_key_cell_reward: float = 0.08
    max_post_key_cells: int = 80
    kill_reward: float = 1.5
    damage_penalty: float = -0.018
    death_penalty: float = -3.0
    door_bounce_penalty: float = -0.25
    use_waste_penalty: float = -0.02
    use_hold_penalty: float = -0.03
    wasted_ammo_penalty: float = -0.025
    weapon_switch_waste_penalty: float = -0.025
    weapon_pickup_reward: float = 0.6
    max_pickup_bonus: float = 2.5
    pbrs_scale: float = 2.0
    pbrs_gamma: float = 0.995
    step_penalty: float = -0.002
    cell_size: float = 64.0
    open_threshold: float = 8.0


def _extract_ammo_and_armor(priv_state: dict[str, Any], obs: dict[str, Any] | None) -> tuple[float | None, float]:
    armor = float(priv_state.get("armor", 0.0))
    if obs is not None and "vitals" in obs:
        vitals = obs["vitals"]
        armor = float(vitals[1])
        ammo = float(np.sum(vitals[4:8]))
        return ammo, armor
    if "ammo_total" in priv_state and priv_state["ammo_total"] is not None:
        return float(priv_state["ammo_total"]), armor
    return None, armor


def _has_visible_enemy(obs: dict[str, Any] | None) -> bool:
    if obs is None or "visible_entities" not in obs:
        return False
    ve = obs["visible_entities"]
    mask = np.asarray(ve["mask"], dtype=bool)
    if not np.any(mask):
        return False
    etypes = np.asarray(ve["entity_type"], dtype=np.int64)[mask]
    return bool(np.isin(etypes, _ENEMY_ENTITY_TYPES).any())


class RewardShaper:
    """Computes dense, bounded per-step training rewards from `env.privileged_state()`."""

    def __init__(self, config: RewardConfig | None = None, *, pbrs: Any = None):
        self.config = config or RewardConfig()
        self.pbrs = pbrs
        self.tracker = PrivilegedEventTracker(
            open_threshold=self.config.open_threshold,
            cell_size=self.config.cell_size,
        )
        self._post_key_visited_cells: set[tuple[int, int]] = set()
        self._post_key_explore_count = 0
        self._prev_ceil: np.ndarray | None = None
        self._rising_doors: set[int] = set()
        self._prev_health: float = 100.0
        self._prev_armor: float = 0.0
        self._prev_ammo: float | None = None
        self._owned_weapons: set[int] = set()
        self._pickup_bonus_earned: float = 0.0

    @property
    def unique_cells_visited(self) -> int:
        return self.tracker.unique_cells_visited

    @property
    def post_key_cells_visited(self) -> int:
        return len(self._post_key_visited_cells)

    def _cell(self, priv_state: dict[str, Any]) -> tuple[int, int]:
        return (
            int(np.floor(float(priv_state["x"]) / self.config.cell_size)),
            int(np.floor(float(priv_state["y"]) / self.config.cell_size)),
        )

    def reset(self, priv_state: dict[str, Any], obs: dict[str, Any] | None = None, *, pbrs: Any = None) -> None:
        if pbrs is not None:
            self.pbrs = pbrs
        self.tracker.reset(priv_state)
        self._post_key_visited_cells = {self._cell(priv_state)}
        self._post_key_explore_count = 0
        ceil = priv_state.get("sector_ceiling")
        self._prev_ceil = None if ceil is None else np.array(ceil, dtype=np.float64, copy=True)
        self._rising_doors.clear()
        self._prev_health = float(priv_state.get("health", 100.0))
        ammo, armor = _extract_ammo_and_armor(priv_state, obs)
        self._prev_armor = armor
        self._prev_ammo = ammo
        self._owned_weapons = set()
        if obs is not None and "vitals" in obs:
            self._owned_weapons.add(int(round(float(obs["vitals"][2]))))
        self._pickup_bonus_earned = 0.0
        if self.pbrs is not None:
            self.pbrs.reset(priv_state)

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
        cell = self._cell(priv_state)

        # 1. Diminishing pre-key exploration up to max_exploration_cells (kept < exit_reward)
        explore = 0.0
        if entered_new_cell and self.tracker.unique_cells_visited - 1 <= cfg.max_exploration_cells:
            cell_idx = max(1, self.tracker.unique_cells_visited - 1)
            explore = cfg.new_cell_reward / (1.0 + float(cell_idx - 1) / max(1.0, cfg.explore_decay_cells))

        # 2. Post-key exploration reset so backtracking earns progress signal on Tier 3-4 maps
        post_key_explore = 0.0
        keys_held = float(np.sum(np.asarray(priv_state.get("keys", 0), dtype=np.float32)))
        if events.keys_picked_up > 0:
            self._post_key_visited_cells = {cell}
            self._post_key_explore_count = 0
        elif keys_held > 0.0 and cell not in self._post_key_visited_cells:
            self._post_key_visited_cells.add(cell)
            if self._post_key_explore_count < cfg.max_post_key_cells:
                self._post_key_explore_count += 1
                post_key_explore = cfg.post_key_cell_reward

        # 3. USE waste & door-bounce penalty (reversing an already-rising door)
        use_penalty = 0.0
        curr_ceil_raw = priv_state.get("sector_ceiling")
        locks_raw = priv_state.get("door_lock")
        bounced = False
        started_or_rising = False
        if curr_ceil_raw is not None and self._prev_ceil is not None and len(curr_ceil_raw) == len(self._prev_ceil):
            curr_ceil = np.asarray(curr_ceil_raw, dtype=np.float64)
            delta_ceil = curr_ceil - self._prev_ceil
            new_rising: set[int] = set()
            for idx in range(len(delta_ceil)):
                is_door = locks_raw is None or (idx < len(locks_raw) and int(locks_raw[idx]) >= 0)
                if not is_door:
                    continue
                if idx in self._rising_doors and delta_ceil[idx] < -0.5:
                    bounced = True
                elif delta_ceil[idx] > 0.5:
                    new_rising.add(idx)
                    started_or_rising = True
            self._rising_doors = new_rising
            self._prev_ceil = curr_ceil.copy()
        elif curr_ceil_raw is not None:
            self._prev_ceil = np.array(curr_ceil_raw, dtype=np.float64, copy=True)

        if bounced:
            use_penalty += cfg.door_bounce_penalty
        elif action is not None and prev_action is not None:
            if (
                events.doors_opened == 0
                and events.locked_doors_unlocked == 0
                and not started_or_rising
            ):
                if int(action[_USE_IDX]) == 1 and int(prev_action[_USE_IDX]) == 0:
                    use_penalty += cfg.use_waste_penalty
                elif int(action[_USE_IDX]) == 1 and int(prev_action[_USE_IDX]) == 1:
                    use_penalty += cfg.use_hold_penalty

        # 4. Exit + time/health efficiency
        efficiency = 0.0
        curr_health = float(priv_state.get("health", self._prev_health))
        if events.exit_reached:
            ep_ticks = max(1, int(priv_state.get("episode_ticks", 1)))
            remaining_time = max(0.0, 1.0 - float(priv_state.get("tic", ep_ticks)) / float(ep_ticks))
            health_frac = min(1.5, max(0.0, curr_health / 100.0))
            efficiency = cfg.efficiency_bonus * remaining_time + cfg.health_efficiency_bonus * health_frac

        # 5. Combat, wasted ammo/action spam, and bounded low-health/armor pickup bonus
        curr_ammo, curr_armor = _extract_ammo_and_armor(priv_state, obs)
        ammo_dropped = (
            curr_ammo is not None and self._prev_ammo is not None and curr_ammo < self._prev_ammo
        )
        attack_pressed = action is not None and int(action[_ATTACK_IDX]) == 1
        ammo_waste = 0.0
        if (ammo_dropped or attack_pressed) and events.kills == 0 and not _has_visible_enemy(obs):
            ammo_waste = cfg.wasted_ammo_penalty

        weapon_switch_penalty = 0.0
        if action is not None and int(action[_WEAPON_SWITCH_IDX]) == 1 and obs is not None and "vitals" in obs:
            sel_wpn = int(round(float(obs["vitals"][2])))
            sel_ammo = float(obs["vitals"][3])
            if sel_wpn >= 2 and sel_ammo > 0.5:
                weapon_switch_penalty = cfg.weapon_switch_waste_penalty

        pickup = 0.0
        if self._pickup_bonus_earned < cfg.max_pickup_bonus:
            raw_pickup = 0.0
            if curr_health > self._prev_health and self._prev_health < 80.0:
                raw_pickup += 0.015 * (curr_health - self._prev_health)
            if curr_armor > self._prev_armor and self._prev_armor < 50.0:
                raw_pickup += 0.008 * (curr_armor - self._prev_armor)
            if obs is not None and "vitals" in obs:
                cur_wpn = int(round(float(obs["vitals"][2])))
                if cur_wpn >= 3 and cur_wpn not in self._owned_weapons:
                    self._owned_weapons.add(cur_wpn)
                    raw_pickup += cfg.weapon_pickup_reward
            if raw_pickup > 0.0:
                pickup = min(raw_pickup, cfg.max_pickup_bonus - self._pickup_bonus_earned)
                self._pickup_bonus_earned += pickup

        pbrs_rew = self.pbrs.step(priv_state) if self.pbrs is not None else 0.0

        self._prev_health = curr_health
        self._prev_armor = curr_armor
        if curr_ammo is not None:
            self._prev_ammo = curr_ammo

        components = {
            "exit": cfg.exit_reward if events.exit_reached else 0.0,
            "efficiency": efficiency,
            "key": cfg.key_pickup_reward * events.keys_picked_up,
            "unlock": cfg.locked_door_unlock_reward * events.locked_doors_unlocked,
            "door": cfg.door_open_reward * events.doors_opened,
            "explore": explore,
            "post_key_explore": post_key_explore,
            "kill": cfg.kill_reward * events.kills,
            "damage": cfg.damage_penalty * events.damage_taken,
            "death": cfg.death_penalty if events.dead else 0.0,
            "use_penalty": use_penalty,
            "ammo_waste": ammo_waste,
            "weapon_switch_penalty": weapon_switch_penalty,
            "pickup": pickup,
            "pbrs": pbrs_rew,
            "time": cfg.step_penalty,
        }
        return float(sum(components.values())), events, components
