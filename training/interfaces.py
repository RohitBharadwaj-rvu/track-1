"""Shared data contracts across Roles 2 (PPO), 3 (Maps/Curriculum), and 4 (Eval/Release).

Locking these dataclasses early lets all 4 roles build in parallel without merge conflicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MapSpec:
    """One playable scenario registered by Role 3 and consumed by Roles 2 & 4."""

    wad: str                       # repo-relative path, e.g. "maps/two_rooms.wad"
    map_name: str = "MAP01"
    family: str = "door_exit"      # "combat" | "multi_room" | "key_return" | "gauntlet" | "door_exit"
    tier: int = 1                  # 1..4 matching docs/EVALUATION.md
    split: str = "train"           # "train" | "dev" | "final"
    episode_ticks: int = 2100      # 2100..4200
    seeds: tuple[int, ...] = (0, 1, 2)


@dataclass
class StepEvents:
    """Per-step progression and combat deltas derived from `env.privileged_state()`."""

    keys_picked_up: int = 0
    doors_opened: int = 0
    locked_doors_unlocked: int = 0
    kills: int = 0
    damage_taken: float = 0.0
    exit_reached: bool = False
    dead: bool = False
    timeout: bool = False


@dataclass
class EpisodeMetrics:
    """Per-episode record emitted by Role 4's evaluator and Role 2's training logger."""

    wad: str
    map_name: str
    family: str
    tier: int
    seed: int
    steps: int = 0
    ticks: int = 0
    outcome: str = "timeout"       # "exit" | "death" | "timeout" | "error"
    keys_picked_up: int = 0
    doors_opened: int = 0
    locked_doors_unlocked: int = 0
    kills: int = 0
    damage_taken: float = 0.0
    final_health: float = 0.0
    unique_cells_visited: int = 0
    total_reward: float = 0.0
    agent_time_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class PrivilegedEventTracker:
    """Tracks pickup, door open, locked-door unlock, kills, and outcome from `privileged_state()`.

    Used identically by Role 2 (reward shaping) and Role 4 (independent evaluation) so
    training logs and evaluation reports measure the exact same events.
    """

    def __init__(self, open_threshold: float = 8.0, cell_size: float = 64.0):
        self.open_threshold = float(open_threshold)
        self.cell_size = float(cell_size)
        self._prev: dict[str, Any] | None = None
        self._initial_ceil: np.ndarray | None = None
        self._opened_sectors: set[int] = set()
        self._unlocked_sectors: set[int] = set()
        self._visited_cells: set[tuple[int, int]] = set()

    def reset(self, priv_state: dict[str, Any]) -> None:
        self._prev = priv_state
        ceil = priv_state.get("sector_ceiling")
        self._initial_ceil = None if ceil is None else np.array(ceil, dtype=np.float64, copy=True)
        self._opened_sectors.clear()
        self._unlocked_sectors.clear()
        self._visited_cells = {self._cell(priv_state["x"], priv_state["y"])}

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(np.floor(x / self.cell_size)), int(np.floor(y / self.cell_size)))

    @property
    def unique_cells_visited(self) -> int:
        return len(self._visited_cells)

    def update(self, priv_state: dict[str, Any]) -> tuple[StepEvents, bool]:
        """Advance tracker with the latest `env.privileged_state()`.

        Returns `(events, entered_new_cell)`.
        """
        if self._prev is None:
            self.reset(priv_state)
            return StepEvents(), False

        prev = self._prev
        prev_keys = np.asarray(prev["keys"], dtype=np.int32)
        curr_keys = np.asarray(priv_state["keys"], dtype=np.int32)
        keys_delta = int(np.maximum(curr_keys - prev_keys, 0).sum())

        doors_opened = 0
        locked_doors_unlocked = 0
        ceil = priv_state.get("sector_ceiling")
        locks = priv_state.get("door_lock")
        if ceil is not None and self._initial_ceil is not None and len(ceil) == len(self._initial_ceil):
            raised = np.flatnonzero(np.asarray(ceil, dtype=np.float64) - self._initial_ceil >= self.open_threshold)
            for idx in raised.tolist():
                is_locked = locks is not None and idx < len(locks) and int(locks[idx]) > 0
                if is_locked:
                    if idx not in self._unlocked_sectors:
                        self._unlocked_sectors.add(idx)
                        locked_doors_unlocked += 1
                else:
                    if idx not in self._opened_sectors:
                        self._opened_sectors.add(idx)
                        doors_opened += 1

        cell = self._cell(priv_state["x"], priv_state["y"])
        new_cell = cell not in self._visited_cells
        if new_cell:
            self._visited_cells.add(cell)

        events = StepEvents(
            keys_picked_up=keys_delta,
            doors_opened=doors_opened,
            locked_doors_unlocked=locked_doors_unlocked,
            kills=max(0, int(priv_state["kills"]) - int(prev["kills"])),
            damage_taken=max(0.0, float(priv_state["damage_taken"]) - float(prev["damage_taken"])),
            exit_reached=bool(priv_state["exit_reached"]),
            dead=bool(priv_state["dead"]),
            timeout=bool(priv_state["timeout"]),
        )
        self._prev = priv_state
        return events, new_cell
