"""Tests for Role 4 independent evaluator and report comparison."""

from __future__ import annotations

import json

import numpy as np

from doom_env import spaces as S
from doom_env.sensors import RAY_DOOR, RAY_LOCKED
from tests.fixtures import fixture_wad
from training.eval import compare_reports, evaluate_episode, evaluate_suite
from training.interfaces import MapSpec


class _ScriptedDoorRunner:
    """Test-only scripted dummy agent that walks east and pulses USE near doors."""

    def __init__(self, observation_space, action_space, device: str):
        self.action_space = action_space
        self.cooldown = 0

    def reset(self) -> None:
        self.cooldown = 0

    def act(self, observation: dict) -> np.ndarray:
        action = np.zeros(self.action_space.shape, dtype=np.int8)
        front_dist = float(observation["rays"]["distance"][0]) * S.RAY_RANGE
        front_type = int(observation["rays"]["hit_type"][0])
        if self.cooldown > 0:
            self.cooldown -= 1
        if front_type in (RAY_DOOR, RAY_LOCKED) and front_dist < 80.0 and self.cooldown == 0:
            action[S.BUTTONS.index("USE")] = 1
            self.cooldown = 12
        else:
            action[S.BUTTONS.index("MOVE_FORWARD")] = 1
        return action


def test_eval_baseline_two_consecutive_maps():
    specs = [
        MapSpec(wad="maps/two_rooms.wad", map_name="MAP01", family="door_exit", tier=1, split="dev", seeds=(0,)),
        MapSpec(wad="maps/two_rooms.wad", map_name="MAP01", family="door_exit", tier=1, split="dev", seeds=(1,)),
    ]
    report = evaluate_suite(specs, device="cpu")
    assert report["summary"]["overall"]["episodes"] == 2
    assert report["summary"]["overall"]["errors"] == 0
    assert report["summary"]["overall"]["all_within_budget"] is True
    for ep in report["episodes"]:
        assert ep["outcome"] in {"exit", "death", "timeout"}
        assert ep["steps"] > 0
        assert ep["agent_time_s"] >= 0.0
    # Must be JSON-serializable for report saving and comparison
    json.dumps(report)


def test_eval_distinguishes_pickup_unlock_exit_and_timeout():
    agent = _ScriptedDoorRunner(None, S.action_space(), "cpu")

    # 1. Key pickup + locked door unlock + exit
    key_wad = fixture_wad("############\n#P.r..R...E#\n############")
    key_spec = MapSpec(wad=key_wad, family="key_return", tier=3, split="dev", episode_ticks=1000)
    m_key = evaluate_episode(agent, key_spec, seed=0)
    assert m_key.keys_picked_up == 1
    assert m_key.locked_doors_unlocked == 1
    assert m_key.outcome == "exit"

    # 2. Unlocked door open + exit
    door_wad = fixture_wad("##########\n#P..D...E#\n##########")
    door_spec = MapSpec(wad=door_wad, family="door_exit", tier=2, split="dev", episode_ticks=1000)
    m_door = evaluate_episode(agent, door_spec, seed=0)
    assert m_door.doors_opened == 1
    assert m_door.locked_doors_unlocked == 0
    assert m_door.outcome == "exit"

    # 3. Timeout on dead-end room
    idle_wad = fixture_wad("#####\n#P..#\n#####")
    idle_spec = MapSpec(wad=idle_wad, family="multi_room", tier=2, split="dev", episode_ticks=40)
    m_timeout = evaluate_episode(agent, idle_spec, seed=0)
    assert m_timeout.outcome == "timeout"
    assert m_timeout.steps == 10


def test_compare_reports_selection_rule():
    def _make_report(*, errors=0, budget=True, tier34=0, unlocks=0, tier12=0, kills=0):
        return {
            "summary": {
                "overall": {
                    "errors": errors,
                    "all_within_budget": budget,
                    "tier34_exits": tier34,
                    "locked_doors_unlocked": unlocks,
                    "tier12_exits": tier12,
                    "kills": kills,
                    "keys_picked_up": 0,
                    "mean_unique_cells": 1.0,
                }
            }
        }

    base = _make_report(tier34=1, unlocks=2, tier12=2, kills=3)
    better_t34 = _make_report(tier34=2, unlocks=1, tier12=1, kills=0)
    better_unlock = _make_report(tier34=1, unlocks=3, tier12=0, kills=0)
    broken = _make_report(errors=1, tier34=5)

    assert compare_reports(base, better_t34)["verdict"] == "promote"
    assert compare_reports(base, better_unlock)["verdict"] == "promote"
    assert compare_reports(base, broken)["verdict"] == "keep_incumbent"
