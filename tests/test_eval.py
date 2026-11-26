"""Tests for Role 4 independent evaluator, diagnostics, comparison, and brief tools."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from doom_env import spaces as S
from doom_env.sensors import RAY_DOOR, RAY_LOCKED
from fixtures import fixture_wad
from training.brief_tools import markdown_to_two_page_html, update_brief_outcomes_row
from training.eval import (
    check_mvp_gate,
    compare_reports,
    diagnose_report,
    evaluate_episode,
    evaluate_suite,
    resolve_maps,
)
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


class _OpposingAndSpamUseAgent:
    """Test-only dummy agent that holds USE and opposing movement buttons every step."""

    def __init__(self, observation_space, action_space, device: str):
        self.action_space = action_space

    def reset(self) -> None:
        pass

    def act(self, observation: dict) -> np.ndarray:
        action = np.zeros(self.action_space.shape, dtype=np.int8)
        action[S.BUTTONS.index("MOVE_FORWARD")] = 1
        action[S.BUTTONS.index("MOVE_BACKWARD")] = 1
        action[S.BUTTONS.index("USE")] = 1
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


def test_action_diagnostics_and_progression_funnel():
    idle_wad = fixture_wad("#####\n#P..#\n#####")
    idle_spec = MapSpec(wad=idle_wad, family="multi_room", tier=2, split="dev", episode_ticks=40, seeds=(0,))
    report = evaluate_suite(
        [idle_spec],
        agent_factory=_OpposingAndSpamUseAgent,
        agent_spec="tests.test_eval:_OpposingAndSpamUseAgent",
        device="cpu",
    )
    overall = report["summary"]["overall"]
    assert overall["total_steps"] == 10
    assert overall["opposing_button_steps"] == 10
    assert overall["use_presses"] == 10
    assert overall["consecutive_use_presses"] == 9
    assert overall["unlock_given_pickup_rate"] is None
    assert overall["exit_given_unlock_rate"] is None
    codes = {d["code"] for d in report["diagnosis"]}
    assert "opposing_buttons" in codes
    assert "use_bounce_or_no_doors" in codes


def test_diagnose_report_flags_key_and_post_pickup_deaths():
    synthetic = {
        "summary": {
            "overall": {
                "errors": 0,
                "all_within_budget": True,
                "total_steps": 100,
                "opposing_button_steps": 0,
                "doors_opened": 1,
                "locked_doors_unlocked": 0,
                "keys_picked_up": 2,
                "deaths_after_pickup": 1,
                "mean_unique_cells": 15.0,
            }
        }
    }
    codes = [d["code"] for d in diagnose_report(synthetic)]
    assert "key_without_unlock" in codes
    assert "deaths_after_pickup" in codes
    assert "opposing_buttons" not in codes


def test_compare_reports_selection_rule_and_per_map_checks():
    def _make_report(*, errors=0, budget=True, tier34=0, unlocks=0, tier12=0, kills=0, episodes=None):
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
            },
            "episodes": episodes or [],
        }

    base = _make_report(
        tier34=1,
        unlocks=2,
        tier12=2,
        kills=3,
        episodes=[
            {"wad": "maps/a.wad", "map_name": "MAP01", "outcome": "exit", "locked_doors_unlocked": 1, "kills": 1},
            {"wad": "maps/b.wad", "map_name": "MAP01", "outcome": "timeout", "locked_doors_unlocked": 1, "kills": 2},
        ],
    )
    better_t34 = _make_report(
        tier34=2,
        unlocks=2,
        tier12=2,
        kills=1,
        episodes=[
            {"wad": "maps/a.wad", "map_name": "MAP01", "outcome": "timeout", "locked_doors_unlocked": 0, "kills": 0},
            {"wad": "maps/b.wad", "map_name": "MAP01", "outcome": "exit", "locked_doors_unlocked": 2, "kills": 1},
        ],
    )
    better_unlock = _make_report(tier34=1, unlocks=3, tier12=0, kills=0)
    broken = _make_report(errors=1, tier34=5)

    cmp_t34 = compare_reports(base, better_t34)
    assert cmp_t34["verdict"] == "promote"
    assert cmp_t34["regressed_maps"] == ["a.wad:MAP01"]
    assert cmp_t34["single_map_gain_warning"] is True
    assert compare_reports(base, better_unlock)["verdict"] == "promote"
    assert compare_reports(base, broken)["verdict"] == "keep_incumbent"


def test_brief_tools_html_and_row_update(tmp_path: Path):
    brief_src = Path("docs/BRIEF.md").read_text(encoding="utf-8")
    html_out = tmp_path / "BRIEF.html"
    html_text = markdown_to_two_page_html(brief_src)
    html_out.write_text(html_text, encoding="utf-8")
    assert '<div class="page-break"></div>' in html_text
    assert "<table>" in html_text

    tmp_brief = tmp_path / "BRIEF.md"
    tmp_brief.write_text(brief_src, encoding="utf-8")
    report = {
        "metadata": {"maps_manifest": [{"wad": "maps/two_rooms.wad", "seeds": [0, 1]}]},
        "summary": {
            "overall": {
                "episodes": 2,
                "exits": 2,
                "keys_picked_up": 0,
                "locked_doors_unlocked": 0,
                "kills": 2,
                "mean_unique_cells": 31.0,
                "errors": 0,
            }
        },
    }
    updated = update_brief_outcomes_row(tmp_brief, "Stage 2 MVP Recurrent PPO", report)
    assert "| Stage 2 MVP Recurrent PPO | `maps/two_rooms.wad` (seeds `0, 1`) | 2 | 2 | 0 | 0 | 2 | 31.0 | 0 |" in updated


def test_resolve_role3_suites_and_check_mvp_gate():
    connect_specs = resolve_maps(suite="connect")
    assert len(connect_specs) == 2
    assert {s.family for s in connect_specs} == {"door_exit", "key_return"}

    mvp_specs = resolve_maps(suite="mvp_gate")
    assert len(mvp_specs) == 4
    assert {s.family for s in mvp_specs} == {"door_exit", "combat"}

    def _gate_report(*, door_exits=2, kills=2, early_deaths=0, errors=0):
        simple_eps = []
        for wad in ("maps/dev/dev_simple_dogleg_door.wad", "maps/dev/dev_simple_chamber_door.wad"):
            for seed in range(3):
                ok = seed < door_exits
                simple_eps.append(
                    {
                        "wad": wad,
                        "map_name": "MAP01",
                        "family": "door_exit",
                        "outcome": "exit" if ok else "timeout",
                        "doors_opened": 1 if ok else 0,
                        "ticks": 500,
                    }
                )
        combat_eps = []
        for wad in ("maps/dev/dev_combat_crossfire.wad", "maps/dev/dev_combat_ambush_ring.wad"):
            for seed in range(3):
                combat_eps.append(
                    {
                        "wad": wad,
                        "map_name": "MAP01",
                        "family": "combat",
                        "outcome": "death" if (seed == 0 and early_deaths > 0) else "timeout",
                        "kills": kills,
                        "ticks": 200 if (seed == 0 and early_deaths > 0) else 2100,
                    }
                )
        return {
            "summary": {"overall": {"errors": errors, "all_within_budget": True}},
            "episodes": simple_eps + combat_eps,
        }

    baseline = _gate_report(door_exits=0, kills=0, early_deaths=1)
    passing = _gate_report(door_exits=2, kills=2, early_deaths=1)
    failing_doors = _gate_report(door_exits=1, kills=2, early_deaths=0)
    failing_combat = _gate_report(door_exits=2, kills=0, early_deaths=0)

    assert check_mvp_gate(passing, baseline)["passed"] is True
    assert check_mvp_gate(failing_doors, baseline)["checks"]["simple_door_exit_pass"] is False
    assert check_mvp_gate(failing_combat, baseline)["checks"]["combat_pass"] is False


def test_wasted_attack_diagnostics_and_eval_mode():
    class _EmptyRoomShooter:
        def __init__(self, observation_space, action_space, device: str):
            self.action_space = action_space
            self.eval_deterministic = False

        def reset(self) -> None:
            pass

        def act(self, observation: dict) -> np.ndarray:
            action = np.zeros(self.action_space.shape, dtype=np.int8)
            action[S.BUTTONS.index("ATTACK")] = 1
            return action

    idle_wad = fixture_wad("#####\n#P..#\n#####")
    spec = MapSpec(wad=idle_wad, family="multi_room", tier=2, split="dev", episode_ticks=40, seeds=(0,))
    report = evaluate_suite(
        [spec],
        agent_factory=_EmptyRoomShooter,
        agent_spec="tests.test_eval:_EmptyRoomShooter",
        device="cpu",
        eval_mode="deterministic",
    )
    overall = report["summary"]["overall"]
    assert overall["attack_steps"] == 10
    assert overall["wasted_attack_steps"] == 10
    assert report["metadata"]["eval_mode"] == "deterministic"
    assert report["metadata"]["eval_mode_applied"] is True
    codes = {d["code"] for d in report["diagnosis"]}
    assert "wasted_ammo" in codes
