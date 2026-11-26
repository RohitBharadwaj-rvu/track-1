"""Independent evaluation and candidate comparison harness (Role 4).

Follows the frozen contracts in `training/interfaces.py` and `docs/EVALUATION.md`.
Usage:
    python -m training.eval --seeds 0 1
    python -m training.eval --split dev --out eval_dev.json --record-dir /tmp/replays --keep-key-replays
    python -m training.eval --compare baseline.json eval_dev.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Callable

import numpy as np

from doom_env import make_env
from doom_env.limits import agent_budget_s
from doom_env.spaces import BUTTONS, FRAME_SKIP
from training.interfaces import EpisodeMetrics, MapSpec, PrivilegedEventTracker

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AGENT_SPEC = "submission.agent:DoomAgent"
DEFAULT_FALLBACK_MAPS = [
    MapSpec(
        wad="maps/two_rooms.wad",
        map_name="MAP01",
        family="door_exit",
        tier=1,
        split="dev",
        episode_ticks=2100,
        seeds=(0, 1),
    )
]

_IDX_FWD = BUTTONS.index("MOVE_FORWARD")
_IDX_BACK = BUTTONS.index("MOVE_BACKWARD")
_IDX_LEFT = BUTTONS.index("MOVE_LEFT")
_IDX_RIGHT = BUTTONS.index("MOVE_RIGHT")
_IDX_TURN_L = BUTTONS.index("TURN_LEFT")
_IDX_TURN_R = BUTTONS.index("TURN_RIGHT")
_IDX_ATTACK = BUTTONS.index("ATTACK")
_IDX_USE = BUTTONS.index("USE")


def load_agent_factory(agent_spec: str = DEFAULT_AGENT_SPEC) -> Callable[[Any, Any, str], Any]:
    """Import `module:ClassName` and return the agent class/factory."""
    if ":" not in agent_spec:
        raise ValueError(f"Expected 'module:ClassName' for agent_spec, got {agent_spec!r}")
    module_name, class_name = agent_spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def parse_map_arg(raw: str, split: str = "dev") -> MapSpec:
    """Parse `wad[:map_name[:family[:tier[:episode_ticks]]]]` into a MapSpec."""
    parts = raw.split(":")
    if len(parts) > 5:
        raise ValueError(f"Invalid --map specification: {raw!r}")
    wad = parts[0]
    map_name = parts[1].upper() if len(parts) > 1 and parts[1] else "MAP01"
    family = parts[2] if len(parts) > 2 and parts[2] else "door_exit"
    tier = int(parts[3]) if len(parts) > 3 and parts[3] else 1
    episode_ticks = int(parts[4]) if len(parts) > 4 and parts[4] else 2100
    return MapSpec(
        wad=wad,
        map_name=map_name,
        family=family,
        tier=tier,
        split=split,
        episode_ticks=episode_ticks,
    )


def resolve_maps(
    split: str = "dev",
    map_args: list[str] | None = None,
    suite: str | None = None,
) -> list[MapSpec]:
    """Resolve maps from CLI --map flags, Role 3's `training.maps`, or Stage 1 fallback."""
    if map_args:
        return [parse_map_arg(m, split=split) for m in map_args]

    target = suite or split
    try:
        maps_mod = importlib.import_module("training.maps")
    except ModuleNotFoundError:
        return list(DEFAULT_FALLBACK_MAPS)

    if target == "connect" and hasattr(maps_mod, "CONNECT_MAPS"):
        specs = list(maps_mod.CONNECT_MAPS)
        if specs:
            return specs

    if target == "mvp_gate":
        simple = list(getattr(maps_mod, "MVP_GATE_SIMPLE_MAPS", ()))
        combat = list(getattr(maps_mod, "MVP_GATE_COMBAT_MAPS", ()))
        if simple or combat:
            return simple + combat

    if hasattr(maps_mod, "get_maps"):
        specs = list(maps_mod.get_maps(split=target))
        if specs:
            return specs

    attr_map = {
        "dev": ("DEV_MAPS", "ALL_MAPS"),
        "final": ("FINAL_MAPS", "ALL_MAPS"),
        "train": ("TRAIN_MAPS", "ALL_MAPS"),
        "all": ("ALL_MAPS", "DEV_MAPS"),
    }
    for attr in attr_map.get(target, ("DEV_MAPS", "ALL_MAPS")):
        if hasattr(maps_mod, attr):
            specs = [s for s in getattr(maps_mod, attr) if target == "all" or s.split == target]
            if specs:
                return specs

    return list(DEFAULT_FALLBACK_MAPS)


def _resolve_wad_path(wad: str) -> str:
    p = Path(wad)
    if p.is_absolute() or p.exists():
        return str(p)
    candidate = ROOT / wad
    return str(candidate if candidate.exists() else p)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _weights_sha256() -> dict[str, str]:
    weights_dir = ROOT / "submission" / "weights"
    if not weights_dir.is_dir():
        return {}
    hashes: dict[str, str] = {}
    for p in sorted(weights_dir.rglob("*")):
        if not p.is_file() or p.name == ".gitkeep":
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        hashes[str(p.relative_to(ROOT)).replace("\\", "/")] = h.hexdigest()
    return hashes


def evaluate_episode(
    agent: Any,
    spec: MapSpec,
    seed: int,
    *,
    render_buffers: bool = False,
    record_dir: Path | None = None,
) -> EpisodeMetrics:
    """Run one episode on `(spec, seed)` and compute metrics via `PrivilegedEventTracker`."""
    wad_path = _resolve_wad_path(spec.wad)
    env = make_env(
        wad_path,
        spec.map_name,
        episode_ticks=spec.episode_ticks,
        render_buffers=render_buffers,
    )
    tracker = PrivilegedEventTracker()
    replay_path: Path | None = None
    if record_dir is not None:
        record_dir.mkdir(parents=True, exist_ok=True)
        replay_path = record_dir / f"{Path(spec.wad).stem}_{spec.map_name}_seed{seed}.lmp"

    steps = 0
    total_reward = 0.0
    keys_picked_up = 0
    doors_opened = 0
    locked_doors_unlocked = 0
    kills = 0
    damage_taken = 0.0
    use_presses = 0
    consecutive_use_presses = 0
    opposing_button_steps = 0
    attack_steps = 0
    wasted_attack_steps = 0
    prev_use = 0
    step_times_ms: list[float] = []
    outcome = "timeout"
    error_msg: str | None = None
    last_priv: dict[str, Any] = {}

    try:
        reset_opts = {"record_path": str(replay_path)} if replay_path is not None else None
        obs, info = env.reset(seed=seed, options=reset_opts)
        last_priv = env.privileged_state()
        tracker.reset(last_priv)
        agent.reset()

        while True:
            t0 = time.perf_counter()
            try:
                raw_action = agent.act(obs)
                action = np.asarray(raw_action)
                if action.shape != env.action_space.shape:
                    raise ValueError(
                        f"Invalid action shape {action.shape}, expected {env.action_space.shape}"
                    )
                if not np.all((action == 0) | (action == 1)):
                    raise ValueError("Action entries must be 0 or 1")
                action = action.astype(np.int8)
            except Exception as exc:  # noqa: BLE001
                step_times_ms.append((time.perf_counter() - t0) * 1000.0)
                outcome = "error"
                error_msg = f"{type(exc).__name__}: {exc}"
                break

            step_times_ms.append((time.perf_counter() - t0) * 1000.0)
            curr_use = int(action[_IDX_USE])
            if curr_use:
                use_presses += 1
                if prev_use:
                    consecutive_use_presses += 1
            prev_use = curr_use

            if (
                (action[_IDX_FWD] and action[_IDX_BACK])
                or (action[_IDX_LEFT] and action[_IDX_RIGHT])
                or (action[_IDX_TURN_L] and action[_IDX_TURN_R])
            ):
                opposing_button_steps += 1

            if action[_IDX_ATTACK]:
                attack_steps += 1
                ent = obs.get("visible_entities", {})
                ent_mask = np.asarray(ent.get("mask", ()), dtype=np.int32)
                ent_type = np.asarray(ent.get("entity_type", ()), dtype=np.int32)
                enemy_in_fov = bool(
                    len(ent_mask) > 0
                    and len(ent_type) == len(ent_mask)
                    and np.any((ent_mask == 1) & ((ent_type == 1) | (ent_type == 2) | (ent_type == 3)))
                )
                if not enemy_in_fov:
                    wasted_attack_steps += 1

            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total_reward += float(reward)

            last_priv = env.privileged_state()
            events, _ = tracker.update(last_priv)
            keys_picked_up += events.keys_picked_up
            doors_opened += events.doors_opened
            locked_doors_unlocked += events.locked_doors_unlocked
            kills += events.kills
            damage_taken += events.damage_taken

            if events.exit_reached or info.get("exit_reached"):
                outcome = "exit"
                break
            if events.dead:
                outcome = "death"
                break
            if events.timeout or truncated:
                outcome = "timeout"
                break
            if terminated:
                outcome = "death"
                break
    except Exception as exc:  # noqa: BLE001
        outcome = "error"
        error_msg = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()

    had_key_pickup = keys_picked_up > 0
    died_after_pickup = outcome == "death" and had_key_pickup
    agent_time_s = sum(step_times_ms) / 1000.0
    budget_s = agent_budget_s(spec.episode_ticks)
    sorted_ms = sorted(step_times_ms)
    p50_ms = statistics.median(sorted_ms) if sorted_ms else 0.0
    p99_idx = min(len(sorted_ms) - 1, int(0.99 * len(sorted_ms))) if sorted_ms else 0
    p99_ms = sorted_ms[p99_idx] if sorted_ms else 0.0

    extra: dict[str, Any] = {
        "first_act_ms": round(step_times_ms[0], 3) if step_times_ms else 0.0,
        "p50_act_ms": round(p50_ms, 3),
        "p99_act_ms": round(p99_ms, 3),
        "max_act_ms": round(max(step_times_ms), 3) if step_times_ms else 0.0,
        "budget_s": round(budget_s, 3),
        "within_budget": (agent_time_s <= budget_s) and (error_msg is None),
        "use_presses": use_presses,
        "consecutive_use_presses": consecutive_use_presses,
        "opposing_button_steps": opposing_button_steps,
        "attack_steps": attack_steps,
        "wasted_attack_steps": wasted_attack_steps,
        "had_key_pickup": had_key_pickup,
        "died_after_pickup": died_after_pickup,
    }
    if replay_path is not None:
        extra["replay_path"] = str(replay_path)
    if error_msg is not None:
        extra["error"] = error_msg

    return EpisodeMetrics(
        wad=spec.wad,
        map_name=spec.map_name,
        family=spec.family,
        tier=spec.tier,
        seed=seed,
        steps=steps,
        ticks=int(last_priv.get("tic", steps * FRAME_SKIP)),
        outcome=outcome,
        keys_picked_up=keys_picked_up,
        doors_opened=doors_opened,
        locked_doors_unlocked=locked_doors_unlocked,
        kills=kills,
        damage_taken=round(damage_taken, 3),
        final_health=float(last_priv.get("health", 0.0)),
        unique_cells_visited=tracker.unique_cells_visited,
        total_reward=round(total_reward, 4),
        agent_time_s=round(agent_time_s, 4),
        extra=extra,
    )


def _aggregate_episodes(episodes: list[EpisodeMetrics]) -> dict[str, Any]:
    n = len(episodes)
    if n == 0:
        return {
            "episodes": 0,
            "exits": 0,
            "deaths": 0,
            "timeouts": 0,
            "errors": 0,
            "exit_rate": 0.0,
            "keys_picked_up": 0,
            "doors_opened": 0,
            "locked_doors_unlocked": 0,
            "kills": 0,
            "episodes_with_pickup": 0,
            "episodes_with_unlock": 0,
            "deaths_after_pickup": 0,
            "unlock_given_pickup_rate": None,
            "exit_given_unlock_rate": None,
            "mean_unique_cells": 0.0,
            "total_steps": 0,
            "use_presses": 0,
            "consecutive_use_presses": 0,
            "opposing_button_steps": 0,
            "attack_steps": 0,
            "wasted_attack_steps": 0,
            "total_agent_time_s": 0.0,
            "max_step_ms": 0.0,
            "all_within_budget": True,
        }
    episodes_with_pickup = sum(1 for e in episodes if e.keys_picked_up > 0)
    episodes_with_unlock = sum(1 for e in episodes if e.locked_doors_unlocked > 0)
    episodes_with_exit_after_unlock = sum(
        1 for e in episodes if e.locked_doors_unlocked > 0 and e.outcome == "exit"
    )
    deaths_after_pickup = sum(1 for e in episodes if bool(e.extra.get("died_after_pickup", False)))
    return {
        "episodes": n,
        "exits": sum(1 for e in episodes if e.outcome == "exit"),
        "deaths": sum(1 for e in episodes if e.outcome == "death"),
        "timeouts": sum(1 for e in episodes if e.outcome == "timeout"),
        "errors": sum(1 for e in episodes if e.outcome == "error"),
        "exit_rate": round(sum(1 for e in episodes if e.outcome == "exit") / n, 4),
        "keys_picked_up": sum(e.keys_picked_up for e in episodes),
        "doors_opened": sum(e.doors_opened for e in episodes),
        "locked_doors_unlocked": sum(e.locked_doors_unlocked for e in episodes),
        "kills": sum(e.kills for e in episodes),
        "episodes_with_pickup": episodes_with_pickup,
        "episodes_with_unlock": episodes_with_unlock,
        "deaths_after_pickup": deaths_after_pickup,
        "unlock_given_pickup_rate": (
            round(episodes_with_unlock / episodes_with_pickup, 4) if episodes_with_pickup > 0 else None
        ),
        "exit_given_unlock_rate": (
            round(episodes_with_exit_after_unlock / episodes_with_unlock, 4)
            if episodes_with_unlock > 0
            else None
        ),
        "mean_unique_cells": round(sum(e.unique_cells_visited for e in episodes) / n, 2),
        "total_steps": sum(e.steps for e in episodes),
        "use_presses": sum(int(e.extra.get("use_presses", 0)) for e in episodes),
        "consecutive_use_presses": sum(int(e.extra.get("consecutive_use_presses", 0)) for e in episodes),
        "opposing_button_steps": sum(int(e.extra.get("opposing_button_steps", 0)) for e in episodes),
        "attack_steps": sum(int(e.extra.get("attack_steps", 0)) for e in episodes),
        "wasted_attack_steps": sum(int(e.extra.get("wasted_attack_steps", 0)) for e in episodes),
        "total_agent_time_s": round(sum(e.agent_time_s for e in episodes), 4),
        "max_step_ms": round(max((e.extra.get("max_act_ms", 0.0) for e in episodes), default=0.0), 3),
        "all_within_budget": all(bool(e.extra.get("within_budget", False)) for e in episodes),
    }


def summarize_episodes(episodes: list[EpisodeMetrics]) -> dict[str, Any]:
    """Compute overall, per-tier (1..4), and per-family summary metrics."""
    overall = _aggregate_episodes(episodes)
    by_tier: dict[str, Any] = {}
    for tier in sorted({e.tier for e in episodes}):
        by_tier[str(tier)] = _aggregate_episodes([e for e in episodes if e.tier == tier])
    by_family: dict[str, Any] = {}
    for family in sorted({e.family for e in episodes}):
        by_family[family] = _aggregate_episodes([e for e in episodes if e.family == family])

    overall["tier12_exits"] = sum(1 for e in episodes if e.tier in (1, 2) and e.outcome == "exit")
    overall["tier34_exits"] = sum(1 for e in episodes if e.tier in (3, 4) and e.outcome == "exit")
    return {"overall": overall, "by_tier": by_tier, "by_family": by_family}


def diagnose_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Diagnose the primary bottlenecks from event counts (`REVISED_PLAN.md §4` & `updates.md §5`)."""
    overall = report["summary"]["overall"]
    findings: list[dict[str, Any]] = []

    if overall.get("errors", 0) > 0 or not overall.get("all_within_budget", True):
        findings.append(
            {
                "priority": 1,
                "code": "runtime_errors_or_budget",
                "evidence": f"errors={overall.get('errors', 0)}, all_within_budget={overall.get('all_within_budget')}, max_step_ms={overall.get('max_step_ms', 0.0)}",
                "recommendation": "Fix agent runtime/action-shape errors or reduce model compute below 40 ms/step before training further.",
            }
        )

    total_steps = int(overall.get("total_steps", 0))
    opposing = int(overall.get("opposing_button_steps", 0))
    if total_steps > 0 and (opposing / total_steps) >= 0.10:
        findings.append(
            {
                "priority": 2,
                "code": "opposing_buttons",
                "evidence": f"opposing_button_steps={opposing}/{total_steps} ({opposing / total_steps:.1%})",
                "recommendation": "Replace independent movement/turn binaries with mutually-exclusive 3-way categorical heads (updates.md §2).",
            }
        )

    if overall.get("doors_opened", 0) == 0 and overall.get("locked_doors_unlocked", 0) == 0:
        consec_use = int(overall.get("consecutive_use_presses", 0))
        findings.append(
            {
                "priority": 3,
                "code": "use_bounce_or_no_doors",
                "evidence": f"doors_opened=0, locked_doors_unlocked=0, consecutive_use_presses={consec_use}",
                "recommendation": "Check USE button-edge handling, add USE-spam penalty in training/rewards.py, and verify door_open_reward (updates.md §2).",
            }
        )

    if overall.get("keys_picked_up", 0) > 0 and overall.get("locked_doors_unlocked", 0) == 0:
        findings.append(
            {
                "priority": 4,
                "code": "key_without_unlock",
                "evidence": f"keys_picked_up={overall['keys_picked_up']}, locked_doors_unlocked=0",
                "recommendation": "Agent finds keys but loses the return route; add key-return curriculum or test learned landmark memory slots (REVISED_PLAN.md §4).",
            }
        )

    if overall.get("deaths_after_pickup", 0) > 0:
        findings.append(
            {
                "priority": 5,
                "code": "deaths_after_pickup",
                "evidence": f"deaths_after_pickup={overall['deaths_after_pickup']}",
                "recommendation": "Add combat encounters along the post-pickup return path in the training curriculum (REVISED_PLAN.md §4).",
            }
        )

    if overall.get("mean_unique_cells", 0.0) < 10.0:
        findings.append(
            {
                "priority": 6,
                "code": "low_exploration",
                "evidence": f"mean_unique_cells={overall.get('mean_unique_cells', 0.0)}",
                "recommendation": "Increase first-visit exploration bonus and add branching/loop training maps.",
            }
        )

    attack_steps = int(overall.get("attack_steps", 0))
    wasted_attacks = int(overall.get("wasted_attack_steps", 0))
    if attack_steps >= 10 and (wasted_attacks / attack_steps) >= 0.60:
        findings.append(
            {
                "priority": 7,
                "code": "wasted_ammo",
                "evidence": f"wasted_attack_steps={wasted_attacks}/{attack_steps} ({wasted_attacks / attack_steps:.1%})",
                "recommendation": "Add a small wasted-ammo penalty in training/rewards.py when firing with no enemy in FOV (TRACK1_IMPROVEMENTS.md §5.3).",
            }
        )

    return findings


def _select_and_prune_replays(
    episodes: list[EpisodeMetrics],
    *,
    keep_key_replays: bool,
) -> dict[str, str | None]:
    recorded = [e for e in episodes if e.extra.get("replay_path")]
    if not recorded:
        return {"best_success": None, "representative_failure": None}

    successes = [e for e in recorded if e.outcome == "exit"]
    best_success_ep = min(successes, key=lambda e: e.steps) if successes else None

    failures = [e for e in recorded if e.outcome != "exit"]
    rep_failure_ep = None
    if failures:
        def _failure_rank(e: EpisodeMetrics) -> tuple[int, int]:
            if e.keys_picked_up > 0 and e.locked_doors_unlocked == 0:
                return (0, -e.unique_cells_visited)
            if e.extra.get("died_after_pickup"):
                return (1, -e.unique_cells_visited)
            if e.outcome == "death":
                return (2, -e.unique_cells_visited)
            if e.outcome == "error":
                return (3, 0)
            return (4, -e.unique_cells_visited)

        rep_failure_ep = min(failures, key=_failure_rank)

    best_path = best_success_ep.extra["replay_path"] if best_success_ep else None
    fail_path = rep_failure_ep.extra["replay_path"] if rep_failure_ep else None
    keep_set = {p for p in (best_path, fail_path) if p is not None}

    if keep_key_replays:
        for e in recorded:
            rp = e.extra.get("replay_path")
            if rp and rp not in keep_set:
                Path(rp).unlink(missing_ok=True)
                e.extra["replay_pruned"] = True

    return {"best_success": best_path, "representative_failure": fail_path}


def _apply_eval_mode(agent: Any, eval_mode: str) -> bool:
    """Apply `'deterministic'` or `'stochastic'` eval mode to an agent if it exposes a flag."""
    if eval_mode == "default":
        return False
    det_flag = eval_mode == "deterministic"
    for attr in ("deterministic", "eval_deterministic", "_deterministic"):
        if hasattr(agent, attr):
            setattr(agent, attr, det_flag)
            return True
    return False


def evaluate_suite(
    specs: list[MapSpec],
    *,
    agent_factory: Callable[[Any, Any, str], Any] | None = None,
    agent_spec: str = DEFAULT_AGENT_SPEC,
    device: str = "cpu",
    seeds_override: tuple[int, ...] | None = None,
    render_buffers: bool = False,
    record_dir: Path | None = None,
    keep_key_replays: bool = False,
    eval_mode: str = "default",
) -> dict[str, Any]:
    """Instantiate an agent once and evaluate all `(spec, seed)` episodes."""
    if not specs:
        raise ValueError("specs must not be empty")
    if eval_mode not in ("default", "deterministic", "stochastic"):
        raise ValueError(f"Unknown eval_mode {eval_mode!r}")
    factory = agent_factory or load_agent_factory(agent_spec)

    probe_env = make_env(
        _resolve_wad_path(specs[0].wad),
        specs[0].map_name,
        episode_ticks=specs[0].episode_ticks,
        render_buffers=render_buffers,
    )
    try:
        obs_space = probe_env.observation_space
        act_space = probe_env.action_space
    finally:
        probe_env.close()

    wall_start = time.perf_counter()
    init_start = time.perf_counter()
    agent = factory(obs_space, act_space, device)
    eval_mode_applied = _apply_eval_mode(agent, eval_mode)
    init_time_s = round(time.perf_counter() - init_start, 4)

    episodes: list[EpisodeMetrics] = []
    for spec in specs:
        seeds = seeds_override if seeds_override is not None else spec.seeds
        for seed in seeds:
            episodes.append(
                evaluate_episode(
                    agent,
                    spec,
                    seed,
                    render_buffers=render_buffers,
                    record_dir=record_dir,
                )
            )
    wall_time_s = round(time.perf_counter() - wall_start, 4)

    summary = summarize_episodes(episodes)
    if record_dir is not None:
        summary["key_replays"] = _select_and_prune_replays(
            episodes, keep_key_replays=keep_key_replays
        )

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "weights_sha256": _weights_sha256(),
            "agent_spec": agent_spec,
            "device": device,
            "eval_mode": eval_mode,
            "eval_mode_applied": eval_mode_applied,
            "render_buffers": render_buffers,
            "init_time_s": init_time_s,
            "wall_time_s": wall_time_s,
            "maps_manifest": [asdict(s) for s in specs],
        },
        "summary": summary,
        "episodes": [asdict(e) for e in episodes],
    }
    report["diagnosis"] = diagnose_report(report)
    return report


def _per_map_totals(report: dict[str, Any]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for ep in report.get("episodes", []):
        key = f"{Path(ep['wad']).name}:{ep['map_name']}"
        row = totals.setdefault(key, {"exits": 0, "unlocks": 0, "kills": 0, "deaths": 0})
        if ep.get("outcome") == "exit":
            row["exits"] += 1
        elif ep.get("outcome") == "death":
            row["deaths"] += 1
        row["unlocks"] += int(ep.get("locked_doors_unlocked", 0))
        row["kills"] += int(ep.get("kills", 0))
    return totals


def compare_reports(baseline: dict[str, Any], challenger: dict[str, Any]) -> dict[str, Any]:
    """Compare two reports using REVISED_PLAN.md §5 selection rules."""
    b = baseline["summary"]["overall"]
    c = challenger["summary"]["overall"]

    deltas = {
        "errors": c["errors"] - b["errors"],
        "tier34_exits": c.get("tier34_exits", 0) - b.get("tier34_exits", 0),
        "locked_doors_unlocked": c["locked_doors_unlocked"] - b["locked_doors_unlocked"],
        "tier12_exits": c.get("tier12_exits", 0) - b.get("tier12_exits", 0),
        "kills": c["kills"] - b["kills"],
        "keys_picked_up": c["keys_picked_up"] - b["keys_picked_up"],
        "mean_unique_cells": round(c["mean_unique_cells"] - b["mean_unique_cells"], 2),
    }

    b_maps = _per_map_totals(baseline)
    c_maps = _per_map_totals(challenger)
    all_map_keys = sorted(set(b_maps) | set(c_maps))
    per_map_deltas: dict[str, dict[str, int]] = {}
    regressed_maps: list[str] = []
    maps_with_prog_gain = 0
    for mk in all_map_keys:
        bm = b_maps.get(mk, {"exits": 0, "unlocks": 0, "kills": 0, "deaths": 0})
        cm = c_maps.get(mk, {"exits": 0, "unlocks": 0, "kills": 0, "deaths": 0})
        md = {
            "exits_delta": cm["exits"] - bm["exits"],
            "unlocks_delta": cm["unlocks"] - bm["unlocks"],
            "kills_delta": cm["kills"] - bm["kills"],
            "deaths_delta": cm["deaths"] - bm["deaths"],
        }
        per_map_deltas[mk] = md
        if md["exits_delta"] < 0 or md["unlocks_delta"] < 0:
            regressed_maps.append(mk)
        if md["exits_delta"] > 0 or md["unlocks_delta"] > 0:
            maps_with_prog_gain += 1

    single_map_gain_warning = (
        len(all_map_keys) >= 2
        and (deltas["tier34_exits"] + deltas["tier12_exits"] > 0 or deltas["locked_doors_unlocked"] > 0)
        and maps_with_prog_gain == 1
    )

    b_clean = b["errors"] == 0 and b.get("all_within_budget", True)
    c_clean = c["errors"] == 0 and c.get("all_within_budget", True)
    if not c_clean:
        verdict = "keep_incumbent"
        reason = "Challenger failed clean-run / compute-budget gate."
    elif not b_clean and c_clean:
        verdict = "promote"
        reason = "Challenger passed clean-run gate where baseline failed."
    else:
        verdict = "tie"
        reason = "No difference across selection metrics."
        for key in ("tier34_exits", "locked_doors_unlocked", "tier12_exits", "kills"):
            diff = deltas[key]
            if diff > 0:
                verdict = "promote"
                reason = f"Challenger improved {key} (+{diff})."
                break
            if diff < 0:
                verdict = "keep_incumbent"
                reason = f"Challenger regressed {key} ({diff})."
                break

    return {
        "verdict": verdict,
        "reason": reason,
        "deltas": deltas,
        "per_map_deltas": per_map_deltas,
        "regressed_maps": regressed_maps,
        "single_map_gain_warning": single_map_gain_warning,
    }


def check_mvp_gate(
    report: dict[str, Any],
    baseline_report: dict[str, Any] | None = None,
    *,
    early_death_ticks: int = 700,
) -> dict[str, Any]:
    """Check the Stage 2 Functional MVP Gate (`REVISED_PLAN.md §2`)."""
    overall = report["summary"]["overall"]
    clean_run = (overall.get("errors", 0) == 0) and bool(overall.get("all_within_budget", True))

    simple_wads: set[str] = set()
    combat_wads: set[str] = set()
    try:
        maps_mod = importlib.import_module("training.maps")
        simple_wads = {Path(s.wad).name for s in getattr(maps_mod, "MVP_GATE_SIMPLE_MAPS", ())}
        combat_wads = {Path(s.wad).name for s in getattr(maps_mod, "MVP_GATE_COMBAT_MAPS", ())}
    except ModuleNotFoundError:
        pass

    episodes = report.get("episodes", [])
    simple_by_map: dict[str, list[dict[str, Any]]] = {}
    combat_eps: list[dict[str, Any]] = []
    for ep in episodes:
        name = Path(ep["wad"]).name
        if name in simple_wads or (not simple_wads and ep.get("family") == "door_exit"):
            simple_by_map.setdefault(name, []).append(ep)
        elif name in combat_wads or (not combat_wads and ep.get("family") == "combat"):
            combat_eps.append(ep)

    per_simple_map: dict[str, dict[str, Any]] = {}
    for map_key, eps in sorted(simple_by_map.items()):
        required = 2 if len(eps) >= 3 else max(1, math.ceil(2 * len(eps) / 3))
        door_exits = sum(
            1 for e in eps if e.get("doors_opened", 0) >= 1 and e.get("outcome") == "exit"
        )
        per_simple_map[map_key] = {
            "episodes": len(eps),
            "door_exits": door_exits,
            "required": required,
            "passed": door_exits >= required,
        }
    simple_door_exit_pass = bool(per_simple_map) and all(v["passed"] for v in per_simple_map.values())

    mean_kills = (
        round(sum(e.get("kills", 0) for e in combat_eps) / len(combat_eps), 4)
        if combat_eps
        else 0.0
    )
    early_deaths = sum(
        1 for e in combat_eps if e.get("outcome") == "death" and e.get("ticks", 0) <= early_death_ticks
    )

    baseline_mean_kills = 0.0
    baseline_early_deaths = 0
    if baseline_report is not None:
        base_combat_eps = [
            e
            for e in baseline_report.get("episodes", [])
            if Path(e["wad"]).name in combat_wads or (not combat_wads and e.get("family") == "combat")
        ]
        if base_combat_eps:
            baseline_mean_kills = round(
                sum(e.get("kills", 0) for e in base_combat_eps) / len(base_combat_eps), 4
            )
            baseline_early_deaths = sum(
                1
                for e in base_combat_eps
                if e.get("outcome") == "death" and e.get("ticks", 0) <= early_death_ticks
            )

    combat_pass = (
        bool(combat_eps)
        and (mean_kills > baseline_mean_kills)
        and (early_deaths <= baseline_early_deaths)
    )

    checks = {
        "clean_run": clean_run,
        "simple_door_exit_pass": simple_door_exit_pass,
        "combat_pass": combat_pass,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "per_simple_map": per_simple_map,
        "combat_stats": {
            "episodes": len(combat_eps),
            "mean_kills": mean_kills,
            "baseline_mean_kills": baseline_mean_kills,
            "early_deaths": early_deaths,
            "baseline_early_deaths": baseline_early_deaths,
        },
    }


def format_report_markdown(report: dict[str, Any]) -> str:
    """Format an evaluation report into a readable Markdown summary."""
    meta = report["metadata"]
    overall = report["summary"]["overall"]
    unlock_rate = overall.get("unlock_given_pickup_rate")
    exit_rate_u = overall.get("exit_given_unlock_rate")
    unlock_str = f"{unlock_rate:.1%}" if unlock_rate is not None else "n/a"
    exit_u_str = f"{exit_rate_u:.1%}" if exit_rate_u is not None else "n/a"

    lines = [
        f"## Evaluation Summary ({meta.get('git_commit') or 'uncommitted'})",
        f"- Agent: `{meta['agent_spec']}` | Device: `{meta['device']}` | Eval mode: `{meta.get('eval_mode', 'default')}` | Init: `{meta['init_time_s']}s` | Wall: `{meta['wall_time_s']}s`",
        f"- Episodes: `{overall['episodes']}` | Exits: `{overall['exits']}` | Deaths: `{overall['deaths']}` | Timeouts: `{overall['timeouts']}` | Errors: `{overall['errors']}`",
        f"- Progression: Keys `{overall['keys_picked_up']}` | Doors `{overall['doors_opened']}` | Unlocks `{overall['locked_doors_unlocked']}` | Kills `{overall['kills']}` | Mean cells `{overall['mean_unique_cells']}`",
        f"- Funnel & Actions: `unlock|pickup={unlock_str}` | `exit|unlock={exit_u_str}` | Deaths after pickup `{overall.get('deaths_after_pickup', 0)}` | Opposing steps `{overall.get('opposing_button_steps', 0)}/{overall.get('total_steps', 0)}` | Consecutive USE `{overall.get('consecutive_use_presses', 0)}` | Wasted attacks `{overall.get('wasted_attack_steps', 0)}/{overall.get('attack_steps', 0)}`",
        f"- Compute: Total agent `{overall['total_agent_time_s']}s` | Max step `{overall['max_step_ms']} ms` | Within budget: `{overall['all_within_budget']}`",
        "",
        "| Map | Tier | Family | Seed | Outcome | Steps | Keys | Doors | Unlocks | Kills | Cells | Agent(s) | Max(ms) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ep in report["episodes"]:
        lines.append(
            f"| {Path(ep['wad']).name}:{ep['map_name']} | {ep['tier']} | {ep['family']} | {ep['seed']} "
            f"| {ep['outcome']} | {ep['steps']} | {ep['keys_picked_up']} | {ep['doors_opened']} "
            f"| {ep['locked_doors_unlocked']} | {ep['kills']} | {ep['unique_cells_visited']} "
            f"| {ep['agent_time_s']:.3f} | {ep['extra'].get('max_act_ms', 0.0):.2f} |"
        )

    key_replays = report["summary"].get("key_replays")
    if key_replays:
        lines.extend(
            [
                "",
                "### Key Replays",
                f"- Best success: `{key_replays.get('best_success')}`",
                f"- Representative failure: `{key_replays.get('representative_failure')}`",
            ]
        )

    diagnosis = report.get("diagnosis") or diagnose_report(report)
    if diagnosis:
        lines.extend(["", "### Automated Diagnosis"])
        for item in diagnosis:
            lines.append(f"- **[{item['code']}]** ({item['evidence']}): {item['recommendation']}")

    mvp_gate = report.get("mvp_gate")
    if mvp_gate is not None:
        status = "PASSED" if mvp_gate["passed"] else "FAILED"
        chk = mvp_gate["checks"]
        cs = mvp_gate["combat_stats"]
        lines.extend(
            [
                "",
                f"### Functional MVP Gate: {status}",
                f"- `clean_run={chk['clean_run']}` | `simple_door_exit_pass={chk['simple_door_exit_pass']}` | `combat_pass={chk['combat_pass']}`",
                f"- Combat: `mean_kills={cs['mean_kills']}` (baseline `{cs['baseline_mean_kills']}`), `early_deaths={cs['early_deaths']}` (baseline `{cs['baseline_early_deaths']}`)",
            ]
        )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Role 4 independent DoomAgent evaluator.")
    parser.add_argument("--split", default="dev", choices=("train", "dev", "final", "all"))
    parser.add_argument(
        "--suite",
        default=None,
        choices=("connect", "mvp_gate", "dev", "final", "train", "all"),
        help="Named map suite (overrides --split)",
    )
    parser.add_argument("--map", action="append", default=None, help="wad[:map[:family[:tier[:ticks]]]]")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Override seed list for all maps")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--eval-mode",
        default="default",
        choices=("default", "deterministic", "stochastic"),
        help="Override agent determinism flag during evaluation",
    )
    parser.add_argument(
        "--compare-determinism",
        action="store_true",
        help="Benchmark stochastic vs deterministic eval modes on the selected specs",
    )
    parser.add_argument("--agent", default=DEFAULT_AGENT_SPEC, help="Agent entry point module:Class")
    parser.add_argument("--render-buffers", action="store_true", help="Render RGB/depth buffers")
    parser.add_argument("--record-dir", type=Path, default=None, help="Directory to save ViZDoom .lmp replays")
    parser.add_argument(
        "--keep-key-replays",
        action="store_true",
        help="Prune non-selected .lmp replays in --record-dir after selecting best success & failure",
    )
    parser.add_argument(
        "--check-mvp-gate",
        nargs="?",
        const="",
        default=None,
        metavar="BASELINE_JSON",
        help="Evaluate Functional MVP Gate (optionally against a saved baseline JSON)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write JSON report to this path")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE_JSON", "CHALLENGER_JSON"), default=None)
    args = parser.parse_args(argv)

    if args.compare:
        baseline = json.loads(Path(args.compare[0]).read_text(encoding="utf-8"))
        challenger = json.loads(Path(args.compare[1]).read_text(encoding="utf-8"))
        cmp_result = compare_reports(baseline, challenger)
        print(json.dumps(cmp_result, indent=2))
        return 0

    specs = resolve_maps(split=args.split, map_args=args.map, suite=args.suite)
    seeds_override = tuple(args.seeds) if args.seeds is not None else None
    if args.compare_determinism:
        stoch = evaluate_suite(
            specs,
            agent_spec=args.agent,
            device=args.device,
            seeds_override=seeds_override,
            render_buffers=args.render_buffers,
            eval_mode="stochastic",
        )
        det = evaluate_suite(
            specs,
            agent_spec=args.agent,
            device=args.device,
            seeds_override=seeds_override,
            render_buffers=args.render_buffers,
            eval_mode="deterministic",
        )
        print("# Stochastic Mode")
        print(format_report_markdown(stoch))
        print("\n# Deterministic Mode")
        print(format_report_markdown(det))
        print("\n# Comparison (stochastic -> deterministic)")
        print(json.dumps(compare_reports(stoch, det), indent=2))
        return 0 if (stoch["summary"]["overall"]["errors"] == 0 and det["summary"]["overall"]["errors"] == 0) else 1

    report = evaluate_suite(
        specs,
        agent_spec=args.agent,
        device=args.device,
        seeds_override=seeds_override,
        render_buffers=args.render_buffers,
        record_dir=args.record_dir,
        keep_key_replays=args.keep_key_replays,
        eval_mode=args.eval_mode,
    )
    if args.check_mvp_gate is not None:
        base_rep = (
            json.loads(Path(args.check_mvp_gate).read_text(encoding="utf-8"))
            if args.check_mvp_gate
            else None
        )
        report["mvp_gate"] = check_mvp_gate(report, base_rep)
    print(format_report_markdown(report))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {args.out}")
    return 0 if report["summary"]["overall"]["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
