"""Independent evaluation and candidate comparison harness (Role 4).

Follows the frozen contracts in `training/interfaces.py` and `docs/EVALUATION.md`.
Usage:
    python -m training.eval --seeds 0 1
    python -m training.eval --split dev --out eval_dev.json
    python -m training.eval --compare baseline.json eval_dev.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np

from doom_env import make_env
from doom_env.limits import agent_budget_s
from doom_env.spaces import FRAME_SKIP
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


def resolve_maps(split: str = "dev", map_args: list[str] | None = None) -> list[MapSpec]:
    """Resolve maps from CLI --map flags, Role 3's `training.maps`, or Stage 1 fallback."""
    if map_args:
        return [parse_map_arg(m, split=split) for m in map_args]

    try:
        maps_mod = importlib.import_module("training.maps")
    except ModuleNotFoundError:
        return list(DEFAULT_FALLBACK_MAPS)

    if hasattr(maps_mod, "get_maps"):
        specs = list(maps_mod.get_maps(split=split))
        if specs:
            return specs

    attr_map = {
        "dev": ("DEV_MAPS", "ALL_MAPS"),
        "final": ("FINAL_MAPS", "ALL_MAPS"),
        "train": ("TRAIN_MAPS", "ALL_MAPS"),
        "all": ("ALL_MAPS", "DEV_MAPS"),
    }
    for attr in attr_map.get(split, ("DEV_MAPS", "ALL_MAPS")):
        if hasattr(maps_mod, attr):
            specs = [s for s in getattr(maps_mod, attr) if split == "all" or s.split == split]
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
            "mean_unique_cells": 0.0,
            "total_agent_time_s": 0.0,
            "max_step_ms": 0.0,
            "all_within_budget": True,
        }
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
        "mean_unique_cells": round(sum(e.unique_cells_visited for e in episodes) / n, 2),
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


def evaluate_suite(
    specs: list[MapSpec],
    *,
    agent_factory: Callable[[Any, Any, str], Any] | None = None,
    agent_spec: str = DEFAULT_AGENT_SPEC,
    device: str = "cpu",
    seeds_override: tuple[int, ...] | None = None,
    render_buffers: bool = False,
    record_dir: Path | None = None,
) -> dict[str, Any]:
    """Instantiate an agent once and evaluate all `(spec, seed)` episodes."""
    if not specs:
        raise ValueError("specs must not be empty")
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
    return {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "weights_sha256": _weights_sha256(),
            "agent_spec": agent_spec,
            "device": device,
            "render_buffers": render_buffers,
            "init_time_s": init_time_s,
            "wall_time_s": wall_time_s,
            "maps_manifest": [asdict(s) for s in specs],
        },
        "summary": summary,
        "episodes": [asdict(e) for e in episodes],
    }


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

    b_clean = b["errors"] == 0 and b.get("all_within_budget", True)
    c_clean = c["errors"] == 0 and c.get("all_within_budget", True)
    if not c_clean:
        return {
            "verdict": "keep_incumbent",
            "reason": "Challenger failed clean-run / compute-budget gate.",
            "deltas": deltas,
        }
    if not b_clean and c_clean:
        return {
            "verdict": "promote",
            "reason": "Challenger passed clean-run gate where baseline failed.",
            "deltas": deltas,
        }

    for key in ("tier34_exits", "locked_doors_unlocked", "tier12_exits", "kills"):
        diff = deltas[key]
        if diff > 0:
            return {"verdict": "promote", "reason": f"Challenger improved {key} (+{diff}).", "deltas": deltas}
        if diff < 0:
            return {"verdict": "keep_incumbent", "reason": f"Challenger regressed {key} ({diff}).", "deltas": deltas}

    return {"verdict": "tie", "reason": "No difference across selection metrics.", "deltas": deltas}


def format_report_markdown(report: dict[str, Any]) -> str:
    """Format an evaluation report into a readable Markdown summary."""
    meta = report["metadata"]
    overall = report["summary"]["overall"]
    lines = [
        f"## Evaluation Summary ({meta.get('git_commit') or 'uncommitted'})",
        f"- Agent: `{meta['agent_spec']}` | Device: `{meta['device']}` | Init: `{meta['init_time_s']}s` | Wall: `{meta['wall_time_s']}s`",
        f"- Episodes: `{overall['episodes']}` | Exits: `{overall['exits']}` | Deaths: `{overall['deaths']}` | Timeouts: `{overall['timeouts']}` | Errors: `{overall['errors']}`",
        f"- Progression: Keys `{overall['keys_picked_up']}` | Doors `{overall['doors_opened']}` | Unlocks `{overall['locked_doors_unlocked']}` | Kills `{overall['kills']}` | Mean cells `{overall['mean_unique_cells']}`",
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
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Role 4 independent DoomAgent evaluator.")
    parser.add_argument("--split", default="dev", choices=("train", "dev", "final", "all"))
    parser.add_argument("--map", action="append", default=None, help="wad[:map[:family[:tier[:ticks]]]]")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Override seed list for all maps")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--agent", default=DEFAULT_AGENT_SPEC, help="Agent entry point module:Class")
    parser.add_argument("--render-buffers", action="store_true", help="Render RGB/depth buffers")
    parser.add_argument("--record-dir", type=Path, default=None, help="Directory to save ViZDoom .lmp replays")
    parser.add_argument("--out", type=Path, default=None, help="Write JSON report to this path")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE_JSON", "CHALLENGER_JSON"), default=None)
    args = parser.parse_args(argv)

    if args.compare:
        baseline = json.loads(Path(args.compare[0]).read_text(encoding="utf-8"))
        challenger = json.loads(Path(args.compare[1]).read_text(encoding="utf-8"))
        cmp_result = compare_reports(baseline, challenger)
        print(json.dumps(cmp_result, indent=2))
        return 0

    specs = resolve_maps(split=args.split, map_args=args.map)
    seeds_override = tuple(args.seeds) if args.seeds is not None else None
    report = evaluate_suite(
        specs,
        agent_spec=args.agent,
        device=args.device,
        seeds_override=seeds_override,
        render_buffers=args.render_buffers,
        record_dir=args.record_dir,
    )
    print(format_report_markdown(report))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {args.out}")
    return 0 if report["summary"]["overall"]["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
