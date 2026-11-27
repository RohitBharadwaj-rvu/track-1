"""Unit & integration tests for game-theoretic verifiable rewards, UED, stress probes, and Nash eval."""

from __future__ import annotations

import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from doom_env import make_env
from doom_env import spaces as S
from submission.agent import DoomAgent
from submission.policy import PolicyConfig, build_policy, save_checkpoint
from training.curriculum import CurriculumConfig, CurriculumScheduler
from training.eval import compare_reports
from training.game_theory import (
    AccelLevelArchive,
    PrimalDualConfig,
    PrimalDualController,
    compare_reports_nash,
    compute_cvar,
    compute_episode_rubric_score,
    compute_polytope_scores,
    solve_zero_sum_nash,
)
from training.interfaces import EpisodeMetrics, MapSpec
from training.maps import STRESS_MAPS
from training.pbrs import GeodesicPBRS, _build_nav_graph
from training.rewards import RewardConfig, RewardShaper
from training.wad_gen.dungeon import (
    DungeonGenome,
    generate_dungeon_from_genome,
    mutate_dungeon_genome,
)
from training.wad_gen.verify import verify_grid_solvability, verify_wad_in_engine

ROOT = Path(__file__).resolve().parent.parent

_NEW_STRESS_STEMS = (
    "aliased_loop_maze",
    "decoy_lock_order",
    "scarce_resource_fork",
    "cover_peek_crossfire",
    "nukage_key_return",
    "switch_dropoff_ambush",
)


def _priv(
    *,
    tic: int = 0,
    episode_ticks: int = 1000,
    x: float = 0.0,
    y: float = 0.0,
    health: float = 100.0,
    armor: float = 0.0,
    kills: int = 0,
    damage_taken: float = 0.0,
    keys: list[int] | None = None,
    sector_ceiling: list[float] | None = None,
    door_lock: list[int] | None = None,
    exit_reached: bool = False,
    dead: bool = False,
    timeout: bool = False,
) -> dict:
    return {
        "tic": tic,
        "episode_ticks": episode_ticks,
        "x": x,
        "y": y,
        "z": 0.0,
        "angle": 0.0,
        "health": health,
        "armor": armor,
        "ammo_total": 50.0,
        "kills": kills,
        "items": 0,
        "damage_taken": damage_taken,
        "keys": np.array(keys if keys is not None else [0, 0, 0, 0, 0, 0], dtype=np.float32),
        "dead": dead,
        "timeout": timeout,
        "exit_reached": exit_reached,
        "finished": exit_reached or dead or timeout,
        "sector_floor": np.zeros(8, dtype=np.float64),
        "sector_ceiling": np.array(
            sector_ceiling if sector_ceiling is not None else [0.0] * 8,
            dtype=np.float64,
        ),
        "door_lock": np.array(door_lock if door_lock is not None else [0] * 8, dtype=np.int32),
    }


def test_pbrs_default_gamma_and_remote_switch_anti_farming():
    pbrs = GeodesicPBRS("maps/stress/remote_door_switch.wad", "MAP01", scale=3.0)
    assert pbrs.gamma == pytest.approx(1.0)
    assert pbrs.graph is not None and len(pbrs.graph.switch_waypoints) >= 1

    env = make_env("maps/stress/remote_door_switch.wad", "MAP01", episode_ticks=600, render_buffers=False)
    try:
        env.reset(seed=0)
        init_priv = env.privileged_state()
        pbrs.reset(init_priv)
        assert pbrs._prev_stage_key is not None and pbrs._prev_stage_key[0] == "switch"
        sw_sid = pbrs._prev_stage_key[1]

        # Death returns 0.0 (no early-suicide bonus/loophole)
        dead_priv = dict(init_priv)
        dead_priv["dead"] = True
        assert pbrs.step(dead_priv) == pytest.approx(0.0)
        pbrs.reset(init_priv)

        # Reverse-Dijkstra must match forward Dijkstra identically on dropoff_return
        p_drop = GeodesicPBRS("maps/stress/dropoff_return.wad", "MAP01", scale=3.0)
        assert p_drop.graph is not None and p_drop.graph.player_start is not None
        gx = min(p_drop.graph.nx - 1, max(0, int((p_drop.graph.player_start[0] - p_drop.graph.min_x) // p_drop.graph.cell_size)))
        gy = min(p_drop.graph.ny - 1, max(0, int((p_drop.graph.player_start[1] - p_drop.graph.min_y) // p_drop.graph.cell_size)))
        c0 = gy * p_drop.graph.nx + gx
        fwd_d = p_drop._dijkstra(c0, set())
        best_fwd_key = min(float(fwd_d[kc]) for kc, _ in p_drop.graph.key_spawns)
        rev_d = p_drop._rev_dijkstra(("test_key",), [kc for kc, _ in p_drop.graph.key_spawns], set())
        assert float(rev_d[c0]) == pytest.approx(best_fwd_key)

        # Simulate opening remote door -> enters exit stage & records switch activation
        opened = dict(init_priv)
        ceil_open = np.array(init_priv["sector_ceiling"], dtype=np.float64, copy=True)
        ceil_open[sw_sid] += 64.0
        opened["sector_ceiling"] = ceil_open
        assert pbrs.step(opened) == pytest.approx(0.0)
        assert pbrs._prev_stage_key == ("exit",)
        assert sw_sid in pbrs._activated_switches

        # Simulate timed remote door re-closing -> must NEVER yield positive repeat switch shaping
        reclosed = dict(init_priv)
        reclosed["x"] = float(init_priv["x"]) + 64.0
        assert pbrs.step(reclosed) == pytest.approx(0.0)
        assert pbrs._prev_stage_key is not None and pbrs._prev_stage_key[0] == "switch"
        reclosed_closer = dict(reclosed)
        reclosed_closer["y"] = float(init_priv["y"]) + 128.0
        assert pbrs.step(reclosed_closer) == pytest.approx(0.0)
    finally:
        env.close()


def test_reward_dominance_cap_and_champion_anchored_delta():
    cfg = RewardConfig()
    assert cfg.pbrs_gamma == pytest.approx(1.0)

    # 1. Opening 5 doors must be capped at max_door_open_bonus (4.0)
    shaper = RewardShaper(cfg)
    shaper.reset(_priv(sector_ceiling=[0.0] * 8, door_lock=[0] * 8))
    _, _, comp = shaper.step(
        _priv(tic=8, sector_ceiling=[64.0, 64.0, 64.0, 64.0, 64.0, 0.0, 0.0, 0.0], door_lock=[0] * 8)
    )
    assert comp["door"] == pytest.approx(cfg.max_door_open_bonus)

    # 2. Cumulative bounded explore (250 cells) + max_door_open_bonus + pickup must stay bounded vs exit progression
    shaper_b = RewardShaper(cfg)
    shaper_b.reset(_priv(tic=0, x=0.0))
    expl_sum = 0.0
    for i in range(1, 260):
        _, _, c = shaper_b.step(_priv(tic=i * 4, x=float(i * 64)))
        expl_sum += c["explore"]
    assert 0.0 < expl_sum < 8.5

    # 3. Champion-anchored delta gives positive bonus for beating incumbent, negative penalty for regressing
    shaper_win = RewardShaper(cfg)
    shaper_win.reset(_priv(tic=0), champion_baseline={"rubric": 0.30, "exit_rate": 0.0}, tier=3)
    _, _, c_beat = shaper_win.step(_priv(tic=200, episode_ticks=1000, exit_reached=True))
    assert c_beat["champion_delta"] > 0.0
    assert shaper_win.last_regressed_vs_champion is False

    shaper_lose = RewardShaper(cfg)
    shaper_lose.reset(_priv(tic=0), champion_baseline={"rubric": 0.85, "exit_rate": 1.0}, tier=3)
    _, _, c_reg = shaper_lose.step(_priv(tic=200, episode_ticks=1000, dead=True, health=0.0))
    assert c_reg["champion_delta"] < 0.0
    assert shaper_lose.last_regressed_vs_champion is True


def test_primal_dual_cmdp_tightens_on_violations_and_relaxes_when_clean():
    ctrl = PrimalDualController(PrimalDualConfig(lr=0.5, window_size=8))
    base = RewardConfig()

    bad_eps = [
        EpisodeMetrics(
            wad="m.wad",
            map_name="MAP01",
            family="combat",
            tier=3,
            seed=i,
            steps=50,
            ticks=200,
            outcome="death",
            keys_picked_up=1,
            damage_taken=120.0,
            extra={"early_death": True, "died_after_pickup": True, "use_waste_rate": 0.4},
        )
        for i in range(8)
    ]
    tight_cfg, metrics_bad = ctrl.observe_episodes(bad_eps, base)
    assert metrics_bad["cmdp_lambda_damage"] > 0.0
    assert metrics_bad["cmdp_lambda_early_death"] > 0.0
    assert metrics_bad["cmdp_lambda_post_key_death"] > 0.0
    assert metrics_bad["cmdp_lambda_use_waste"] > 0.0
    assert tight_cfg.damage_penalty < base.damage_penalty
    assert tight_cfg.death_penalty < base.death_penalty
    assert tight_cfg.post_key_death_penalty < 0.0
    assert tight_cfg.use_waste_penalty < base.use_waste_penalty

    clean_eps = [
        EpisodeMetrics(
            wad="m.wad",
            map_name="MAP01",
            family="combat",
            tier=3,
            seed=100 + i,
            steps=100,
            ticks=900,
            outcome="exit",
            keys_picked_up=1,
            damage_taken=0.0,
            extra={"early_death": False, "died_after_pickup": False, "use_waste_rate": 0.0},
        )
        for i in range(16)
    ]
    for _ in range(12):
        relaxed_cfg, metrics_good = ctrl.observe_episodes(clean_eps, base)
    assert metrics_good["cmdp_lambda_damage"] == pytest.approx(0.0)
    assert metrics_good["cmdp_lambda_early_death"] == pytest.approx(0.0)
    assert relaxed_cfg.damage_penalty == pytest.approx(base.damage_penalty)


def test_curriculum_sfl_pvl_peaks_at_frontier_and_preserves_tier_budgets():
    specs = [
        MapSpec("m_zero.wad", family="key_return", tier=3, split="train"),
        MapSpec("m_front.wad", family="key_return", tier=3, split="train"),
        MapSpec("m_easy.wad", family="key_return", tier=3, split="train"),
        MapSpec("m_t1.wad", family="combat", tier=1, split="train"),
        MapSpec("m_t2.wad", family="multi_room", tier=2, split="train"),
        MapSpec("m_t4.wad", family="gauntlet", tier=4, split="train"),
    ]
    sched = CurriculumScheduler(
        CurriculumConfig(
            auto_curriculum=False,
            adaptive_weights=True,
            adaptive_mode="sfl_pvl",
            tier_weights={1: 0.2, 2: 0.2, 3: 0.4, 4: 0.2},
            adaptive_window=8,
        )
    )
    eps = (
        [EpisodeMetrics("m_zero.wad", "MAP01", "key_return", 3, i, outcome="timeout") for i in range(8)]
        + [
            EpisodeMetrics("m_front.wad", "MAP01", "key_return", 3, i, outcome="exit" if i % 2 == 0 else "timeout")
            for i in range(8)
        ]
        + [EpisodeMetrics("m_easy.wad", "MAP01", "key_return", 3, i, outcome="exit") for i in range(8)]
        + [EpisodeMetrics("m_t1.wad", "MAP01", "combat", 1, 0, outcome="exit")]
        + [EpisodeMetrics("m_t2.wad", "MAP01", "multi_room", 2, 0, outcome="exit")]
        + [EpisodeMetrics("m_t4.wad", "MAP01", "gauntlet", 4, 0, outcome="timeout")]
    )
    sched.observe_episodes(eps)
    w = sched.compute_map_weights(specs)
    # Within Tier 3: p=0.5 frontier (m_front) outranks flat p=0.0 (m_zero) and p=1.0 (m_easy)
    assert w[1] > w[0] > w[2]
    assert sum(w[:3]) == pytest.approx(0.4, abs=1e-5)

    # High PVL on m_zero lifts its priority when partial-progress signal appears
    sched.observe_pvl({"m_zero.wad": 2.5, "m_front.wad": 0.1, "m_easy.wad": 0.05})
    w_pvl = sched.compute_map_weights(specs)
    assert w_pvl[0] > w[0]

    # Champion regression boosts regressed map above equal peer
    sched.observe_episodes(
        [EpisodeMetrics("m_easy.wad", "MAP01", "key_return", 3, 99, outcome="timeout", extra={"regressed_vs_champion": True})]
    )
    scores = sched.get_map_scores(specs)
    assert scores["m_easy.wad"] > sched.config.adaptive_floor + 0.5


def test_accel_dungeon_mutations_and_archive_yield_100pct_valid_pbrs(tmp_path):
    rng = np.random.default_rng(42)
    initial_pairs = []
    for fam_idx, fam in enumerate(("door_exit", "combat", "multi_room", "key_return", "gauntlet")):
        g = DungeonGenome(family=fam, seed=7000 + fam_idx)
        spec = generate_dungeon_from_genome(g, tmp_path)
        initial_pairs.append((spec, g))
        m_g = mutate_dungeon_genome(g, rng)
        assert m_g.generation == g.generation + 1

    archive = AccelLevelArchive()
    archive.seed_pool(initial_pairs)
    archive.update_scores({s.wad: (1.8 if s.family in ("key_return", "gauntlet") else 0.4) for s, _ in initial_pairs})

    evolved = archive.evolve_pool(
        count_per_family=2,
        families=("door_exit", "combat", "multi_room", "key_return", "gauntlet"),
        out_dir=tmp_path,
        rng=rng,
        generation_seed=8000,
        genome_cls=DungeonGenome,
        generator_fn=generate_dungeon_from_genome,
        mutate_fn=mutate_dungeon_genome,
    )
    assert len(evolved) == 10
    for spec in evolved:
        nav = _build_nav_graph(spec.wad, spec.map_name)
        assert nav is not None and len(nav.exit_cells) >= 1
        pbrs = GeodesicPBRS(spec.wad, spec.map_name, scale=3.0)
        env = make_env(spec.wad, spec.map_name, episode_ticks=200, render_buffers=False)
        try:
            env.reset(seed=0)
            pbrs.reset(env.privileged_state())
            assert pbrs._prev_stage_key != ("none",), f"Unreachable initial PBRS stage on {spec.wad}"
        finally:
            env.close()


def test_new_adversarial_stress_probes_exist_engine_verified_and_pbrs_reachable():
    by_stem = {Path(s.wad).stem: s for s in STRESS_MAPS}
    assert len(STRESS_MAPS) >= 20
    for stem in _NEW_STRESS_STEMS:
        assert stem in by_stem, f"Missing new stress probe {stem}"
        spec = by_stem[stem]
        wad_path = ROOT / spec.wad
        assert wad_path.is_file(), f"Missing generated WAD file {wad_path}"
        res = verify_wad_in_engine(wad_path, map_name=spec.map_name, episode_ticks=spec.episode_ticks, step_count=2)
        assert res.ok, f"Engine check failed for {stem}: {res.errors}"
        pbrs = GeodesicPBRS(wad_path, spec.map_name, scale=3.0)
        assert pbrs.graph is not None and len(pbrs.graph.exit_cells) >= 1
        env = make_env(str(wad_path), spec.map_name, episode_ticks=200, render_buffers=False)
        try:
            env.reset(seed=0)
            pbrs.reset(env.privileged_state())
            assert pbrs._prev_stage_key != ("none",), f"No reachable initial PBRS stage on {stem}"
        finally:
            env.close()


def test_nash_averaging_redundancy_invariance_polytope_and_cvar():
    # Agent A solves 10 redundant easy maps (0.76) but fails Tier 4 frontier maps (~0.02);
    # Agent B solves both easy maps (0.76) AND hard Tier 4 frontier maps (~0.66).
    easy_eps_a = [
        {"wad": f"maps/easy_{k}.wad", "map_name": "MAP01", "tier": 1, "outcome": "exit", "ticks": 300, "episode_ticks": 2100, "final_health": 100.0, "unique_cells_visited": 60, "kills": 3}
        for k in range(10)
    ]
    easy_eps_b = [
        {"wad": f"maps/easy_{k}.wad", "map_name": "MAP01", "tier": 1, "outcome": "exit", "ticks": 300, "episode_ticks": 2100, "final_health": 100.0, "unique_cells_visited": 60, "kills": 3}
        for k in range(10)
    ]
    hard_eps_a = [
        {"wad": f"maps/hard_{k}.wad", "map_name": "MAP01", "tier": 4, "outcome": "timeout", "ticks": 4200, "episode_ticks": 4200, "final_health": 50.0, "unique_cells_visited": 10, "kills": 0}
        for k in range(2)
    ]
    hard_eps_b = [
        {"wad": f"maps/hard_{k}.wad", "map_name": "MAP01", "tier": 4, "outcome": "exit", "ticks": 3200, "episode_ticks": 4200, "final_health": 60.0, "keys_picked_up": 1, "locked_doors_unlocked": 1, "unique_cells_visited": 30, "kills": 1}
        for k in range(2)
    ]

    rep_a = {"episodes": easy_eps_a + hard_eps_a, "summary": {"overall": {"errors": 0, "all_within_budget": True, "tier34_exits": 0, "locked_doors_unlocked": 0, "tier12_exits": 10, "kills": 30, "keys_picked_up": 0, "mean_unique_cells": 51.7}}}
    rep_b = {"episodes": easy_eps_b + hard_eps_b, "summary": {"overall": {"errors": 0, "all_within_budget": True, "tier34_exits": 2, "locked_doors_unlocked": 2, "tier12_exits": 10, "kills": 32, "keys_picked_up": 2, "mean_unique_cells": 55.0}}}

    nash = compare_reports_nash([("agent_a", rep_a), ("agent_b", rep_b)])
    assert nash["agents"][0]["name"] == "agent_b"
    assert nash["agents"][0]["minimax_regret"] < nash["agents"][1]["minimax_regret"]
    frontier_names = [f["map"] for f in nash["frontier_maps"][:2]]
    assert any("hard_" in m for m in frontier_names)

    # Duplicating easy_0 10x or adding 10 dead-tie Tier-1 maps must leave Nash regret game value and winner invariant!
    rep_a_dup = {"episodes": rep_a["episodes"] + [{**easy_eps_a[0], "wad": f"maps/easy_clone_{j}.wad"} for j in range(10)]}
    rep_b_dup = {"episodes": rep_b["episodes"] + [{**easy_eps_b[0], "wad": f"maps/easy_clone_{j}.wad"} for j in range(10)]}
    nash_dup = compare_reports_nash([("agent_a", rep_a_dup), ("agent_b", rep_b_dup)])
    assert nash_dup["agents"][0]["name"] == "agent_b"
    assert nash_dup["game_value"] == pytest.approx(nash["game_value"], abs=0.02)

    dead_t1 = [
        {"wad": f"maps/dead_tie_{j}.wad", "map_name": "MAP01", "tier": 1, "outcome": "timeout", "ticks": 2100, "episode_ticks": 2100, "final_health": 0.0, "unique_cells_visited": 1, "kills": 0}
        for j in range(10)
    ]
    nash_dead = compare_reports_nash([("agent_a", {"episodes": rep_a["episodes"] + dead_t1}), ("agent_b", {"episodes": rep_b["episodes"] + dead_t1})])
    assert nash_dead["agents"][0]["name"] == "agent_b"
    assert nash_dead["minimax_regret_game_value"] == pytest.approx(nash["minimax_regret_game_value"], abs=0.02)
    assert all("dead_tie_" not in f["map"] for f in nash_dead["frontier_maps"][:2])

    # Polytope dominance & CVaR-25
    assert compute_cvar({"a": 0.1, "b": 0.9, "c": 0.9, "d": 0.9}, alpha=0.25) == pytest.approx(0.1)
    poly_a = compute_polytope_scores(rep_a["episodes"])
    poly_b = compute_polytope_scores(rep_b["episodes"])
    assert poly_b["min_corner"] > poly_a["max_corner"]
    cmp = compare_reports(rep_a, rep_b)
    assert cmp["game_theory"]["polytope_dominates"] is True
    assert cmp["game_theory"]["cvar_25_rubric_delta"] > 0.0

    # Zero-sum Nash solver sanity on matching pennies -> (0.5, 0.5)
    p_mp, q_mp, val_mp = solve_zero_sum_nash(np.array([[1.0, -1.0], [-1.0, 1.0]]), iters=2000)
    assert val_mp == pytest.approx(0.0, abs=0.05)
    assert p_mp[0] == pytest.approx(0.5, abs=0.05)
    assert q_mp[0] == pytest.approx(0.5, abs=0.05)


def test_eval_temperature_scaling_and_doom_agent_integration(tmp_path):
    torch.manual_seed(0)
    cfg = PolicyConfig(arch="telemetry_gru_v1", action_head="factorized", eval_temperature=0.05)
    policy = build_policy(cfg)
    policy.eval()

    logits = torch.tensor([[3.0, -2.0, -2.0, 0.5, -1.0, -1.0, 2.0, -2.0, -2.0, 3.0, -3.0, 2.5]])
    det_act, det_lp = policy.sample_from_logits(logits, deterministic=True)
    # Near-zero temperature must match argmax identically
    cold_act, cold_lp = policy.sample_from_logits(logits, deterministic=False, temperature=1e-4)
    torch.testing.assert_close(cold_act, det_act)
    torch.testing.assert_close(cold_lp, det_lp)

    # Save checkpoint & ensure DoomAgent respects config eval_temperature
    ckpt = tmp_path / "temp_policy.pt"
    save_checkpoint(ckpt, policy, cfg)
    prev = os.environ.get("DOOM_AGENT_WEIGHTS")
    os.environ["DOOM_AGENT_WEIGHTS"] = str(ckpt)
    try:
        agent = DoomAgent(S.observation_space(), S.action_space(), device="cpu")
        assert agent.temperature == pytest.approx(0.05)
    finally:
        if prev is None:
            os.environ.pop("DOOM_AGENT_WEIGHTS", None)
        else:
            os.environ["DOOM_AGENT_WEIGHTS"] = prev
