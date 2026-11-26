"""Tests for Role 3: verified WAD generators, Train/Dev/Final splits, curriculum, and playthroughs."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from doom_env import make_env
from doom_env import spaces as S
from doom_env.mapinfo import load_mapinfo
from doom_env.sensors import RAY_DOOR, RAY_LOCKED
from training.interfaces import PrivilegedEventTracker
from training.maps import (
    ALL_MAPS,
    CONNECT_MAPS,
    CURRICULUM_STAGES,
    DEV_MAPS,
    FINAL_MAPS,
    FINAL_POLYGON_MAPS,
    MVP_GATE_COMBAT_MAPS,
    MVP_GATE_SIMPLE_MAPS,
    ROOT,
    TRAIN_MAPS,
    generate_training_variations,
    get_curriculum_maps,
    get_curriculum_weights,
    get_definition,
    get_maps,
    get_reserve_seeds,
)
from training.wad_gen.families import ALL_DEFINITIONS
from training.wad_gen.verify import (
    verify_grid_solvability,
    verify_polygon_solvability,
    verify_wad_in_engine,
)

FWD = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
USE = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], np.int8)
TURN_L = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], np.int8)
TURN_R = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0], np.int8)
NOOP = np.zeros(9, np.int8)


def _ray_ahead_type(obs: dict) -> int:
    return int(obs["rays"]["hit_type"][0])


def _turn_toward(env, target_deg: float, max_steps: int = 80) -> dict:
    """Test-only helper using privileged angle to face a cardinal heading cleanly."""
    obs = None
    for _ in range(max_steps):
        ang = float(env.privileged_state()["angle"]) % 360.0
        diff = (target_deg - ang + 180.0) % 360.0 - 180.0
        if abs(diff) <= 6.0:
            break
        act = TURN_L if diff > 0 else TURN_R
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            break
    if obs is None:
        obs, *_ = env.step(NOOP)
    return obs


def _walk_toward(env, tracker: PrivilegedEventTracker, tx: float, ty: float, max_steps: int = 200):
    """Test-only scripted walker (uses privileged coords ONLY inside pytest) to verify map playability."""
    events_seen = {"doors": 0, "unlocks": 0, "keys": 0, "exit": False, "done": False}
    last_pos = None
    stuck_ticks = 0
    for _ in range(max_steps):
        st = env.privileged_state()
        if st["exit_reached"] or st["dead"] or st["timeout"]:
            events_seen["done"] = True
            return events_seen
        dx, dy = tx - float(st["x"]), ty - float(st["y"])
        dist = math.hypot(dx, dy)
        if dist < 24.0:
            return events_seen
        desired = math.degrees(math.atan2(dy, dx)) % 360.0
        ang = float(st["angle"]) % 360.0
        diff = (desired - ang + 180.0) % 360.0 - 180.0

        cur_pos = (float(st["x"]), float(st["y"]))
        if last_pos is not None and math.hypot(cur_pos[0] - last_pos[0], cur_pos[1] - last_pos[1]) < 1.0:
            stuck_ticks += 1
        else:
            stuck_ticks = 0
        last_pos = cur_pos

        if abs(diff) > 10.0:
            act = TURN_L if diff > 0 else TURN_R
        elif stuck_ticks > 0 and stuck_ticks % 12 == 2:
            # Pulse USE once then wait for door to open (vanilla DOOM door rule)
            act = USE
        else:
            act = FWD

        _, _, term, trunc, _ = env.step(act)
        ev, _ = tracker.update(env.privileged_state())
        events_seen["doors"] += ev.doors_opened
        events_seen["unlocks"] += ev.locked_doors_unlocked
        events_seen["keys"] += ev.keys_picked_up
        if ev.exit_reached:
            events_seen["exit"] = True
        if term or trunc:
            events_seen["done"] = True
            return events_seen
    return events_seen


def test_registry_splits_and_zero_overlap():
    assert len(TRAIN_MAPS) == 20
    assert len(DEV_MAPS) == 10
    assert len(FINAL_MAPS) == 8
    assert len(ALL_MAPS) == 38

    # Stage 1 Connect pair & Stage 2 MVP gate subsets
    assert len(CONNECT_MAPS) == 2
    assert len(MVP_GATE_SIMPLE_MAPS) >= 2
    assert len(MVP_GATE_COMBAT_MAPS) >= 2
    assert len(FINAL_POLYGON_MAPS) == 4

    # Every split covers all 4 evaluation tiers
    for split_name in ("train", "dev", "final"):
        specs = get_maps(split=split_name)
        assert {s.tier for s in specs} == {1, 2, 3, 4}, f"Split {split_name} missing tiers"
        assert all(s.split == split_name for s in specs)

    # Zero overlap in WAD paths and structural layout hashes across splits
    split_wads = {sp: {s.wad for s in get_maps(split=sp)} for sp in ("train", "dev", "final")}
    assert split_wads["train"].isdisjoint(split_wads["dev"])
    assert split_wads["train"].isdisjoint(split_wads["final"])
    assert split_wads["dev"].isdisjoint(split_wads["final"])

    split_hashes = {
        sp: {get_definition(s.wad).blueprint_hash() for s in get_maps(split=sp)}
        for sp in ("train", "dev", "final")
    }
    assert split_hashes["train"].isdisjoint(split_hashes["dev"])
    assert split_hashes["train"].isdisjoint(split_hashes["final"])
    assert split_hashes["dev"].isdisjoint(split_hashes["final"])

    # Reserve seeds never overlap nominal seeds
    for spec in DEV_MAPS:
        res = get_reserve_seeds(spec, count=3)
        assert len(res) == 3
        assert set(res).isdisjoint(set(spec.seeds))

    # Curriculum stages return valid maps + normalized probability weights
    for stage in CURRICULUM_STAGES:
        c_maps = get_curriculum_maps(stage)
        c_weights = get_curriculum_weights(stage)
        assert len(c_maps) == len(c_weights) >= 2
        assert sum(c_weights) == pytest.approx(1.0, abs=1e-6)
        assert all(w > 0.0 for w in c_weights)


def test_all_registered_wads_pass_static_and_mapinfo_checks():
    total_bytes = 0
    for defn in ALL_DEFINITIONS:
        wad_path = ROOT / defn.spec.wad
        assert wad_path.exists(), f"Missing generated WAD: {wad_path}"
        size = wad_path.stat().st_size
        total_bytes += size
        assert size < 256_000, f"Unexpectedly large WAD {wad_path.name}: {size} bytes"

        if defn.construction == "grid":
            assert defn.grid is not None
            res = verify_grid_solvability(defn.grid)
            assert res.ok, f"{defn.spec.wad} failed static solvability: {res.errors}"
        else:
            ok, errs = verify_polygon_solvability(defn.poly_sectors, defn.poly_things)
            assert ok, f"{defn.spec.wad} failed polygon solvability: {errs}"

        info = load_mapinfo(wad_path, defn.spec.map_name)
        assert info.num_sectors < 2048
        unlocked = sum(1 for lock in info.door_sectors.values() if lock == 0)
        locked = sum(1 for lock in info.door_sectors.values() if lock > 0)
        assert unlocked >= defn.min_unlocked_doors, f"{defn.spec.wad}: unlocked {unlocked} < {defn.min_unlocked_doors}"
        assert locked >= defn.min_locked_doors, f"{defn.spec.wad}: locked {locked} < {defn.min_locked_doors}"

    # Entire maps/ suite stays well under 1 MB (budget is 1 GB)
    assert total_bytes < 1_000_000


def test_connect_door_exit_scripted_playthrough():
    """Stage 1 Connect check: door/exit map emits RAY_DOOR, opens door, reaches exit."""
    spec = CONNECT_MAPS[0]
    env = make_env(ROOT / spec.wad, spec.map_name, episode_ticks=spec.episode_ticks, render_buffers=False)
    try:
        obs, _ = env.reset(seed=0)
        assert _ray_ahead_type(obs) == RAY_DOOR
        tracker = PrivilegedEventTracker()
        tracker.reset(env.privileged_state())

        # Straight east to exit cell (col 8, row 2 -> x=544, y=160)
        ev = _walk_toward(env, tracker, 544.0, 160.0, max_steps=260)
        assert ev["doors"] >= 1
        assert ev["unlocks"] == 0
        assert ev["exit"] is True
    finally:
        env.close()


def test_connect_key_lock_exit_scripted_playthrough():
    """Stage 1 Connect check: key/lock map emits RAY_LOCKED, picks up Red key, unlocks Red door, exits."""
    spec = CONNECT_MAPS[1]
    env = make_env(ROOT / spec.wad, spec.map_name, episode_ticks=spec.episode_ticks, render_buffers=False)
    try:
        obs, _ = env.reset(seed=0)
        assert _ray_ahead_type(obs) == RAY_LOCKED
        assert obs["vitals"][S.VITALS_INDEX["key_red_card"]] == 0.0

        tracker = PrivilegedEventTracker()
        tracker.reset(env.privileged_state())

        # Step 1: Walk east to hub center (x=224, y=160), then north into alcove for RedCard (col 5 -> x=352? Let's walk to (256, 352))
        # Grid rows (height=7, cell=64):
        # row 1: ###..r..### -> r is at col 5 -> x=(5.5)*64=352, y=(7-1-1+0.5)*64=352
        # row 4: #.P..R..E.# -> P at col 2 (x=160, y=160), R at col 5 (x=352, y=160), E at col 8 (x=544, y=160)
        ev1 = _walk_toward(env, tracker, 256.0, 160.0, max_steps=100)
        ev2 = _walk_toward(env, tracker, 256.0, 352.0, max_steps=120)
        ev3 = _walk_toward(env, tracker, 352.0, 352.0, max_steps=120)
        # Return back down to hub and through Red locked door to exit
        ev4 = _walk_toward(env, tracker, 256.0, 352.0, max_steps=120)
        ev5 = _walk_toward(env, tracker, 256.0, 160.0, max_steps=120)
        ev6 = _walk_toward(env, tracker, 544.0, 160.0, max_steps=260)

        total_keys = sum(e["keys"] for e in (ev1, ev2, ev3, ev4, ev5, ev6))
        total_unlocks = sum(e["unlocks"] for e in (ev1, ev2, ev3, ev4, ev5, ev6))
        exited = any(e["exit"] for e in (ev1, ev2, ev3, ev4, ev5, ev6))

        assert total_keys == 1
        assert total_unlocks == 1
        assert exited is True
    finally:
        env.close()


def test_final_polygon_maps_engine_verified():
    """Verify the distinct-construction polygonal Final family loads with exact WAD geometry."""
    for spec in FINAL_POLYGON_MAPS:
        defn = get_definition(spec.wad)
        res = verify_wad_in_engine(
            ROOT / spec.wad,
            map_name=spec.map_name,
            episode_ticks=spec.episode_ticks,
            min_unlocked_doors=defn.min_unlocked_doors,
            min_locked_doors=defn.min_locked_doors,
        )
        assert res.ok, f"Polygon final map {spec.wad} failed engine check: {res.errors}"
        assert res.geometry_source == "wad"

    # Also verify a real playthrough through a 45-degree diagonal door to the diagonal exit
    env = make_env(
        ROOT / "maps/final/final_poly_hex_hub_explore.wad",
        "MAP01",
        episode_ticks=2100,
        render_buffers=False,
    )
    try:
        env.reset(seed=0)
        tracker = PrivilegedEventTracker()
        tracker.reset(env.privileged_state())
        ev1 = _walk_toward(env, tracker, 0.0, 0.0, max_steps=160)
        ev2 = _walk_toward(env, tracker, 350.0, 350.0, max_steps=300)
        assert ev1["doors"] + ev2["doors"] >= 1
        assert ev1["exit"] or ev2["exit"]
    finally:
        env.close()


def test_procedural_generator_all_families_engine_valid(tmp_path: Path):
    """Verify on-the-fly procedural WAD generator produces engine-valid maps for all 4 families."""
    specs = generate_training_variations(tmp_path, count_per_family=1, base_seed=42)
    assert len(specs) == 4
    for spec in specs:
        res = verify_wad_in_engine(spec.wad, map_name=spec.map_name, episode_ticks=spec.episode_ticks)
        assert res.ok, f"Procedural map {spec.wad} failed engine verification: {res.errors}"
