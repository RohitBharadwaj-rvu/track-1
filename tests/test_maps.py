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
    FINAL_OOD_MAPS,
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


def _walk_toward(env, tracker: PrivilegedEventTracker, tx: float, ty: float, max_steps: int = 200, *, attack: bool = False):
    """Test-only scripted walker (uses privileged coords ONLY inside pytest) to verify map playability."""
    events_seen = {"doors": 0, "unlocks": 0, "keys": 0, "exit": False, "done": False}
    last_pos = None
    stuck_ticks = 0
    for step_i in range(max_steps):
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
            act = FWD.copy()
            if attack and step_i % 2 == 0:
                act[S.BUTTONS.index("ATTACK")] = 1

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


def _clear_visible_enemies(env, tracker: PrivilegedEventTracker, max_steps: int = 80) -> bool:
    """Aim at visible enemies using obs['visible_entities'] and fire until none remain."""
    atk_idx = S.BUTTONS.index("ATTACK")
    obs, _, term, trunc, _ = env.step(NOOP)
    tracker.update(env.privileged_state())
    if term or trunc:
        return True
    scan = 0
    for _ in range(max_steps):
        ve = obs["visible_entities"]
        best = None
        for k in range(len(ve["mask"])):
            if ve["mask"][k] > 0 and int(ve["entity_type"][k]) in (1, 2, 3):
                dist = float(ve["features"][k][0])
                if best is None or dist < best[0]:
                    ang = math.degrees(math.atan2(float(ve["features"][k][1]), float(ve["features"][k][2])))
                    best = (dist, ang)
        if best is None:
            scan += 1
            if scan > 24:
                return False
            act = TURN_R
        else:
            scan = 0
            _, ang = best
            if abs(ang) > 16.0:
                act = TURN_L if ang > 0 else TURN_R
            else:
                act = (TURN_L if ang > 8.0 else (TURN_R if ang < -8.0 else NOOP)).copy()
                act[atk_idx] = 1
        obs, _, term, trunc, _ = env.step(act)
        tracker.update(env.privileged_state())
        if term or trunc:
            return True
    return False


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

    stress_hashes = {get_definition(s.wad).blueprint_hash() for s in get_maps("stress")}
    assert len(stress_hashes) == len(get_maps("stress")) == 20
    assert all(stress_hashes.isdisjoint(group) for group in split_hashes.values())

    ood_hashes = {get_definition(s.wad).blueprint_hash() for s in FINAL_OOD_MAPS}
    assert len(ood_hashes) == len(get_maps("final_ood")) == 8
    assert {s.tier for s in FINAL_OOD_MAPS} == {1, 2, 3, 4}
    assert all(ood_hashes.isdisjoint(group) for group in split_hashes.values())
    assert ood_hashes.isdisjoint(stress_hashes)

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
    """Verify on-the-fly procedural WAD generator produces engine-valid maps for all 5 families."""
    specs = generate_training_variations(tmp_path, count_per_family=1, base_seed=42)
    assert len(specs) == 5
    for spec in specs:
        res = verify_wad_in_engine(spec.wad, map_name=spec.map_name, episode_ticks=spec.episode_ticks)
        assert res.ok, f"Procedural map {spec.wad} failed engine verification: {res.errors}"


def test_procedural_generator_covers_verticality_polygons_switches_and_roster(tmp_path: Path):
    """Verify broadened procedural generation produces verticality, windows/grates, switches, polygons, and expanded roster."""
    from doom_env.mapinfo import parse_udmf
    from doom_env.wad import map_lumps, read_wad

    specs = generate_training_variations(tmp_path, count_per_family=8, base_seed=200)
    assert len(specs) == 40

    has_stairs_or_ledge = False
    has_non_axis = False
    has_exit_switch = False
    has_remote_door_switch = False
    has_skull_key = False
    has_window_or_grate = False
    has_new_monster = False
    has_new_weapon = False

    engine_checked_families: set[str] = set()
    for spec in specs:
        if spec.family not in engine_checked_families:
            res = verify_wad_in_engine(spec.wad, map_name=spec.map_name, episode_ticks=spec.episode_ticks, step_count=2)
            assert res.ok, f"Procedural map {spec.wad} failed engine verification: {res.errors}"
            engine_checked_families.add(spec.family)

        b = parse_udmf(map_lumps(read_wad(spec.wad), spec.map_name)["TEXTMAP"].decode("utf-8", "replace"))
        verts = [(float(v["x"]), float(v["y"])) for v in b.get("vertex", [])]
        secs = b.get("sector", [])
        if any(int(s.get("heightfloor", 0)) > 0 for s in secs):
            has_stairs_or_ledge = True
        if any(0 < int(s.get("heightceiling", 128)) < 56 for s in secs):
            has_window_or_grate = True
        for ld in b.get("linedef", []):
            v1, v2 = verts[int(ld["v1"])], verts[int(ld["v2"])]
            if abs(v1[0] - v2[0]) > 1e-3 and abs(v1[1] - v2[1]) > 1e-3:
                has_non_axis = True
            if int(ld.get("special", 0) or 0) == 243 and ld.get("playeruse"):
                has_exit_switch = True
            if int(ld.get("special", 0) or 0) == 12 and int(ld.get("arg0", 0) or 0) > 0:
                has_remote_door_switch = True
            if ld.get("twosided") and ld.get("blocking"):
                has_window_or_grate = True
        for th in b.get("thing", []):
            tp = int(th["type"])
            if tp in (38, 39, 40):
                has_skull_key = True
            if tp in (65, 3005, 69, 3006, 58):
                has_new_monster = True
            if tp in (82, 2003, 2004):
                has_new_weapon = True

    assert has_stairs_or_ledge, "Expected procedural maps with non-zero floor heights (stairs/ledges)"
    assert has_non_axis, "Expected procedural maps with diagonal/polygonal walls"
    assert has_exit_switch, "Expected procedural maps with wall-mounted exit switches"
    assert has_remote_door_switch, "Expected procedural maps with remote switch-triggered doors"
    assert has_skull_key, "Expected procedural maps with skull keys"
    assert has_window_or_grate, "Expected procedural maps with low windows or impassable grates"
    assert has_new_monster, "Expected procedural maps with expanded monster roster"
    assert has_new_weapon, "Expected procedural maps with expanded weapon roster"


@pytest.mark.parametrize(
    "name",
    [
        "exit_ahead",
        "exit_left",
        "exit_behind",
        "exit_switch",
        "key_open",
        "key_window",
        "junction_return",
        "stairs_ledge",
        "remote_door_switch",
        "grate_combat",
        "nukage_hazard",
        "dropoff_return",
        "open_arena_horde",
        "three_lock_citadel",
    ],
)
def test_stress_playthrough(name):
    spec = next(s for s in get_maps("stress") if Path(s.wad).stem == name)
    env = make_env(str(ROOT / spec.wad), spec.map_name, episode_ticks=spec.episode_ticks,
                   render_buffers=False)
    try:
        obs, _ = env.reset(seed=0)
        tracker = PrivilegedEventTracker()
        tracker.reset(env.privileged_state())
        if name.startswith("exit_"):
            _turn_toward(env, 0)
            if name == "exit_switch":
                for _ in range(60):
                    _, _, term, trunc, _ = env.step(FWD)
                    assert not (term or trunc)
                env.step(USE)
            else:
                _walk_toward(env, tracker, 432, 96)
        elif name.startswith("key_"):
            assert 9 in obs["visible_entities"]["entity_type"]
            if name == "key_window":
                assert _ray_ahead_type(obs) == 1
                for _ in range(30):
                    env.step(FWD)
                assert env.privileged_state()["x"] < 192
                for x, y in ((128, 240), (352, 240)):
                    _walk_toward(env, tracker, x, y)
            _walk_toward(env, tracker, 352, 80)
            obs, *_ = env.step(NOOP)
            assert obs["vitals"][11] == 1
            if name == "key_window":
                for x, y in ((352, 240), (96, 240)):
                    _walk_toward(env, tracker, x, y)
            _walk_toward(env, tracker, 80, 80)
            _walk_toward(env, tracker, -220, 80)
        elif name == "stairs_ledge":
            # Direct east path is blocked by 48-unit cliff (RAY_WALL) while YellowSkull is visible
            assert _ray_ahead_type(obs) == 1
            assert 9 in obs["visible_entities"]["entity_type"]
            for _ in range(30):
                env.step(FWD)
            assert env.privileged_state()["x"] < 192
            # Climb 3-step stairs north (z rises to 48), cross upper bridge, grab YellowSkull, unlock & exit
            for x, y in ((96, 352), (352, 352), (352, 80)):
                _walk_toward(env, tracker, x, y)
            obs, *_ = env.step(NOOP)
            assert env.privileged_state()["z"] >= 47.0
            assert obs["vitals"][12] == 1  # YellowSkull
            _walk_toward(env, tracker, 656, 80)
        elif name == "remote_door_switch":
            # Remote door emits RAY_DOOR but pressing USE directly on it cannot open it
            assert _ray_ahead_type(obs) == RAY_DOOR
            _walk_toward(env, tracker, 232, 80, max_steps=60)
            env.step(USE)
            for _ in range(12):
                env.step(FWD)
            assert env.privileged_state()["x"] < 256
            # Walk into north alcove, press USE on wall switch, then cross raised door to exit
            _walk_toward(env, tracker, 128, 296, max_steps=120)
            _turn_toward(env, 90)
            for _ in range(10):
                env.step(FWD)
            env.step(USE)
            for _ in range(10):
                env.step(NOOP)
            _walk_toward(env, tracker, 128, 80, max_steps=120)
            _walk_toward(env, tracker, 528, 80, max_steps=200)
        elif name == "grate_combat":
            # Impassable mid-grate emits RAY_WALL while enemies remain visible in LOS
            assert _ray_ahead_type(obs) == 1
            ve_types = set(obs["visible_entities"]["entity_type"][obs["visible_entities"]["mask"] > 0].tolist())
            assert ve_types & {2, 3}  # enemy_hitscan / enemy_projectile visible behind grate
            for _ in range(35):
                env.step(FWD)
            assert env.privileged_state()["x"] < 256
            # Flank through angled north corridor to exit
            for x, y in ((160, 336), (400, 336), (480, 128), (608, 128)):
                ev = _walk_toward(env, tracker, x, y, max_steps=160)
                if ev["done"]:
                    break
        elif name == "nukage_hazard":
            # Walk over RadSuit and across nukage pool directly to east exit
            _walk_toward(env, tracker, 144, 128, max_steps=80)
            _walk_toward(env, tracker, 672, 128, max_steps=220)
        elif name == "dropoff_return":
            # Climb south/east 16u stairs to +64u high ledge, grab RedCard, drop off 64u cliff back down
            for x, y in ((280, 32), (480, 32), (480, 128), (384, 128)):
                _walk_toward(env, tracker, x, y, max_steps=180, attack=True)
            assert env.privileged_state()["z"] >= 63.0
            _walk_toward(env, tracker, 160, 128, max_steps=140, attack=True)
            assert env.privileged_state()["z"] <= 1.0
            _walk_toward(env, tracker, -208, 128, max_steps=220, attack=True)
        elif name == "open_arena_horde":
            # Wide >1024u arena must produce RAY_NONE rays (type 0)
            assert 0 in set(obs["rays"]["hit_type"].tolist())
            for x, y in ((160, 128), (1216, 128), (1312, 640)):
                ev = _walk_toward(env, tracker, x, y, max_steps=280)
                if ev["done"]:
                    break
        elif name == "three_lock_citadel":
            # Grab SSG, clear West wing (RedCard), collect hub armor/shells,
            # clear East wing (PlasmaRifle + CellPack + BlueCard), heal at hub Medikit,
            # clear North wing (Revenant + YellowCard), then South sanctum (Baron + exit).
            steps = (
                ("walk", 384, 448, True),
                ("clear", 35),
                ("walk", 160, 320, True),
                ("clear", 25),
                ("walk", 96, 384, True),
                ("walk", 384, 384, False),
                ("walk", 448, 448, False),
                ("walk", 384, 320, False),
                ("walk", 384, 384, False),
                ("walk", 544, 384, False),
                ("clear", 45),
                ("walk", 640, 320, False),
                ("walk", 704, 320, False),
                ("clear", 45),
                ("walk", 736, 384, True),
                ("walk", 384, 384, False),
                ("walk", 320, 320, False),
                ("walk", 384, 384, False),
                ("walk", 384, 544, False),
                ("walk", 384, 624, False),
                ("clear", 90),
                ("walk", 320, 704, False),
                ("walk", 384, 736, True),
                ("walk", 384, 384, False),
                ("walk", 384, 224, False),
                ("clear", 140),
                ("walk", 384, 160, True),
                ("clear", 140),
                ("walk", 336, 128, True),
                ("walk", 384, 32, True),
            )
            for item in steps:
                if item[0] == "clear":
                    if _clear_visible_enemies(env, tracker, max_steps=item[1]):
                        break
                else:
                    _, x, y, atk = item
                    ev = _walk_toward(env, tracker, x, y, max_steps=320, attack=atk)
                    if ev["done"]:
                        break
        else:
            for x in range(240, 1681, 160):
                _walk_toward(env, tracker, x, 64)
            _walk_toward(env, tracker, 1680, 320)
            _walk_toward(env, tracker, 1680, 64)
            for x in range(1520, 79, -160):
                _walk_toward(env, tracker, x, 64)
            _walk_toward(env, tracker, -220, 64)
        assert env.privileged_state()["exit_reached"]
    finally:
        env.close()


def test_final_ood_suite_engine_verified():
    assert len(FINAL_OOD_MAPS) == 8
    for spec in FINAL_OOD_MAPS:
        res = verify_wad_in_engine(ROOT / spec.wad, map_name=spec.map_name, episode_ticks=spec.episode_ticks, step_count=2)
        assert res.ok, f"OOD map {spec.wad} failed engine check: {res.errors}"

@pytest.mark.parametrize(("archetype", "route"), [
    ("two_key_join", ("key_b", "key_a", "switch_1", "exit")),
    ("remote_key_relay", ("switch_1", "key_a", "switch_2", "exit")),
    ("revisit_wing", ("key_a", "switch_1", "key_b", "switch_2", "exit")),
])
def test_progression_wings_complete_in_engine(tmp_path, archetype, route):
    import random
    from training.wad_gen.progression import build_progression
    from training.wad_gen.udmf import build_polygon_wad

    sectors, things, delay = build_progression(archetype, random.Random(2026), 2026)
    wad = build_polygon_wad(tmp_path / "route.wad", sectors, things, door_close_delay=delay)
    env = make_env(str(wad), "MAP01", episode_ticks=4200, render_buffers=False)
    try:
        env.reset(seed=0)
        tracker = PrivilegedEventTracker()
        tracker.reset(env.privileged_state())
        for goal in route:
            target = next(s for s in sectors if s.name.rsplit("_", 1)[0] == goal)
            arm = target.name.split("_")[-1]
            path = []
            for s in sectors[1:]:
                if s.name.startswith(f"approach_{arm}_") or (
                    not s.name.startswith("approach_") and s.name.endswith(f"_{arm}")
                ):
                    path.append(np.mean(s.vertices, axis=0))
                if s == target:
                    break
            for pos in path:
                _walk_toward(env, tracker, *pos)
            if goal == "exit":
                break
            if target.door_switch:
                wall = np.mean(target.door_switch[:2], axis=0)
                _walk_toward(env, tracker, *wall)
                env.step(NOOP)
                env.step(USE)
            for pos in [*reversed(path[:-1]), (0, 0)]:
                _walk_toward(env, tracker, *pos)
        assert env.privileged_state()["exit_reached"]
    finally:
        env.close()
