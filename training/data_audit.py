"""Exhaustive 12-metric data quality & D4-invariant leakage audit (no torch dependency)."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Sequence

from doom_env.mapinfo import load_mapinfo, parse_udmf
from doom_env.sensors import MAX_STEP_UP, PLAYER_HEIGHT
from doom_env.wad import map_lumps, read_wad
from doom_env.wrapper import IWAD
from training.curriculum import CurriculumConfig, CurriculumScheduler
from training.eval import EpisodeMetrics, resolve_maps, summarize_episodes
from training.interfaces import MapSpec
from training.maps import (
    ALL_MAPS,
    DEV_MAPS,
    FINAL_MAPS,
    FINAL_OOD_MAPS,
    STRESS_MAPS,
    TRAIN_MAPS,
    generate_training_variations,
    get_definition,
)
from training.pbrs import GeodesicPBRS, _REMOTE_DOOR_LOCK, _build_nav_graph
from training.rollout import SyncVecDoomEnv

ROOT = Path(__file__).resolve().parent.parent


def _read_train_config_defaults() -> dict[str, Any]:
    """Parse `TrainConfig` literal defaults via AST so data_audit stays torch-free."""
    tree = ast.parse((ROOT / "training" / "train.py").read_text(encoding="utf-8"))
    defaults: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "TrainConfig":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                    try:
                        defaults[stmt.target.id] = ast.literal_eval(stmt.value)
                    except (ValueError, SyntaxError):
                        pass
    return defaults

_ALL_MONSTER_TYPES = frozenset({3004, 9, 65, 3001, 3002, 58, 3006, 3005, 69, 3003, 68, 67, 66})
_ALL_WEAPON_TYPES = frozenset({2001, 2002, 82, 2003, 2004, 2005})
_ALL_PICKUP_TYPES = frozenset({2011, 2012, 2018, 2019, 2013, 2024, 2025, 2048, 2049, 2046, 2047, 17, 2035})
_SKULL_KEY_TYPES = frozenset({38, 39, 40})
_ALL_KEY_TYPES = frozenset({5, 6, 13, 38, 39, 40})
_DAMAGING_SPECIALS = frozenset({69, 71, 75, 80, 105, 115, 196})


def canonical_d4_vertex_signature(wad_path: str | Path, map_name: str = "MAP01") -> tuple[tuple[int, int], ...]:
    """Compute a D4-invariant normalized vertex signature for geometric shape leakage checks."""
    lumps = map_lumps(read_wad(str(wad_path)), map_name)
    if "TEXTMAP" not in lumps:
        return ()
    blocks = parse_udmf(lumps["TEXTMAP"].decode("utf-8", "replace"))
    raw = [(float(v["x"]), float(v["y"])) for v in blocks.get("vertex", [])]
    if not raw:
        return ()

    candidates: list[tuple[tuple[int, int], ...]] = []
    for flip in (False, True):
        pts = [(-x if flip else x, y) for x, y in raw]
        for rot in range(4):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            span_x = max(1e-6, max_x - min_x)
            span_y = max(1e-6, max_y - min_y)
            norm = tuple(sorted((round((x - min_x) / span_x * 1000), round((y - min_y) / span_y * 1000)) for x, y in pts))
            candidates.append(norm)
            pts = [(-y, x) for x, y in pts]
    return min(candidates)


def extract_map_features(wad_path: str | Path, map_name: str = "MAP01") -> dict[str, Any]:
    """Inspect static UDMF geometry, mechanics, entities, and bearing from spawn to exit."""
    resolved = Path(wad_path) if Path(wad_path).exists() else (ROOT / wad_path)
    lumps = map_lumps(read_wad(str(resolved)), map_name)
    text = lumps["TEXTMAP"].decode("utf-8", "replace")
    b = parse_udmf(text)
    verts = [(float(v["x"]), float(v["y"])) for v in b.get("vertex", [])]
    sides = b.get("sidedef", [])
    secs = b.get("sector", [])
    linedefs = b.get("linedef", [])
    things = b.get("thing", [])

    xs = [p[0] for p in verts] or [0.0]
    ys = [p[1] for p in verts] or [0.0]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)

    non_axis = False
    has_exit_switch = False
    has_remote_door = False
    has_grate = False
    has_cliff_24 = False
    has_window_56 = False
    door_delays: set[int] = set()
    locks_seen: set[int] = set()
    exit_pts: list[tuple[float, float]] = []

    sec_hf = [int(s.get("heightfloor", 0) or 0) for s in secs]
    sec_hc = [int(s.get("heightceiling", 128) or 128) for s in secs]
    sec_sp = [int(s.get("special", 0) or 0) for s in secs]
    info = load_mapinfo(resolved, map_name)
    door_secs = set(info.door_sectors.keys())

    for ld in linedefs:
        v1 = verts[int(ld["v1"])]
        v2 = verts[int(ld["v2"])]
        if abs(v1[0] - v2[0]) > 1e-3 and abs(v1[1] - v2[1]) > 1e-3:
            non_axis = True
        sp = int(ld.get("special", 0) or 0)
        arg0 = int(ld.get("arg0", 0) or 0)
        arg2 = int(ld.get("arg2", 0) or 0)
        arg3 = int(ld.get("arg3", 0) or 0)
        if sp in (243, 244):
            exit_pts.append(((v1[0] + v2[0]) * 0.5, (v1[1] + v2[1]) * 0.5))
            if ld.get("playeruse"):
                has_exit_switch = True
        if sp in (11, 12) and arg0 > 0:
            has_remote_door = True
            door_delays.add(arg2)
        if sp == 12 and arg0 == 0:
            door_delays.add(arg2)
        if sp == 13 and arg3 > 0:
            locks_seen.add(arg3)
        if ld.get("twosided") and ld.get("blocking"):
            has_grate = True
        sf = ld.get("sidefront", -1)
        sb = ld.get("sideback", -1)
        if isinstance(sf, int) and isinstance(sb, int) and 0 <= sf < len(sides) and 0 <= sb < len(sides):
            sa = int(sides[sf]["sector"])
            sbb = int(sides[sb]["sector"])
            if sa not in door_secs and sbb not in door_secs:
                op = min(sec_hc[sa], sec_hc[sbb]) - max(sec_hf[sa], sec_hf[sbb])
                step = abs(sec_hf[sa] - sec_hf[sbb])
                if 0 < op < int(PLAYER_HEIGHT):
                    has_window_56 = True
                if step > int(MAX_STEP_UP):
                    has_cliff_24 = True

    floor_textures = {str(s.get("texturefloor", "")).upper() for s in secs}
    wall_textures = {str(sd.get("texturemiddle", "")).upper() for sd in sides if str(sd.get("texturemiddle", "-")) != "-"}
    has_floor_var = len(set(sec_hf)) > 1 or any(h != 0 for h in sec_hf)
    has_damaging = any(s in _DAMAGING_SPECIALS for s in sec_sp)

    player_xy: tuple[float, float, int] | None = None
    monsters: list[int] = []
    weapons: list[int] = []
    pickups: list[int] = []
    keys: list[int] = []
    skull_keys = False
    has_radsuit = False
    has_ammo = False

    for th in things:
        t = int(th["type"])
        if t == 1:
            player_xy = (float(th["x"]), float(th["y"]), int(th.get("angle", 0)))
        elif t in _ALL_MONSTER_TYPES:
            monsters.append(t)
        elif t in _ALL_WEAPON_TYPES:
            weapons.append(t)
        elif t in _ALL_KEY_TYPES:
            keys.append(t)
            if t in _SKULL_KEY_TYPES:
                skull_keys = True
        if t in _ALL_PICKUP_TYPES:
            pickups.append(t)
        if t == 2025:
            has_radsuit = True
        if t in (2048, 2049, 2046, 2047, 17):
            has_ammo = True

    bearing_bin = "ahead"
    if player_xy is not None and exit_pts:
        px, py, pang = player_xy
        # Average exit center
        ex = sum(p[0] for p in exit_pts) / len(exit_pts)
        ey = sum(p[1] for p in exit_pts) / len(exit_pts)
        edeg = math.degrees(math.atan2(ey - py, ex - px)) % 360.0
        rel = (edeg - pang + 180.0) % 360.0 - 180.0
        if -45.0 <= rel <= 45.0:
            bearing_bin = "ahead"
        elif 45.0 < rel <= 135.0:
            bearing_bin = "left"
        elif -135.0 <= rel < -45.0:
            bearing_bin = "right"
        else:
            bearing_bin = "behind"

    return {
        "sectors": len(secs),
        "span_max": round(max(span_x, span_y), 1),
        "wide_gt_1024": max(span_x, span_y) > 1024.0,
        "non_axis": non_axis,
        "floor_var": has_floor_var,
        "cliff_gt_24": has_cliff_24,
        "window_lt_56": has_window_56,
        "grate": has_grate,
        "damaging_floor": has_damaging,
        "radsuit": has_radsuit,
        "exit_switch": has_exit_switch,
        "remote_door": has_remote_door,
        "skull_key": skull_keys,
        "num_locks": len(locks_seen),
        "door_delays": tuple(sorted(door_delays)),
        "monsters": tuple(monsters),
        "weapons": tuple(weapons),
        "pickups": tuple(pickups),
        "has_ammo": has_ammo,
        "floor_textures": floor_textures,
        "wall_textures": wall_textures,
        "bearing_bin": bearing_bin,
    }


def _cap10(x: float) -> float:
    return round(max(0.0, min(10.0, float(x))), 2)


def run_data_audit(proc_count_per_family: int = 20, base_seed: int = 5000) -> dict[str, Any]:
    """Compute the exhaustive 12-metric data quality scorecard and D4 leakage audit."""
    with tempfile.TemporaryDirectory() as td:
        proc_specs = generate_training_variations(
            Path(td),
            count_per_family=proc_count_per_family,
            base_seed=base_seed,
        )
        proc_feats = [extract_map_features(s.wad, s.map_name) for s in proc_specs]
        proc_sigs = [canonical_d4_vertex_signature(s.wad, s.map_name) for s in proc_specs]

        stress_feats = [extract_map_features(s.wad, s.map_name) for s in STRESS_MAPS]
        final_ood_feats = [extract_map_features(s.wad, s.map_name) for s in FINAL_OOD_MAPS]

        # Check D4 leakage between PROC vs held-out/test sets
        held_out: list[tuple[str, MapSpec]] = (
            [("dev", s) for s in DEV_MAPS]
            + [("stress", s) for s in STRESS_MAPS]
            + [("final", s) for s in FINAL_MAPS]
            + [("final_ood", s) for s in FINAL_OOD_MAPS]
        )
        held_sig_map: dict[tuple[tuple[int, int], ...], list[str]] = {}
        for split_lbl, spec in held_out:
            sig = canonical_d4_vertex_signature(ROOT / spec.wad, spec.map_name)
            held_sig_map.setdefault(sig, []).append(f"{split_lbl}:{spec.wad}")

        d4_collisions: list[dict[str, Any]] = []
        for spec, sig in zip(proc_specs, proc_sigs):
            if sig in held_sig_map:
                d4_collisions.append({"proc_wad": Path(spec.wad).name, "matches": held_sig_map[sig]})

        # PBRS check across PROC + STRESS + FINAL_OOD + Freedoom MAP01..08
        pbrs_proc_ok = 0
        for spec in proc_specs:
            g = _build_nav_graph(spec.wad, spec.map_name)
            if g is not None and len(g.exit_cells) > 0:
                pbrs_proc_ok += 1

        pbrs_stress_ok = 0
        remote_sw_ok = False
        dropoff_asym_ok = False
        dmg_cost_ok = False
        for spec in STRESS_MAPS:
            w = ROOT / spec.wad
            g = _build_nav_graph(w, spec.map_name)
            if g is not None and len(g.exit_cells) > 0:
                pbrs_stress_ok += 1
                stem = Path(spec.wad).stem
                if stem == "remote_door_switch" and len(g.switch_waypoints) >= 1 and any(
                    lk == _REMOTE_DOOR_LOCK for lk, _ in g.locked_door_cells.values()
                ):
                    remote_sw_ok = True
                if stem == "dropoff_return":
                    # Edge asymmetry check: at least one directed edge u->v lacks reverse v->u
                    has_oneway = any(
                        not any(rev_v == u for rev_v, _, _ in g.adj[v])
                        for u, edges in enumerate(g.adj)
                        for v, _, _ in edges
                    )
                    dropoff_asym_ok = has_oneway
                if stem == "nukage_hazard":
                    costs = {round(c / g.cell_size, 1) for edges in g.adj for _, c, _ in edges}
                    dmg_cost_ok = any(c >= 2.9 for c in costs)

        pbrs_freedoom_ok = 0
        for spec in resolve_maps(suite="freedoom"):
            g = _build_nav_graph(IWAD, spec.map_name)
            if g is not None and len(g.exit_cells) > 0:
                pbrs_freedoom_ok += 1

    # 1. Geometric & topological diversity
    n_p = len(proc_feats)
    frac_non_axis = sum(1 for f in proc_feats if f["non_axis"]) / n_p
    frac_wide = sum(1 for f in proc_feats if f["wide_gt_1024"]) / n_p
    sector_counts = sorted({f["sectors"] for f in proc_feats})
    m1_geom = _cap10(
        4.0 * min(1.0, frac_non_axis / 0.30)
        + 3.0 * min(1.0, frac_wide / 0.15)
        + 3.0 * min(1.0, len(sector_counts) / 10.0)
    )

    # 2. Verticality & sight != move
    frac_floor = sum(1 for f in proc_feats if f["floor_var"]) / n_p
    frac_cliff = sum(1 for f in proc_feats if f["cliff_gt_24"]) / n_p
    frac_win_grate = sum(1 for f in proc_feats if (f["window_lt_56"] or f["grate"])) / n_p
    m2_vert = _cap10(
        3.5 * min(1.0, frac_floor / 0.45)
        + 3.5 * min(1.0, frac_cliff / 0.25)
        + 3.0 * min(1.0, frac_win_grate / 0.28)
    )

    # 3. Mechanical progression & hazards
    frac_dmg = sum(1 for f in proc_feats if f["damaging_floor"]) / n_p
    frac_ex_sw = sum(1 for f in proc_feats if f["exit_switch"]) / n_p
    frac_rem = sum(1 for f in proc_feats if f["remote_door"]) / n_p
    frac_skull = sum(1 for f in proc_feats if f["skull_key"]) / n_p
    frac_2plus_lock = sum(1 for f in proc_feats if f["num_locks"] >= 2) / n_p
    all_delays = {d for f in proc_feats for d in f["door_delays"]}
    m3_mech = _cap10(
        2.0 * min(1.0, frac_dmg / 0.18)
        + 1.8 * min(1.0, frac_ex_sw / 0.08)
        + 1.8 * min(1.0, frac_rem / 0.10)
        + 1.6 * min(1.0, frac_skull / 0.15)
        + 1.6 * min(1.0, frac_2plus_lock / 0.15)
        + 1.2 * min(1.0, len(all_delays) / 4.0)
    )

    # 4. Observation, texture & entity roster coverage
    seen_monsters = {m for f in proc_feats for m in f["monsters"]}
    seen_weapons = {w for f in proc_feats for w in f["weapons"]}
    seen_pickups = {p for f in proc_feats for p in f["pickups"]}
    m4_obs = _cap10(
        4.5 * (len(seen_monsters) / len(_ALL_MONSTER_TYPES))
        + 3.0 * (len(seen_weapons) / len(_ALL_WEAPON_TYPES))
        + 2.5 * min(1.0, len(seen_pickups) / 10.0)
    )

    # 5. Combat intensity & resource balance
    monster_counts = [len(f["monsters"]) for f in proc_feats]
    max_monsters = max(monster_counts)
    has_dense = sum(1 for c in monster_counts if c >= 7)
    has_zero = sum(1 for c in monster_counts if c == 0)
    frac_ammo = sum(1 for f in proc_feats if f["has_ammo"]) / n_p
    m5_combat = _cap10(
        4.0 * min(1.0, max_monsters / 8.0)
        + 3.0 * min(1.0, has_dense / max(1, n_p // 10))
        + 1.5 * (1.0 if has_zero >= 2 else has_zero / 2.0)
        + 1.5 * min(1.0, frac_ammo / 0.45)
    )

    # 6. Ego bearing symmetry
    proc_bearings = Counter(f["bearing_bin"] for f in proc_feats)
    ood_bearings = Counter(f["bearing_bin"] for f in final_ood_feats)
    min_proc_share = min(proc_bearings.get(k, 0) for k in ("ahead", "left", "right", "behind")) / n_p
    min_ood_cnt = min(ood_bearings.get(k, 0) for k in ("ahead", "left", "right", "behind"))
    m6_sym = _cap10(5.5 * min(1.0, min_proc_share / 0.16) + 4.5 * min(1.0, min_ood_cnt / 2.0))

    # 7. Split integrity & D4 anti-leakage
    split_sets = [
        {s.wad for s in TRAIN_MAPS},
        {s.wad for s in DEV_MAPS},
        {s.wad for s in FINAL_MAPS},
        {s.wad for s in STRESS_MAPS},
        {s.wad for s in FINAL_OOD_MAPS},
    ]
    all_disjoint = sum(len(x) for x in split_sets) == len(set().union(*split_sets))
    hash_sets = [{get_definition(s.wad).blueprint_hash() for s in grp} for grp in (TRAIN_MAPS, DEV_MAPS, FINAL_MAPS, STRESS_MAPS, FINAL_OOD_MAPS)]
    all_hash_disjoint = sum(len(x) for x in hash_sets) == len(set().union(*hash_sets))
    m7_split = 10.0 if (all_disjoint and all_hash_disjoint and len(d4_collisions) == 0) else _cap10(10.0 - 2.0 * len(d4_collisions))

    # 8. Eval & test representativeness
    ood_tiers = Counter(s.tier for s in FINAL_OOD_MAPS)
    freedoom_cnt = len(resolve_maps(suite="freedoom"))
    stress_tier4 = any(s.tier == 4 for s in STRESS_MAPS)
    stress_dmg = any(f["damaging_floor"] for f in stress_feats)
    ood_dmg = any(f["damaging_floor"] for f in final_ood_feats)
    ood_3lock = any(f["num_locks"] >= 3 for f in final_ood_feats)
    m8_eval = _cap10(
        2.5 * (1.0 if len(STRESS_MAPS) >= 14 and stress_tier4 and stress_dmg else 0.5)
        + 3.5 * (1.0 if len(FINAL_OOD_MAPS) >= 8 and all(ood_tiers.get(t, 0) == 2 for t in (1, 2, 3, 4)) and ood_dmg and ood_3lock else 0.5)
        + 2.0 * min(1.0, freedoom_cnt / 8.0)
        + 2.0 * (1.0 if min_ood_cnt >= 2 else min_ood_cnt / 2.0)
    )

    # 9. Training pool freshness & LRU cache
    cfg_def = _read_train_config_defaults()
    has_swap_sync = hasattr(SyncVecDoomEnv, "set_map_specs")
    m9_fresh = _cap10(
        3.5 * min(1.0, float(cfg_def.get("procedural_per_family", 0)) / 4.0)
        + 3.0 * min(1.0, float(cfg_def.get("max_cached_envs", 0)) / 48.0)
        + 2.0 * (1.0 if int(cfg_def.get("procedural_refresh_updates", 0)) > 0 else 0.0)
        + 1.5 * (1.0 if has_swap_sync else 0.0)
    )

    # 10. Curriculum sampling parity & unseen exploration
    sched = CurriculumScheduler(CurriculumConfig(adaptive_weights=True, adaptive_floor=0.25))
    sched.phase = 2  # {1: 0.15, 2: 0.25, 3: 0.35, 4: 0.25}
    synthetic = [
        MapSpec("m_t1_a", tier=1),
        MapSpec("m_t1_b", tier=1),
        MapSpec("m_t1_c", tier=1),
        MapSpec("m_t2_a", tier=2),
        MapSpec("m_t3_a", tier=3),
        MapSpec("m_t4_a", tier=4),
    ]
    w_norm = sched.compute_map_weights(synthetic)
    t_sums = {t: sum(w_norm[i] for i, s in enumerate(synthetic) if s.tier == t) for t in (1, 2, 3, 4)}
    target_tb = sched.current_tier_weights()
    tier_err = max(abs(t_sums[t] - target_tb[t]) for t in (1, 2, 3, 4))
    sched.observe_episodes([EpisodeMetrics("m_t1_a", "MAP01", "combat", 1, 0, 10, 40, "exit")])
    w_after = sched.compute_map_weights(synthetic)
    unseen_bonus_ok = w_after[1] > w_after[0]
    m10_curr = 10.0 if (tier_err < 1e-5 and unseen_bonus_ok) else 6.0

    # 11. PBRS geometric fidelity
    m11_pbrs = _cap10(
        3.0 * (pbrs_proc_ok / n_p)
        + 2.0 * (pbrs_stress_ok / len(STRESS_MAPS))
        + 2.0 * (pbrs_freedoom_ok / freedoom_cnt)
        + 1.0 * (1.0 if remote_sw_ok else 0.0)
        + 1.0 * (1.0 if dropoff_asym_ok else 0.0)
        + 1.0 * (1.0 if dmg_cost_ok else 0.0)
    )

    # 12. Evaluation statistical rigor & gate
    dummy_eps = [
        EpisodeMetrics("a.wad", "MAP01", "combat", 1, 0, 10, 40, "exit"),
        EpisodeMetrics("b.wad", "MAP01", "gauntlet", 4, 0, 10, 40, "timeout"),
    ]
    summ = summarize_episodes(dummy_eps)["overall"]
    has_se = "exit_rate_se" in summ and abs(summ["exit_rate_se"] - 0.3536) < 0.01
    has_proxy = "proxy_leaderboard_score" in summ and abs(summ["proxy_leaderboard_score"] - 0.25) < 0.01
    m12_eval_rigor = 10.0 if (has_se and has_proxy and freedoom_cnt >= 8 and len(FINAL_OOD_MAPS) == 8) else 7.0

    scores = {
        "geometric_topo": m1_geom,
        "verticality_sight": m2_vert,
        "mechanics_depth": m3_mech,
        "obs_entity_coverage": m4_obs,
        "combat_resource": m5_combat,
        "ego_symmetry": m6_sym,
        "split_integrity": m7_split,
        "eval_test_representativeness": m8_eval,
        "training_pool_freshness": m9_fresh,
        "curriculum_sampling": m10_curr,
        "pbrs_fidelity": m11_pbrs,
        "eval_rigor_gate": m12_eval_rigor,
    }
    overall = round(sum(scores.values()) / len(scores), 2)

    return {
        "overall_score": overall,
        "scores": scores,
        "stats": {
            "proc_count": n_p,
            "frac_non_axis": round(frac_non_axis, 3),
            "frac_wide_gt_1024": round(frac_wide, 3),
            "frac_floor_var": round(frac_floor, 3),
            "frac_cliff_gt_24": round(frac_cliff, 3),
            "frac_window_or_grate": round(frac_win_grate, 3),
            "frac_damaging_floor": round(frac_dmg, 3),
            "frac_exit_switch": round(frac_ex_sw, 3),
            "frac_remote_door": round(frac_rem, 3),
            "frac_skull_key": round(frac_skull, 3),
            "frac_multi_lock": round(frac_2plus_lock, 3),
            "monster_types": sorted(seen_monsters),
            "weapon_types": sorted(seen_weapons),
            "pickup_types": sorted(seen_pickups),
            "max_monsters": max_monsters,
            "proc_bearings": dict(proc_bearings),
            "final_ood_bearings": dict(ood_bearings),
            "d4_collisions": d4_collisions,
            "pbrs_proc_ok": f"{pbrs_proc_ok}/{n_p}",
            "pbrs_stress_ok": f"{pbrs_stress_ok}/{len(STRESS_MAPS)}",
            "pbrs_freedoom_ok": f"{pbrs_freedoom_ok}/{freedoom_cnt}",
            "pbrs_remote_sw_ok": remote_sw_ok,
            "pbrs_dropoff_asym_ok": dropoff_asym_ok,
            "pbrs_dmg_cost_ok": dmg_cost_ok,
        },
    }


def main() -> int:
    report = run_data_audit()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
