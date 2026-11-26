"""Parallel 5,000+ map high-variety & very-high-difficulty WAD corpus generator, auditor, and pool sampler.

Generates verified unique training WADs into a gitignored directory (default:
`training/runs/corpus_5k/`), enforces zero D4 geometric leakage against all held-out
suites, verifies 100% static solvability & PBRS reachability, computes objective
architectural/topological/mechanical/difficulty metrics across every map, spot-checks
a stratified sample inside real ViZDoom `doom_env`, and exposes `CorpusPoolSampler`
for streaming rotating subsets into `training/train.py`.

CLI usage:
    uv run python -m training.generate_corpus --count 5000 --out training/runs/corpus_5k --workers 12 --engine-spotcheck 30
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import time
from typing import Any, Sequence

import numpy as np

from doom_env.mapinfo import load_mapinfo, parse_udmf
from doom_env.sensors import PLAYER_HEIGHT
from doom_env.wad import map_lumps, read_wad
from doom_env.wrapper import IWAD
from training.data_audit import canonical_d4_vertex_signature
from training.interfaces import MapSpec
from training.maps import DEV_MAPS, FINAL_MAPS, STRESS_MAPS
from training.pbrs import compute_pbrs_geodesic_summary
from training.wad_gen.dungeon import DungeonGenome, generate_dungeon_from_genome, generate_dungeon_wad
from training.wad_gen.extreme import EXTREME_ARCHETYPES, generate_extreme_wad
from training.wad_gen.progression import WINGS
from training.wad_gen.final_ood import FINAL_OOD_MAPS
from training.wad_gen.procedural import generate_procedural_wad
from training.wad_gen.verify import verify_wad_in_engine

ROOT = Path(__file__).resolve().parent.parent

# Verified PBRS-valid Freedoom2 maps strictly disjoint from held-out MAP01..MAP08
DISJOINT_FREEDOOM_MAPS: tuple[int, ...] = (9, 11, 12, 15, 24, 25, 27, 29, 31)

_ALL_MONSTER_TYPES = frozenset({3004, 9, 65, 3001, 3002, 58, 3006, 3005, 69, 3003, 68, 67, 66})
_ALL_WEAPON_TYPES = frozenset({2001, 2002, 82, 2003, 2004, 2005})
_ALL_PICKUP_TYPES = frozenset({2011, 2012, 2018, 2019, 2013, 2024, 2025, 2048, 2049, 2046, 2047, 17, 2035})
_SKULL_KEY_TYPES = frozenset({38, 39, 40})
_ALL_KEY_TYPES = frozenset({5, 6, 13, 38, 39, 40})
_UDMF_DAMAGING_SPECIALS = frozenset({69, 71, 75, 80, 105, 115, 196})
_DOOM_DAMAGING_SPECIALS = frozenset({4, 5, 7, 11, 16})

_MONSTER_HP: dict[int, int] = {
    3004: 20,   # Zombieman
    9: 30,      # ShotgunGuy
    3001: 60,   # DoomImp
    65: 70,     # ChaingunGuy
    3006: 100,  # LostSoul
    3002: 150,  # Demon
    58: 150,    # Spectre
    66: 300,    # Revenant
    3005: 400,  # Cacodemon
    69: 500,    # HellKnight
    68: 500,    # Arachnotron
    67: 600,    # Mancubus
    3003: 1000, # BaronOfHell
}

_WORKER_HELD_OUT_HASHES: frozenset[str] = frozenset()


def compute_held_out_d4_hashes() -> frozenset[str]:
    """Compute SHA-1 digests of D4-canonical vertex shapes for ALL held-out evaluation splits."""
    hashes: set[str] = set()
    for spec in (*DEV_MAPS, *STRESS_MAPS, *FINAL_MAPS, *FINAL_OOD_MAPS):
        p = Path(spec.wad) if Path(spec.wad).exists() else (ROOT / spec.wad)
        sig = canonical_d4_vertex_signature(p, spec.map_name)
        if sig:
            hashes.add(hashlib.sha1(repr(sig).encode("utf-8")).hexdigest()[:16])
    return frozenset(hashes)


def _shannon_entropy_bits(weights: Sequence[float]) -> float:
    tot = float(sum(w for w in weights if w > 0.0))
    if tot <= 1e-12:
        return 0.0
    ent = 0.0
    for w in weights:
        if w > 0.0:
            p = float(w) / tot
            ent -= p * math.log2(p)
    return round(ent, 3)


def analyze_corpus_wad(
    wad_path: str | Path,
    map_name: str = "MAP01",
    *,
    source: str,
    archetype: str,
    family: str,
    tier: int,
    episode_ticks: int,
    stored_wad: str | None = None,
) -> tuple[dict[str, Any], list[float]]:
    """Extract objective geometry, topology, wall-angle entropy, difficulty, and PBRS features."""
    resolved = Path(wad_path)
    if str(wad_path) == "vizdoom/freedoom2.wad":
        resolved = Path(IWAD)
    elif not resolved.exists():
        resolved = ROOT / wad_path

    lumps = map_lumps(read_wad(str(resolved)), map_name.upper())
    info = load_mapinfo(resolved, map_name.upper())
    pbrs_sum = compute_pbrs_geodesic_summary(resolved, map_name=map_name.upper(), cell_size=32.0)

    wall_hist = [0.0] * 18  # 18 bins of 10 degrees in [0, 180)
    non_axis = False
    has_exit_switch = False
    has_remote_door = False
    has_grate = False
    has_cliff_24 = False
    has_window_56 = False
    door_delays: set[int] = set()
    locks_seen: set[int] = set()
    exit_pts: list[tuple[float, float]] = []
    monsters: list[int] = []
    weapons: list[int] = []
    pickups: list[int] = []
    skull_keys = False
    has_radsuit = False
    has_ammo = False
    has_damaging = False
    player_xy: tuple[float, float, int] | None = None

    sec_adj: dict[int, set[int]] = {}
    active_secs: set[int] = set()

    if "TEXTMAP" in lumps:
        text = lumps["TEXTMAP"].decode("utf-8", "replace")
        b = parse_udmf(text)
        verts = [(float(v["x"]), float(v["y"])) for v in b.get("vertex", [])]
        sides = b.get("sidedef", [])
        secs = b.get("sector", [])
        linedefs = b.get("linedef", [])
        things = b.get("thing", [])

        xs = [p[0] for p in verts] or [0.0]
        ys = [p[1] for p in verts] or [0.0]
        span_max = round(max(max(xs) - min(xs), max(ys) - min(ys)), 1)
        num_sectors = len(secs)

        sec_hf = [int(s.get("heightfloor", 0) or 0) for s in secs]
        sec_hc = [int(s.get("heightceiling", 128) or 128) for s in secs]
        sec_sp = [int(s.get("special", 0) or 0) for s in secs]
        has_damaging = any(s in _UDMF_DAMAGING_SPECIALS for s in sec_sp)
        door_secs = set(info.door_sectors.keys())

        for ld in linedefs:
            v1 = verts[int(ld["v1"])]
            v2 = verts[int(ld["v2"])]
            dx, dy = v2[0] - v1[0], v2[1] - v1[1]
            seg_len = math.hypot(dx, dy)
            if seg_len > 1e-6:
                ang = math.degrees(math.atan2(dy, dx)) % 180.0
                bin_idx = min(17, int(ang // 10.0))
                wall_hist[bin_idx] += seg_len
            if abs(dx) > 1e-3 and abs(dy) > 1e-3:
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
            if isinstance(sf, int) and 0 <= sf < len(sides):
                sa = int(sides[sf]["sector"])
                active_secs.add(sa)
                if isinstance(sb, int) and 0 <= sb < len(sides):
                    sbb = int(sides[sb]["sector"])
                    active_secs.add(sbb)
                    if sa != sbb and not (ld.get("twosided") and ld.get("blocking")):
                        op = min(sec_hc[sa], sec_hc[sbb]) - max(sec_hf[sa], sec_hf[sbb])
                        if sa in door_secs or sbb in door_secs or op >= int(PLAYER_HEIGHT):
                            sec_adj.setdefault(sa, set()).add(sbb)
                            sec_adj.setdefault(sbb, set()).add(sa)
                        if sa not in door_secs and sbb not in door_secs:
                            step = abs(sec_hf[sa] - sec_hf[sbb])
                            if 0 < op < int(PLAYER_HEIGHT):
                                has_window_56 = True
                            if step > 24:
                                has_cliff_24 = True

        for th in things:
            t = int(th["type"])
            if t == 1:
                player_xy = (float(th["x"]), float(th["y"]), int(th.get("angle", 0)))
            elif t in _ALL_MONSTER_TYPES:
                monsters.append(t)
            elif t in _ALL_WEAPON_TYPES:
                weapons.append(t)
            elif t in _SKULL_KEY_TYPES:
                skull_keys = True
            if t in _ALL_PICKUP_TYPES:
                pickups.append(t)
            if t == 2025:
                has_radsuit = True
            if t in (2048, 2049, 2046, 2047, 17):
                has_ammo = True

        sig = canonical_d4_vertex_signature(resolved, map_name)
        d4_sig_hash = hashlib.sha1(repr(sig).encode("utf-8")).hexdigest()[:16] if sig else ""
    else:
        # Standard Doom binary lumps (Freedoom2 MAP09..31)
        v_raw = lumps["VERTEXES"]
        l_raw = lumps["LINEDEFS"]
        sd_raw = lumps["SIDEDEFS"]
        sc_raw = lumps["SECTORS"]
        t_raw = lumps.get("THINGS", b"")

        n_v = len(v_raw) // 4
        verts = [(float(x), float(y)) for x, y in (struct.unpack_from("<hh", v_raw, i * 4) for i in range(n_v))]
        xs = [p[0] for p in verts] or [0.0]
        ys = [p[1] for p in verts] or [0.0]
        span_max = round(max(max(xs) - min(xs), max(ys) - min(ys)), 1)

        n_sd = len(sd_raw) // 30
        side_secs = [int(struct.unpack_from("<hh8s8s8sh", sd_raw, i * 30)[5]) for i in range(n_sd)]
        n_sc = len(sc_raw) // 26
        num_sectors = n_sc
        sec_hf: list[int] = []
        sec_hc: list[int] = []
        for i in range(n_sc):
            hf, hc, _ft, _ct, _lt, sp, _tag = struct.unpack_from("<hh8s8shhh", sc_raw, i * 26)
            sec_hf.append(int(hf))
            sec_hc.append(int(hc))
            if int(sp) in _DOOM_DAMAGING_SPECIALS:
                has_damaging = True

        door_secs = set(info.door_sectors.keys())
        for lk in info.door_sectors.values():
            if lk > 0:
                locks_seen.add(lk)

        n_l = len(l_raw) // 14
        for i in range(n_l):
            v1i, v2i, flags, spec, _tag, sf_u, sb_u = struct.unpack_from("<HHHHHHH", l_raw, i * 14)
            if 0 <= v1i < n_v and 0 <= v2i < n_v:
                dx = verts[v2i][0] - verts[v1i][0]
                dy = verts[v2i][1] - verts[v1i][1]
                seg_len = math.hypot(dx, dy)
                if seg_len > 1e-6:
                    ang = math.degrees(math.atan2(dy, dx)) % 180.0
                    wall_hist[min(17, int(ang // 10.0))] += seg_len
                if abs(dx) > 1e-3 and abs(dy) > 1e-3:
                    non_axis = True
                if spec in (11, 51, 52, 124, 197, 198):
                    exit_pts.append(((verts[v1i][0] + verts[v2i][0]) * 0.5, (verts[v1i][1] + verts[v2i][1]) * 0.5))
                    if spec in (11, 51):
                        has_exit_switch = True
            sf = -1 if sf_u == 0xFFFF else int(sf_u)
            sb = -1 if sb_u == 0xFFFF else int(sb_u)
            blocking = bool(flags & 0x0001)
            if 0 <= sf < n_sd:
                sa = side_secs[sf]
                active_secs.add(sa)
                if 0 <= sb < n_sd:
                    sbb = side_secs[sb]
                    active_secs.add(sbb)
                    if blocking:
                        has_grate = True
                    elif sa != sbb and 0 <= sa < n_sc and 0 <= sbb < n_sc:
                        op = min(sec_hc[sa], sec_hc[sbb]) - max(sec_hf[sa], sec_hf[sbb])
                        if sa in door_secs or sbb in door_secs or op >= int(PLAYER_HEIGHT):
                            sec_adj.setdefault(sa, set()).add(sbb)
                            sec_adj.setdefault(sbb, set()).add(sa)
                        if sa not in door_secs and sbb not in door_secs:
                            if 0 < op < int(PLAYER_HEIGHT):
                                has_window_56 = True
                            if abs(sec_hf[sa] - sec_hf[sbb]) > 24:
                                has_cliff_24 = True

        n_t = len(t_raw) // 10
        for i in range(n_t):
            tx, ty, tang, t, _tf = struct.unpack_from("<hhhhh", t_raw, i * 10)
            if t == 1:
                player_xy = (float(tx), float(ty), int(tang))
            elif t in _ALL_MONSTER_TYPES:
                monsters.append(t)
            elif t in _ALL_WEAPON_TYPES:
                weapons.append(t)
            elif t in _SKULL_KEY_TYPES:
                skull_keys = True
            if t in _ALL_PICKUP_TYPES:
                pickups.append(t)
            if t == 2025:
                has_radsuit = True
            if t in (2048, 2049, 2046, 2047, 17):
                has_ammo = True

        d4_sig_hash = hashlib.sha1(f"freedoom:{map_name}".encode("utf-8")).hexdigest()[:16]

    # Cycle rank beta1 = |E| - |V| + C on walkable sector adjacency graph
    edge_count = sum(len(nbrs) for nbrs in sec_adj.values()) // 2
    comp_count = 0
    seen_secs: set[int] = set()
    for s in active_secs:
        if s in seen_secs:
            continue
        comp_count += 1
        dq = deque([s])
        seen_secs.add(s)
        while dq:
            u = dq.popleft()
            for v in sec_adj.get(u, ()):
                if v not in seen_secs:
                    seen_secs.add(v)
                    dq.append(v)
    cycle_rank_beta1 = max(0, edge_count - len(active_secs) + comp_count)

    bearing_bin = "ahead"
    if player_xy is not None and exit_pts:
        px, py, pang = player_xy
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

    num_locks = len(locks_seen)
    monster_count = len(monsters)
    monster_threat_hp = int(sum(_MONSTER_HP.get(m, 50) for m in monsters))
    wall_angle_entropy = _shannon_entropy_bits(wall_hist)
    geodesic_horizon = float(pbrs_sum["geodesic_horizon_u"])
    has_oneway = bool(pbrs_sum["has_oneway_dropoff"])

    locks_pts = min(
        28.0,
        9.0 * num_locks
        + 7.0 * int(has_remote_door)
        + 4.0 * int(has_exit_switch)
        + 2.0 * int(skull_keys),
    )
    combat_pts = min(
        28.0,
        18.0 * min(1.0, monster_threat_hp / 2200.0) + 10.0 * min(1.0, monster_count / 10.0),
    )
    horizon_pts = min(
        18.0,
        12.0 * min(1.0, geodesic_horizon / 2800.0) + 6.0 * min(1.0, span_max / 1600.0),
    )
    hazard_pts = min(
        16.0,
        4.0 * int(has_cliff_24)
        + 3.0 * int(has_oneway)
        + 4.0 * int(has_damaging)
        + 3.0 * int(has_window_56 or has_grate)
        + 2.0 * int(non_axis),
    )
    topo_pts = min(
        10.0,
        6.0 * min(1.0, cycle_rank_beta1 / 4.0) + 4.0 * int(monster_count >= 5 and not has_ammo),
    )
    difficulty_score = round(locks_pts + combat_pts + horizon_pts + hazard_pts + topo_pts, 2)
    if difficulty_score >= 72.0:
        difficulty_band = "nightmare"
    elif difficulty_score >= 55.0:
        difficulty_band = "very_high"
    elif difficulty_score >= 35.0:
        difficulty_band = "hard"
    else:
        difficulty_band = "standard"

    wad_str = stored_wad if stored_wad is not None else str(wad_path).replace("\\", "/")
    rec = {
        "wad": wad_str,
        "map_name": map_name.upper(),
        "family": family,
        "tier": int(tier),
        "episode_ticks": int(episode_ticks),
        "source": source,
        "archetype": archetype,
        "difficulty_score": difficulty_score,
        "difficulty_band": difficulty_band,
        "pbrs_valid": bool(pbrs_sum["pbrs_valid"]),
        "geodesic_horizon_u": geodesic_horizon,
        "wall_angle_entropy_bits": wall_angle_entropy,
        "cycle_rank_beta1": int(cycle_rank_beta1),
        "d4_sig_hash": d4_sig_hash,
        "sectors": int(num_sectors),
        "span_max": float(span_max),
        "wide_gt_1024": bool(span_max > 1024.0),
        "non_axis": bool(non_axis),
        "num_locks": int(num_locks),
        "skull_key": bool(skull_keys),
        "remote_door": bool(has_remote_door),
        "exit_switch": bool(has_exit_switch),
        "window_lt_56": bool(has_window_56),
        "grate": bool(has_grate),
        "cliff_gt_24": bool(has_cliff_24),
        "has_oneway_dropoff": has_oneway,
        "damaging_floor": bool(has_damaging),
        "radsuit": bool(has_radsuit),
        "monster_count": int(monster_count),
        "monster_threat_hp": monster_threat_hp,
        "monsters": sorted(set(monsters)),
        "weapons": sorted(set(weapons)),
        "pickups": sorted(set(pickups)),
        "has_ammo": bool(has_ammo),
        "bearing_bin": bearing_bin,
        "door_delays": sorted(door_delays),
    }
    return rec, wall_hist


def _worker_init(held_out_hashes: frozenset[str]) -> None:
    global _WORKER_HELD_OUT_HASHES
    _WORKER_HELD_OUT_HASHES = held_out_hashes


def _build_job_schedule(count: int, seed: int) -> list[tuple[int, str, str, int]]:
    """Create a stratified, objectively diverse + 45% Very-High/Nightmare task schedule."""
    rng = random.Random(int(seed))
    n_std = int(round(count * 0.20))
    n_hard = int(round(count * 0.35))
    n_ext = max(0, count - n_std - n_hard)

    # Very high / nightmare 45%: ~30% extreme_poly (balanced across combat and progression archetypes), ~10% extreme_dungeon, ~5% poly gauntlet
    n_ext_poly = int(round(n_ext * (30.0 / 45.0)))
    n_ext_dng = int(round(n_ext * (10.0 / 45.0)))
    n_ext_proc = max(0, n_ext - n_ext_poly - n_ext_dng)

    jobs_raw: list[tuple[str, str]] = []
    std_fams = ("door_exit", "combat", "multi_room")
    for i in range(n_std):
        kind = "proc_poly" if (i % 2 == 0) else "dungeon"
        jobs_raw.append((kind, std_fams[i % len(std_fams)]))

    hard_choices = [
        ("proc_poly", "key_return"),
        ("proc_poly", "multi_room"),
        ("proc_poly", "combat"),
        ("dungeon_hard", "key_return"),
        ("dungeon", "key_return"),
        ("dungeon", "multi_room"),
    ]
    for i in range(n_hard):
        jobs_raw.append(hard_choices[i % len(hard_choices)])

    for i in range(n_ext_poly):
        jobs_raw.append(("extreme_poly", EXTREME_ARCHETYPES[i % len(EXTREME_ARCHETYPES)]))
    for _ in range(n_ext_dng):
        jobs_raw.append(("extreme_dungeon", "gauntlet"))
    for _ in range(n_ext_proc):
        jobs_raw.append(("proc_poly", "gauntlet"))

    rng.shuffle(jobs_raw)
    base = int(seed) * 100_000 + 10_000
    return [(idx, kind, target, base + idx * 13) for idx, (kind, target) in enumerate(jobs_raw)]


def _generate_one_job(task: tuple[int, str, str, int, str, str]) -> tuple[int, dict[str, Any], list[float]]:
    idx, kind, target, base_seed, out_root_str, abs_wad_str = task
    out_root = Path(out_root_str)
    abs_wad = Path(abs_wad_str)
    abs_wad.parent.mkdir(parents=True, exist_ok=True)
    rel_wad = str(abs_wad.relative_to(ROOT)).replace("\\", "/") if abs_wad.is_relative_to(ROOT) else str(abs_wad).replace("\\", "/")

    last_reason = "unknown"
    for attempt in range(16):
        s = base_seed + attempt * 104729
        try:
            if kind == "extreme_poly":
                spec = generate_extreme_wad(target, s, abs_wad.parent, split="train", out_path=abs_wad)
                src = "extreme"
                arch = target
            elif kind == "extreme_dungeon":
                rng = random.Random(s)
                genome = DungeonGenome(
                    family="gauntlet",
                    seed=s,
                    extra_rooms=rng.choice((1, 2)),
                    extra_locks=rng.choice((1, 2)),
                    force_window=True,
                    force_vertical=True,
                    force_nukage=rng.random() < 0.70,
                    force_low_ammo=rng.random() < 0.45,
                    monster_delta=rng.choice((2, 3, 4)),
                    door_delay=rng.choice((0, 45, 80, 120)),
                )
                spec = generate_dungeon_from_genome(genome, abs_wad.parent, split="train", out_path=abs_wad)
                src = "extreme_dungeon"
                arch = "dungeon_nightmare"
            elif kind == "dungeon_hard":
                rng = random.Random(s)
                genome = DungeonGenome(
                    family=target,
                    seed=s,
                    extra_rooms=1,
                    extra_locks=1 if target == "key_return" else 0,
                    force_window=rng.random() < 0.60,
                    force_vertical=rng.random() < 0.65,
                    force_nukage=rng.random() < 0.50,
                    force_low_ammo=rng.random() < 0.30,
                    monster_delta=rng.choice((1, 2)),
                )
                spec = generate_dungeon_from_genome(genome, abs_wad.parent, split="train", out_path=abs_wad)
                src = "dungeon_mutated"
                arch = f"dungeon_{target}_hard"
            elif kind == "proc_poly":
                spec = generate_procedural_wad(
                    target,
                    s,
                    abs_wad.parent,
                    split="train",
                    with_combat=(target in ("key_return", "gauntlet")),
                    out_path=abs_wad,
                    continuous_angle=True,
                    poly_only=True,
                )
                src = "procedural_poly"
                arch = f"poly_{target}"
            else:
                spec = generate_dungeon_wad(
                    target,
                    seed=s,
                    out_dir=abs_wad.parent,
                    split="train",
                    with_combat=(target != "door_exit"),
                    out_path=abs_wad,
                )
                src = "dungeon"
                arch = f"dungeon_{target}"

            rec, hist = analyze_corpus_wad(
                abs_wad,
                spec.map_name,
                source=src,
                archetype=arch,
                family=spec.family,
                tier=spec.tier,
                episode_ticks=spec.episode_ticks,
                stored_wad=rel_wad,
            )
            if not rec["pbrs_valid"]:
                last_reason = "pbrs_invalid"
                continue
            if rec["d4_sig_hash"] in _WORKER_HELD_OUT_HASHES:
                last_reason = "held_out_collision"
                continue
            return idx, rec, hist
        except Exception as exc:
            last_reason = f"{type(exc).__name__}:{exc}"

    raise RuntimeError(f"Failed job idx={idx} kind={kind} target={target}: {last_reason}")


def generate_corpus(
    *,
    count: int = 5000,
    out_dir: str | Path = "training/runs/corpus_5k",
    seed: int = 2026,
    workers: int | None = None,
    include_freedoom: bool = True,
    engine_spotcheck: int = 30,
) -> dict[str, Any]:
    """Generate `count` unique PBRS-verified WADs across 6+ architectures & save manifest + audit scorecard."""
    t0 = time.perf_counter()
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.mkdir(parents=True, exist_ok=True)
    wads_root = out_path / "wads"
    wads_root.mkdir(parents=True, exist_ok=True)

    held_out_hashes = compute_held_out_d4_hashes()
    schedule = _build_job_schedule(int(count), seed=int(seed))
    n_workers = max(1, min(int(workers or (os.cpu_count() or 4)), max(1, int(count))))

    def _wad_path_for(idx: int, kind: str, target: str) -> Path:
        bucket = f"b{idx // 500:02d}"
        return wads_root / bucket / f"c{idx:05d}_{kind}_{target}.wad"

    results_by_idx: dict[int, tuple[dict[str, Any], list[float]]] = {}
    tasks = [
        (idx, kind, target, jseed, str(out_path), str(_wad_path_for(idx, kind, target)))
        for idx, kind, target, jseed in schedule
    ]

    if n_workers == 1:
        _worker_init(held_out_hashes)
        for task in tasks:
            idx, rec, hist = _generate_one_job(task)
            results_by_idx[idx] = (rec, hist)
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init, initargs=(held_out_hashes,)) as ex:
            futs = [ex.submit(_generate_one_job, t) for t in tasks]
            for fut in as_completed(futs):
                idx, rec, hist = fut.result()
                results_by_idx[idx] = (rec, hist)

    # Deduplicate intra-corpus exact D4 vertex shapes via targeted polygonal replacements (bounded rounds)
    seen_hashes: set[str] = set()
    dup_indices: list[int] = []
    for idx in range(len(schedule)):
        rec, _ = results_by_idx[idx]
        h = rec["d4_sig_hash"]
        if h in seen_hashes:
            dup_indices.append(idx)
        else:
            seen_hashes.add(h)

    if dup_indices and len(dup_indices) <= count:
        retry_round = 0
        while dup_indices and retry_round < 2:
            retry_round += 1
            regen_tasks = []
            for idx in dup_indices:
                _i, kind, target, jseed = schedule[idx]
                # Switch pure-grid duplicate into continuous-angle poly so shape vertices become unique!
                eff_kind = "proc_poly" if kind in ("dungeon", "dungeon_hard") else ("extreme_poly" if kind == "extreme_dungeon" else kind)
                eff_target = EXTREME_ARCHETYPES[idx % len(EXTREME_ARCHETYPES)] if eff_kind == "extreme_poly" else target
                regen_tasks.append((
                    idx,
                    eff_kind,
                    eff_target,
                    jseed + retry_round * 9_000_000 + idx * 31,
                    str(out_path),
                    str(_wad_path_for(idx, kind, target)),
                ))
            next_dups: list[int] = []
            if n_workers == 1:
                _worker_init(held_out_hashes)
                regen_res = [_generate_one_job(t) for t in regen_tasks]
            else:
                with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init, initargs=(held_out_hashes,)) as ex:
                    regen_res = [f.result() for f in as_completed([ex.submit(_generate_one_job, t) for t in regen_tasks])]
            for idx, rec, hist in regen_res:
                results_by_idx[idx] = (rec, hist)
                if rec["d4_sig_hash"] in seen_hashes:
                    next_dups.append(idx)
                else:
                    seen_hashes.add(rec["d4_sig_hash"])
            dup_indices = next_dups

    ordered_records: list[dict[str, Any]] = []
    pooled_wall_hist = [0.0] * 18
    for idx in range(len(schedule)):
        rec, hist = results_by_idx[idx]
        ordered_records.append(rec)
        for bi in range(18):
            pooled_wall_hist[bi] += hist[bi]

    freedoom_records: list[dict[str, Any]] = []
    if include_freedoom and Path(IWAD).exists():
        for m_num in DISJOINT_FREEDOOM_MAPS:
            m_name = f"MAP{m_num:02d}"
            rec, hist = analyze_corpus_wad(
                "vizdoom/freedoom2.wad",
                m_name,
                source="freedoom2_disjoint",
                archetype=f"freedoom2_{m_name.lower()}",
                family="gauntlet",
                tier=4,
                episode_ticks=4200,
                stored_wad="vizdoom/freedoom2.wad",
            )
            if rec["pbrs_valid"] or rec["geodesic_horizon_u"] > 0.0:
                freedoom_records.append(rec)

    all_records = [*ordered_records, *freedoom_records]
    manifest_path = out_path / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec) + "\n")

    with (out_path / "progression.jsonl").open("w", encoding="utf-8") as fh:
        for rec in ordered_records:
            if rec["archetype"] in WINGS:
                fh.write(json.dumps(rec) + "\n")

    # Stratified real ViZDoom engine spot-check across archetypes & difficulty bands
    spot_ok = 0
    spot_total = 0
    spot_errors: list[str] = []
    if int(engine_spotcheck) > 0 and ordered_records:
        spot_rng = random.Random(int(seed) + 77)
        by_group: dict[str, list[dict[str, Any]]] = {}
        for r in ordered_records:
            by_group.setdefault(f"{r['source']}:{r['archetype']}", []).append(r)
        picked: list[dict[str, Any]] = []
        group_keys = sorted(by_group)
        for k in group_keys:
            picked.append(spot_rng.choice(by_group[k]))
        while len(picked) < min(int(engine_spotcheck), len(ordered_records)):
            picked.append(spot_rng.choice(ordered_records))
        for rec in picked[: int(engine_spotcheck)]:
            spot_total += 1
            w_p = Path(rec["wad"]) if Path(rec["wad"]).exists() else (ROOT / rec["wad"])
            vres = verify_wad_in_engine(
                w_p,
                map_name=rec["map_name"],
                episode_ticks=int(rec["episode_ticks"]),
                step_count=2,
            )
            if vres.ok:
                spot_ok += 1
            else:
                spot_errors.append(f"{rec['wad']}: {vres.errors}")

    gen_s = round(time.perf_counter() - t0, 2)
    disk_bytes = sum(
        (Path(r["wad"]) if Path(r["wad"]).exists() else (ROOT / r["wad"])).stat().st_size
        for r in ordered_records
    )

    n_gen = max(1, len(ordered_records))
    diffs = sorted(float(r["difficulty_score"]) for r in ordered_records)
    horizons = sorted(float(r["geodesic_horizon_u"]) for r in ordered_records)
    spans = sorted(float(r["span_max"]) for r in ordered_records)
    beta1_vals = sorted(int(r["cycle_rank_beta1"]) for r in ordered_records)
    d4_unique = len({r["d4_sig_hash"] for r in ordered_records if r["d4_sig_hash"]})
    held_out_collisions = sum(1 for r in ordered_records if r["d4_sig_hash"] in held_out_hashes)

    def _pct(arr: list[float], p: float) -> float:
        if not arr:
            return 0.0
        idx = min(len(arr) - 1, max(0, int(round(p * (len(arr) - 1)))))
        return round(float(arr[idx]), 2)

    band_counts = Counter(r["difficulty_band"] for r in ordered_records)
    n_very_high_plus = band_counts["very_high"] + band_counts["nightmare"]
    beta1_hist = {
        "0": sum(1 for b in beta1_vals if b == 0),
        "1": sum(1 for b in beta1_vals if b == 1),
        "2": sum(1 for b in beta1_vals if b == 2),
        "3": sum(1 for b in beta1_vals if b == 3),
        "4+": sum(1 for b in beta1_vals if b >= 4),
    }

    seen_monsters = sorted({m for r in ordered_records for m in r["monsters"]})
    seen_weapons = sorted({w for r in ordered_records for w in r["weapons"]})
    seen_pickups = sorted({p for r in ordered_records for p in r["pickups"]})

    audit = {
        "count_generated": len(ordered_records),
        "count_disjoint_freedoom": len(freedoom_records),
        "total_manifest_entries": len(all_records),
        "multiplier_vs_static_train": round(len(ordered_records) / 20.0, 1),
        "generation_seconds": gen_s,
        "disk_size_mb": round(disk_bytes / (1024.0 * 1024.0), 2),
        "manifest_path": str(manifest_path.relative_to(ROOT)).replace("\\", "/") if manifest_path.is_relative_to(ROOT) else str(manifest_path),
        "integrity": {
            "held_out_d4_collisions": held_out_collisions,
            "unique_d4_shapes": d4_unique,
            "unique_d4_ratio": round(d4_unique / n_gen, 4),
            "pbrs_valid_count": sum(1 for r in ordered_records if r["pbrs_valid"]),
            "pbrs_valid_rate": round(sum(1 for r in ordered_records if r["pbrs_valid"]) / n_gen, 4),
            "engine_spotcheck_passed": spot_ok,
            "engine_spotcheck_total": spot_total,
            "engine_spotcheck_errors": spot_errors,
        },
        "difficulty_summary": {
            "bands": dict(band_counts),
            "very_high_plus_count": n_very_high_plus,
            "very_high_plus_frac": round(n_very_high_plus / n_gen, 4),
            "nightmare_count": band_counts["nightmare"],
            "nightmare_frac": round(band_counts["nightmare"] / n_gen, 4),
            "score_p10": _pct(diffs, 0.10),
            "score_p25": _pct(diffs, 0.25),
            "score_p50": _pct(diffs, 0.50),
            "score_p75": _pct(diffs, 0.75),
            "score_p90": _pct(diffs, 0.90),
            "score_p95": _pct(diffs, 0.95),
            "score_max": max(diffs, default=0.0),
        },
        "objective_variety": {
            "by_source": dict(Counter(r["source"] for r in ordered_records)),
            "by_archetype": dict(Counter(r["archetype"] for r in ordered_records)),
            "by_tier": dict(sorted(Counter(r["tier"] for r in ordered_records).items())),
            "frac_non_axis": round(sum(1 for r in ordered_records if r["non_axis"]) / n_gen, 4),
            "mean_wall_angle_entropy_bits": round(sum(float(r["wall_angle_entropy_bits"]) for r in ordered_records) / n_gen, 3),
            "pooled_wall_angle_entropy_bits": _shannon_entropy_bits(pooled_wall_hist),
            "wall_angle_bins_nonzero": sum(1 for x in pooled_wall_hist if x > 0.0),
            "cycle_rank_beta1_dist": beta1_hist,
            "cycle_rank_beta1_max": max(beta1_vals, default=0),
            "frac_wide_gt_1024": round(sum(1 for r in ordered_records if r["wide_gt_1024"]) / n_gen, 4),
            "span_min": min(spans, default=0.0),
            "span_p50": _pct(spans, 0.50),
            "span_max": max(spans, default=0.0),
            "geodesic_horizon_p50_u": _pct(horizons, 0.50),
            "geodesic_horizon_p90_u": _pct(horizons, 0.90),
            "geodesic_horizon_max_u": max(horizons, default=0.0),
            "frac_locked_1plus": round(sum(1 for r in ordered_records if r["num_locks"] >= 1) / n_gen, 4),
            "frac_locked_2plus": round(sum(1 for r in ordered_records if r["num_locks"] >= 2) / n_gen, 4),
            "frac_locked_3plus": round(sum(1 for r in ordered_records if r["num_locks"] >= 3) / n_gen, 4),
            "frac_skull_key": round(sum(1 for r in ordered_records if r["skull_key"]) / n_gen, 4),
            "frac_remote_door": round(sum(1 for r in ordered_records if r["remote_door"]) / n_gen, 4),
            "frac_exit_switch": round(sum(1 for r in ordered_records if r["exit_switch"]) / n_gen, 4),
            "frac_window_lt_56": round(sum(1 for r in ordered_records if r["window_lt_56"]) / n_gen, 4),
            "frac_grate": round(sum(1 for r in ordered_records if r["grate"]) / n_gen, 4),
            "frac_cliff_gt_24": round(sum(1 for r in ordered_records if r["cliff_gt_24"]) / n_gen, 4),
            "frac_oneway_dropoff": round(sum(1 for r in ordered_records if r["has_oneway_dropoff"]) / n_gen, 4),
            "frac_damaging_floor": round(sum(1 for r in ordered_records if r["damaging_floor"]) / n_gen, 4),
            "frac_radsuit": round(sum(1 for r in ordered_records if r["radsuit"]) / n_gen, 4),
            "frac_scarce_combat": round(sum(1 for r in ordered_records if r["monster_count"] >= 5 and not r["has_ammo"]) / n_gen, 4),
            "monster_types_count": len(seen_monsters),
            "weapon_types_count": len(seen_weapons),
            "pickup_types_count": len(seen_pickups),
            "bearings": dict(Counter(r["bearing_bin"] for r in ordered_records)),
        },
    }
    audit_path = out_path / "corpus_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


class CorpusPoolSampler:
    """Streams stratified active working subsets from a pre-generated corpus manifest JSONL."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        include_freedoom: bool = True,
        min_difficulty: float = 0.0,
    ) -> None:
        mp = Path(manifest_path)
        if not mp.exists():
            mp = ROOT / manifest_path
        if not mp.is_file():
            raise FileNotFoundError(f"Corpus manifest not found: {manifest_path}")
        self.manifest_path = mp
        self.records: list[dict[str, Any]] = []
        with mp.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("split", "train") != "train":
                    raise ValueError(f"Training sampler received a held-out map in {mp}")
                if not include_freedoom and rec.get("source") == "freedoom2_disjoint":
                    continue
                if float(rec.get("difficulty_score", 0.0)) < float(min_difficulty):
                    continue
                self.records.append(rec)
        if not self.records:
            raise ValueError(f"No usable records loaded from corpus manifest {mp}")

        self._unseen_std: list[int] = []
        self._unseen_high: list[int] = []
        self._std_indices: list[int] = []
        self._high_indices: list[int] = []
        self._freedoom_indices: list[int] = []
        for idx, rec in enumerate(self.records):
            if rec.get("source") == "freedoom2_disjoint":
                self._freedoom_indices.append(idx)
            elif rec.get("difficulty_band") in ("very_high", "nightmare") or int(rec.get("tier", 1)) >= 4:
                self._high_indices.append(idx)
            else:
                self._std_indices.append(idx)
        self._seen_counts: Counter[str] = Counter()
        self._current_pool: list[MapSpec] = []

    def _to_spec(self, rec: dict[str, Any]) -> MapSpec:
        return MapSpec(
            wad=str(rec["wad"]),
            map_name=str(rec.get("map_name", "MAP01")),
            family=str(rec.get("family", "gauntlet")),
            tier=int(rec.get("tier", 4)),
            split="train",
            episode_ticks=int(rec.get("episode_ticks", 4200 if int(rec.get("tier", 1)) >= 3 else 2100)),
            seeds=(0, 1, 2),
        )

    def _pop_unseen(self, unseen: list[int], all_idxs: list[int], k: int, rng: Any, exclude: set[str]) -> list[int]:
        if not all_idxs or k <= 0:
            return []
        picked: list[int] = []
        eligible = {i for i in all_idxs
                    if (self.records[i]["wad"] + ":" + self.records[i].get("map_name", "MAP01")) not in exclude}
        while eligible and len(picked) < k:
            if not unseen:
                unseen.extend(all_idxs)
                perm = rng.permutation(len(unseen)) if hasattr(rng, "permutation") else None
                if perm is not None:
                    unseen[:] = [unseen[int(i)] for i in perm]
                else:
                    random.Random(0).shuffle(unseen)
            cand = unseen.pop()
            key = f"{self.records[cand]['wad']}:{self.records[cand].get('map_name', 'MAP01')}"
            if cand not in eligible:
                continue
            eligible.remove(cand)
            exclude.add(key)
            picked.append(cand)
        return picked

    def sample_pool(
        self,
        *,
        pool_size: int = 60,
        high_diff_frac: float = 0.45,
        rng: Any = None,
        scores_by_wad: dict[str, float] | None = None,
        elite_keep_frac: float = 0.20,
    ) -> list[MapSpec]:
        """Return a stratified working pool (retaining top SFL+PVL elites + unseen rotation)."""
        if rng is None:
            rng = np.random.default_rng(0)
        k_total = max(1, int(pool_size))
        chosen_specs: list[MapSpec] = []
        used_keys: set[str] = set()

        # 1. Optionally carry over top elite high-regret maps from current pool
        if self._current_pool and scores_by_wad:
            n_keep = min(len(self._current_pool), max(0, int(round(k_total * elite_keep_frac))))
            ranked = sorted(self._current_pool, key=lambda s: -float(scores_by_wad.get(s.wad, 0.0)))
            for s in ranked[:n_keep]:
                key = f"{s.wad}:{s.map_name}"
                if key not in used_keys:
                    chosen_specs.append(s)
                    used_keys.add(key)

        # 2. Optionally include up to 2 rotating disjoint Freedoom2 maps for real-WAD grounding
        rem = k_total - len(chosen_specs)
        if self._freedoom_indices and rem >= 12:
            n_fd = min(2, len(self._freedoom_indices))
            perm_fd = rng.permutation(len(self._freedoom_indices)) if hasattr(rng, "permutation") else list(range(len(self._freedoom_indices)))
            for pi in perm_fd[:n_fd]:
                rec = self.records[self._freedoom_indices[int(pi)]]
                s = self._to_spec(rec)
                key = f"{s.wad}:{s.map_name}"
                if key not in used_keys:
                    chosen_specs.append(s)
                    used_keys.add(key)

        # 3. Fill remainder stratified between Very High/Nightmare and Standard/Hard unseen-first
        rem = max(0, k_total - len(chosen_specs))
        n_high = min(rem, int(round(rem * float(np.clip(high_diff_frac, 0.0, 1.0)))))
        n_std = rem - n_high
        if not self._high_indices:
            n_std += n_high
            n_high = 0
        if not self._std_indices:
            n_high += n_std
            n_std = 0

        for idx in self._pop_unseen(self._unseen_high, self._high_indices, n_high, rng, used_keys):
            s = self._to_spec(self.records[idx])
            chosen_specs.append(s)
            self._seen_counts[f"{s.wad}:{s.map_name}"] += 1
        for idx in self._pop_unseen(self._unseen_std, self._std_indices, n_std, rng, used_keys):
            s = self._to_spec(self.records[idx])
            chosen_specs.append(s)
            self._seen_counts[f"{s.wad}:{s.map_name}"] += 1

        self._current_pool = list(chosen_specs)
        return chosen_specs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate & objectively audit a large multi-archetype WAD corpus.")
    p.add_argument("--count", type=int, default=5000, help="Number of unique procedural WADs to generate.")
    p.add_argument("--out", default="training/runs/corpus_5k", help="Output directory (recommended inside gitignored training/runs/).")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--include-freedoom", dest="include_freedoom", action="store_true", default=True)
    p.add_argument("--no-include-freedoom", dest="include_freedoom", action="store_false")
    p.add_argument("--engine-spotcheck", type=int, default=30, help="Number of WADs to boot and step inside real ViZDoom.")
    args = p.parse_args(argv)

    audit = generate_corpus(
        count=args.count,
        out_dir=args.out,
        seed=args.seed,
        workers=args.workers,
        include_freedoom=args.include_freedoom,
        engine_spotcheck=args.engine_spotcheck,
    )
    print(json.dumps(audit, indent=2), flush=True)
    integ = audit["integrity"]
    if integ["held_out_d4_collisions"] > 0:
        return 1
    if integ["pbrs_valid_rate"] < 1.0:
        return 1
    if integ["engine_spotcheck_passed"] < integ["engine_spotcheck_total"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
