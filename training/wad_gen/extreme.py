"""Very-high-difficulty & nightmare continuous-angle polygonal WAD archetypes (Role 3).

Generates parameterized Tier-4 compound and multi-wing dependency maps with arbitrary
continuous rotation (`angle_deg in [0, 360)`), zero T-junctions, multi-lock & decoy-lock
progression, remote door switches, one-way drop-offs, low windows, impassable grates,
damaging nukage + dry catwalks, high cycle-rank (`beta1 >= 2`) aliasing loops, wide
`>1024u` `RAY_NONE` crossfire arenas, scarce ammo splits, and balanced 4-quadrant spawn
bearings. All outputs are verified via `verify_polygon_solvability(..., check_t_junctions=True)`
and strictly offline/training-only.
"""

from __future__ import annotations

from collections import deque
from functools import partial
import hashlib
import math
from pathlib import Path
import random
from typing import Sequence

from training.interfaces import MapSpec
from training.wad_gen.progression import WINGS, build_progression
from training.wad_gen.udmf import (
    THEME_NAMES,
    PolygonHole,
    PolygonSector,
    ThingSpec,
    build_polygon_wad,
    transform_polygon_map,
)
from training.wad_gen.verify import verify_polygon_solvability

EXTREME_ARCHETYPES: tuple[str, ...] = (
    "decoy_multilock",
    "vertical_switch_citadel",
    "aliased_labyrinth",
    "scarce_crossfire_gauntlet",
    "nukage_compound_fortress",
    "grate_window_backtrack",
    *WINGS,
)

_KEY_SPECS: tuple[tuple[str, int, int], ...] = (
    ("door_red", 13, 38),
    ("door_blue", 5, 40),
    ("door_yellow", 6, 39),
)

# Heavy & compound monster roster (DoomEdNums) weighted toward challenging foes
_HEAVY_MONSTERS: tuple[int, ...] = (
    65,    # ChaingunGuy (hitscan)
    3001,  # DoomImp
    3002,  # Demon
    58,    # Spectre
    3006,  # LostSoul
    3005,  # Cacodemon
    69,    # HellKnight
    66,    # Revenant
    68,    # Arachnotron
    67,    # Fatso / Mancubus
    3003,  # BaronOfHell
    9,     # ShotgunGuy
    3004,  # Zombieman
)

_WEAPON_AMMO_PAIRS: tuple[tuple[int, int, int], ...] = (
    (2001, 2049, 2049),  # Shotgun + ShellBox
    (82, 2049, 2049),    # SuperShotgun + ShellBox
    (2002, 2048, 2048),  # Chaingun + ClipBox
    (2003, 2046, 2046),  # RocketLauncher + RocketBox
    (2004, 17, 2047),    # PlasmaRifle + CellPack + Cell
    (2005, 2001, 2049),  # Chainsaw + Shotgun + ShellBox
)


def _det_rng(archetype: str, seed: int, attempt: int = 0) -> random.Random:
    digest = hashlib.sha256(f"extreme:{archetype}:{seed}:{attempt}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "little"))


def _rect_splits(
    name: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    south_xs: Sequence[float] = (),
    north_xs: Sequence[float] = (),
    west_ys: Sequence[float] = (),
    east_ys: Sequence[float] = (),
    **kwargs,
) -> PolygonSector:
    """Build an axis rectangle with exact boundary split vertices so adjacent edges stitch 1:1."""
    xa, xb = (float(min(x0, x1)), float(max(x0, x1)))
    ya, yb = (float(min(y0, y1)), float(max(y0, y1)))
    pts: list[tuple[float, float]] = [(xa, ya)]
    for x in sorted({float(v) for v in south_xs if xa + 0.5 < float(v) < xb - 0.5}):
        pts.append((x, ya))
    pts.append((xb, ya))
    for y in sorted({float(v) for v in east_ys if ya + 0.5 < float(v) < yb - 0.5}):
        pts.append((xb, y))
    pts.append((xb, yb))
    for x in sorted({float(v) for v in north_xs if xa + 0.5 < float(v) < xb - 0.5}, reverse=True):
        pts.append((x, yb))
    pts.append((xa, yb))
    for y in sorted({float(v) for v in west_ys if ya + 0.5 < float(v) < yb - 0.5}, reverse=True):
        pts.append((xa, y))
    return PolygonSector(name, tuple(pts), **kwargs)


def _octagon(cx: float, cy: float, r: float, a: float) -> tuple[tuple[float, float], ...]:
    return (
        (cx - r, cy - a),
        (cx - r, cy + a),
        (cx - a, cy + r),
        (cx + a, cy + r),
        (cx + r, cy + a),
        (cx + r, cy - a),
        (cx + a, cy - r),
        (cx - a, cy - r),
    )


def _extrude(
    p1: tuple[float, float],
    p2: tuple[float, float],
    dx: float,
    dy: float,
) -> tuple[tuple[float, float], ...]:
    q1 = (round(p1[0] + dx, 1), round(p1[1] + dy, 1))
    q2 = (round(p2[0] + dx, 1), round(p2[1] + dy, 1))
    return (p2, p1, q1, q2)


def _poly_center(pts: Sequence[tuple[float, float]]) -> tuple[float, float]:
    n = float(len(pts))
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)


def _pick_bearing(
    spawn_xy: tuple[float, float],
    exit_xy: tuple[float, float],
    rng: random.Random,
    seed: int,
) -> int:
    dx = exit_xy[0] - spawn_xy[0]
    dy = exit_xy[1] - spawn_xy[1]
    exit_deg = math.degrees(math.atan2(dy, dx)) % 360.0
    quad = (0, 90, 180, 270)[int(seed) % 4]
    return int(round((exit_deg + quad + rng.randint(-25, 25)) % 360.0))


def _sample_continuous_rot(rng: random.Random) -> float:
    base_quad = rng.randrange(4) * 90.0
    return round(base_quad + rng.uniform(11.0, 79.0), 1)


def _theme_walls(rng: random.Random) -> tuple[str, str, str, int]:
    idx = rng.randrange(len(THEME_NAMES))
    wall, flr, ceil, light = (
        ("STARTAN2", "FLOOR0_1", "CEIL1_1", 192),
        ("STONE2", "FLOOR4_8", "CEIL3_5", 176),
        ("BROWN1", "FLAT14", "CEIL1_1", 184),
        ("METAL1", "FLOOR5_1", "CEIL5_1", 192),
        ("GRAY1", "FLOOR0_3", "FLAT19", 184),
        ("BRICK1", "FLOOR7_1", "CEIL3_2", 176),
    )[idx]
    return wall, flr, ceil, light


def _build_decoy_multilock(
    rng: random.Random, seed: int
) -> tuple[tuple[PolygonSector, ...], tuple[ThingSpec, ...], int]:
    """Octagonal citadel with visible decoy final lock, 2-3 sequential/parallel locks, nukage & heavy bosses."""
    keys = list(_KEY_SPECS)
    rng.shuffle(keys)
    three_lock = rng.random() < 0.72
    use_nukage = rng.random() < 0.65
    use_switch_exit = rng.random() < 0.45
    (lock1, c1, s1), (lock2, c2, s2), (lock3, c3, s3) = keys[0], keys[1], keys[2]
    k1 = s1 if rng.random() < 0.6 else c1
    k2 = s2 if rng.random() < 0.6 else c2
    k3 = s3 if rng.random() < 0.6 else c3
    wpn, ammo1, ammo2 = rng.choice(_WEAPON_AMMO_PAIRS)
    m_ang = rng.randrange(0, 360, 45)

    r_hub = float(rng.choice((240, 256, 280)))
    a_hub = float(rng.choice((68, 76, 84)))
    hub = _octagon(0.0, 0.0, r_hub, a_hub)
    center_pillar = PolygonHole(vertices=((0.0, -56.0), (56.0, 0.0), (0.0, 56.0), (-56.0, 0.0)), wall_texture="BRICK1")

    s_spawn = _extrude(hub[6], hub[7], 0.0, -160.0)
    w_door = _extrude(hub[0], hub[1], -56.0, 0.0)
    w_room = _extrude(w_door[3], w_door[2], -224.0, 0.0)
    e_lock1 = _extrude(hub[4], hub[5], 56.0, 0.0)
    e_room = _extrude(e_lock1[3], e_lock1[2], 240.0, 0.0)

    nw_conn = _extrude(hub[1], hub[2], -56.0, 56.0)
    nw_room = _extrude(nw_conn[3], nw_conn[2], -176.0, 176.0)

    ne_alc = _extrude(hub[3], hub[4], 56.0, -56.0)
    # Wait: hub[3]->hub[4] is (+a, +r) -> (+r, +a), outward normal is (+1, +1) (NE)
    ne_conn = _extrude(hub[3], hub[4], 56.0, 56.0)
    ne_room = _extrude(ne_conn[3], ne_conn[2], 160.0, 160.0)

    # Decoy lock directly North of hub facing the spawn!
    n_lock_decoy = _extrude(hub[2], hub[3], 0.0, 56.0)
    n_s1 = _extrude(n_lock_decoy[3], n_lock_decoy[2], 0.0, 56.0)
    n_s2 = _extrude(n_s1[3], n_s1[2], 0.0, 56.0)
    n_sanctum = _extrude(n_s2[3], n_s2[2], 0.0, 192.0)
    n_exit = _extrude(n_sanctum[3], n_sanctum[2], 0.0, 80.0)
    ex_far = (n_exit[2], n_exit[3])

    sw_far = (nw_room[2], nw_room[3])
    use_remote_n = rng.random() < 0.40

    sectors: list[PolygonSector] = [
        PolygonSector("hub", hub, "room", wall_texture="STONE2", height_floor=0, height_ceiling=200, holes=(center_pillar,)),
        PolygonSector("s_spawn", s_spawn, "room", wall_texture="STARTAN2", height_floor=0, height_ceiling=200),
        PolygonSector("w_door", w_door, "door", height_floor=0, height_ceiling=200),
        PolygonSector("w_room", w_room, "room", wall_texture="BROWN1", height_floor=0, height_ceiling=200),
        PolygonSector("e_lock1", e_lock1, lock1, height_floor=0, height_ceiling=200),
        PolygonSector(
            "e_room",
            e_room,
            "room",
            wall_texture="BRICK1",
            floor_texture="NUKAGE1" if use_nukage else "FLOOR7_1",
            height_floor=0,
            height_ceiling=200,
            sector_special=71 if use_nukage else 0,
        ),
        PolygonSector(
            "nw_conn",
            nw_conn,
            lock2 if three_lock else "door",
            height_floor=0,
            height_ceiling=200,
        ),
        PolygonSector(
            "nw_room",
            nw_room,
            "room",
            wall_texture="GRAY1",
            height_floor=0,
            height_ceiling=200,
            door_switch=(sw_far[0], sw_far[1], 1) if use_remote_n else None,
        ),
        PolygonSector(
            "ne_conn",
            ne_conn,
            "remote_door" if use_remote_n else "door",
            height_floor=0,
            height_ceiling=200,
            sector_id=1 if use_remote_n else 0,
        ),
        PolygonSector("ne_room", ne_room, "room", wall_texture="METAL1", height_floor=0, height_ceiling=200),
        PolygonSector("n_lock_decoy", n_lock_decoy, lock3 if three_lock else lock2, height_floor=0, height_ceiling=200),
        PolygonSector("n_s1", n_s1, "room", wall_texture="STONE2", height_floor=20, height_ceiling=200),
        PolygonSector("n_s2", n_s2, "room", wall_texture="STONE2", height_floor=40, height_ceiling=200),
        PolygonSector("n_sanctum", n_sanctum, "room", wall_texture="METAL1", height_floor=40, height_ceiling=200),
        PolygonSector(
            "n_exit",
            n_exit,
            "room" if use_switch_exit else "exit",
            height_floor=40,
            height_ceiling=200,
            exit_switch=ex_far if use_switch_exit else None,
        ),
    ]

    spawn_xy = _poly_center(s_spawn)
    exit_xy = _poly_center(n_exit)
    p_ang = _pick_bearing(spawn_xy, exit_xy, rng, seed)
    w_xy = _poly_center(w_room)
    e_xy = _poly_center(e_room)
    nw_xy = _poly_center(nw_room)
    ne_xy = _poly_center(ne_room)
    san_xy = _poly_center(n_sanctum)

    scarce = rng.random() < 0.35
    things: list[ThingSpec] = [
        ThingSpec(spawn_xy[0], spawn_xy[1], 1, p_ang),
        ThingSpec(spawn_xy[0] - 32.0, spawn_xy[1] + 24.0, wpn, 0),
        ThingSpec(w_xy[0], w_xy[1], k1, 0),
        ThingSpec(e_xy[0], e_xy[1], k2, 0),
        ThingSpec(ne_xy[0], ne_xy[1], 2019 if not scarce else 2018, 0),
        ThingSpec(0.0, -r_hub * 0.65, 2025 if use_nukage else 2013, 0),
        ThingSpec(san_xy[0], san_xy[1], 2012, 0),
    ]
    if three_lock:
        things.append(ThingSpec(nw_xy[0], nw_xy[1], k3, 0))
    if not scarce:
        things.extend([
            ThingSpec(spawn_xy[0] + 32.0, spawn_xy[1] + 24.0, ammo1, 0),
            ThingSpec(w_xy[0] + 40.0, w_xy[1] - 40.0, ammo2, 0),
            ThingSpec(e_xy[0] - 48.0, e_xy[1] + 40.0, 2012, 0),
        ])

    enemy_spots = [
        (-r_hub * 0.58, -r_hub * 0.55),
        (r_hub * 0.58, -r_hub * 0.55),
        (0.0, r_hub * 0.62),
        (w_xy[0] + 56.0, w_xy[1] + 32.0),
        (w_xy[0] - 40.0, w_xy[1] - 36.0),
        (e_xy[0] - 56.0, e_xy[1] - 32.0),
        (e_xy[0] + 48.0, e_xy[1] + 36.0),
        (nw_xy[0] + 40.0, nw_xy[1] - 40.0),
        (ne_xy[0] - 40.0, ne_xy[1] - 40.0),
        (san_xy[0] - 48.0, san_xy[1]),
        (san_xy[0] + 48.0, san_xy[1]),
        (san_xy[0], san_xy[1] + 40.0),
    ]
    rng.shuffle(enemy_spots)
    n_mon = rng.randint(9, 12)
    for ex, ey in enemy_spots[:n_mon]:
        things.append(ThingSpec(ex, ey, rng.choice(_HEAVY_MONSTERS[:10]), m_ang))
    things.append(ThingSpec(-r_hub * 0.45, 0.0, 2035, 0))
    things.append(ThingSpec(r_hub * 0.45, 0.0, 2035, 0))

    rot_k = rng.randrange(4)
    flip_x = bool(rng.randrange(2))
    angle_deg = _sample_continuous_rot(rng)
    rot_s, rot_t = transform_polygon_map(sectors, things, rot_k=rot_k, flip_x=flip_x, angle_deg=angle_deg)
    return rot_s, rot_t, rng.choice((0, 45, 80, 120))


def _build_vertical_switch_citadel(
    rng: random.Random, seed: int
) -> tuple[tuple[PolygonSector, ...], tuple[ThingSpec, ...], int]:
    """Multi-tier stairs + true 64u one-way cliff drop-off + remote switch + 2-3 locks + bosses."""
    keys = list(_KEY_SPECS)
    rng.shuffle(keys)
    three_lock = rng.random() < 0.55
    use_nukage = rng.random() < 0.50
    use_switch_exit = rng.random() < 0.50
    (lock1, c1, s1), (lock2, c2, s2), (lock3, c3, s3) = keys[0], keys[1], keys[2]
    k1 = s1 if rng.random() < 0.55 else c1
    k2 = s2 if rng.random() < 0.55 else c2
    k3 = s3 if rng.random() < 0.55 else c3
    wpn, ammo1, _ = rng.choice(_WEAPON_AMMO_PAIRS)
    m_ang = rng.randrange(0, 360, 45)

    w_c = float(rng.choice((320, 384, 448)))
    h_c = float(rng.choice((320, 360)))
    stair_h = 112.0
    w_top = float(rng.choice((208, 256)))

    # West courtyard (0..w_c, 0..h_c): south stairs to east high_ledge (hf=64), drop-off back west
    # North wing via lock1 (for k2 + nukage); South-west wing via remote_door(tag=1) -> lock2 -> exit!
    y_n0, y_n1 = 112.0, 224.0
    far_ex_wall = ((-432.0, y_n0), (-432.0, y_n1))
    rem_sw_wall = ((w_c + 240.0 + w_top, stair_h), (w_c + 240.0 + w_top, h_c))

    pillar = PolygonHole(
        vertices=((w_c * 0.5, h_c * 0.5 - 40), (w_c * 0.5 + 40, h_c * 0.5), (w_c * 0.5, h_c * 0.5 + 40), (w_c * 0.5 - 40, h_c * 0.5)),
        wall_texture="STONE2",
    )
    sectors: list[PolygonSector] = [
        _rect_splits(
            "courtyard",
            0.0,
            0.0,
            w_c,
            h_c,
            north_xs=(96.0, 224.0),
            west_ys=(y_n0, y_n1),
            east_ys=(stair_h,),
            wall_texture="STONE2",
            height_floor=0,
            height_ceiling=208,
            holes=(pillar,),
        ),
        _rect_splits("step1", w_c, 0.0, w_c + 80.0, stair_h, wall_texture="STEP1", height_floor=16, height_ceiling=208),
        _rect_splits("step2", w_c + 80.0, 0.0, w_c + 160.0, stair_h, wall_texture="STEP1", height_floor=32, height_ceiling=208),
        _rect_splits("step3", w_c + 160.0, 0.0, w_c + 240.0, stair_h, wall_texture="STEP1", height_floor=48, height_ceiling=208),
        _rect_splits("step4_top", w_c + 240.0, 0.0, w_c + 240.0 + w_top, stair_h, wall_texture="BRICK1", height_floor=64, height_ceiling=208),
        _rect_splits(
            "high_ledge",
            w_c,
            stair_h,
            w_c + 240.0 + w_top,
            h_c,
            south_xs=(w_c + 80.0, w_c + 160.0, w_c + 240.0),
            wall_texture="METAL1",
            floor_texture="FLOOR7_1",
            height_floor=64,
            height_ceiling=208,
            door_switch=(rem_sw_wall[0], rem_sw_wall[1], 1),
        ),
        _rect_splits("n_lock1", 96.0, h_c, 224.0, h_c + 56.0, kind=lock1, height_floor=0, height_ceiling=208),
        _rect_splits(
            "n_vault",
            48.0,
            h_c + 56.0,
            272.0,
            h_c + 280.0,
            south_xs=(96.0, 224.0),
            floor_texture="NUKAGE1" if use_nukage else "FLOOR4_8",
            wall_texture="BROWN1",
            height_floor=0,
            height_ceiling=208,
            sector_special=71 if use_nukage else 0,
        ),
        _rect_splits("w_rdoor", -56.0, y_n0, 0.0, y_n1, kind="remote_door", sector_id=1, height_floor=0, height_ceiling=208),
        _rect_splits("w_hall", -208.0, y_n0, -56.0, y_n1, wall_texture="GRAY1", height_floor=0, height_ceiling=208),
        _rect_splits("w_lock2", -264.0, y_n0, -208.0, y_n1, kind=lock2, height_floor=0, height_ceiling=208),
        _rect_splits("w_sanctum", -368.0, y_n0, -264.0, y_n1, wall_texture="BRICK1", height_floor=0, height_ceiling=208),
        _rect_splits(
            "w_exit",
            -432.0,
            y_n0,
            -368.0,
            y_n1,
            kind="room" if use_switch_exit else "exit",
            height_floor=0,
            height_ceiling=208,
            exit_switch=far_ex_wall if use_switch_exit else None,
        ),
    ]
    if three_lock:
        # Add South-West branch behind lock3 containing key3 before final exit!
        # Keep courtyards clean: swap lock2 to lock3 on w_lock2 and put lock2 east of n_vault!
        sectors[10] = _rect_splits("w_lock2", -264.0, y_n0, -208.0, y_n1, kind=lock3, height_floor=0, height_ceiling=208)
        sectors[7] = _rect_splits(
            "n_vault",
            48.0,
            h_c + 56.0,
            272.0,
            h_c + 280.0,
            south_xs=(96.0, 224.0),
            east_ys=(h_c + 120.0, h_c + 216.0),
            floor_texture="NUKAGE1" if use_nukage else "FLOOR4_8",
            wall_texture="BROWN1",
            height_floor=0,
            height_ceiling=208,
            sector_special=71 if use_nukage else 0,
        )
        sectors.extend([
            _rect_splits("ne_lock2", 272.0, h_c + 120.0, 328.0, h_c + 216.0, kind=lock2, height_floor=0, height_ceiling=208),
            _rect_splits("ne_crypt", 328.0, h_c + 96.0, 512.0, h_c + 240.0, west_ys=(h_c + 120.0, h_c + 216.0), wall_texture="BRICK1", height_floor=0, height_ceiling=208),
        ])

    spawn_xy = (64.0, h_c * 0.5)
    exit_xy = (-400.0, (y_n0 + y_n1) * 0.5)
    p_ang = _pick_bearing(spawn_xy, exit_xy, rng, seed)
    things: list[ThingSpec] = [
        ThingSpec(spawn_xy[0], spawn_xy[1], 1, p_ang),
        ThingSpec(128.0, 64.0, wpn, 0),
        ThingSpec(w_c + 120.0, (stair_h + h_c) * 0.5, k1, 0),
        ThingSpec(160.0, h_c + 180.0, k2, 0),
        ThingSpec(160.0, h_c - 64.0, 2025 if use_nukage else 2018, 0),
        ThingSpec(-132.0, (y_n0 + y_n1) * 0.5, 2012, 0),
        ThingSpec(w_c + 200.0, 48.0, ammo1, 0),
    ]
    if three_lock:
        things.append(ThingSpec(430.0, h_c + 168.0, k3, 0))
        things.append(ThingSpec(420.0, h_c + 132.0, rng.choice(_HEAVY_MONSTERS[:10]), m_ang))
        things.append(ThingSpec(440.0, h_c + 204.0, rng.choice(_HEAVY_MONSTERS[:10]), m_ang))

    enemy_spots = [
        (w_c * 0.72, h_c * 0.75),
        (w_c * 0.72, h_c * 0.25),
        (w_c + 80.0, (stair_h + h_c) * 0.5),
        (w_c + 220.0, h_c - 56.0),
        (w_c + 280.0, stair_h + 64.0),
        (100.0, h_c + 120.0),
        (220.0, h_c + 220.0),
        (-132.0, y_n0 + 36.0),
        (-316.0, (y_n0 + y_n1) * 0.5),
    ]
    rng.shuffle(enemy_spots)
    for ex, ey in enemy_spots[: rng.randint(7, 9)]:
        things.append(ThingSpec(ex, ey, rng.choice(_HEAVY_MONSTERS[:10]), m_ang))
    things.append(ThingSpec(w_c * 0.5, 48.0, 2035, 0))
    things.append(ThingSpec(w_c + 40.0, h_c - 48.0, 2035, 0))

    rot_k = rng.randrange(4)
    flip_x = bool(rng.randrange(2))
    angle_deg = _sample_continuous_rot(rng)
    rot_s, rot_t = transform_polygon_map(sectors, things, rot_k=rot_k, flip_x=flip_x, angle_deg=angle_deg)
    return rot_s, rot_t, rng.choice((0, 60, 100, 150))


def _build_aliased_labyrinth(
    rng: random.Random, seed: int
) -> tuple[tuple[PolygonSector, ...], tuple[ThingSpec, ...], int]:
    """Zone-partitioned `3x3`/`3x4` polygonal labyrinth with high cycle rank beta1>=2, 2-3 locks, grates, windows & cliff drop-offs."""
    mr, mc = rng.choice(((3, 3), (3, 4), (4, 3)))
    n_nodes = mr * mc
    gap = 72.0  # room-to-room portal thickness >= 72u (wide for smooth continuous angle PBRS & motion)
    col_ws = [float(rng.choice((192, 224, 256))) for _ in range(mc)]
    row_hs = [float(rng.choice((192, 224, 256))) for _ in range(mr)]

    col_x0: list[float] = []
    cx = 0.0
    for w in col_ws:
        col_x0.append(cx)
        cx += w + gap
    row_y0: list[float] = []
    cy = 0.0
    for h in row_hs:
        row_y0.append(cy)
        cy += h + gap

    def nid(r: int, c: int) -> int:
        return r * mc + c

    def ncoords(u: int) -> tuple[int, int]:
        return divmod(u, mc)

    # Randomized spanning tree rooted at spawn_node
    spawn_node = nid(rng.randrange(mr), rng.randrange(mc))
    visited = {spawn_node}
    tree_adj: dict[int, list[int]] = {u: [] for u in range(n_nodes)}
    tree_children: dict[int, list[int]] = {u: [] for u in range(n_nodes)}
    frontier: list[tuple[int, int]] = []
    sr, sc = ncoords(spawn_node)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        if 0 <= sr + dr < mr and 0 <= sc + dc < mc:
            frontier.append((spawn_node, nid(sr + dr, sc + dc)))
    while frontier:
        u, v = frontier.pop(rng.randrange(len(frontier)))
        if v in visited:
            continue
        visited.add(v)
        tree_adj[u].append(v)
        tree_adj[v].append(u)
        tree_children[u].append(v)
        vr, vc = ncoords(v)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if 0 <= vr + dr < mr and 0 <= vc + dc < mc:
                nxt = nid(vr + dr, vc + dc)
                if nxt not in visited:
                    frontier.append((v, nxt))

    # Pick farthest node from spawn_node as exit_node
    parent: dict[int, int | None] = {spawn_node: None}
    dists: dict[int, int] = {spawn_node: 0}
    q = deque([spawn_node])
    while q:
        u = q.popleft()
        for v in tree_adj[u]:
            if v not in dists:
                dists[v] = dists[u] + 1
                parent[v] = u
                q.append(v)
    exit_node = max(dists.keys(), key=lambda u: (dists[u], rng.random()))
    path_to_exit: list[int] = []
    cur: int | None = exit_node
    while cur is not None:
        path_to_exit.append(cur)
        cur = parent[cur]
    path_to_exit.reverse()

    # Subtree helper
    def _subtree(root: int) -> set[int]:
        out: set[int] = set()
        dq = deque([root])
        while dq:
            x = dq.popleft()
            if x not in out:
                out.add(x)
                dq.extend(tree_children[x])
        return out

    # Assign 2 or 3 progression locks along path_to_exit
    keys = list(_KEY_SPECS)
    rng.shuffle(keys)
    n_edges = len(path_to_exit) - 1
    n_locks = min(3 if n_edges >= 4 else 2, n_edges)
    lock_edge_map: dict[tuple[int, int], str] = {}
    zone_id: dict[int, int] = {u: 0 for u in range(n_nodes)}
    key_places: list[tuple[int, int]] = []  # (doomednum, node)

    if n_locks >= 1:
        lock_indices = list(range(n_edges - n_locks, n_edges))
        for z_idx, ei in enumerate(lock_indices, start=1):
            for node in _subtree(path_to_exit[ei + 1]):
                zone_id[node] = z_idx
        zone_nodes: dict[int, list[int]] = {}
        for u, z in zone_id.items():
            if u != exit_node:
                zone_nodes.setdefault(z, []).append(u)
        z0_cands = [u for u in zone_nodes.get(0, [spawn_node]) if u != spawn_node] or [spawn_node]
        for li, ei in enumerate(lock_indices):
            u, v = path_to_exit[ei], path_to_exit[ei + 1]
            door_kind, card_id, skull_id = keys[li]
            lock_edge_map[(min(u, v), max(u, v))] = door_kind
            cands = zone_nodes.get(li) or z0_cands
            if li == 0 and len(cands) > 1 and spawn_node in cands:
                cands = [x for x in cands if x != spawn_node]
            k_id = skull_id if rng.random() < 0.55 else card_id
            key_places.append((k_id, rng.choice(cands)))

    # Edge catalog: tree edges + intra-zone loop edges (high beta1) + sight-only windows/grates
    tree_edges = {(min(u, v), max(u, v)) for u in range(n_nodes) for v in tree_adj[u]}
    all_portals: dict[tuple[int, int], str] = {}
    for ek in tree_edges:
        if ek in lock_edge_map:
            all_portals[ek] = lock_edge_map[ek]
        elif rng.random() < 0.45:
            all_portals[ek] = "door"
        else:
            all_portals[ek] = "open"

    # Collect intra-zone non-tree edges for cycles & windows/grates
    same_zone_cands: list[tuple[int, int]] = []
    other_cands: list[tuple[int, int]] = []
    for rr in range(mr):
        for cc in range(mc):
            u = nid(rr, cc)
            for dr, dc in ((0, 1), (1, 0)):
                if rr + dr < mr and cc + dc < mc:
                    v = nid(rr + dr, cc + dc)
                    ek = (min(u, v), max(u, v))
                    if ek not in tree_edges:
                        if zone_id[u] == zone_id[v]:
                            same_zone_cands.append(ek)
                        else:
                            other_cands.append(ek)
    rng.shuffle(same_zone_cands)
    rng.shuffle(other_cands)
    loop_count = min(len(same_zone_cands), rng.randint(2, max(2, len(same_zone_cands))))
    loop_edges = same_zone_cands[:loop_count]
    for ek in loop_edges:
        all_portals[ek] = "door" if rng.random() < 0.30 else "open"

    # Add windows / grates on unused adjacent pairs for perceptual sight != move!
    sight_cands = same_zone_cands[loop_count:] + other_cands
    for idx_s, ek in enumerate(sight_cands[: rng.randint(1, 3)]):
        all_portals[ek] = "window" if idx_s % 2 == 0 else "grate"

    # Assign zone heights (0 or 48u with safe 24u stair step OR one-way 48u drop-off on loop edge)
    comp_id = list(range(n_nodes))

    def _find(x: int) -> int:
        while comp_id[x] != x:
            comp_id[x] = comp_id[comp_id[x]]
            x = comp_id[x]
        return x

    for ek, ptype in all_portals.items():
        if ptype in ("open", "door"):
            # Separate components across height-steppable connectors when desired; merge across locks to share base or allow stairs
            pass
    # Alternate elevation on some zones/nodes cleanly
    room_hf: dict[int, int] = {}
    for u in range(n_nodes):
        room_hf[u] = 48 if (zone_id[u] % 2 == 1 and rng.random() < 0.65) else 0

    # Track split coordinates on every room's 4 walls so stitched portals match vertex-by-vertex
    splits_n: dict[int, list[float]] = {u: [] for u in range(n_nodes)}
    splits_s: dict[int, list[float]] = {u: [] for u in range(n_nodes)}
    splits_e: dict[int, list[float]] = {u: [] for u in range(n_nodes)}
    splits_w: dict[int, list[float]] = {u: [] for u in range(n_nodes)}

    portal_sectors: list[PolygonSector] = []
    for (u, v), ptype in sorted(all_portals.items()):
        ur, uc = ncoords(u)
        vr, vc = ncoords(v)
        pw = 96.0
        if ur == vr:
            # Horizontal neighbors: u left (uc), v right (uc+1)
            y_base = row_y0[ur]
            h_avail = row_hs[ur]
            p_lo = round(y_base + (h_avail - pw) * 0.5, 1)
            p_hi = round(p_lo + pw, 1)
            x_a = round(col_x0[uc] + col_ws[uc], 1)
            x_b = round(col_x0[vc], 1)
            splits_e[u].extend([p_lo, p_hi])
            splits_w[v].extend([p_lo, p_hi])
            coords = (x_a, p_lo, x_b, p_hi)
            grate_edge = ((x_a, p_lo), (x_a, p_hi))
        else:
            # Vertical neighbors: u bottom (ur), v top (ur+1)
            x_base = col_x0[uc]
            w_avail = col_ws[uc]
            p_lo = round(x_base + (w_avail - pw) * 0.5, 1)
            p_hi = round(p_lo + pw, 1)
            y_a = round(row_y0[ur] + row_hs[ur], 1)
            y_b = round(row_y0[vr], 1)
            splits_n[u].extend([p_lo, p_hi])
            splits_s[v].extend([p_lo, p_hi])
            coords = (p_lo, y_a, p_hi, y_b)
            grate_edge = ((p_lo, y_a), (p_hi, y_a))

        x0, y0, x1, y1 = coords
        min_hf = min(room_hf[u], room_hf[v])
        max_hf = max(room_hf[u], room_hf[v])
        if ptype == "window":
            portal_sectors.append(
                _rect_splits(f"win_{u}_{v}", x0, y0, x1, y1, wall_texture="GRAY1", height_floor=max_hf + 32, height_ceiling=max_hf + 72)
            )
        elif ptype == "grate":
            portal_sectors.append(
                _rect_splits(f"grate_{u}_{v}", x0, y0, x1, y1, wall_texture="METAL1", height_floor=min_hf, height_ceiling=max_hf + 192, blocking_edges=(grate_edge,))
            )
        elif ptype in ("door", "door_red", "door_blue", "door_yellow"):
            # Keep door frame floor at midpoint if step-diff else min_hf (doors open ceiling; step height <= 24u if needed)
            d_hf = (room_hf[u] + room_hf[v]) // 2 if abs(room_hf[u] - room_hf[v]) > 24 else min_hf
            portal_sectors.append(
                _rect_splits(f"door_{u}_{v}", x0, y0, x1, y1, kind=ptype, height_floor=d_hf, height_ceiling=max_hf + 192)
            )
        else:
            # Open passage: either 24u stair bridge or one-way dropoff on loop edge!
            is_dropoff = ((u, v) in loop_edges) and (abs(room_hf[u] - room_hf[v]) > 24) and (rng.random() < 0.65)
            p_hf = min_hf if is_dropoff else ((room_hf[u] + room_hf[v]) // 2 if abs(room_hf[u] - room_hf[v]) > 24 else min_hf)
            portal_sectors.append(
                _rect_splits(f"port_{u}_{v}", x0, y0, x1, y1, wall_texture="STONE2", height_floor=p_hf, height_ceiling=max_hf + 192)
            )

    # Optional wall exit switch on exterior wall of exit_node that doesn't touch any portal
    use_switch_exit = rng.random() < 0.40
    nukage_node = rng.choice([u for u in range(n_nodes) if u not in (spawn_node, exit_node)])
    use_nukage = rng.random() < 0.55

    room_sectors: list[PolygonSector] = []
    for u in range(n_nodes):
        ur, uc = ncoords(u)
        x0, x1 = round(col_x0[uc], 1), round(col_x0[uc] + col_ws[uc], 1)
        y0, y1 = round(row_y0[ur], 1), round(row_y0[ur] + row_hs[ur], 1)
        wall_t, flr_t, ceil_t, _ = _theme_walls(rng)

        holes: tuple[PolygonHole, ...] = ()
        if u not in (spawn_node, exit_node) and min(x1 - x0, y1 - y0) >= 224.0 and rng.random() < 0.55:
            cx_r, cy_r = (x0 + x1) * 0.5, (y0 + y1) * 0.5
            holes = (
                PolygonHole(
                    vertices=((cx_r, cy_r - 36.0), (cx_r + 36.0, cy_r), (cx_r, cy_r + 36.0), (cx_r - 36.0, cy_r)),
                    wall_texture="METAL1",
                ),
            )

        is_ex = u == exit_node
        ex_sw = None
        kind = "room"
        if is_ex:
            if use_switch_exit and not splits_n[u]:
                ex_sw = ((x0, y1), (x1, y1))
            elif use_switch_exit and not splits_s[u]:
                ex_sw = ((x0, y0), (x1, y0))
            elif use_switch_exit and not splits_e[u]:
                ex_sw = ((x1, y0), (x1, y1))
            elif use_switch_exit and not splits_w[u]:
                ex_sw = ((x0, y0), (x0, y1))
            else:
                kind = "exit"

        sec_sp = 71 if (use_nukage and u == nukage_node) else 0
        if sec_sp:
            flr_t = "NUKAGE1"

        room_sectors.append(
            _rect_splits(
                f"room_{u}",
                x0,
                y0,
                x1,
                y1,
                south_xs=splits_s[u],
                north_xs=splits_n[u],
                west_ys=splits_w[u],
                east_ys=splits_e[u],
                kind=kind,
                wall_texture=wall_t,
                floor_texture=flr_t,
                ceiling_texture=ceil_t,
                height_floor=room_hf[u],
                height_ceiling=max(room_hf[u] + 144, 192),
                holes=holes,
                exit_switch=ex_sw,
                sector_special=sec_sp,
            )
        )

    all_sectors = (*room_sectors, *portal_sectors)

    def _room_pt(u: int, ox: float = 0.0, oy: float = 0.0) -> tuple[float, float]:
        ur, uc = ncoords(u)
        return (round(col_x0[uc] + col_ws[uc] * 0.5 + ox, 1), round(row_y0[ur] + row_hs[ur] * 0.5 + oy, 1))

    # Avoid central hole radius (+/-36u) when placing items in rooms that got a hole
    has_hole = {u: bool(room_sectors[u].holes) for u in range(n_nodes)}
    spawn_xy = _room_pt(spawn_node)
    exit_xy = _room_pt(exit_node)
    p_ang = _pick_bearing(spawn_xy, exit_xy, rng, seed)
    wpn, ammo1, ammo2 = rng.choice(_WEAPON_AMMO_PAIRS)
    m_ang = rng.randrange(0, 360, 45)

    things: list[ThingSpec] = [
        ThingSpec(spawn_xy[0], spawn_xy[1], 1, p_ang),
        ThingSpec(spawn_xy[0] + 48.0, spawn_xy[1] + 40.0, wpn, 0),
        ThingSpec(spawn_xy[0] - 48.0, spawn_xy[1] - 40.0, ammo1, 0),
    ]
    if use_nukage:
        rad_xy = _room_pt(spawn_node, 48.0, -40.0)
        things.append(ThingSpec(rad_xy[0], rad_xy[1], 2025, 0))

    for idx_k, (k_id, k_node) in enumerate(key_places):
        off = (-52.0 if has_hole[k_node] else 0.0, 52.0 if (idx_k % 2 == 0 and has_hole[k_node]) else 0.0)
        kx, ky = _room_pt(k_node, off[0], off[1])
        things.append(ThingSpec(kx, ky, k_id, 0))

    other_nodes = [u for u in range(n_nodes) if u != spawn_node]
    rng.shuffle(other_nodes)
    for pu_id, node in zip((2019, 2012, ammo2, 2035), other_nodes):
        ox, oy = (56.0, -56.0) if has_hole[node] else (40.0, -32.0)
        px, py = _room_pt(node, ox, oy)
        things.append(ThingSpec(px, py, pu_id, 0))

    n_mon = rng.randint(8, 12)
    for mi in range(n_mon):
        u = other_nodes[mi % len(other_nodes)]
        ox = rng.choice((-60.0, 60.0)) if has_hole[u] else rng.choice((-48.0, 48.0))
        oy = rng.choice((-60.0, 60.0)) if has_hole[u] else rng.choice((-48.0, 48.0))
        mx, my = _room_pt(u, ox, oy)
        things.append(ThingSpec(mx, my, rng.choice(_HEAVY_MONSTERS[:10]), m_ang))

    rot_k = rng.randrange(4)
    flip_x = bool(rng.randrange(2))
    angle_deg = _sample_continuous_rot(rng)
    rot_s, rot_t = transform_polygon_map(all_sectors, things, rot_k=rot_k, flip_x=flip_x, angle_deg=angle_deg)
    return rot_s, rot_t, rng.choice((0, 60, 100, 150, 200))


def _build_scarce_crossfire_gauntlet(
    rng: random.Random, seed: int
) -> tuple[tuple[PolygonSector, ...], tuple[ThingSpec, ...], int]:
    """Wide >1150u RAY_NONE crossfire hall + pillar cover + low-window supply flank + 2 locks + scarce ammo."""
    keys = list(_KEY_SPECS)
    rng.shuffle(keys)
    (lock1, c1, s1), (lock2, c2, s2), _ = keys
    k1 = s1 if rng.random() < 0.6 else c1
    k2 = s2 if rng.random() < 0.6 else c2
    use_nukage = rng.random() < 0.45
    use_switch_exit = rng.random() < 0.45
    scarce = rng.random() < 0.55
    wpn, ammo1, _ = rng.choice(_WEAPON_AMMO_PAIRS)
    m_ang = rng.randrange(0, 360, 45)

    w_a = float(rng.choice((1184, 1280, 1376)))
    h_a = float(rng.choice((896, 1024)))
    flank_h = 224.0
    x_win0, x_win1 = 360.0, 560.0
    x_door0, x_door1 = w_a - 240.0, w_a - 112.0
    y_ex0, y_ex1 = h_a * 0.5 - 80.0, h_a * 0.5 + 80.0

    # Pillars inside wide crossfire arena
    cx1, cy1 = w_a * 0.32, h_a * 0.30
    cx2, cy2 = w_a * 0.32, h_a * 0.70
    cx3, cy3 = w_a * 0.62, h_a * 0.50
    cx4, cy4 = w_a * 0.78, h_a * 0.28
    pr = 56.0
    holes = tuple(
        PolygonHole(
            vertices=((x, y - pr), (x + pr, y), (x, y + pr), (x - pr, y)),
            wall_texture=tex,
        )
        for (x, y), tex in (
            ((cx1, cy1), "STONE2"),
            ((cx2, cy2), "METAL1"),
            ((cx3, cy3), "BRICK1"),
            ((cx4, cy4), "STONE2"),
        )
    )

    far_wall = ((w_a + 328.0, y_ex0), (w_a + 328.0, y_ex1))
    sw_wall = ((x_win0 - 64.0, h_a + 64.0 + flank_h), (w_a, h_a + 64.0 + flank_h))

    sectors = (
        _rect_splits(
            "wide_arena",
            0.0,
            0.0,
            w_a,
            h_a,
            north_xs=(x_win0, x_win1, x_door0, x_door1),
            east_ys=(y_ex0, y_ex1),
            wall_texture="BRICK1",
            floor_texture="NUKAGE1" if use_nukage else "FLOOR7_1",
            height_ceiling=224,
            holes=holes,
            sector_special=71 if use_nukage else 0,
        ),
        _rect_splits("flank_win", x_win0, h_a, x_win1, h_a + 64.0, wall_texture="GRAY1", height_floor=0, height_ceiling=40),
        _rect_splits("flank_door", x_door0, h_a, x_door1, h_a + 64.0, kind="door", height_ceiling=224),
        _rect_splits(
            "flank_armory",
            x_win0 - 64.0,
            h_a + 64.0,
            w_a,
            h_a + 64.0 + flank_h,
            south_xs=(x_win0, x_win1, x_door0, x_door1),
            wall_texture="METAL1",
            height_ceiling=224,
            door_switch=(sw_wall[0], sw_wall[1], 1),
        ),
        _rect_splits("east_lock1", w_a, y_ex0, w_a + 56.0, y_ex1, kind=lock1, height_ceiling=224),
        _rect_splits("east_rdoor", w_a + 56.0, y_ex0, w_a + 184.0, y_ex1, wall_texture="STONE2", height_ceiling=224),
        _rect_splits("east_lock2", w_a + 184.0, y_ex0, w_a + 240.0, y_ex1, kind=lock2, height_ceiling=224),
        _rect_splits(
            "east_exit",
            w_a + 240.0,
            y_ex0,
            w_a + 328.0,
            y_ex1,
            kind="room" if use_switch_exit else "exit",
            height_ceiling=224,
            exit_switch=far_wall if use_switch_exit else None,
        ),
    )

    # Note: put remote_door + key2 inside south alcove or keep remote_door cleanly stitched at east_rdoor?
    # Let's make east_rdoor -> kind="remote_door", sector_id=1 on east_lock1's next hop, or swap east_lock1 & flank door!
    # Wait: if flank_armory has door_switch(tag=1) and east_lock1 is kind="remote_door" (with k2 in flank_armory?
    # Better: put k1 in south-east arena corner guarded by bosses, lock1 gates flank_door, flank_armory holds k2!
    # Yes! Then Player fights/sneaks across wide crossfire arena -> gets Key1 -> unlocks Side Armory -> gets Key2 + Weapon/Health -> crosses back -> unlocks East Lock2 -> Exit!)
    sectors = (
        sectors[0],
        sectors[1],
        _rect_splits("flank_lock1", x_door0, h_a, x_door1, h_a + 64.0, kind=lock1, height_ceiling=224),
        _rect_splits(
            "flank_armory",
            x_win0 - 64.0,
            h_a + 64.0,
            w_a,
            h_a + 64.0 + flank_h,
            south_xs=(x_win0, x_win1, x_door0, x_door1),
            wall_texture="METAL1",
            height_ceiling=224,
            door_switch=(sw_wall[0], sw_wall[1], 1),
        ),
        _rect_splits("east_rdoor", w_a, y_ex0, w_a + 56.0, y_ex1, kind="remote_door", sector_id=1, height_ceiling=224),
        _rect_splits("east_hall", w_a + 56.0, y_ex0, w_a + 184.0, y_ex1, wall_texture="STONE2", height_ceiling=224),
        _rect_splits("east_lock2", w_a + 184.0, y_ex0, w_a + 240.0, y_ex1, kind=lock2, height_ceiling=224),
        sectors[7],
    )

    spawn_xy = (112.0, h_a * 0.5)
    exit_xy = (w_a + 284.0, h_a * 0.5)
    p_ang = _pick_bearing(spawn_xy, exit_xy, rng, seed)
    things: list[ThingSpec] = [
        ThingSpec(spawn_xy[0], spawn_xy[1], 1, p_ang),
        ThingSpec(160.0, h_a * 0.5 + 80.0, 2001 if scarce else wpn, 0),
        ThingSpec(w_a * 0.85, h_a * 0.14, k1, 0),
        ThingSpec(x_win0 + 64.0, h_a + 64.0 + flank_h * 0.5, k2, 0),
        ThingSpec(w_a * 0.72, h_a + 64.0 + flank_h * 0.5, wpn, 0),
        ThingSpec(w_a * 0.58, h_a + 64.0 + flank_h * 0.5, 2019 if not scarce else 2012, 0),
    ]
    if use_nukage:
        things.append(ThingSpec(160.0, h_a * 0.5 - 80.0, 2025, 0))
    if not scarce:
        things.append(ThingSpec(200.0, h_a * 0.5, ammo1, 0))
        things.append(ThingSpec(w_a * 0.80, h_a + 64.0 + flank_h * 0.5, ammo1, 0))

    enemy_spots = [
        (w_a * 0.28, h_a * 0.14),
        (w_a * 0.28, h_a * 0.86),
        (w_a * 0.48, h_a * 0.20),
        (w_a * 0.48, h_a * 0.80),
        (w_a * 0.65, h_a * 0.16),
        (w_a * 0.65, h_a * 0.84),
        (w_a * 0.86, h_a * 0.22),
        (w_a * 0.86, h_a * 0.78),
        (w_a * 0.82, h_a * 0.50),
        (w_a * 0.52, h_a + 64.0 + flank_h * 0.5),
        (w_a + 120.0, h_a * 0.5),
    ]
    for idx_e, (ex, ey) in enumerate(enemy_spots):
        if idx_e < 10:
            things.append(ThingSpec(ex, ey, rng.choice(_HEAVY_MONSTERS[:10]), m_ang))
    # Barrels near cover pillars
    things.extend([
        ThingSpec(w_a * 0.42, h_a * 0.38, 2035, 0),
        ThingSpec(w_a * 0.70, h_a * 0.62, 2035, 0),
    ])

    rot_k = rng.randrange(4)
    flip_x = bool(rng.randrange(2))
    angle_deg = _sample_continuous_rot(rng)
    rot_s, rot_t = transform_polygon_map(sectors, things, rot_k=rot_k, flip_x=flip_x, angle_deg=angle_deg)
    return rot_s, rot_t, rng.choice((0, 60, 120))


def _build_nukage_compound_fortress(
    rng: random.Random, seed: int
) -> tuple[tuple[PolygonSector, ...], tuple[ThingSpec, ...], int]:
    """Damaging nukage gorge (hf=-16, special=71) with narrow dry catwalks, RadSuit, 2-3 locks & bosses."""
    keys = list(_KEY_SPECS)
    rng.shuffle(keys)
    three_lock = rng.random() < 0.60
    use_switch_exit = rng.random() < 0.45
    (lock1, c1, s1), (lock2, c2, s2), (lock3, c3, s3) = keys[0], keys[1], keys[2]
    k1, k2, k3 = s1, s2, s3  # emphasize skull keys!
    wpn, ammo1, _ = rng.choice(_WEAPON_AMMO_PAIRS)
    m_ang = rng.randrange(0, 360, 45)

    # Central basin broken cleanly into nukage halves + horizontal dry bridge!
    # West dry ledge (0..192, 0..512) -> South Nukage (192..512, 0..208) + Dry Bridge (192..512, 208..304) + North Nukage (192..512, 304..512) -> East dry ledge (512..704, 0..512)
    sectors: list[PolygonSector] = [
        _rect_splits("w_bank", 0.0, 0.0, 192.0, 512.0, south_xs=(), north_xs=(48.0, 144.0), east_ys=(208.0, 304.0), wall_texture="STONE2", height_floor=0, height_ceiling=200),
        _rect_splits("nuk_s", 192.0, 0.0, 512.0, 208.0, wall_texture="BROWN1", floor_texture="NUKAGE1", height_floor=-16, height_ceiling=200, sector_special=71),
        _rect_splits("dry_bridge", 192.0, 208.0, 512.0, 304.0, wall_texture="METAL1", floor_texture="FLOOR5_1", height_floor=0, height_ceiling=200),
        _rect_splits("nuk_n", 192.0, 304.0, 512.0, 512.0, wall_texture="BROWN1", floor_texture="NUKAGE1", height_floor=-16, height_ceiling=200, sector_special=71),
        _rect_splits(
            "e_bank",
            512.0,
            0.0,
            704.0,
            512.0,
            north_xs=(560.0, 656.0),
            south_xs=(560.0, 656.0),
            west_ys=(208.0, 304.0),
            wall_texture="STONE2",
            height_floor=0,
            height_ceiling=200,
        ),
        _rect_splits("s_e_lock1", 560.0, -56.0, 656.0, 0.0, kind=lock1, height_floor=0, height_ceiling=200),
        _rect_splits("s_e_vault", 512.0, -240.0, 704.0, -56.0, north_xs=(560.0, 656.0), wall_texture="BRICK1", height_floor=0, height_ceiling=200),
        _rect_splits("n_w_door", 48.0, 512.0, 144.0, 568.0, kind=lock2 if three_lock else "door", height_floor=0, height_ceiling=200),
        _rect_splits("n_w_vault", 0.0, 568.0, 224.0, 768.0, south_xs=(48.0, 144.0), wall_texture="GRAY1", height_floor=0, height_ceiling=200),
        _rect_splits("n_e_lock_ex", 560.0, 512.0, 656.0, 568.0, kind=lock3 if three_lock else lock2, height_floor=0, height_ceiling=200),
        _rect_splits("n_e_sanctum", 512.0, 568.0, 704.0, 720.0, south_xs=(560.0, 656.0), west_ys=(624.0,), wall_texture="METAL1", height_floor=0, height_ceiling=200),
        _rect_splits(
            "n_e_exit",
            432.0,
            624.0,
            512.0,
            720.0,
            kind="room" if use_switch_exit else "exit",
            height_floor=0,
            height_ceiling=200,
            exit_switch=((432.0, 624.0), (432.0, 720.0)) if use_switch_exit else None,
        ),
    ]

    spawn_xy = (96.0, 256.0)
    exit_xy = (472.0, 672.0)
    p_ang = _pick_bearing(spawn_xy, exit_xy, rng, seed)
    things: list[ThingSpec] = [
        ThingSpec(spawn_xy[0], spawn_xy[1], 1, p_ang),
        ThingSpec(96.0, 192.0, wpn, 0),
        ThingSpec(96.0, 320.0, 2025, 0),   # RadSuit on west bank
        ThingSpec(144.0, 128.0, ammo1, 0),
        ThingSpec(352.0, 96.0, k1, 0),     # Key1 in south nukage!
        ThingSpec(608.0, -148.0, k2, 0),   # Key2 behind lock1!
        ThingSpec(352.0, 416.0, 2025, 0),  # Second RadSuit on north nukage
        ThingSpec(608.0, 128.0, 2012, 0),
    ]
    if three_lock:
        things.append(ThingSpec(112.0, 668.0, k3, 0))
        things.append(ThingSpec(144.0, 640.0, rng.choice(_HEAVY_MONSTERS[:10]), m_ang))

    enemy_spots = [
        (352.0, 256.0),
        (440.0, 96.0),
        (264.0, 416.0),
        (608.0, 256.0),
        (608.0, 400.0),
        (560.0, -148.0),
        (656.0, -160.0),
        (72.0, 660.0),
        (608.0, 644.0),
        (560.0, 660.0),
    ]
    rng.shuffle(enemy_spots)
    for ex, ey in enemy_spots[: rng.randint(8, 10)]:
        things.append(ThingSpec(ex, ey, rng.choice(_HEAVY_MONSTERS[:10]), m_ang))
    things.append(ThingSpec(608.0, 208.0, 2035, 0))

    rot_k = rng.randrange(4)
    flip_x = bool(rng.randrange(2))
    angle_deg = _sample_continuous_rot(rng)
    rot_s, rot_t = transform_polygon_map(sectors, things, rot_k=rot_k, flip_x=flip_x, angle_deg=angle_deg)
    return rot_s, rot_t, rng.choice((0, 60, 100, 150))


def _build_grate_window_backtrack(
    rng: random.Random, seed: int
) -> tuple[tuple[PolygonSector, ...], tuple[ThingSpec, ...], int]:
    """Dual sight!=move trap: low window lures Key1 on East, impassable grate taunts Lock2+Exit on West."""
    keys = list(_KEY_SPECS)
    rng.shuffle(keys)
    (lock1, c1, s1), (lock2, c2, s2), _ = keys
    k1 = s1 if rng.random() < 0.55 else c1
    k2 = s2 if rng.random() < 0.55 else c2
    use_nukage = rng.random() < 0.50
    use_switch_exit = rng.random() < 0.45
    wpn, ammo1, _ = rng.choice(_WEAPON_AMMO_PAIRS)
    m_ang = rng.randrange(0, 360, 45)

    # Central spawn room [0..240, 0..240]
    # East: window [240..312, 64..176] -> east_key_room [312..544, 0..240] (with Key1)
    # Bypass North detour around window -> nukage + stairs + enemies!
    # West: grate [-72..0, 64..176] (impassable MIDBARS3) -> exit_sanctum [-320..-72, 0..240]
    # South detour around grate: door -> w_vault (Key2 behind Lock1!) -> Lock2 -> exit_sanctum!
    far_ex_wall = ((-416.0, 64.0), (-416.0, 176.0))
    sectors = (
        _rect_splits(
            "spawn",
            0.0,
            0.0,
            240.0,
            240.0,
            south_xs=(64.0, 176.0),
            north_xs=(64.0, 176.0),
            west_ys=(64.0, 176.0),
            east_ys=(64.0, 176.0),
            wall_texture="STONE2",
            height_ceiling=200,
            blocking_edges=(((0.0, 64.0), (0.0, 176.0)),),
        ),
        _rect_splits("e_win", 240.0, 64.0, 312.0, 176.0, wall_texture="BRICK1", height_floor=0, height_ceiling=40),
        _rect_splits("e_key", 312.0, 0.0, 544.0, 240.0, north_xs=(360.0, 480.0), west_ys=(64.0, 176.0), wall_texture="BRICK1", height_ceiling=200),
        _rect_splits("n_det_w", 64.0, 240.0, 176.0, 416.0, east_ys=(304.0,), wall_texture="BROWN1", height_ceiling=200),
        _rect_splits(
            "n_det_mid",
            176.0,
            304.0,
            360.0,
            416.0,
            floor_texture="NUKAGE1" if use_nukage else "FLOOR4_8",
            wall_texture="BROWN1",
            height_ceiling=200,
            sector_special=71 if use_nukage else 0,
        ),
        _rect_splits("n_det_e", 360.0, 240.0, 480.0, 416.0, west_ys=(304.0,), wall_texture="STONE2", height_ceiling=200),
        _rect_splits("w_grate", -72.0, 64.0, 0.0, 176.0, wall_texture="METAL1", height_ceiling=200),
        _rect_splits("s_det_e", 64.0, -192.0, 176.0, 0.0, west_ys=(-192.0, -80.0), wall_texture="GRAY1", height_ceiling=200),
        _rect_splits("s_lock1", -72.0, -192.0, 64.0, -80.0, kind=lock1, height_ceiling=200),
        _rect_splits("sw_vault", -320.0, -192.0, -72.0, -80.0, north_xs=(-240.0, -128.0), wall_texture="METAL1", height_ceiling=200),
        _rect_splits("w_lock2", -240.0, -80.0, -128.0, 0.0, kind=lock2, height_ceiling=200),
        _rect_splits(
            "w_sanctum",
            -320.0,
            0.0,
            -72.0,
            240.0,
            south_xs=(-240.0, -128.0),
            west_ys=(64.0, 176.0),
            east_ys=(64.0, 176.0),
            wall_texture="METAL1",
            height_ceiling=200,
        ),
        _rect_splits(
            "w_exit",
            -416.0,
            64.0,
            -320.0,
            176.0,
            kind="room" if use_switch_exit else "exit",
            height_ceiling=200,
            exit_switch=far_ex_wall if use_switch_exit else None,
        ),
    )

    spawn_xy = (120.0, 120.0)
    exit_xy = (-368.0, 120.0)
    p_ang = _pick_bearing(spawn_xy, exit_xy, rng, seed)
    things = [
        ThingSpec(spawn_xy[0], spawn_xy[1], 1, p_ang),
        ThingSpec(120.0, 184.0, wpn, 0),
        ThingSpec(184.0, 120.0, ammo1, 0),
        ThingSpec(424.0, 120.0, k1, 0),
        ThingSpec(-256.0, -136.0, k2, 0),
        ThingSpec(120.0, 280.0, 2025 if use_nukage else 2018, 0),
        ThingSpec(-184.0, 120.0, 2012, 0),
        # Monsters across both detours + ambush behind Lock2!
        ThingSpec(120.0, 360.0, rng.choice(_HEAVY_MONSTERS[:10]), m_ang),
        ThingSpec(420.0, 356.0, rng.choice(_HEAVY_MONSTERS[:10]), m_ang),
        ThingSpec(440.0, 184.0, rng.choice(_HEAVY_MONSTERS[:10]), m_ang),
        ThingSpec(440.0, 56.0, rng.choice(_HEAVY_MONSTERS[:10]), m_ang),
        ThingSpec(120.0, -136.0, rng.choice(_HEAVY_MONSTERS[:10]), m_ang),
        ThingSpec(-120.0, -136.0, rng.choice(_HEAVY_MONSTERS[:10]), m_ang),
        ThingSpec(-160.0, 180.0, rng.choice(_HEAVY_MONSTERS[:10]), m_ang),
        ThingSpec(-260.0, 60.0, rng.choice(_HEAVY_MONSTERS[:10]), m_ang),
        ThingSpec(-180.0, 120.0, rng.choice(_HEAVY_MONSTERS[:10]), m_ang),
    ]

    rot_k = rng.randrange(4)
    flip_x = bool(rng.randrange(2))
    angle_deg = _sample_continuous_rot(rng)
    rot_s, rot_t = transform_polygon_map(sectors, things, rot_k=rot_k, flip_x=flip_x, angle_deg=angle_deg)
    return rot_s, rot_t, rng.choice((0, 60, 100, 150))


def generate_extreme_wad(
    archetype: str,
    seed: int,
    out_dir: str | Path,
    *,
    split: str = "train",
    out_path: str | Path | None = None,
) -> MapSpec:
    """Synthesize and verify one Very-High-Difficulty / Nightmare continuous-angle WAD."""
    if archetype not in EXTREME_ARCHETYPES:
        raise ValueError(f"Unknown extreme archetype {archetype!r}; expected one of {EXTREME_ARCHETYPES}")

    target_path = Path(out_path) if out_path is not None else (Path(out_dir) / f"ext_{archetype}_s{seed}.wad")
    builders = {
        "decoy_multilock": _build_decoy_multilock,
        "vertical_switch_citadel": _build_vertical_switch_citadel,
        "aliased_labyrinth": _build_aliased_labyrinth,
        "scarce_crossfire_gauntlet": _build_scarce_crossfire_gauntlet,
        "nukage_compound_fortress": _build_nukage_compound_fortress,
        "grate_window_backtrack": _build_grate_window_backtrack,
    }
    builder = partial(build_progression, archetype) if archetype in WINGS else builders[archetype]
    last_err: tuple[str, ...] = ()
    for attempt in range(12):
        rng = _det_rng(archetype, int(seed), attempt)
        sectors, things, door_delay = builder(rng, int(seed) + attempt)
        ok, errors = verify_polygon_solvability(sectors, things, check_t_junctions=True)
        if not ok:
            last_err = errors
            continue
        all_xs = [v[0] for ps in sectors for v in ps.vertices]
        all_ys = [v[1] for ps in sectors for v in ps.vertices]
        span = max(max(all_xs) - min(all_xs), max(all_ys) - min(all_ys))
        if span > 2400.0:
            last_err = (f"span_too_large:{span:.1f}",)
            continue
        build_polygon_wad(target_path, sectors, things, door_close_delay=door_delay)
        return MapSpec(
            wad=str(target_path),
            map_name="MAP01",
            family="gauntlet",
            tier=4,
            split=split,
            episode_ticks=4200,
            seeds=(0, 1, 2),
        )
    raise RuntimeError(f"Failed extreme WAD generation ({archetype}, seed={seed}): {last_err}")
