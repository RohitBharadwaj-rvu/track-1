"""On-the-fly procedural WAD generator for Role 2 training variation (Role 3).

Generates fresh, statically verified, guaranteed-solvable variations across all 4 tiers
(`door_exit`, `combat`, `multi_room`, `key_return`, `gauntlet`) from an integer seed so the PPO
trainer can scale variation count on demand without bloating the repository.
"""

from __future__ import annotations

from pathlib import Path
import random

from training.interfaces import MapSpec
from training.wad_gen.dungeon import generate_dungeon_wad
from training.wad_gen.udmf import (
    THEME_NAMES,
    PolygonHole,
    PolygonSector,
    ThingSpec,
    build_grid_wad,
    build_polygon_wad,
    transform_polygon_map,
)
from training.wad_gen.verify import verify_grid_solvability, verify_polygon_solvability

_KEY_SPECS: tuple[tuple[str, int, int], ...] = (
    ("door_red", 13, 38),
    ("door_blue", 5, 40),
    ("door_yellow", 6, 39),
)

_COMBAT_MONSTER_IDS: tuple[int, ...] = (3004, 9, 3001, 3002, 65, 3005, 69, 3006, 58, 3003, 68, 67, 66)
_WEAPON_AMMO_PAIRS: tuple[tuple[int, int], ...] = (
    (2001, 2049),  # Shotgun + ShellBox
    (2002, 2048),  # Chaingun + ClipBox
    (82, 2049),    # SuperShotgun + ShellBox
    (2003, 2046),  # RocketLauncher + RocketBox
    (2004, 17),    # PlasmaRifle + CellPack
    (2005, 2048),  # Chainsaw + ClipBox
)


def _to_grid_str(buf: list[list[str]]) -> str:
    return "\n".join("".join(row) for row in buf)


def _transform_grid(grid: str, rng: random.Random) -> str:
    """Apply a random D4 symmetry (flips / 90-deg transpose) to eliminate spatial bias."""
    rows = [list(line) for line in grid.splitlines() if line]
    if rng.random() < 0.5:
        rows = [list(reversed(r)) for r in rows]
    if rng.random() < 0.5:
        rows = list(reversed(rows))
    if rng.random() < 0.5:
        rows = [list(col) for col in zip(*rows)]
    return _to_grid_str(rows)


def _rect(name: str, x0: float, y0: float, x1: float, y1: float, **kwargs) -> PolygonSector:
    xa, xb = (x0, x1) if x0 <= x1 else (x1, x0)
    ya, yb = (y0, y1) if y0 <= y1 else (y1, y0)
    return PolygonSector(name, ((xa, ya), (xa, yb), (xb, yb), (xb, ya)), **kwargs)


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
    q1 = (round(p1[0] + dx, 2), round(p1[1] + dy, 2))
    q2 = (round(p2[0] + dx, 2), round(p2[1] + dy, 2))
    return (p2, p1, q1, q2)


def _carve_rect(buf: list[list[str]], r0: int, c0: int, r1: int, c1: int, ch: str = ".") -> None:
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            buf[r][c] = ch


def _gen_door_exit_grid(rng: random.Random) -> tuple[str, int, int]:
    """Generate varied Tier 2 single-door layouts (doglegs, L-turns, pillar chambers, N/S/E/W doors)."""
    arch = rng.choice(["dogleg", "lturn", "pillar_chamber"])
    cell_size = rng.choice([64, 80])
    if arch == "dogleg":
        w = rng.randint(9, 13)
        buf = [["#"] * w for _ in range(7)]
        _carve_rect(buf, 1, 1, 2, w - 2)
        door_c = rng.randint(2, w - 3)
        buf[3][door_c] = "D"
        ex_c0 = max(1, door_c - rng.randint(1, 2))
        ex_c1 = min(w - 2, door_c + rng.randint(1, 2))
        _carve_rect(buf, 4, ex_c0, 5, ex_c1)
        if rng.random() < 0.5:
            buf[1][1] = "P"
            buf[5][ex_c1] = "E"
            angle = rng.choice([0, 270, 315])
        else:
            buf[5][ex_c0] = "P"
            buf[1][w - 2] = "E"
            angle = rng.choice([90, 0, 180])
    elif arch == "lturn":
        w = rng.randint(9, 11)
        buf = [["#"] * w for _ in range(7)]
        _carve_rect(buf, 1, 1, 2, w - 2)
        _carve_rect(buf, 4, 1, 5, w - 2)
        dc = rng.randint(2, w - 3)
        buf[3][dc] = "D"
        if rng.random() < 0.5:
            buf[4][1] = "P"
            buf[1][w - 2] = "E"
            buf[5][w - 2] = rng.choice([".", "h"])
            angle = rng.choice([0, 90, 180])
        else:
            buf[1][w - 2] = "P"
            buf[5][1] = "E"
            angle = rng.choice([180, 270])
    else:
        w = rng.randint(9, 13)
        buf = [["#"] * w for _ in range(8)]
        _carve_rect(buf, 1, 1, 2, w - 2)
        _carve_rect(buf, 4, 1, 6, w - 2)
        dc = rng.randint(2, w - 3)
        buf[3][dc] = "D"
        # Optional central pillar in spawn chamber
        pc = w // 2
        if pc != dc:
            buf[5][pc] = "#"
        buf[6][rng.choice([1, w - 2])] = "P"
        buf[1][rng.choice([1, w - 2])] = "E"
        buf[5][max(1, pc - 1)] = rng.choice([".", "h"])
        angle = rng.choice([0, 90, 180, 270])
    return _to_grid_str(buf), angle, cell_size


def _gen_combat_grid(rng: random.Random) -> tuple[str, int, int]:
    h = rng.choice([7, 9])
    w = rng.choice([11, 13, 15])
    buf = [["#"] * w for _ in range(h)]
    _carve_rect(buf, 1, 1, h - 2, w - 2, ".")

    # Place 2-4 symmetric or staggered cover pillars
    mid_r = h // 2
    for c in range(4, w - 4, 3):
        if rng.random() < 0.7:
            pr = rng.choice([2, h - 3])
            buf[pr][c] = "#"
    if rng.random() < 0.35 and w >= 13:
        buf[mid_r][w // 2] = "#"
        if h >= 9:
            buf[mid_r - 1][w // 2] = "#"

    # Player spawn (left or right) and exit on opposite side
    if rng.random() < 0.65:
        buf[mid_r][2] = "P"
        buf[mid_r][w - 2] = "E"
        e_lo, e_hi = w // 2, w - 2
        sup_c = 3
    else:
        buf[mid_r][w - 2] = "P"
        buf[mid_r][1] = "E"
        e_lo, e_hi = 2, w // 2
        sup_c = w - 3

    # Weapon & supplies near player (covers all weapon slots, ammo types, armor, powerups)
    wpn_ch, ammo_ch = rng.choice([("g", "e"), ("G", "a"), ("W", "e"), ("L", "o"), ("F", "u")])
    buf[max(1, mid_r - 1)][sup_c] = wpn_ch
    buf[min(h - 2, mid_r + 1)][sup_c] = rng.choice([ammo_ch, "H", "A", "M", "O"])

    # 2-4 enemies on opposite half (full monster roster incl. ChaingunGuy, Cacodemon, HellKnight, LostSoul, Spectre)
    enemy_pool = ["z", "s", "i", "m", "c", "k", "n", "l", "S"]
    num_enemies = rng.randint(2, 4)
    candidates = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(e_lo, e_hi + 1)
        if buf[r][c] == "." and not (r == mid_r and c in (1, w - 2))
    ]
    rng.shuffle(candidates)
    for r, c in candidates[:num_enemies]:
        buf[r][c] = rng.choice(enemy_pool)

    angle = rng.choice([0, 45, 90, 180, 270, 315])
    cell_size = rng.choice([64, 80])
    return _to_grid_str(buf), angle, cell_size


def _gen_multi_room_grid(rng: random.Random) -> tuple[str, int, int]:
    arch = rng.choice(["chain", "double_loop", "spiral"])
    if arch == "double_loop":
        w, h = 15, 9
        buf = [["#"] * w for _ in range(h)]
        # Left loop, center corridor, right wing + dead end
        _carve_rect(buf, 1, 1, 3, 3)
        _carve_rect(buf, 5, 1, 7, 3)
        _carve_rect(buf, 1, 5, 7, 9)
        buf[2][6] = "#"
        buf[3][6] = "#"
        buf[5][6] = "#"
        buf[6][6] = "#"
        _carve_rect(buf, 1, 11, 3, 13)
        _carve_rect(buf, 5, 11, 7, 13)
        buf[4][2] = "." if rng.random() < 0.7 else "D"
        buf[rng.choice([2, 6])][4] = "D"
        buf[rng.choice([2, 6])][10] = "D"
        buf[4][12] = "D" if rng.random() < 0.5 else "."
        buf[2][2] = "P"
        buf[6][12] = "E"
        buf[2][12] = rng.choice(["h", "a", "z"])
        buf[4][8] = rng.choice([".", "z", "i"])
        return _to_grid_str(buf), rng.choice([0, 90, 270]), 64

    if arch == "spiral":
        w, h = 14, 9
        buf = [["#"] * w for _ in range(h)]
        _carve_rect(buf, 1, 1, 2, 4)
        _carve_rect(buf, 1, 6, 3, 12)
        _carve_rect(buf, 4, 10, 7, 12)
        _carve_rect(buf, 6, 1, 7, 8)
        _carve_rect(buf, 4, 1, 5, 4)
        buf[2][5] = "D"
        buf[3][11] = "D"
        buf[6][9] = "D"
        if rng.random() < 0.5:
            buf[5][2] = "D"
            buf[4][3] = "h"
        buf[1][2] = "P"
        buf[4][6] = "."
        _carve_rect(buf, 4, 6, 4, 8)
        buf[4][7] = "E"
        buf[5][7] = "."
        buf[6][7] = "."
        buf[2][9] = rng.choice([".", "z"])
        return _to_grid_str(buf), rng.choice([0, 270]), 64

    # 3 rooms horizontally + optional dead-end branch north or south
    room_w = rng.choice([3, 4])
    room_h = rng.choice([3, 4])
    has_branch = rng.random() < 0.85
    branch_north = rng.choice([True, False])

    total_w = 1 + room_w + 1 + room_w + 1 + room_w + 1
    extra_h = (room_h + 1) if has_branch else 0
    total_h = 1 + room_h + 1 + extra_h
    buf = [["#"] * total_w for _ in range(total_h)]

    base_r = (room_h + 1) if (has_branch and branch_north) else 1
    c1 = 1
    c2 = c1 + room_w + 1
    c3 = c2 + room_w + 1

    _carve_rect(buf, base_r, c1, base_r + room_h - 1, c1 + room_w - 1)
    _carve_rect(buf, base_r, c2, base_r + room_h - 1, c2 + room_w - 1)
    _carve_rect(buf, base_r, c3, base_r + room_h - 1, c3 + room_w - 1)

    d1_r = rng.randint(base_r, base_r + room_h - 1)
    d2_r = rng.randint(base_r, base_r + room_h - 1)
    buf[d1_r][c1 + room_w] = "D"
    buf[d2_r][c2 + room_w] = "D"

    if has_branch:
        br_r = 1 if branch_north else (base_r + room_h + 1)
        _carve_rect(buf, br_r, c2, br_r + room_h - 1, c2 + room_w - 1)
        door_r = base_r - 1 if branch_north else base_r + room_h
        door_c = rng.randint(c2, c2 + room_w - 1)
        buf[door_r][door_c] = "D"
        buf[br_r + room_h // 2][c2 + room_w // 2] = rng.choice(["h", "a", "z"])

    spawn_left = rng.choice([True, False])
    if spawn_left:
        buf[base_r + room_h // 2][c1 + 1] = "P"
        buf[base_r + room_h // 2][c3 + room_w - 1] = "E"
        angle = rng.choice([0, 90, 270])
    else:
        buf[base_r + room_h // 2][c3 + room_w - 1] = "P"
        buf[base_r + room_h // 2][c1] = "E"
        angle = rng.choice([180, 90, 270])

    if buf[base_r + room_h // 2][c2 + room_w // 2] == ".":
        buf[base_r + room_h // 2][c2 + room_w // 2] = rng.choice(["z", "i"])

    return _to_grid_str(buf), angle, 64


def _gen_key_return_grid(
    rng: random.Random,
    *,
    return_distance: int | None = None,
    with_combat: bool = False,
) -> tuple[str, int, int]:
    """Generate parameterized key-and-return or mirrored H-corridor layouts."""
    lock_char, key_char = rng.choice([("R", "r"), ("B", "b"), ("Y", "y")])
    if return_distance is None and rng.random() < 0.35:
        # Mirrored symmetric H-corridor with repeated doors
        w, h = 17, 11
        buf = [["#"] * w for _ in range(h)]
        _carve_rect(buf, 1, 1, 3, 3)
        _carve_rect(buf, 1, 5, 3, 11)
        _carve_rect(buf, 1, 13, 3, 15)
        _carve_rect(buf, 4, 6, 6, 10)
        _carve_rect(buf, 7, 1, 9, 3)
        _carve_rect(buf, 7, 5, 9, 11)
        _carve_rect(buf, 7, 13, 9, 15)

        wings = [(2, 2), (2, 14), (8, 2), (8, 14)]
        door_pos = [(2, 4), (2, 12), (8, 4), (8, 12)]
        key_idx = rng.randrange(4)
        exit_idx = (key_idx + rng.choice([1, 2, 3])) % 4
        for idx, (dr, dc) in enumerate(door_pos):
            buf[dr][dc] = lock_char if idx == exit_idx else "D"
        kr, kc = wings[key_idx]
        er, ec = wings[exit_idx]
        buf[kr][kc] = key_char
        buf[er][ec] = "E"
        buf[5][8] = "P"
        for idx, (wr, wc) in enumerate(wings):
            if idx not in (key_idx, exit_idx):
                buf[wr][wc] = rng.choice(["h", "a", "."])
        if with_combat or rng.random() < 0.5:
            buf[2][8] = rng.choice(["z", "i"])
            buf[8][8] = rng.choice(["z", "s"])
        return _to_grid_str(buf), rng.choice([0, 90, 180, 270]), 64

    branch_len = return_distance if return_distance is not None else rng.choice([3, 5, 7, 18, 24])
    key_on_north = rng.choice([True, False])
    use_mid_door = rng.random() < 0.5
    add_distractor = rng.random() < 0.7

    hub_w = 5
    hub_h = 3
    exit_w = 3
    total_w = 1 + hub_w + 1 + exit_w + 1 + max(0, branch_len - 3)
    north_h = (branch_len + 1) if key_on_north else (3 if add_distractor else 0)
    south_h = (branch_len + 1) if (not key_on_north) else (3 if add_distractor else 0)
    total_h = 1 + north_h + hub_h + south_h + 1

    buf = [["#"] * total_w for _ in range(total_h)]
    hub_r0 = 1 + north_h
    hub_r1 = hub_r0 + hub_h - 1
    hub_c0 = 1
    hub_c1 = hub_c0 + hub_w - 1

    _carve_rect(buf, hub_r0, hub_c0, hub_r1, hub_c1)
    exit_c0 = hub_c1 + 2
    _carve_rect(buf, hub_r0, exit_c0, hub_r1, exit_c0 + exit_w - 1)
    buf[hub_r0 + 1][hub_c1 + 1] = lock_char
    buf[hub_r0 + 1][exit_c0 + exit_w - 1] = "E"
    buf[hub_r0 + 1][hub_c0 + 1] = "P"

    cor_c = hub_c0 + hub_w // 2
    if key_on_north:
        _carve_rect(buf, 1, cor_c - 1, hub_r0 - 1, cor_c + 1)
        if use_mid_door and branch_len >= 4:
            door_r = hub_r0 - 2
            buf[door_r][cor_c - 1] = "#"
            buf[door_r][cor_c + 1] = "#"
            buf[door_r][cor_c] = "D"
        buf[1][cor_c] = key_char
        if with_combat:
            buf[2][cor_c] = rng.choice(["z", "i"])
    else:
        _carve_rect(buf, hub_r1 + 1, cor_c - 1, total_h - 2, cor_c + 1)
        if use_mid_door and branch_len >= 4:
            door_r = hub_r1 + 2
            buf[door_r][cor_c - 1] = "#"
            buf[door_r][cor_c + 1] = "#"
            buf[door_r][cor_c] = "D"
        buf[total_h - 2][cor_c] = key_char
        if with_combat:
            buf[total_h - 3][cor_c] = rng.choice(["z", "i"])

    # Long routes place the key around a corner among repeated side passages.
    if branch_len >= 18:
        end_row = 1 if key_on_north else total_h - 2
        buf[end_row][cor_c] = "."
        toward_hub = 1 if key_on_north else -1
        for offset in range(0, branch_len - 3, 4):
            row = end_row + toward_hub * offset
            reach = rng.randint(4, 8)
            adjacent = row + toward_hub
            _carve_rect(buf, min(row, adjacent), cor_c + 1, max(row, adjacent), cor_c + reach)
            if offset == 0:
                buf[row][cor_c + reach] = key_char

    if add_distractor:
        if key_on_north and south_h >= 3:
            _carve_rect(buf, hub_r1 + 2, cor_c - 1, total_h - 2, cor_c + 1)
            buf[hub_r1 + 1][cor_c] = "D"
            buf[total_h - 2][cor_c] = "h"
        elif (not key_on_north) and north_h >= 3:
            _carve_rect(buf, 1, cor_c - 1, hub_r0 - 2, cor_c + 1)
            buf[hub_r0 - 1][cor_c] = "D"
            buf[1][cor_c] = "h"

    if with_combat or rng.random() < 0.4:
        buf[hub_r0][hub_c0 + 2] = "g"
        buf[hub_r1][hub_c1] = rng.choice(["z", "s"])

    angle = rng.choice([0, 90, 270])
    return _to_grid_str(buf), angle, 64


def _gen_gauntlet_grid(rng: random.Random) -> tuple[str, int, int]:
    """Generate 3 distinct randomized Tier-4 gauntlet archetypes."""
    pairs = [("R", "r"), ("B", "b"), ("Y", "y")]
    rng.shuffle(pairs)
    (lock1, key1), (lock2, key2) = pairs[0], pairs[1]
    arch = rng.choice(["return_ambush", "seq_two_lock", "hub_two_lock"])

    if arch == "return_ambush":
        buf = [list(row) for row in [
            "#################",
            "#.....#...#.....#",
            "#.....#...#.....#",
            "#.....#...#.....#",
            "#####.##.##.#####",
            "#...........#...#",
            "#...........#...#",
            "#...........#...#",
            "#################",
        ]]
        if rng.random() < 0.5:
            buf[1][2] = key1
            buf[2][2] = rng.choice(["i", "s"])
            buf[3][2] = "H"
            buf[2][6] = "D"
            buf[2][10] = "D"
            buf[1][14] = "h"
            buf[2][14] = rng.choice(["z", "m"])
        else:
            buf[1][14] = key1
            buf[2][14] = rng.choice(["i", "s"])
            buf[3][14] = "H"
            buf[2][6] = "D"
            buf[2][10] = "D"
            buf[1][2] = "h"
            buf[2][2] = rng.choice(["z", "m"])
        buf[6][2] = "P"
        buf[5][4] = "g"
        buf[7][4] = "e"
        buf[7][2] = "A"
        buf[5][8] = rng.choice(["z", "s"])
        buf[6][10] = rng.choice(["i", "m"])
        buf[6][12] = lock1
        buf[6][15] = "E"
        return _to_grid_str(buf), rng.choice([0, 90, 270]), 80

    if arch == "hub_two_lock":
        buf = [list(row) for row in [
            "#################",
            "#.....#...#.....#",
            "#.....#...#.....#",
            "#.....#...#.....#",
            "#####.#...#.#####",
            "#.........#.....#",
            "#.........#.....#",
            "#.........#.....#",
            "#################",
        ]]
        buf[2][2] = key1
        buf[2][4] = rng.choice(["z", "i"])
        buf[3][2] = "H"
        buf[2][6] = "D"
        buf[2][10] = lock1
        buf[2][14] = key2
        buf[2][12] = rng.choice(["i", "s"])
        buf[3][14] = "h"
        buf[6][2] = "P"
        buf[6][3] = "g"
        buf[7][3] = "e"
        buf[5][8] = "x"
        buf[6][7] = rng.choice(["z", "s", "m"])
        buf[6][10] = lock2
        buf[6][14] = "E"
        return _to_grid_str(buf), rng.choice([0, 90]), 80

    # Sequential 2-lock multi-wing fortress
    buf = [list(row) for row in [
        "###################",
        "#.....#.....#.....#",
        "#.....#.....#.....#",
        "#.....#.....#.....#",
        "#####.###.###.#####",
        "#.....#.....#.....#",
        "#.....#.....#.....#",
        "#.....#.....#.....#",
        "#.....#.....#.....#",
        "###################",
    ]]
    # South-center spawn hub; West wing (D) has key1; NW wing (lock1) has key2; NE/SE behind lock2 -> E
    buf[7][9] = "P"
    buf[7][10] = "g"
    buf[8][10] = "e"
    buf[7][14] = rng.choice(["m", "i"])
    buf[8][16] = "A"
    # Open portal between south-center and SE
    buf[6][12] = "."
    buf[7][12] = "."
    buf[6][6] = "D"
    buf[6][2] = key1
    buf[7][2] = rng.choice(["z", "s"])
    buf[8][2] = "h"
    buf[2][6] = lock1
    buf[2][2] = key2
    buf[2][4] = rng.choice(["i", "s"])
    buf[3][2] = "H"
    buf[2][9] = "x"
    buf[3][9] = rng.choice(["s", "z"])
    buf[2][12] = lock2
    buf[2][16] = "E"
    return _to_grid_str(buf), rng.choice([0, 90, 180, 270]), 80


def _build_angled_door_exit(path: Path, rng: random.Random) -> None:
    """Manual/remote door + angled passage with optional stairs/nukage to walk-over pad or wall switch."""
    width = rng.choice((96, 112, 128, 160, 192))
    run = rng.randrange(200, 480, 24)
    bend = rng.choice((-160, -96, -48, 48, 96, 160))
    switch = rng.random() < 0.45
    rot_k = rng.randrange(4)
    flip_x = bool(rng.randrange(2))
    p_ang = rng.choice((0, 90, 180, 270))

    if rng.random() < 0.65:
        end = 304 + run
        far_wall = ((end + 80, bend), (end + 80, bend + width))
        use_nukage = rng.random() < 0.35
        alc_h = width + rng.choice((144, 160, 192))
        sectors = (
            _rect("start", 0, 0, 256, width),
            _rect(
                "switch_alcove",
                0,
                width,
                256,
                alc_h,
                wall_texture="METAL1",
                door_switch=((0, alc_h), (256, alc_h), 1),
            ),
            _rect("rdoor", 256, 0, 304, width, kind="remote_door", sector_id=1),
            PolygonSector(
                "passage",
                ((304, 0), (end, bend), (end, bend + width), (304, width)),
                floor_texture="NUKAGE1" if use_nukage else "FLOOR0_1",
                sector_special=71 if use_nukage else 0,
            ),
            PolygonSector(
                "destination",
                ((end, bend), *far_wall, (end, bend + width)),
                kind="room" if switch else "exit",
                exit_switch=far_wall if switch else None,
            ),
        )
        things = [ThingSpec(80, width // 2, 1, p_ang)]
        if use_nukage:
            things.append(ThingSpec(176, width // 2, 2025, 0))
    else:
        end = 48 + run
        far_wall = ((end + 80, bend), (end + 80, bend + width))
        use_stairs = rng.random() < 0.55
        bevel = rng.choice((64, 96, 128))
        if use_stairs:
            mid_x = 48 + run // 2
            mid_y = bend // 2
            sectors = (
                PolygonSector("start", ((-256, -bevel), (0, 0), (0, width), (-256, width + bevel)), height_floor=0, height_ceiling=160),
                PolygonSector("door", ((0, 0), (48, 0), (48, width), (0, width)), kind="door", height_floor=0, height_ceiling=160),
                PolygonSector("step1", ((48, 0), (mid_x, mid_y), (mid_x, mid_y + width), (48, width)), height_floor=16, height_ceiling=160, wall_texture="BROWN1"),
                PolygonSector("step2", ((mid_x, mid_y), (end, bend), (end, bend + width), (mid_x, mid_y + width)), height_floor=32, height_ceiling=160, wall_texture="STONE2"),
                PolygonSector(
                    "destination",
                    ((end, bend), *far_wall, (end, bend + width)),
                    kind="room" if switch else "exit",
                    height_floor=32,
                    height_ceiling=160,
                    exit_switch=far_wall if switch else None,
                ),
            )
        else:
            sectors = (
                PolygonSector("start", ((-256, -bevel), (0, 0), (0, width), (-256, width + bevel))),
                PolygonSector("door", ((0, 0), (48, 0), (48, width), (0, width)), kind="door"),
                PolygonSector("passage", ((48, 0), (end, bend), (end, bend + width), (48, width))),
                PolygonSector(
                    "destination",
                    ((end, bend), *far_wall, (end, bend + width)),
                    kind="room" if switch else "exit",
                    exit_switch=far_wall if switch else None,
                ),
            )
        things = [ThingSpec(-160, width // 2, 1, p_ang)]

    sectors, things = transform_polygon_map(sectors, things, rot_k=rot_k, flip_x=flip_x)
    ok, errors = verify_polygon_solvability(sectors, things)
    if not ok:
        raise ValueError(f"Invalid procedural polygon door_exit map: {errors}")
    build_polygon_wad(path, sectors, things, door_close_delay=rng.choice((0, 60, 100, 150, 200)))


def _build_poly_combat(path: Path, rng: random.Random) -> None:
    """Angled combat arena (including >1024u wide halls), pillars, stairs/grates, nukage, full roster."""
    mode = rng.choice(("stairs", "grate", "wide_arena"))
    switch = rng.random() < 0.40
    wpn_id, ammo_id = rng.choice(_WEAPON_AMMO_PAIRS)
    m_ang = rng.randrange(0, 360, 45)
    p_ang = rng.choice((0, 90, 180, 270))
    rot_k = rng.randrange(4)
    flip_x = bool(rng.randrange(2))
    use_nukage = rng.random() < 0.35

    if mode == "wide_arena":
        w_arena = rng.choice((1152.0, 1280.0))
        h_arena = rng.choice((896.0, 1024.0))
        cx, cy = w_arena * 0.5, h_arena * 0.5
        pr = rng.choice((56.0, 72.0))
        p1 = PolygonHole(((cx - 192 - pr, cy - pr), (cx - 192, cy + pr), (cx - 192 + pr, cy), (cx - 192, cy - pr * 1.3)), wall_texture="STONE2")
        p2 = PolygonHole(((cx + 192, cy - pr), (cx + 192 - pr, cy), (cx + 192, cy + pr), (cx + 192 + pr, cy)), wall_texture="METAL1")
        far_wall = ((w_arena + 96, cy - 96), (w_arena + 96, cy + 96))
        sectors = (
            PolygonSector(
                "wide_hull",
                ((0, 128), (0, h_arena - 128), (160, h_arena), (w_arena, h_arena), (w_arena, cy + 96), (w_arena, cy - 96), (w_arena, 0), (160, 0)),
                wall_texture="BRICK1",
                floor_texture="NUKAGE1" if use_nukage else "FLOOR7_1",
                height_ceiling=192,
                holes=(p1, p2),
                sector_special=71 if use_nukage else 0,
            ),
            _rect(
                "exit",
                w_arena,
                cy - 96,
                w_arena + 96,
                cy + 96,
                kind="room" if switch else "exit",
                height_ceiling=192,
                exit_switch=far_wall if switch else None,
            ),
        )
        enemy_spots = [
            (w_arena * 0.42, h_arena * 0.20),
            (w_arena * 0.42, h_arena * 0.80),
            (w_arena * 0.62, h_arena * 0.22),
            (w_arena * 0.62, h_arena * 0.78),
            (w_arena * 0.82, h_arena * 0.28),
            (w_arena * 0.82, h_arena * 0.72),
        ]
        n_enemies = rng.randint(4, 6)
        things: list[ThingSpec] = [
            ThingSpec(128, cy, 1, p_ang),
            ThingSpec(200, cy - 96, wpn_id, 0),
            ThingSpec(200, cy + 96, ammo_id, 0),
            ThingSpec(160, cy, 2025 if use_nukage else rng.choice((2012, 2018, 2019, 2024)), 0),
        ]
        for ex, ey in enemy_spots[:n_enemies]:
            things.append(ThingSpec(ex, ey, rng.choice(_COMBAT_MONSTER_IDS), m_ang))
    elif mode == "stairs":
        w0 = float(rng.choice((208, 240, 272)))
        w1 = float(rng.choice((288, 336, 384)))
        h0 = float(rng.choice((224, 256, 288)))
        lip = 64.0
        px, py = w0 + lip + w1 * 0.5, h0 * 0.5
        pillar = PolygonHole(vertices=((px, py - 40), (px + 40, py), (px, py + 40), (px - 40, py)), wall_texture="METAL1")
        far_x = w0 + lip + w1 + 80
        far_wall = ((far_x, 0.0), (far_x, h0))
        sectors = (
            _rect("west", 0, 0, w0, h0, height_floor=0, height_ceiling=176, wall_texture="STARTAN2"),
            _rect("east_lip", w0, 0, w0 + lip, h0, height_floor=48, height_ceiling=176, wall_texture="STONE2"),
            _rect(
                "east_arena",
                w0 + lip,
                0,
                w0 + lip + w1,
                h0,
                height_floor=48,
                height_ceiling=176,
                wall_texture="BRICK1",
                floor_texture="NUKAGE1" if use_nukage else "FLOOR0_1",
                holes=(pillar,),
                sector_special=71 if use_nukage else 0,
            ),
            _rect("step1", 0, h0, w0, h0 + 64, height_floor=16, height_ceiling=176, wall_texture="BROWN1"),
            _rect("step2", 0, h0 + 64, w0, h0 + 128, height_floor=32, height_ceiling=176, wall_texture="BROWN1"),
            _rect("top_w", 0, h0 + 128, w0, h0 + 256, height_floor=48, height_ceiling=176, wall_texture="STONE2"),
            _rect("top_mid", w0, h0 + 128, w0 + lip, h0 + 256, height_floor=48, height_ceiling=176, wall_texture="STONE2"),
            _rect("top_e", w0 + lip, h0 + 128, w0 + lip + w1, h0 + 256, height_floor=48, height_ceiling=176, wall_texture="STONE2"),
            _rect("east_north", w0 + lip, h0, w0 + lip + w1, h0 + 128, height_floor=48, height_ceiling=176, wall_texture="STONE2"),
            _rect(
                "exit",
                w0 + lip + w1,
                0,
                far_x,
                h0,
                kind="room" if switch else "exit",
                height_floor=48,
                height_ceiling=176,
                exit_switch=far_wall if switch else None,
            ),
        )
        enemy_spots = [
            (w0 + lip + 64, 56),
            (w0 + lip + 64, h0 - 56),
            (w0 + lip + w1 - 64, 56),
            (w0 + lip + w1 - 64, h0 - 56),
        ]
        rng.shuffle(enemy_spots)
        things = [
            ThingSpec(80, h0 * 0.5, 1, p_ang),
            ThingSpec(128, 72, wpn_id, 0),
            ThingSpec(128, h0 - 72, ammo_id, 0),
            ThingSpec(w0 - 48, h0 * 0.5, 2025 if use_nukage else rng.choice((2012, 2018, 2019, 2024)), 0),
        ]
        for ex, ey in enemy_spots[: rng.randint(3, 4)]:
            things.append(ThingSpec(ex, ey, rng.choice(_COMBAT_MONSTER_IDS), m_ang))
    else:
        w0 = float(rng.choice((224, 256, 288)))
        w1 = float(rng.choice((352, 400, 448)))
        h0 = float(rng.choice((224, 256, 304)))
        grate = ((w0, 0.0), (w0, h0))
        px, py = w0 + w1 * 0.5, h0 * 0.5
        pillar = PolygonHole(vertices=((px, py - 44), (px + 44, py), (px, py + 44), (px - 44, py)), wall_texture="METAL1")
        far_x = w0 + w1 + 80
        far_wall = ((far_x, 0.0), (far_x, h0))
        flank_h = float(rng.choice((144, 160, 192)))
        sectors = (
            _rect("west", 0, 0, w0, h0, wall_texture="BRICK1", blocking_edges=(grate,)),
            _rect("east", w0, 0, w0 + w1, h0, wall_texture="STONE2", holes=(pillar,)),
            PolygonSector("flank_w", ((0, h0), (64, h0 + flank_h), (w0, h0 + flank_h), (w0, h0)), wall_texture="METAL1"),
            PolygonSector("flank_e", ((w0, h0), (w0, h0 + flank_h), (w0 + w1 - 64, h0 + flank_h), (w0 + w1, h0)), wall_texture="METAL1"),
            _rect(
                "exit",
                w0 + w1,
                0,
                far_x,
                h0,
                kind="room" if switch else "exit",
                exit_switch=far_wall if switch else None,
            ),
        )
        enemy_spots = [
            (w0 + 72, 64),
            (w0 + 72, h0 - 64),
            (w0 + w1 - 72, 64),
            (w0 + w1 - 72, h0 - 64),
        ]
        rng.shuffle(enemy_spots)
        things = [
            ThingSpec(80, h0 * 0.5, 1, p_ang),
            ThingSpec(128, 72, wpn_id, 0),
            ThingSpec(128, h0 - 72, ammo_id, 0),
            ThingSpec(w0 - 48, 64, rng.choice((2012, 2018, 2019, 2024)), 0),
        ]
        for ex, ey in enemy_spots[: rng.randint(3, 4)]:
            things.append(ThingSpec(ex, ey, rng.choice(_COMBAT_MONSTER_IDS), m_ang))

    sectors, things = transform_polygon_map(sectors, things, rot_k=rot_k, flip_x=flip_x)
    ok, errors = verify_polygon_solvability(sectors, things)
    if not ok:
        raise ValueError(f"Invalid procedural polygon combat map: {errors}")
    build_polygon_wad(path, sectors, things, door_close_delay=rng.choice((0, 100, 150, 200)))


def _build_poly_multi_room(path: Path, rng: random.Random) -> None:
    """Octagonal hub with diagonal branches, stairs, low-window lure, optional nukage + switch exit."""
    r_hub = rng.choice((176, 192, 224))
    a_hub = rng.choice((48, 64, 72))
    hub = _octagon(0, 0, r_hub, a_hub)
    s_entry = _extrude(hub[6], hub[7], 0, -128)
    exit_on_ne = rng.choice((True, False))
    switch = rng.random() < 0.45
    remote_exit_door = rng.random() < 0.70
    use_nukage = rng.random() < 0.45
    rot_k = rng.randrange(4)
    flip_x = bool(rng.randrange(2))
    p_ang = rng.choice((0, 90, 180, 270))

    nw_door = _extrude(hub[1], hub[2], -48, 48)
    ne_door = _extrude(hub[3], hub[4], 48, 48)
    se_win = _extrude(hub[5], hub[6], 48, -48)
    se_room = _extrude(se_win[3], se_win[2], 128, -128)

    if exit_on_ne:
        nw_room = _extrude(nw_door[3], nw_door[2], -144, 144)
        nw_far = (nw_room[2], nw_room[3])
        ne_s1 = _extrude(ne_door[3], ne_door[2], 64, 64)
        ne_s2 = _extrude(ne_s1[3], ne_s1[2], 64, 64)
        ne_top = _extrude(ne_s2[3], ne_s2[2], 128, 128)
        ne_ex = _extrude(ne_top[3], ne_top[2], 64, 64)
        ex_far = (ne_ex[2], ne_ex[3])
        sectors = (
            PolygonSector(
                "hub",
                hub,
                "room",
                wall_texture="STONE2",
                floor_texture="NUKAGE1" if use_nukage else "FLOOR4_8",
                height_floor=0,
                height_ceiling=176,
                sector_special=71 if use_nukage else 0,
            ),
            PolygonSector("s_entry", s_entry, "room", wall_texture="STARTAN2", height_floor=0, height_ceiling=176),
            PolygonSector("nw_door", nw_door, "door", height_floor=0, height_ceiling=176),
            PolygonSector(
                "nw_dead",
                nw_room,
                "room",
                wall_texture="BROWN1",
                height_floor=0,
                height_ceiling=176,
                door_switch=(nw_far[0], nw_far[1], 1) if remote_exit_door else None,
            ),
            PolygonSector(
                "ne_door",
                ne_door,
                "remote_door" if remote_exit_door else "door",
                height_floor=0,
                height_ceiling=176,
                sector_id=1 if remote_exit_door else 0,
            ),
            PolygonSector("ne_s1", ne_s1, "room", wall_texture="GRAY1", height_floor=16, height_ceiling=176),
            PolygonSector("ne_s2", ne_s2, "room", wall_texture="GRAY1", height_floor=32, height_ceiling=176),
            PolygonSector("ne_top", ne_top, "room", wall_texture="METAL1", height_floor=48, height_ceiling=176),
            PolygonSector(
                "ne_exit",
                ne_ex,
                "room" if switch else "exit",
                wall_texture="GRAY1",
                height_floor=48,
                height_ceiling=176,
                exit_switch=ex_far if switch else None,
            ),
            PolygonSector("se_win", se_win, "room", wall_texture="BRICK1", height_floor=0, height_ceiling=40),
            PolygonSector("se_lure", se_room, "room", wall_texture="BRICK1", height_floor=0, height_ceiling=176),
        )
        cx_nw = sum(v[0] for v in nw_room) / 4.0
        cy_nw = sum(v[1] for v in nw_room) / 4.0
        cx_top = sum(v[0] for v in ne_top) / 4.0
        cy_top = sum(v[1] for v in ne_top) / 4.0
        cx_lure = sum(v[0] for v in se_room) / 4.0
        cy_lure = sum(v[1] for v in se_room) / 4.0
        things = [
            ThingSpec(0, -(r_hub + 64), 1, p_ang),
            ThingSpec(0, -(r_hub + 24), 2025 if use_nukage else 2001, 0),
            ThingSpec(cx_nw, cy_nw, 2012, 0),
            ThingSpec(cx_nw - 32, cy_nw - 32, rng.choice((3004, 9, 3001)), rng.randrange(0, 360, 45)),
            ThingSpec(cx_top, cy_top, rng.choice((3004, 3001, 65)), rng.randrange(0, 360, 45)),
            ThingSpec(cx_lure, cy_lure, 2024, 0),
        ]
    else:
        ne_room = _extrude(ne_door[3], ne_door[2], 144, 144)
        ne_far = (ne_room[2], ne_room[3])
        nw_s1 = _extrude(nw_door[3], nw_door[2], -64, 64)
        nw_s2 = _extrude(nw_s1[3], nw_s1[2], -64, 64)
        nw_top = _extrude(nw_s2[3], nw_s2[2], -128, 128)
        nw_ex = _extrude(nw_top[3], nw_top[2], -64, 64)
        ex_far = (nw_ex[2], nw_ex[3])
        sectors = (
            PolygonSector(
                "hub",
                hub,
                "room",
                wall_texture="STONE2",
                floor_texture="NUKAGE1" if use_nukage else "FLOOR4_8",
                height_floor=0,
                height_ceiling=176,
                sector_special=71 if use_nukage else 0,
            ),
            PolygonSector("s_entry", s_entry, "room", wall_texture="STARTAN2", height_floor=0, height_ceiling=176),
            PolygonSector("ne_door", ne_door, "door", height_floor=0, height_ceiling=176),
            PolygonSector(
                "ne_dead",
                ne_room,
                "room",
                wall_texture="BROWN1",
                height_floor=0,
                height_ceiling=176,
                door_switch=(ne_far[0], ne_far[1], 1) if remote_exit_door else None,
            ),
            PolygonSector(
                "nw_door",
                nw_door,
                "remote_door" if remote_exit_door else "door",
                height_floor=0,
                height_ceiling=176,
                sector_id=1 if remote_exit_door else 0,
            ),
            PolygonSector("nw_s1", nw_s1, "room", wall_texture="GRAY1", height_floor=16, height_ceiling=176),
            PolygonSector("nw_s2", nw_s2, "room", wall_texture="GRAY1", height_floor=32, height_ceiling=176),
            PolygonSector("nw_top", nw_top, "room", wall_texture="METAL1", height_floor=48, height_ceiling=176),
            PolygonSector(
                "nw_exit",
                nw_ex,
                "room" if switch else "exit",
                wall_texture="GRAY1",
                height_floor=48,
                height_ceiling=176,
                exit_switch=ex_far if switch else None,
            ),
            PolygonSector("se_win", se_win, "room", wall_texture="BRICK1", height_floor=0, height_ceiling=40),
            PolygonSector("se_lure", se_room, "room", wall_texture="BRICK1", height_floor=0, height_ceiling=176),
        )
        cx_ne = sum(v[0] for v in ne_room) / 4.0
        cy_ne = sum(v[1] for v in ne_room) / 4.0
        cx_top = sum(v[0] for v in nw_top) / 4.0
        cy_top = sum(v[1] for v in nw_top) / 4.0
        cx_lure = sum(v[0] for v in se_room) / 4.0
        cy_lure = sum(v[1] for v in se_room) / 4.0
        things = [
            ThingSpec(0, -(r_hub + 64), 1, p_ang),
            ThingSpec(0, -(r_hub + 24), 2025 if use_nukage else 2001, 0),
            ThingSpec(cx_ne, cy_ne, 2012, 0),
            ThingSpec(cx_ne - 32, cy_ne - 32, rng.choice((3004, 9, 3001)), rng.randrange(0, 360, 45)),
            ThingSpec(cx_top, cy_top, rng.choice((3004, 3001, 65)), rng.randrange(0, 360, 45)),
            ThingSpec(cx_lure, cy_lure, 2024, 0),
        ]

    sectors, things = transform_polygon_map(sectors, things, rot_k=rot_k, flip_x=flip_x)
    ok, errors = verify_polygon_solvability(sectors, things)
    if not ok:
        raise ValueError(f"Invalid procedural polygon multi_room map: {errors}")
    build_polygon_wad(path, sectors, things, door_close_delay=rng.choice((0, 60, 100, 150, 200)))


def _build_poly_key_return(path: Path, rng: random.Random, *, with_combat: bool = False) -> None:
    """Polygonal key-return: windows/grates, one-way drop-offs, octagonal stairs, D4 transformed."""
    door_kind, card_id, skull_id = rng.choice(_KEY_SPECS)
    key_id = skull_id if rng.choice((True, False)) else card_id
    switch = rng.random() < 0.30
    p_ang = rng.choice((0, 90, 180, 270))
    m_ang = rng.randrange(0, 360, 45)
    rot_k = rng.randrange(4)
    flip_x = bool(rng.randrange(2))

    mode = rng.choice(("window", "grate", "dropoff_cliff", "octagonal_stairs"))
    if mode in ("window", "grate"):
        detour_north = rng.choice((True, False))
        w_sp = float(rng.choice((160, 192, 224)))
        w_key = float(rng.choice((176, 208, 240)))
        gap = float(rng.choice((48, 64, 80)))
        h_main = float(rng.choice((144, 160, 176)))
        h_det = float(rng.choice((144, 176, 192)))
        y_det0, y_det1 = (h_main, h_main + h_det) if detour_north else (-h_det, 0.0)
        grate_edge = (((w_sp, 0.0), (w_sp, h_main)),) if mode == "grate" else ()
        far_wall = ((-240.0, 0.0), (-240.0, h_main))
        use_nuk = rng.random() < 0.35
        x_mid = w_sp + gap
        x_end = x_mid + w_key
        sectors = (
            _rect("spawn", 0, 0, w_sp, h_main, blocking_edges=grate_edge),
            _rect("divider", w_sp, 0, x_mid, h_main, height_ceiling=40 if mode == "window" else 128),
            PolygonSector(
                "key_room",
                ((x_mid, 0), (x_mid, h_main), (x_end, h_main), (x_end + 32, h_main * 0.5), (x_end, 0)),
                wall_texture="BRICK1",
            ),
            _rect("det_w", 0, y_det0, w_sp, y_det1, wall_texture="STONE2"),
            _rect(
                "det_m",
                w_sp,
                y_det0,
                x_mid,
                y_det1,
                wall_texture="STONE2",
                floor_texture="NUKAGE1" if use_nuk else "FLOOR0_1",
                sector_special=71 if use_nuk else 0,
            ),
            _rect("det_e", x_mid, y_det0, x_end, y_det1, wall_texture="STONE2"),
            _rect("lock", -48, 0, 0, h_main, kind=door_kind),
            _rect("vestibule", -176, 0, -48, h_main, wall_texture="METAL1"),
            _rect(
                "exit",
                -240,
                0,
                -176,
                h_main,
                kind="room" if switch else "exit",
                exit_switch=far_wall if switch else None,
            ),
        )
        things = [ThingSpec(w_sp * 0.45, h_main * 0.5, 1, p_ang), ThingSpec(x_mid + w_key * 0.5, h_main * 0.5, key_id, 0)]
        if use_nuk:
            things.append(ThingSpec(w_sp * 0.5, (y_det0 + y_det1) * 0.5, 2025, 0))
        if with_combat or rng.random() < 0.5:
            wpn_id, ammo_id = rng.choice(_WEAPON_AMMO_PAIRS)
            things.extend([
                ThingSpec(w_sp * 0.6, (y_det0 + y_det1) * 0.5, wpn_id, 0),
                ThingSpec(x_mid + 32, (y_det0 + y_det1) * 0.5, ammo_id, 0),
                ThingSpec(x_mid + w_key * 0.5, (y_det0 + y_det1) * 0.5, rng.choice(_COMBAT_MONSTER_IDS), m_ang),
            ])
    elif mode == "dropoff_cliff":
        w_court = float(rng.choice((288, 320, 368)))
        h_c = float(rng.choice((240, 272)))
        w_top = float(rng.choice((160, 192)))
        far_wall = ((-240.0, 64.0), (-240.0, h_c - 64.0))
        sectors = (
            PolygonSector(
                "courtyard",
                ((0, 0), (0, 64), (0, h_c - 64), (0, h_c), (w_court, h_c), (w_court, 64), (w_court, 0)),
                wall_texture="STONE2",
                height_ceiling=192,
            ),
            _rect("step1", w_court, 0, w_court + 64, 64, height_floor=16, height_ceiling=192, wall_texture="STEP1"),
            _rect("step2", w_court + 64, 0, w_court + 128, 64, height_floor=32, height_ceiling=192, wall_texture="STEP1"),
            _rect("step3_top", w_court + 128, 0, w_court + 128 + w_top, 64, height_floor=48, height_ceiling=192, wall_texture="BRICK1"),
            PolygonSector(
                "high_rampart",
                ((w_court, 64), (w_court, h_c), (w_court + 128 + w_top, h_c), (w_court + 128 + w_top, 64), (w_court + 128, 64)),
                height_floor=48,
                height_ceiling=192,
                wall_texture="BRICK1",
                floor_texture="FLOOR7_1",
            ),
            _rect("lock", -48, 64, 0, h_c - 64, kind=door_kind, height_ceiling=192),
            _rect("vestibule", -176, 64, -48, h_c - 64, wall_texture="METAL1", height_ceiling=192),
            _rect(
                "exit",
                -240,
                64,
                -176,
                h_c - 64,
                kind="room" if switch else "exit",
                height_ceiling=192,
                exit_switch=far_wall if switch else None,
            ),
        )
        things = [
            ThingSpec(w_court * 0.45, h_c * 0.5, 1, p_ang),
            ThingSpec(w_court + w_top * 0.5, h_c * 0.65, key_id, 0),
        ]
        if with_combat or rng.random() < 0.5:
            wpn_id, ammo_id = rng.choice(_WEAPON_AMMO_PAIRS)
            things.extend([
                ThingSpec(w_court * 0.35, 64, wpn_id, 0),
                ThingSpec(w_court + 96, h_c * 0.55, rng.choice(_COMBAT_MONSTER_IDS), m_ang),
                ThingSpec(-112, h_c * 0.5, rng.choice((9, 3001, 65)), 0),
            ])
    else:
        r_hub = float(rng.choice((176, 192, 224)))
        a_hub = float(rng.choice((56, 64, 72)))
        hub = _octagon(0, 0, r_hub, a_hub)
        w_entry = _extrude(hub[0], hub[1], -128, 0)
        nw_door = _extrude(hub[1], hub[2], -48, 48)
        nw_trap = _extrude(nw_door[3], nw_door[2], -128, 128)
        se_s1 = _extrude(hub[5], hub[6], 64, -64)
        se_s2 = _extrude(se_s1[3], se_s1[2], 64, -64)
        se_key = _extrude(se_s2[3], se_s2[2], 144, -144)
        ne_lock = _extrude(hub[3], hub[4], 48, 48)
        ne_vault = _extrude(ne_lock[3], ne_lock[2], 128, 128)
        ne_exit = _extrude(ne_vault[3], ne_vault[2], 64, 64)
        hub_pillar = PolygonHole(vertices=((0, -40), (40, 0), (0, 40), (-40, 0)), wall_texture="METAL1")
        ex_far = (ne_exit[2], ne_exit[3])
        sectors = (
            PolygonSector("hub", hub, "room", wall_texture="GRAY1", height_floor=0, height_ceiling=176, holes=(hub_pillar,)),
            PolygonSector("w_entry", w_entry, "room", wall_texture="STARTAN2", height_floor=0, height_ceiling=176),
            PolygonSector("nw_door", nw_door, "door", height_floor=0, height_ceiling=176),
            PolygonSector("nw_trap", nw_trap, "room", wall_texture="BROWN1", height_floor=0, height_ceiling=176),
            PolygonSector("se_s1", se_s1, "room", wall_texture="STONE2", height_floor=16, height_ceiling=176),
            PolygonSector("se_s2", se_s2, "room", wall_texture="STONE2", height_floor=32, height_ceiling=176),
            PolygonSector("se_key", se_key, "room", wall_texture="BRICK1", height_floor=48, height_ceiling=176),
            PolygonSector("ne_lock", ne_lock, door_kind, height_floor=0, height_ceiling=176),
            PolygonSector("ne_vault", ne_vault, "room", wall_texture="METAL1", height_floor=0, height_ceiling=176),
            PolygonSector(
                "ne_exit",
                ne_exit,
                "room" if switch else "exit",
                height_floor=0,
                height_ceiling=176,
                exit_switch=ex_far if switch else None,
            ),
        )
        wpn_id, ammo_id = rng.choice(_WEAPON_AMMO_PAIRS)
        cx_k = sum(v[0] for v in se_key) / 4.0
        cy_k = sum(v[1] for v in se_key) / 4.0
        cx_t = sum(v[0] for v in nw_trap) / 4.0
        cy_t = sum(v[1] for v in nw_trap) / 4.0
        things = [
            ThingSpec(-(r_hub + 64), 0, 1, p_ang),
            ThingSpec(-(r_hub + 16), 0, wpn_id, 0),
            ThingSpec(-(r_hub + 16), 32, ammo_id, 0),
            ThingSpec(cx_k, cy_k, key_id, 0),
            ThingSpec(cx_k - 32, cy_k + 32, rng.choice(_COMBAT_MONSTER_IDS), m_ang),
            ThingSpec(cx_t, cy_t, 2011, 0),
            ThingSpec(cx_t + 32, cy_t - 32, rng.choice((3004, 3001, 65)), m_ang),
        ]

    sectors, things = transform_polygon_map(sectors, things, rot_k=rot_k, flip_x=flip_x)
    ok, errors = verify_polygon_solvability(sectors, things)
    if not ok:
        raise ValueError(f"Invalid procedural polygon key_return map ({mode}): {errors}")
    build_polygon_wad(path, sectors, things, door_close_delay=rng.choice((0, 100, 150, 200)))


def _build_poly_gauntlet(path: Path, rng: random.Random) -> None:
    """Two-lock or three-lock octagonal citadel with stairs, optional nukage, and 7-10 heavy foes."""
    keys = list(_KEY_SPECS)
    rng.shuffle(keys)
    three_lock = rng.random() < 0.45
    use_nukage = rng.random() < 0.40
    (lock1, c1, s1), (lock2, c2, s2), (lock3, c3, s3) = keys[0], keys[1], keys[2]
    key1_id = s1 if rng.choice((True, False)) else c1
    key2_id = s2 if rng.choice((True, False)) else c2
    key3_id = s3 if rng.choice((True, False)) else c3
    wpn_id, ammo_id = rng.choice(_WEAPON_AMMO_PAIRS)
    switch = rng.random() < 0.25
    m_ang = rng.randrange(0, 360, 45)
    p_ang = rng.choice((0, 90, 180, 270))
    rot_k = rng.randrange(4)
    flip_x = bool(rng.randrange(2))

    r_hub = float(rng.choice((224, 240, 256)))
    a_hub = float(rng.choice((64, 72)))
    hub = _octagon(0, 0, r_hub, a_hub)
    s_spawn = _extrude(hub[6], hub[7], 0, -128)
    w_door = _extrude(hub[0], hub[1], -48, 0)
    w_room = _extrude(w_door[3], w_door[2], -192, 0)
    e_lock1 = _extrude(hub[4], hub[5], 48, 0)
    e_room = _extrude(e_lock1[3], e_lock1[2], 224, 0)
    nw_conn = _extrude(hub[1], hub[2], -48, 48)
    nw_room = _extrude(nw_conn[3], nw_conn[2], -144, 144)

    n_s1 = _extrude(hub[2], hub[3], 0, 48)
    n_s2 = _extrude(n_s1[3], n_s1[2], 0, 48)
    n_land = _extrude(n_s2[3], n_s2[2], 0, 64)
    n_lock_end = _extrude(n_land[3], n_land[2], 0, 48)
    n_sanctum = _extrude(n_lock_end[3], n_lock_end[2], 0, 160)
    n_exit = _extrude(n_sanctum[3], n_sanctum[2], 0, 64)
    ex_far = (n_exit[2], n_exit[3])

    center_diamond = PolygonHole(vertices=((0, -48), (48, 0), (0, 48), (-48, 0)), wall_texture="BRICK1")
    cx_nw = sum(v[0] for v in nw_room) / 4.0
    cy_nw = sum(v[1] for v in nw_room) / 4.0
    sectors = (
        PolygonSector("hub", hub, "room", wall_texture="STONE2", height_floor=0, height_ceiling=192, holes=(center_diamond,)),
        PolygonSector("s_spawn", s_spawn, "room", wall_texture="STARTAN2", height_floor=0, height_ceiling=192),
        PolygonSector("w_door", w_door, "door", height_floor=0, height_ceiling=192),
        PolygonSector("w_room", w_room, "room", wall_texture="BROWN1", height_floor=0, height_ceiling=192),
        PolygonSector("e_lock1", e_lock1, lock1, height_floor=0, height_ceiling=192),
        PolygonSector(
            "e_room",
            e_room,
            "room",
            wall_texture="BRICK1",
            floor_texture="NUKAGE1" if use_nukage else "FLOOR7_1",
            height_floor=0,
            height_ceiling=192,
            sector_special=71 if use_nukage else 0,
        ),
        PolygonSector("nw_conn", nw_conn, lock2 if three_lock else "door", height_floor=0, height_ceiling=192),
        PolygonSector("nw_room", nw_room, "room", wall_texture="GRAY1", height_floor=0, height_ceiling=192),
        PolygonSector("n_s1", n_s1, "room", wall_texture="BROWN1", height_floor=16, height_ceiling=192),
        PolygonSector("n_s2", n_s2, "room", wall_texture="BROWN1", height_floor=32, height_ceiling=192),
        PolygonSector("n_land", n_land, "room", wall_texture="STONE2", height_floor=48, height_ceiling=192),
        PolygonSector("n_lock_end", n_lock_end, lock3 if three_lock else lock2, height_floor=48, height_ceiling=192),
        PolygonSector("n_sanctum", n_sanctum, "room", wall_texture="METAL1", height_floor=48, height_ceiling=192),
        PolygonSector(
            "n_exit",
            n_exit,
            "room" if switch else "exit",
            height_floor=48,
            height_ceiling=192,
            exit_switch=ex_far if switch else None,
        ),
    )
    things: list[ThingSpec] = [
        ThingSpec(0, -(r_hub + 64), 1, p_ang),
        ThingSpec(0, -(r_hub + 24), wpn_id, 0),
        ThingSpec(-32, -(r_hub + 40), ammo_id, 0),
        ThingSpec(-(r_hub + 144), 0, key1_id, 0),
        ThingSpec(r_hub + 144, 0, key2_id, 0),
        ThingSpec(cx_nw, cy_nw, 2019, 0),
        ThingSpec(0, -128, 2025 if use_nukage else 2013, 0),
        ThingSpec(r_hub + 96, 32, 2012, 0),
    ]
    if three_lock:
        things.append(ThingSpec(cx_nw - 24, cy_nw + 24, key3_id, 0))
    enemy_spots = [
        (-(r_hub + 96), -32),
        (-(r_hub + 96), 32),
        (r_hub + 96, -32),
        (r_hub + 144, 48),
        (cx_nw + 24, cy_nw - 24),
        (128, -96),
        (-128, -96),
        (0, r_hub + 240),
        (32, r_hub + 288),
    ]
    for ex, ey in enemy_spots[: rng.randint(7, 9)]:
        things.append(ThingSpec(ex, ey, rng.choice(_COMBAT_MONSTER_IDS), m_ang))

    sectors, things = transform_polygon_map(sectors, things, rot_k=rot_k, flip_x=flip_x)
    ok, errors = verify_polygon_solvability(sectors, things)
    if not ok:
        raise ValueError(f"Invalid procedural polygon gauntlet map: {errors}")
    build_polygon_wad(path, sectors, things, door_close_delay=rng.choice((0, 100, 150, 200)))


def _build_key_window_detour(path: Path, rng: random.Random) -> None:
    """Asymmetric key-window detour (D4 transformed, never matching static stress templates)."""
    door_kind, key_card, key_skull = rng.choice(_KEY_SPECS)
    key_num = key_skull if rng.random() < 0.5 else key_card
    bypass_north = rng.choice((True, False))
    use_bypass_door = rng.random() < 0.5
    win_ceil = rng.choice((36, 40, 44))
    span_w = float(rng.choice((152, 176, 208, 232)))
    span_e = float(rng.choice((160, 184, 216, 240)))
    bypass_h = float(rng.choice((136, 168, 200)))
    room_h = float(rng.choice((144, 168, 192)))
    gap_w = float(rng.choice((48, 64, 80)))
    rot_k = rng.randrange(4)
    flip_x = bool(rng.randrange(2))
    p_ang = rng.choice((0, 90, 180, 270))

    by0, by1 = (room_h, room_h + bypass_h) if bypass_north else (-bypass_h, 0.0)
    x1 = span_w
    x2 = span_w + gap_w
    x3 = x2 + span_e
    sectors: tuple[PolygonSector, ...] = (
        _rect("spawn", 0.0, 0.0, x1, room_h),
        _rect("window", x1, 0.0, x2, room_h, height_ceiling=win_ceil),
        PolygonSector(
            "key_room",
            ((x2, 0.0), (x2, room_h), (x3, room_h), (x3 + 32.0, room_h * 0.5), (x3, 0.0)),
        ),
        _rect("bypass_west", 0.0, by0, x1, by1),
        _rect("bypass_mid", x1, by0, x2, by1, kind="door" if use_bypass_door else "room"),
        _rect("bypass_east", x2, by0, x3, by1),
        _rect("lock", -48.0, 0.0, 0.0, room_h, kind=door_kind),
        _rect("vestibule", -176.0, 0.0, -48.0, room_h),
        _rect("exit", -240.0, 0.0, -176.0, room_h, kind="exit"),
    )
    things: list[ThingSpec] = [
        ThingSpec(span_w * 0.45, room_h * 0.5, 1, p_ang),
        ThingSpec(x2 + span_e * 0.5, room_h * 0.5, key_num, 0),
    ]
    if rng.random() < 0.6:
        things.append(ThingSpec(x2 + span_e * 0.5, (by0 + by1) * 0.5, rng.choice(_COMBAT_MONSTER_IDS), 180))
    sectors, things = transform_polygon_map(sectors, things, rot_k=rot_k, flip_x=flip_x)
    ok, errors = verify_polygon_solvability(sectors, things)
    if not ok:
        raise ValueError(f"Invalid procedural key-window map: {errors}")
    build_polygon_wad(path, sectors, things, door_close_delay=rng.choice((0, 100, 150, 200)))


def generate_procedural_wad(
    family: str,
    seed: int,
    out_dir: str | Path,
    *,
    split: str = "train",
    return_distance: int | None = None,
    with_combat: bool = False,
) -> MapSpec:
    """Synthesize, statically verify, and write one procedural WAD for training variation.

    Combines connected room-graph dungeons (~48%), D4-transformed polygonal maps (~40%),
    and D4-transformed structured grids (~12%) across all 4 tiers.
    """
    rng = random.Random(int(seed))
    theme = rng.choice(THEME_NAMES)
    out_path = Path(out_dir) / f"proc_{family}_s{seed}.wad"

    if return_distance is None:
        roll_mode = rng.random()
        if roll_mode < 0.48:
            return generate_dungeon_wad(
                family,
                seed=int(seed),
                out_dir=out_dir,
                split=split,
                with_combat=with_combat,
                out_path=out_path,
            )
        if roll_mode < 0.88:
            if family == "door_exit":
                _build_angled_door_exit(out_path, rng)
                return MapSpec(str(out_path), family=family, tier=2, split=split, episode_ticks=2100)
            if family == "combat":
                _build_poly_combat(out_path, rng)
                return MapSpec(str(out_path), family=family, tier=1, split=split, episode_ticks=2100)
            if family == "multi_room":
                _build_poly_multi_room(out_path, rng)
                return MapSpec(str(out_path), family=family, tier=2, split=split, episode_ticks=2400)
            if family == "key_return":
                if rng.random() < 0.35:
                    _build_key_window_detour(out_path, rng)
                else:
                    _build_poly_key_return(out_path, rng, with_combat=with_combat)
                return MapSpec(str(out_path), family=family, tier=3, split=split, episode_ticks=3150)
            if family == "gauntlet":
                _build_poly_gauntlet(out_path, rng)
                return MapSpec(str(out_path), family=family, tier=4, split=split, episode_ticks=4200)

    if family == "door_exit":
        grid, angle, cell_size = _gen_door_exit_grid(rng)
        tier, ticks = 2, 2100
    elif family == "combat":
        grid, angle, cell_size = _gen_combat_grid(rng)
        tier, ticks = 1, 2100
    elif family == "multi_room":
        grid, angle, cell_size = _gen_multi_room_grid(rng)
        tier, ticks = 2, 2100
    elif family == "key_return":
        grid, angle, cell_size = _gen_key_return_grid(
            rng, return_distance=return_distance, with_combat=with_combat
        )
        tier = 3
        span = max(len(grid.splitlines()), max(map(len, grid.splitlines()))) * cell_size
        ticks = 4200 if span >= 1500 else (3150 if (return_distance or 0) >= 6 else 2100)
    elif family == "gauntlet":
        grid, angle, cell_size = _gen_gauntlet_grid(rng)
        tier, ticks = 4, 4200
    else:
        raise ValueError(f"Unsupported procedural family: {family!r}")

    grid = _transform_grid(grid, rng)
    check = verify_grid_solvability(grid)
    if not check.ok:
        raise RuntimeError(f"Procedural generator produced unsolvable {family} seed={seed}: {check.errors}")

    build_grid_wad(
        out_path,
        grid,
        map_name="MAP01",
        cell_size=cell_size,
        player_angle=(angle + rng.choice((0, 90, 180, 270))) % 360,
        monster_angle=rng.randrange(0, 360, 45),
        skull_keys=rng.choice((False, True)),
        door_close_delay=rng.choice((0, 100, 150, 200)),
        theme=theme,
    )
    return MapSpec(
        wad=str(out_path),
        map_name="MAP01",
        family=family,
        tier=tier,
        split=split,
        episode_ticks=ticks,
        seeds=(0, 1, 2),
    )
