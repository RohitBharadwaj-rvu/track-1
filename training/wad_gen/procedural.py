"""On-the-fly procedural WAD generator for Role 2 training variation (Role 3).

Generates fresh, statically verified, guaranteed-solvable variations across all 4 tiers
(`door_exit`, `combat`, `multi_room`, `key_return`, `gauntlet`) from an integer seed so the PPO
trainer can scale variation count on demand without bloating the repository.
"""

from __future__ import annotations

from pathlib import Path
import random

from training.interfaces import MapSpec
from training.wad_gen.udmf import THEME_NAMES, build_grid_wad
from training.wad_gen.verify import verify_grid_solvability


def _to_grid_str(buf: list[list[str]]) -> str:
    return "\n".join("".join(row) for row in buf)


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

    # Weapon & supplies near player
    buf[max(1, mid_r - 1)][sup_c] = rng.choice(["g", "G", "a", "h"])
    buf[min(h - 2, mid_r + 1)][sup_c] = rng.choice(["e", "h", "H", "A"])

    # 2-4 enemies on opposite half
    enemy_pool = ["z", "z", "i", "s", "m"]
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

    branch_len = return_distance if return_distance is not None else rng.choice([3, 5, 7])
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

    Supported families: `"door_exit"`, `"combat"`, `"multi_room"`, `"key_return"`, `"gauntlet"`.
    """
    rng = random.Random(int(seed))
    theme = rng.choice(THEME_NAMES)
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
        tier, ticks = 3, 3150 if (return_distance or 0) >= 6 else 2100
    elif family == "gauntlet":
        grid, angle, cell_size = _gen_gauntlet_grid(rng)
        tier, ticks = 4, 4200
    else:
        raise ValueError(f"Unsupported procedural family: {family!r}")

    check = verify_grid_solvability(grid)
    if not check.ok:
        raise RuntimeError(f"Procedural generator produced unsolvable {family} seed={seed}: {check.errors}")

    out_path = Path(out_dir) / f"proc_{family}_s{seed}.wad"
    build_grid_wad(
        out_path,
        grid,
        map_name="MAP01",
        cell_size=cell_size,
        player_angle=angle,
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
