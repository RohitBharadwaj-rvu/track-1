"""On-the-fly procedural WAD generator for Role 2 training variation (Role 3).

Generates fresh, statically verified, guaranteed-solvable variations across all 4 tiers
(`combat`, `multi_room`, `key_return`, `gauntlet`) from an integer seed so the PPO
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

    # Player spawn on left, exit on right
    buf[mid_r][2] = "P"
    buf[mid_r][w - 2] = "E"

    # Weapon & supplies near player
    buf[max(1, mid_r - 1)][3] = rng.choice(["g", "a", "h"])
    buf[min(h - 2, mid_r + 1)][3] = rng.choice(["e", "h", "H", "A"])

    # 2-4 enemies on right half
    enemy_pool = ["z", "z", "i", "s", "m"]
    num_enemies = rng.randint(2, 4)
    candidates = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(w // 2, w - 2)
        if buf[r][c] == "." and not (r == mid_r and c == w - 2)
    ]
    rng.shuffle(candidates)
    for r, c in candidates[:num_enemies]:
        buf[r][c] = rng.choice(enemy_pool)

    angle = rng.choice([0, 45, 90, 270, 315])
    cell_size = rng.choice([64, 80])
    return _to_grid_str(buf), angle, cell_size


def _gen_multi_room_grid(rng: random.Random) -> tuple[str, int, int]:
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

    # Doors between room 1-2 and room 2-3 at varied vertical offsets
    d1_r = rng.randint(base_r, base_r + room_h - 1)
    d2_r = rng.randint(base_r, base_r + room_h - 1)
    buf[d1_r][c1 + room_w] = "D"
    buf[d2_r][c2 + room_w] = "D"

    # Optional dead-end room off middle room
    if has_branch:
        br_r = 1 if branch_north else (base_r + room_h + 1)
        _carve_rect(buf, br_r, c2, br_r + room_h - 1, c2 + room_w - 1)
        door_r = base_r - 1 if branch_north else base_r + room_h
        door_c = rng.randint(c2, c2 + room_w - 1)
        buf[door_r][door_c] = "D"
        buf[br_r + room_h // 2][c2 + room_w // 2] = rng.choice(["h", "a", "z"])

    # Flip left-right randomly for bearing variety
    spawn_left = rng.choice([True, False])
    if spawn_left:
        buf[base_r + room_h // 2][c1 + 1] = "P"
        buf[base_r + room_h // 2][c3 + room_w - 1] = "E"
        angle = rng.choice([0, 90, 270])
    else:
        buf[base_r + room_h // 2][c3 + room_w - 1] = "P"
        buf[base_r + room_h // 2][c1] = "E"
        angle = rng.choice([180, 90, 270])

    # Place 1-2 guards in middle room
    if buf[base_r + room_h // 2][c2 + room_w // 2] == ".":
        buf[base_r + room_h // 2][c2 + room_w // 2] = rng.choice(["z", "i"])

    return _to_grid_str(buf), angle, 64


def _gen_key_return_grid(
    rng: random.Random,
    *,
    return_distance: int | None = None,
    with_combat: bool = False,
) -> tuple[str, int, int]:
    """Generate a parameterized key-and-return layout with variable branch length, side, and lock color."""
    lock_char, key_char = rng.choice([("R", "r"), ("B", "b"), ("Y", "y")])
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

    # Hub + locked exit room on East
    _carve_rect(buf, hub_r0, hub_c0, hub_r1, hub_c1)
    exit_c0 = hub_c1 + 2
    _carve_rect(buf, hub_r0, exit_c0, hub_r1, exit_c0 + exit_w - 1)
    buf[hub_r0 + 1][hub_c1 + 1] = lock_char
    buf[hub_r0 + 1][exit_c0 + exit_w - 1] = "E"
    buf[hub_r0 + 1][hub_c0 + 1] = "P"

    # Key corridor (North or South)
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

    # Optional opposite distractor alcove
    if add_distractor:
        if key_on_north and south_h >= 3:
            _carve_rect(buf, hub_r1 + 2, cor_c - 1, total_h - 2, cor_c + 1)
            buf[hub_r1 + 1][cor_c] = "D"
            buf[total_h - 2][cor_c] = "h"
        elif (not key_on_north) and north_h >= 3:
            _carve_rect(buf, 1, cor_c - 1, hub_r0 - 2, cor_c + 1)
            buf[hub_r0 - 1][cor_c] = "D"
            buf[1][cor_c] = "h"

    if with_combat:
        buf[hub_r0][hub_c0 + 2] = "g"
        buf[hub_r1][hub_c1] = rng.choice(["z", "s"])

    angle = rng.choice([0, 90, 270])
    return _to_grid_str(buf), angle, 64


def _gen_gauntlet_grid(rng: random.Random) -> tuple[str, int, int]:
    """Generate a two-lock gauntlet with combat and scarce resources."""
    pairs = [("R", "r"), ("B", "b"), ("Y", "y")]
    rng.shuffle(pairs)
    (lock1, key1), (lock2, key2) = pairs[0], pairs[1]

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
    # Wing 1 (NW): regular door -> key1 + guard
    buf[2][6] = "D"
    buf[2][2] = key1
    buf[2][4] = rng.choice(["z", "i"])
    buf[3][2] = "H"
    # Wing 2 (NE): lock1 -> key2 + tougher guard
    buf[2][10] = lock1
    buf[2][14] = key2
    buf[2][12] = rng.choice(["i", "s"])
    buf[3][14] = "h"
    # Hub & Exit (SE): lock2 -> exit
    buf[6][2] = "P"
    buf[6][3] = "g"
    buf[7][3] = "e"
    buf[6][7] = rng.choice(["z", "s", "m"])
    buf[6][10] = lock2
    buf[6][14] = "E"

    angle = rng.choice([0, 90])
    return _to_grid_str(buf), angle, 80


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

    Supported families: `"combat"`, `"multi_room"`, `"key_return"`, `"gauntlet"`.
    """
    rng = random.Random(int(seed))
    theme = rng.choice(THEME_NAMES)
    if family == "combat":
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
