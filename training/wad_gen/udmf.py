"""Zero-dependency ZDoom UDMF WAD compilers for Role 3.

Supports two completely independent map construction methods:
  1. `build_grid_wad`: Orthogonal grid-to-UDMF compiler with merged sectors,
     arbitrary cell scale, N/S/E/W doors & locked doors, cover pillars, and things.
  2. `build_polygon_wad`: Non-grid polygonal/angled-wall UDMF compiler for convex
     polygonal chambers (octagons, hexagons, diamonds, trapezoids, angled corridors)
     and interior polygonal pillars — used for the distinct-construction `final` family.

Both emit pure `namespace = "zdoom"` UDMF WADs (`MAP01`, `TEXTMAP`, `ENDMAP`) that
ViZDoom 1.3.3 builds nodes for automatically at load time and `doom_env.mapinfo`
parses with exact static geometry (`source == "wad"`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Sequence

from doom_env.wad import Lump, write_wad

# ZDoom lock numbers (card or skull of matching color)
LOCK_CODES: dict[str, int] = {
    "R": 129,  # Red card or skull
    "B": 130,  # Blue card or skull
    "Y": 131,  # Yellow card or skull
}

DOOR_TEXTURES: dict[str, str] = {
    "D": "BIGDOOR2",
    "R": "DOORRED",
    "B": "DOORBLU",
    "Y": "DOORYEL",
}

DOOR_CHARS = frozenset({"D", "R", "B", "Y"})
LOCKED_DOOR_CHARS = frozenset({"R", "B", "Y"})
KEY_CHARS = frozenset({"r", "b", "y"})
PLAYER_CHARS: dict[str, int | None] = {
    "P": None,  # uses caller-specified player_angle
    ">": 0,
    "^": 90,
    "<": 180,
    "v": 270,
}

# Standard Doom2 / Freedoom2 Thing DoomEdNums
THING_DOOMEDNUM: dict[str, int] = {
    # Keys
    "b": 5,     # BlueCard
    "y": 6,     # YellowCard
    "r": 13,    # RedCard
    # Monsters
    "z": 3004,  # Zombieman (hitscan, 20 HP)
    "s": 9,     # ShotgunGuy (hitscan, 30 HP)
    "c": 65,    # ChaingunGuy (hitscan, 70 HP)
    "i": 3001,  # DoomImp (projectile, 60 HP)
    "m": 3002,  # Demon (melee, 150 HP)
    "S": 58,    # Spectre (melee, 150 HP)
    "l": 3006,  # LostSoul (melee, 100 HP)
    "k": 3005,  # Cacodemon (projectile, 400 HP)
    "n": 69,    # HellKnight (projectile, 500 HP)
    # Pickups & props
    "h": 2011,  # Stimpack (+10 HP)
    "H": 2012,  # Medikit (+25 HP)
    "A": 2018,  # GreenArmor (+100 armor)
    "M": 2019,  # BlueArmor (+200 armor)
    "a": 2048,  # ClipBox (+50 bullets)
    "e": 2049,  # ShellBox (+20 shells)
    "o": 2046,  # RocketBox (+5 rockets)
    "u": 2047,  # CellPack (+100 cells)
    "g": 2001,  # Shotgun
    "G": 2002,  # Chaingun
    "W": 82,    # SuperShotgun
    "L": 2003,  # RocketLauncher
    "F": 2004,  # PlasmaRifle
    "O": 2024,  # BlurSphere (Powerup -> other)
    "x": 2035,  # ExplosiveBarrel
}

MONSTER_CHARS = frozenset({"z", "s", "c", "i", "m", "S", "l", "k", "n"})

# Visual texture themes (all standard Freedoom2 / Doom2 textures)
THEMES: dict[str, tuple[str, str, str, int]] = {
    # name -> (wall_tex, floor_tex, ceil_tex, lightlevel)
    "techbase": ("STARTAN2", "FLOOR0_1", "CEIL1_1", 192),
    "stone": ("STONE2", "FLOOR4_8", "CEIL3_5", 176),
    "brown": ("BROWN1", "FLAT14", "CEIL1_1", 184),
    "metal": ("METAL1", "FLOOR5_1", "CEIL5_1", 192),
    "gray": ("GRAY1", "FLOOR0_3", "FLAT19", 184),
    "brick": ("BRICK1", "FLOOR7_1", "CEIL3_2", 176),
}

THEME_NAMES = tuple(THEMES.keys())


@dataclass(frozen=True)
class ThingSpec:
    """Explicit thing placement in world coordinates."""

    x: float
    y: float
    doomednum: int
    angle: int = 0


@dataclass(frozen=True)
class PolygonHole:
    """Interior solid pillar inside a polygonal sector (vertices in any order; normalized CCW)."""

    vertices: tuple[tuple[float, float], ...]
    wall_texture: str = "STONE2"


@dataclass(frozen=True)
class PolygonSector:
    """One convex or simple polygonal sector for the non-grid polygonal UDMF builder.

    `kind` is one of:
      - "room": regular playable space (ceiling 128)
      - "door": unlocked manual door (ceiling starts at height_floor, special=12)
      - "remote_door": switch-triggered door (ceiling starts at height_floor, opened via `door_switch` matching `sector_id`)
      - "door_red" / "door_blue" / "door_yellow": locked door (ceiling starts at height_floor, special=13)
      - "exit": exit pad (ceiling 128, GATE3 floor, special=243 playercross)
    """

    name: str
    vertices: tuple[tuple[float, float], ...]
    kind: str = "room"
    wall_texture: str = "STARTAN2"
    floor_texture: str = "FLOOR0_1"
    ceiling_texture: str = "CEIL1_1"
    height_floor: int = 0
    height_ceiling: int = 128
    light_level: int = 192
    holes: tuple[PolygonHole, ...] = field(default_factory=tuple)
    # Exterior edge activating Exit_Normal on a USE pulse (either vertex order).
    exit_switch: tuple[tuple[float, float], tuple[float, float]] | None = None
    # Optional UDMF sector tag ID (used by remote switch-triggered doors).
    sector_id: int = 0
    # Exterior edge activating Door_Raise on `target_sector_id`: (p1, p2, target_sector_id).
    door_switch: tuple[tuple[float, float], tuple[float, float], int] | None = None
    # Two-sided shared edges marked impassable (sight passes through, movement/rays stop).
    blocking_edges: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = field(default_factory=tuple)


def layout_hash(spec_text: str) -> str:
    """Stable SHA-1 digest of a normalized layout definition for split-overlap checks."""
    normalized = "\n".join(line.rstrip() for line in spec_text.strip().splitlines())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _parse_grid(grid: str) -> list[str]:
    rows = [line.strip() for line in grid.strip("\n").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Grid cannot be empty")
    valid_chars = {"#", ".", "E", *DOOR_CHARS, *PLAYER_CHARS, *THING_DOOMEDNUM}
    for r_idx, row in enumerate(rows):
        unknown = set(row) - valid_chars
        if unknown:
            raise ValueError(f"Unknown grid character(s) {sorted(unknown)} on row {r_idx}: {row!r}")
    width = max(len(r) for r in rows)
    # Pad ragged rows with walls and add an outer wall border if not already closed
    padded = [r.ljust(width, "#") for r in rows]
    needs_border = (
        any(ch != "#" for ch in padded[0])
        or any(ch != "#" for ch in padded[-1])
        or any(r[0] != "#" or r[-1] != "#" for r in padded)
    )
    if needs_border:
        w2 = width + 2
        padded = ["#" * w2, *[f"#{r}#" for r in padded], "#" * w2]
    return padded


def _sector_kind_for_cell(ch: str) -> str:
    if ch == "#":
        return "void"
    if ch == "D":
        return "door"
    if ch == "R":
        return "door_red"
    if ch == "B":
        return "door_blue"
    if ch == "Y":
        return "door_yellow"
    if ch == "E":
        return "exit"
    return "room"


def _door_code_from_kind(kind: str) -> str | None:
    return {
        "door": "D",
        "door_red": "R",
        "door_blue": "B",
        "door_yellow": "Y",
    }.get(kind)


def _signed_area(pts: Sequence[tuple[float, float]]) -> float:
    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _snap_pt(pt: tuple[float, float]) -> tuple[float, float]:
    return (round(float(pt[0]), 2), round(float(pt[1]), 2))


def _format_thing(t: ThingSpec) -> str:
    return (
        f"thing {{ x = {t.x:.1f}; y = {t.y:.1f}; angle = {int(t.angle) % 360}; "
        f"type = {int(t.doomednum)}; "
        f"skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }}"
    )


def _emit_udmf(
    vertices: list[tuple[float, float]],
    sectors: list[dict[str, object]],
    sidedefs: list[dict[str, object]],
    linedefs: list[dict[str, object]],
    things: list[ThingSpec],
) -> str:
    lines: list[str] = ['namespace = "zdoom";']
    for x, y in vertices:
        lines.append(f"vertex {{ x = {x:.1f}; y = {y:.1f}; }}")
    for sec in sectors:
        id_part = f' id = {int(sec["id"])};' if int(sec.get("id", 0) or 0) != 0 else ""
        lines.append(
            f'sector {{ texturefloor = "{sec["floor"]}"; textureceiling = "{sec["ceil"]}"; '
            f'heightfloor = {int(sec["hf"])}; heightceiling = {int(sec["hc"])}; '
            f'lightlevel = {int(sec["light"])};{id_part} }}'
        )
    for sd in sidedefs:
        lines.append(
            f'sidedef {{ sector = {int(sd["sector"])}; '
            f'texturetop = "{sd["top"]}"; texturemiddle = "{sd["mid"]}"; texturebottom = "{sd["bot"]}"; }}'
        )
    for ld in linedefs:
        parts = [f"v1 = {int(ld['v1'])}", f"v2 = {int(ld['v2'])}", f"sidefront = {int(ld['front'])}"]
        if ld.get("back") is not None:
            parts.append(f"sideback = {int(ld['back'])}")
        if ld.get("twosided"):
            parts.append("twosided = true")
        if ld.get("blocking"):
            parts.append("blocking = true")
        if ld.get("dontpegbottom"):
            parts.append("dontpegbottom = true")
        special = int(ld.get("special", 0) or 0)
        if special:
            parts.append(f"special = {special}")
            for k in range(5):
                ak = f"arg{k}"
                if ak in ld:
                    parts.append(f"{ak} = {int(ld[ak])}")
            for flag in ("playeruse", "monsteruse", "repeatspecial", "playercross"):
                if flag in ld:
                    parts.append(f"{flag} = {'true' if ld[flag] else 'false'}")
        lines.append("linedef { " + "; ".join(parts) + "; }")
    for t in things:
        lines.append(_format_thing(t))
    return "\n".join(lines) + "\n"


def _write_textmap_wad(path: str | Path, textmap: str, map_name: str = "MAP01") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lumps = [
        Lump(map_name.upper(), b""),
        Lump("TEXTMAP", textmap.encode("utf-8")),
        Lump("ENDMAP", b""),
    ]
    write_wad(out, lumps, kind=b"PWAD")
    return out


def build_grid_textmap(
    grid: str,
    *,
    cell_size: int = 64,
    player_angle: int = 0,
    theme: str = "techbase",
    monster_angle: int = 180,
    extra_things: Sequence[ThingSpec] = (),
    skull_keys: bool = False,
    door_close_delay: int = 150,
) -> str:
    """Compile a 2D ASCII grid into a valid ZDoom UDMF `TEXTMAP` string.

    Each connected component of normal open cells becomes one sector (with varied
    room textures), each connected component of identical door cells becomes a door
    sector (ceiling=0), and each connected component of exit cells becomes an exit
    sector (`GATE3`, `special=243 playercross`).
    """
    rows = _parse_grid(grid)
    height = len(rows)
    width = len(rows[0])
    s = float(cell_size)

    # 1. Assign sector IDs by flood-filling connected components of identical sector kind.
    cell_sector: list[list[int]] = [[-1] * width for _ in range(height)]
    sector_kinds: list[str] = []
    for r in range(height):
        for c in range(width):
            ch = rows[r][c]
            kind = _sector_kind_for_cell(ch)
            if kind == "void" or cell_sector[r][c] != -1:
                continue
            sid = len(sector_kinds)
            sector_kinds.append(kind)
            q: deque[tuple[int, int]] = deque([(r, c)])
            cell_sector[r][c] = sid
            while q:
                cr, cc = q.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if cell_sector[nr][nc] == -1 and _sector_kind_for_cell(rows[nr][nc]) == kind:
                            cell_sector[nr][nc] = sid
                            q.append((nr, nc))

    if not sector_kinds:
        raise ValueError("Grid has no playable cells")
    if len(sector_kinds) >= 2048:
        raise ValueError(f"Too many sectors ({len(sector_kinds)} >= 2048)")

    # 2. Build sector definitions with clean theme variation per room.
    base_idx = THEME_NAMES.index(theme) if theme in THEME_NAMES else 0
    room_counter = 0
    sectors: list[dict[str, object]] = []
    sector_wall_tex: list[str] = []
    for kind in sector_kinds:
        if kind == "room":
            tname = THEME_NAMES[(base_idx + room_counter) % len(THEME_NAMES)]
            room_counter += 1
            wall_tex, floor_tex, ceil_tex, light = THEMES[tname]
            sectors.append({"floor": floor_tex, "ceil": ceil_tex, "hf": 0, "hc": 128, "light": light})
            sector_wall_tex.append(wall_tex)
        elif kind == "exit":
            sectors.append({"floor": "GATE3", "ceil": "CEIL1_1", "hf": 0, "hc": 128, "light": 208})
            sector_wall_tex.append("GRAY1")
        else:
            # Door sector: ceiling starts closed at 0
            sectors.append({"floor": "FLAT5_4", "ceil": "CEIL1_1", "hf": 0, "hc": 0, "light": 192})
            sector_wall_tex.append("DOORTRAK")

    # 3. Collect all directed cell boundary segments.
    # For each cell (r, c) in sector sid, check its 4 edges:
    #   West  (neighbor c-1): (x0, y0) -> (x0, y1)  [front = East = inside (r,c)]
    #   North (neighbor r-1): (x0, y1) -> (x1, y1)  [front = South = inside (r,c)]
    #   East  (neighbor c+1): (x1, y1) -> (x1, y0)  [front = West = inside (r,c)]
    #   South (neighbor r+1): (x1, y0) -> (x0, y0)  [front = North = inside (r,c)]
    # Each undirected segment between two different sectors is emitted once, oriented so
    # that `front` is the non-door/non-exit room sector whenever one side is a door/exit.
    raw_edges: list[tuple[tuple[float, float], tuple[float, float], int, int]] = []
    seen_two_sided: set[tuple[tuple[float, float], tuple[float, float]]] = set()

    for r in range(height):
        y0 = (height - 1 - r) * s
        y1 = (height - r) * s
        for c in range(width):
            sid = cell_sector[r][c]
            if sid < 0:
                continue
            x0 = c * s
            x1 = (c + 1) * s
            neighbors = (
                (r, c - 1, (x0, y0), (x0, y1)),  # West
                (r - 1, c, (x0, y1), (x1, y1)),  # North
                (r, c + 1, (x1, y1), (x1, y0)),  # East
                (r + 1, c, (x1, y0), (x0, y0)),  # South
            )
            for nr, nc, p1, p2 in neighbors:
                other = cell_sector[nr][nc] if (0 <= nr < height and 0 <= nc < width) else -1
                if other == sid:
                    continue
                if other < 0:
                    raw_edges.append((p1, p2, sid, -1))
                else:
                    ukey = (min(p1, p2), max(p1, p2))
                    if ukey in seen_two_sided:
                        continue
                    seen_two_sided.add(ukey)
                    k_front = sector_kinds[sid]
                    k_back = sector_kinds[other]
                    # Ensure sidefront is the room and sideback is the door or exit
                    if k_front != "room" and k_back == "room":
                        raw_edges.append((p2, p1, other, sid))
                    elif k_front == "exit" and k_back != "exit":
                        raw_edges.append((p2, p1, other, sid))
                    else:
                        raw_edges.append((p1, p2, sid, other))

    # 4. Merge collinear adjacent segments with identical (front_sector, back_sector).
    merged_edges = _merge_collinear_edges(raw_edges)

    # 5. Build vertices, sidedefs, linedefs.
    vert_index: dict[tuple[float, float], int] = {}
    vertices: list[tuple[float, float]] = []

    def get_v(pt: tuple[float, float]) -> int:
        pt = _snap_pt(pt)
        idx = vert_index.get(pt)
        if idx is None:
            idx = len(vertices)
            vert_index[pt] = idx
            vertices.append(pt)
        return idx

    sidedefs: list[dict[str, object]] = []
    linedefs: list[dict[str, object]] = []

    for p1, p2, front_sec, back_sec in merged_edges:
        v1 = get_v(p1)
        v2 = get_v(p2)
        if back_sec < 0:
            sd_front = len(sidedefs)
            sidedefs.append({
                "sector": front_sec,
                "top": "-",
                "mid": sector_wall_tex[front_sec],
                "bot": "-",
            })
            linedefs.append({
                "v1": v1,
                "v2": v2,
                "front": sd_front,
                "blocking": True,
                "dontpegbottom": True,
            })
        else:
            back_kind = sector_kinds[back_sec]
            front_kind = sector_kinds[front_sec]
            door_code = _door_code_from_kind(back_kind)
            if door_code is not None and _door_code_from_kind(front_kind) is not None:
                raise ValueError("Two different door sectors cannot directly touch each other")

            if door_code is not None:
                sd_front = len(sidedefs)
                sidedefs.append({
                    "sector": front_sec,
                    "top": DOOR_TEXTURES[door_code],
                    "mid": "-",
                    "bot": "-",
                })
                sd_back = len(sidedefs)
                sidedefs.append({
                    "sector": back_sec,
                    "top": "-",
                    "mid": "-",
                    "bot": "-",
                })
                if door_code == "D":
                    linedefs.append({
                        "v1": v1,
                        "v2": v2,
                        "front": sd_front,
                        "back": sd_back,
                        "twosided": True,
                        "special": 12,
                        "arg0": 0,
                        "arg1": 16,
                        "arg2": int(door_close_delay),
                        "playeruse": True,
                        "monsteruse": True,
                        "repeatspecial": True,
                    })
                else:
                    linedefs.append({
                        "v1": v1,
                        "v2": v2,
                        "front": sd_front,
                        "back": sd_back,
                        "twosided": True,
                        "special": 13,
                        "arg0": 0,
                        "arg1": 16,
                        "arg2": 0,  # stay open once unlocked
                        "arg3": LOCK_CODES[door_code],
                        "playeruse": True,
                        "monsteruse": False,
                        "repeatspecial": True,
                    })
            elif back_kind == "exit":
                sd_front = len(sidedefs)
                sidedefs.append({"sector": front_sec, "top": "-", "mid": "-", "bot": "-"})
                sd_back = len(sidedefs)
                sidedefs.append({"sector": back_sec, "top": "-", "mid": "-", "bot": "-"})
                linedefs.append({
                    "v1": v1,
                    "v2": v2,
                    "front": sd_front,
                    "back": sd_back,
                    "twosided": True,
                    "special": 243,
                    "playercross": True,
                })
            else:
                sd_front = len(sidedefs)
                sidedefs.append({"sector": front_sec, "top": "-", "mid": "-", "bot": "-"})
                sd_back = len(sidedefs)
                sidedefs.append({"sector": back_sec, "top": "-", "mid": "-", "bot": "-"})
                linedefs.append({
                    "v1": v1,
                    "v2": v2,
                    "front": sd_front,
                    "back": sd_back,
                    "twosided": True,
                })

    # 6. Place Things from grid characters + optional extra_things.
    things: list[ThingSpec] = []
    player_count = 0
    for r in range(height):
        cy = (height - 1 - r + 0.5) * s
        for c in range(width):
            ch = rows[r][c]
            cx = (c + 0.5) * s
            if ch in PLAYER_CHARS:
                player_count += 1
                ang = PLAYER_CHARS[ch] if PLAYER_CHARS[ch] is not None else player_angle
                things.append(ThingSpec(x=cx, y=cy, doomednum=1, angle=int(ang)))
            elif ch in THING_DOOMEDNUM:
                ang = monster_angle if ch in MONSTER_CHARS else 0
                number = THING_DOOMEDNUM[ch]
                if skull_keys and ch in KEY_CHARS:
                    number = {"b": 40, "y": 39, "r": 38}[ch]
                things.append(ThingSpec(x=cx, y=cy, doomednum=number, angle=ang))
    things.extend(extra_things)

    if player_count != 1:
        raise ValueError(f"Grid must contain exactly 1 player spawn (P/>/^/</v), found {player_count}")

    return _emit_udmf(vertices, sectors, sidedefs, linedefs, things)


def _merge_collinear_edges(
    edges: list[tuple[tuple[float, float], tuple[float, float], int, int]],
) -> list[tuple[tuple[float, float], tuple[float, float], int, int]]:
    """Merge consecutive axis-aligned collinear edges sharing identical (front, back) sectors."""
    # Group by (direction, line_coord, front, back)
    groups: dict[tuple[str, float, int, int], list[tuple[float, float]]] = {}
    for (x1, y1), (x2, y2), front, back in edges:
        if x1 == x2:
            direction = "N" if y2 > y1 else "S"
            a, b = (y1, y2) if direction == "N" else (-y1, -y2)
            groups.setdefault((direction, x1, front, back), []).append((a, b))
        elif y1 == y2:
            direction = "E" if x2 > x1 else "W"
            a, b = (x1, x2) if direction == "E" else (-x1, -x2)
            groups.setdefault((direction, y1, front, back), []).append((a, b))
        else:
            raise ValueError("Grid edges must be axis-aligned")

    out: list[tuple[tuple[float, float], tuple[float, float], int, int]] = []
    for (direction, coord, front, back), intervals in sorted(groups.items()):
        intervals.sort()
        cur_a, cur_b = intervals[0]
        for a, b in intervals[1:]:
            if abs(a - cur_b) < 1e-6:
                cur_b = b
            else:
                out.append(_reconstruct_edge(direction, coord, cur_a, cur_b, front, back))
                cur_a, cur_b = a, b
        out.append(_reconstruct_edge(direction, coord, cur_a, cur_b, front, back))
    return out


def _reconstruct_edge(
    direction: str, coord: float, a: float, b: float, front: int, back: int
) -> tuple[tuple[float, float], tuple[float, float], int, int]:
    if direction == "N":
        return ((coord, a), (coord, b), front, back)
    if direction == "S":
        return ((coord, -a), (coord, -b), front, back)
    if direction == "E":
        return ((a, coord), (b, coord), front, back)
    return ((-a, coord), (-b, coord), front, back)


def build_grid_wad(
    path: str | Path,
    grid: str,
    *,
    map_name: str = "MAP01",
    cell_size: int = 64,
    player_angle: int = 0,
    theme: str = "techbase",
    monster_angle: int = 180,
    extra_things: Sequence[ThingSpec] = (),
    skull_keys: bool = False,
    door_close_delay: int = 150,
) -> Path:
    """Compile a 2D ASCII grid and write a complete UDMF PWAD file."""
    textmap = build_grid_textmap(
        grid,
        cell_size=cell_size,
        player_angle=player_angle,
        theme=theme,
        monster_angle=monster_angle,
        extra_things=extra_things,
        skull_keys=skull_keys,
        door_close_delay=door_close_delay,
    )
    return _write_textmap_wad(path, textmap, map_name=map_name)


def build_polygon_textmap(
    poly_sectors: Sequence[PolygonSector],
    things: Sequence[ThingSpec],
    *,
    door_close_delay: int = 150,
) -> str:
    """Compile non-grid polygonal sectors (with optional angled walls & pillar holes) into UDMF.

    Adjacent polygons that share an identical boundary edge `(p, q)` / `(q, p)` are
    automatically stitched with a two-sided linedef (door, locked door, exit, or open portal).
    Exterior polygon boundaries are normalized to clockwise winding so Doom's right-hand
    front side always faces inward; interior pillar holes are normalized to CCW winding
    so their front side faces outward into the enclosing room sector.
    """
    if not poly_sectors:
        raise ValueError("poly_sectors cannot be empty")
    if len(poly_sectors) >= 2048:
        raise ValueError(f"Too many sectors ({len(poly_sectors)} >= 2048)")
    if sum(1 for t in things if t.doomednum == 1) != 1:
        raise ValueError("Polygonal map must contain exactly 1 player start (type=1)")

    sectors: list[dict[str, object]] = []
    for ps in poly_sectors:
        is_door = (_door_code_from_kind(ps.kind) is not None) or (ps.kind == "remote_door")
        hc = ps.height_floor if is_door else ps.height_ceiling
        floor = "GATE3" if ps.kind == "exit" else ("FLAT5_4" if is_door else ps.floor_texture)
        sectors.append({
            "floor": floor,
            "ceil": ps.ceiling_texture,
            "hf": ps.height_floor,
            "hc": hc,
            "light": ps.light_level,
            "id": int(ps.sector_id),
        })

    # Collect directed boundary edges from every polygon (normalized CW -> interior on right).
    # Key by undirected segment (min(p1,p2), max(p1,p2)) to stitch shared edges.
    edge_owners: dict[
        tuple[tuple[float, float], tuple[float, float]],
        list[tuple[tuple[float, float], tuple[float, float], int, str]],
    ] = {}

    for sid, ps in enumerate(poly_sectors):
        pts = [_snap_pt(p) for p in ps.vertices]
        if len(pts) < 3:
            raise ValueError(f"Polygon sector {ps.name!r} has fewer than 3 vertices")
        # Normalize exterior boundary to CW (signed area < 0) so interior is on the RIGHT.
        if _signed_area(pts) > 0:
            pts.reverse()
        n = len(pts)
        for i in range(n):
            p1, p2 = pts[i], pts[(i + 1) % n]
            if p1 == p2:
                continue
            ukey = (min(p1, p2), max(p1, p2))
            is_door = (_door_code_from_kind(ps.kind) is not None) or (ps.kind == "remote_door")
            wall_tex = "DOORTRAK" if is_door else ps.wall_texture
            edge_owners.setdefault(ukey, []).append((p1, p2, sid, wall_tex))

    vert_index: dict[tuple[float, float], int] = {}
    vertices: list[tuple[float, float]] = []

    def get_v(pt: tuple[float, float]) -> int:
        pt = _snap_pt(pt)
        idx = vert_index.get(pt)
        if idx is None:
            idx = len(vertices)
            vert_index[pt] = idx
            vertices.append(pt)
        return idx

    sidedefs: list[dict[str, object]] = []
    linedefs: list[dict[str, object]] = []

    blocking_edge_keys: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for sid, ps in enumerate(poly_sectors):
        if ps.exit_switch is not None:
            edge = tuple(sorted(_snap_pt(p) for p in ps.exit_switch))
            owners = edge_owners.get(edge, [])
            if len(owners) != 1 or owners[0][2] != sid:
                raise ValueError(f"Exit switch in {ps.name} requires an exterior edge")
        if ps.door_switch is not None:
            sp1, sp2, target_id = ps.door_switch
            if int(target_id) <= 0:
                raise ValueError(f"Door switch in {ps.name} requires positive target sector_id")
            edge = tuple(sorted((_snap_pt(sp1), _snap_pt(sp2))))
            owners = edge_owners.get(edge, [])
            if len(owners) != 1 or owners[0][2] != sid:
                raise ValueError(f"Door switch in {ps.name} requires an exterior edge")
        for bp1, bp2 in ps.blocking_edges:
            blocking_edge_keys.add(tuple(sorted((_snap_pt(bp1), _snap_pt(bp2)))))

    for ukey, owners in edge_owners.items():
        if len(owners) == 1:
            p1, p2, sid, wall_tex = owners[0]
            ps_cur = poly_sectors[sid]
            switch = ps_cur.exit_switch
            is_exit_switch = switch is not None and ukey == tuple(sorted(_snap_pt(p) for p in switch))
            dswitch = ps_cur.door_switch
            is_door_switch = (
                dswitch is not None and ukey == tuple(sorted((_snap_pt(dswitch[0]), _snap_pt(dswitch[1]))))
            )
            mid_tex = "SW1EXIT" if is_exit_switch else ("SW1COMP" if is_door_switch else wall_tex)
            sd_front = len(sidedefs)
            sidedefs.append({"sector": sid, "top": "-", "mid": mid_tex, "bot": "-"})
            ld_dict: dict[str, object] = {
                "v1": get_v(p1),
                "v2": get_v(p2),
                "front": sd_front,
                "blocking": True,
                "dontpegbottom": True,
            }
            if is_exit_switch:
                ld_dict.update({"special": 243, "playeruse": True})
            elif is_door_switch:
                assert dswitch is not None
                ld_dict.update({
                    "special": 12,
                    "arg0": int(dswitch[2]),
                    "arg1": 16,
                    "arg2": 0,  # stay open once triggered by remote switch
                    "playeruse": True,
                    "repeatspecial": True,
                })
            linedefs.append(ld_dict)
        elif len(owners) == 2:
            (p1a, p2a, sa, _), (p1b, p2b, sb, _) = owners
            if sa == sb:
                continue
            # Ensure opposite directions (valid non-overlapping stitch)
            if not (p1a == p2b and p2a == p1b):
                raise ValueError(f"Overlapping polygons share same-direction edge {ukey}")
            ka, kb = poly_sectors[sa].kind, poly_sectors[sb].kind
            # Pick front sector to be the regular room (so back sector is door/exit)
            if ka != "room" and kb == "room":
                p1, p2, front_sec, back_sec = p1b, p2b, sb, sa
            elif ka == "exit" and kb != "exit":
                p1, p2, front_sec, back_sec = p1b, p2b, sb, sa
            else:
                p1, p2, front_sec, back_sec = p1a, p2a, sa, sb

            back_kind = poly_sectors[back_sec].kind
            door_code = _door_code_from_kind(back_kind)
            v1, v2 = get_v(p1), get_v(p2)
            if door_code is not None:
                sd_front = len(sidedefs)
                sidedefs.append({"sector": front_sec, "top": DOOR_TEXTURES[door_code], "mid": "-", "bot": "-"})
                sd_back = len(sidedefs)
                sidedefs.append({"sector": back_sec, "top": "-", "mid": "-", "bot": "-"})
                if door_code == "D":
                    linedefs.append({
                        "v1": v1,
                        "v2": v2,
                        "front": sd_front,
                        "back": sd_back,
                        "twosided": True,
                        "special": 12,
                        "arg0": 0,
                        "arg1": 16,
                        "arg2": int(door_close_delay),
                        "playeruse": True,
                        "monsteruse": True,
                        "repeatspecial": True,
                    })
                else:
                    linedefs.append({
                        "v1": v1,
                        "v2": v2,
                        "front": sd_front,
                        "back": sd_back,
                        "twosided": True,
                        "special": 13,
                        "arg0": 0,
                        "arg1": 16,
                        "arg2": 0,
                        "arg3": LOCK_CODES[door_code],
                        "playeruse": True,
                        "monsteruse": False,
                        "repeatspecial": True,
                    })
            elif back_kind == "remote_door":
                sd_front = len(sidedefs)
                sidedefs.append({"sector": front_sec, "top": DOOR_TEXTURES["D"], "mid": "-", "bot": "-"})
                sd_back = len(sidedefs)
                sidedefs.append({"sector": back_sec, "top": "-", "mid": "-", "bot": "-"})
                linedefs.append({
                    "v1": v1,
                    "v2": v2,
                    "front": sd_front,
                    "back": sd_back,
                    "twosided": True,
                })
            elif back_kind == "exit":
                sd_front = len(sidedefs)
                sidedefs.append({"sector": front_sec, "top": "-", "mid": "-", "bot": "-"})
                sd_back = len(sidedefs)
                sidedefs.append({"sector": back_sec, "top": "-", "mid": "-", "bot": "-"})
                linedefs.append({
                    "v1": v1,
                    "v2": v2,
                    "front": sd_front,
                    "back": sd_back,
                    "twosided": True,
                    "special": 243,
                    "playercross": True,
                })
            else:
                is_grate = ukey in blocking_edge_keys
                sd_front = len(sidedefs)
                sd_back = sd_front + 1
                for sid, opposite in ((front_sec, back_sec), (back_sec, front_sec)):
                    room, neighbor = poly_sectors[sid], poly_sectors[opposite]
                    sidedefs.append({
                        "sector": sid,
                        "mid": "MIDBARS3" if is_grate else "-",
                        "top": neighbor.wall_texture if room.height_ceiling > neighbor.height_ceiling else "-",
                        "bot": neighbor.wall_texture if room.height_floor < neighbor.height_floor else "-",
                    })
                ld_portal: dict[str, object] = {
                    "v1": v1,
                    "v2": v2,
                    "front": sd_front,
                    "back": sd_back,
                    "twosided": True,
                }
                if is_grate:
                    ld_portal["blocking"] = True
                linedefs.append(ld_portal)
        else:
            raise ValueError(f"Edge {ukey} shared by {len(owners)} polygons (max 2)")

    # Emit interior pillar holes (normalized CCW -> right-hand front faces outward into room)
    for sid, ps in enumerate(poly_sectors):
        for hole in ps.holes:
            hpts = [_snap_pt(p) for p in hole.vertices]
            if len(hpts) < 3:
                raise ValueError(f"Hole in sector {ps.name!r} has fewer than 3 vertices")
            if _signed_area(hpts) < 0:
                hpts.reverse()  # ensure CCW
            hn = len(hpts)
            for i in range(hn):
                p1, p2 = hpts[i], hpts[(i + 1) % hn]
                sd_front = len(sidedefs)
                sidedefs.append({"sector": sid, "top": "-", "mid": hole.wall_texture, "bot": "-"})
                linedefs.append({
                    "v1": get_v(p1),
                    "v2": get_v(p2),
                    "front": sd_front,
                    "blocking": True,
                    "dontpegbottom": True,
                })

    return _emit_udmf(vertices, sectors, sidedefs, linedefs, list(things))


def build_polygon_wad(
    path: str | Path,
    poly_sectors: Sequence[PolygonSector],
    things: Sequence[ThingSpec],
    *,
    map_name: str = "MAP01",
    door_close_delay: int = 150,
) -> Path:
    """Compile a non-grid polygonal map and write a complete UDMF PWAD file."""
    textmap = build_polygon_textmap(poly_sectors, things, door_close_delay=door_close_delay)
    return _write_textmap_wad(path, textmap, map_name=map_name)
