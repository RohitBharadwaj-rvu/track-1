"""Static playability & real ViZDoom/doom_env verification for Role 3 WADs.

Verifies every map before adding it to training/dev/final suites (REVISED_PLAN.md §3, §7):
  - < 2,048 sectors, exact WAD static geometry (`env._geometry.source == "wad"`)
  - Valid door and locked-door linedef specials emitting RAY_DOOR (2) / RAY_LOCKED (3)
  - Reachability: every key is reachable before its lock, every door is traversable,
    and the exit is reachable from spawn (zero soft-locks!)
  - Real `doom_env` + `PrivilegedEventTracker` smoke execution on freedoom2.wad / skill 3
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from doom_env import make_env
from doom_env.mapinfo import load_mapinfo
from doom_env.sensors import MAX_STEP_UP, PLAYER_HEIGHT, RAY_DOOR, RAY_LOCKED
from training.interfaces import PrivilegedEventTracker
from training.wad_gen.udmf import (
    DOOR_CHARS,
    KEY_CHARS,
    LOCKED_DOOR_CHARS,
    MONSTER_CHARS,
    PLAYER_CHARS,
    PolygonSector,
    ThingSpec,
    _door_code_from_kind,
    _parse_grid,
    _signed_area,
    _snap_pt,
)

_LOCK_TO_KEY = {"R": "r", "B": "b", "Y": "y"}
_KEY_DOOMEDNUM_TO_CHAR = {13: "r", 5: "b", 6: "y", 38: "r", 40: "b", 39: "y"}
_DOOR_KIND_TO_KEY = {"door_red": "r", "door_blue": "b", "door_yellow": "y"}


@dataclass(frozen=True)
class GridVerificationResult:
    ok: bool
    exit_reachable: bool
    keys_present: tuple[str, ...]
    keys_reachable: tuple[str, ...]
    unlocked_doors: int
    locked_doors: int
    monsters: int
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class WadVerificationResult:
    ok: bool
    wad: str
    map_name: str
    num_sectors: int
    geometry_source: str
    num_unlocked_doors: int
    num_locked_doors: int
    has_door_ray_lines: bool
    has_locked_ray_lines: bool
    errors: tuple[str, ...] = ()


def verify_grid_solvability(grid: str) -> GridVerificationResult:
    """Verify via progression-aware graph search that a grid map has no soft-locks.

    (Offline map verification only — never imported or used by `submission/`.)
    """
    rows = _parse_grid(grid)
    h, w = len(rows), len(rows[0])
    errors: list[str] = []

    spawns: list[tuple[int, int]] = []
    exits: list[tuple[int, int]] = []
    keys_pos: dict[str, list[tuple[int, int]]] = {}
    monsters = 0

    for r in range(h):
        for c in range(w):
            ch = rows[r][c]
            if ch in PLAYER_CHARS:
                spawns.append((r, c))
            elif ch == "E":
                exits.append((r, c))
            elif ch in KEY_CHARS:
                keys_pos.setdefault(ch, []).append((r, c))
            elif ch in MONSTER_CHARS:
                monsters += 1

    if len(spawns) != 1:
        errors.append(f"Expected exactly 1 player spawn, found {len(spawns)}")
    if not exits:
        errors.append("No exit cell 'E' found")

    # Count connected door components and check every door component touches >= 2 non-wall sides
    visited_door: set[tuple[int, int]] = set()
    unlocked_doors = 0
    locked_doors = 0
    for r in range(h):
        for c in range(w):
            ch = rows[r][c]
            if ch not in DOOR_CHARS or (r, c) in visited_door:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            visited_door.add((r, c))
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and rows[nr][nc] == ch and (nr, nc) not in visited_door:
                        visited_door.add((nr, nc))
                        q.append((nr, nc))
            if ch == "D":
                unlocked_doors += 1
            else:
                locked_doors += 1
            # Check door doesn't touch another door type or exit directly
            adj_open = 0
            for cr, cc in comp:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    nch = rows[nr][nc]
                    if nch in DOOR_CHARS and nch != ch:
                        errors.append(f"Door {ch!r} at ({cr},{cc}) directly touches door {nch!r}")
                    if nch == "E":
                        errors.append(f"Door {ch!r} at ({cr},{cc}) directly touches exit 'E'")
                    if nch != "#" and nch != ch:
                        adj_open += 1
            if adj_open < 2:
                errors.append(f"Door {ch!r} at {comp[0]} is not flanked by open space on both sides")

    if errors or not spawns:
        return GridVerificationResult(
            ok=False,
            exit_reachable=False,
            keys_present=tuple(sorted(keys_pos)),
            keys_reachable=(),
            unlocked_doors=unlocked_doors,
            locked_doors=locked_doors,
            monsters=monsters,
            errors=tuple(errors),
        )

    # Iterative progression BFS: expand reachable cells, collect keys, unlock matching doors
    held_keys: set[str] = set()
    reachable: set[tuple[int, int]] = set()
    changed = True
    while changed:
        changed = False
        q = deque([spawns[0]])
        reachable = {spawns[0]}
        while q:
            cr, cc = q.popleft()
            ch = rows[cr][cc]
            if ch in KEY_CHARS and ch not in held_keys:
                held_keys.add(ch)
                changed = True
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in reachable:
                    continue
                nch = rows[nr][nc]
                if nch == "#":
                    continue
                if nch in LOCKED_DOOR_CHARS and _LOCK_TO_KEY[nch] not in held_keys:
                    continue
                reachable.add((nr, nc))
                q.append((nr, nc))

    exit_reachable = any(ep in reachable for ep in exits)
    if not exit_reachable:
        errors.append("Exit is NOT reachable from player spawn with available keys")
    missing_keys = sorted(set(keys_pos) - held_keys)
    if missing_keys:
        errors.append(f"Unreachable keys: {missing_keys}")

    # Check every locked door present has a matching key in the map
    for r in range(h):
        for c in range(w):
            ch = rows[r][c]
            if ch in LOCKED_DOOR_CHARS and _LOCK_TO_KEY[ch] not in keys_pos:
                errors.append(f"Locked door {ch!r} has no matching key {_LOCK_TO_KEY[ch]!r} in map")

    return GridVerificationResult(
        ok=len(errors) == 0,
        exit_reachable=exit_reachable,
        keys_present=tuple(sorted(keys_pos)),
        keys_reachable=tuple(sorted(held_keys)),
        unlocked_doors=unlocked_doors,
        locked_doors=locked_doors,
        monsters=monsters,
        errors=tuple(errors),
    )


def _point_in_polygon(pt: tuple[float, float], poly: Sequence[tuple[float, float]]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def verify_polygon_solvability(
    poly_sectors: Sequence[PolygonSector],
    things: Sequence[ThingSpec],
) -> tuple[bool, tuple[str, ...]]:
    """Verify polygonal map adjacency, thing containment, and key-before-lock exit reachability."""
    errors: list[str] = []
    # Build sector adjacency via shared edges
    edge_map: dict[tuple[tuple[float, float], tuple[float, float]], list[int]] = {}
    for sid, ps in enumerate(poly_sectors):
        pts = [_snap_pt(p) for p in ps.vertices]
        if _signed_area(pts) > 0:
            pts.reverse()
        for i in range(len(pts)):
            p1, p2 = pts[i], pts[(i + 1) % len(pts)]
            edge_map.setdefault((min(p1, p2), max(p1, p2)), []).append(sid)

    blocking_edge_keys: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for ps in poly_sectors:
        for bp1, bp2 in ps.blocking_edges:
            blocking_edge_keys.add(tuple(sorted((_snap_pt(bp1), _snap_pt(bp2)))))

    adj: dict[int, set[int]] = {i: set() for i in range(len(poly_sectors))}
    for ukey, owners in edge_map.items():
        if len(owners) == 2 and ukey not in blocking_edge_keys:
            a, b = owners
            first, second = poly_sectors[a], poly_sectors[b]
            opening = min(first.height_ceiling, second.height_ceiling) - max(first.height_floor, second.height_floor)
            if opening >= PLAYER_HEIGHT and abs(first.height_floor - second.height_floor) <= MAX_STEP_UP:
                adj[a].add(b)
                adj[b].add(a)

    def locate_sector(pt: tuple[float, float]) -> int | None:
        for sid, ps in enumerate(poly_sectors):
            if _point_in_polygon(pt, ps.vertices):
                in_hole = any(_point_in_polygon(pt, hole.vertices) for hole in ps.holes)
                if not in_hole:
                    return sid
        return None

    spawn_sec = None
    sector_keys: dict[int, set[str]] = {}
    for t in things:
        sid = locate_sector((t.x, t.y))
        if sid is None:
            errors.append(f"Thing type={t.doomednum} at ({t.x}, {t.y}) is outside all walkable sectors")
            continue
        if t.doomednum == 1:
            spawn_sec = sid
        elif t.doomednum in _KEY_DOOMEDNUM_TO_CHAR:
            sector_keys.setdefault(sid, set()).add(_KEY_DOOMEDNUM_TO_CHAR[t.doomednum])

    if spawn_sec is None:
        errors.append("Player start is missing or outside walkable sectors")
        return False, tuple(errors)

    exit_secs = {i for i, ps in enumerate(poly_sectors) if ps.kind == "exit" or ps.exit_switch is not None}
    if not exit_secs:
        errors.append("No exit polygon sector defined")
        return False, tuple(errors)

    # Progression BFS across sectors (keys + remote door switches)
    held_keys: set[str] = set()
    triggered_door_ids: set[int] = set()
    reachable: set[int] = set()
    changed = True
    while changed:
        changed = False
        q = deque([spawn_sec])
        reachable = {spawn_sec}
        while q:
            cur = q.popleft()
            for k in sector_keys.get(cur, ()):
                if k not in held_keys:
                    held_keys.add(k)
                    changed = True
            dsw = poly_sectors[cur].door_switch
            if dsw is not None and int(dsw[2]) not in triggered_door_ids:
                triggered_door_ids.add(int(dsw[2]))
                changed = True
            for nxt in adj[cur]:
                if nxt in reachable:
                    continue
                nxt_ps = poly_sectors[nxt]
                req_key = _DOOR_KIND_TO_KEY.get(nxt_ps.kind)
                if req_key is not None and req_key not in held_keys:
                    continue
                if nxt_ps.kind == "remote_door" and int(nxt_ps.sector_id) not in triggered_door_ids:
                    continue
                reachable.add(nxt)
                q.append(nxt)

    if not (exit_secs & reachable):
        errors.append("Exit polygon sector is NOT reachable from player spawn")
    return len(errors) == 0, tuple(errors)


def verify_wad_in_engine(
    wad_path: str | Path,
    *,
    map_name: str = "MAP01",
    episode_ticks: int = 2100,
    min_unlocked_doors: int = 0,
    min_locked_doors: int = 0,
    step_count: int = 5,
) -> WadVerificationResult:
    """Load a WAD in real `doom_env` (freedoom2.wad, skill 3) and validate all contract checks."""
    wad = str(wad_path)
    errors: list[str] = []

    info = load_mapinfo(wad, map_name)
    if info.num_sectors >= 2048:
        errors.append(f"Sector count {info.num_sectors} exceeds ViZDoom limit 2048")
    unlocked = sum(1 for lock in info.door_sectors.values() if lock == 0)
    locked = sum(1 for lock in info.door_sectors.values() if lock > 0)
    if unlocked < min_unlocked_doors:
        errors.append(f"Expected >= {min_unlocked_doors} unlocked door sectors, got {unlocked}")
    if locked < min_locked_doors:
        errors.append(f"Expected >= {min_locked_doors} locked door sectors, got {locked}")

    env = make_env(wad, map_name, episode_ticks=episode_ticks, render_buffers=False)
    geom_source = "unknown"
    has_door_lines = False
    has_locked_lines = False
    try:
        obs, _ = env.reset(seed=0)
        geom_source = getattr(env._geometry, "source", "unknown")
        if geom_source != "wad":
            errors.append(f"Expected geometry source 'wad', got {geom_source!r}")
        if not env.observation_space.contains(obs):
            errors.append("Initial observation does not match observation_space")

        hit_types = getattr(env._geometry, "hit_type", np.array([], dtype=np.int64))
        has_door_lines = bool(np.any(hit_types == RAY_DOOR))
        has_locked_lines = bool(np.any(hit_types == RAY_LOCKED))
        if min_unlocked_doors > 0 and not has_door_lines:
            errors.append("No RAY_DOOR linedefs detected in LevelGeometry")
        if min_locked_doors > 0 and not has_locked_lines:
            errors.append("No RAY_LOCKED linedefs detected in LevelGeometry")

        tracker = PrivilegedEventTracker()
        priv = env.privileged_state()
        tracker.reset(priv)
        if priv["health"] <= 0 or priv["dead"]:
            errors.append("Player spawned dead")

        noop = np.zeros(9, dtype=np.int8)
        for _ in range(step_count):
            obs, _, term, trunc, _ = env.step(noop)
            if not env.observation_space.contains(obs):
                errors.append("Step observation failed observation_space check")
                break
            tracker.update(env.privileged_state())
            if term or trunc:
                break
    finally:
        env.close()

    return WadVerificationResult(
        ok=len(errors) == 0,
        wad=wad,
        map_name=map_name,
        num_sectors=info.num_sectors,
        geometry_source=geom_source,
        num_unlocked_doors=unlocked,
        num_locked_doors=locked,
        has_door_ray_lines=has_door_lines,
        has_locked_ray_lines=has_locked_lines,
        errors=tuple(errors),
    )
