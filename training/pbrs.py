"""Training-only geodesic Potential-Based Reward Shaping (`docs/RULES.md` §2).

Parses UDMF WAD geometry + `doom_env.mapinfo` to compute exact progression-aware
shortest-path potentials from spawn -> keys -> locked doors -> exit.
Lives strictly under `training/` and is NEVER imported by `submission/`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from doom_env.mapinfo import (
    LOCK_ALL,
    LOCK_ANY,
    LOCK_BLUE,
    LOCK_RED,
    LOCK_YELLOW,
    MapInfo,
    load_mapinfo,
    parse_udmf,
)
from doom_env.wad import map_lumps, read_wad

# ZDoom UDMF lock numbers + binary lock numbers -> canonical color index {0: blue, 1: yellow, 2: red}
_LOCK_TO_COLOR: dict[int, int] = {
    LOCK_BLUE: 0,
    130: 0,
    LOCK_YELLOW: 1,
    131: 1,
    LOCK_RED: 2,
    129: 2,
}

# DoomEdNums for key items -> color index {0: blue, 1: yellow, 2: red}
_KEY_THING_TO_COLOR: dict[int, int] = {
    5: 0,   # BlueCard
    40: 0,  # BlueSkull
    6: 1,   # YellowCard
    39: 1,  # YellowSkull
    13: 2,  # RedCard
    38: 2,  # RedSkull
}

_EXIT_SPECIALS = {11, 52, 243, 244}
_INF = 1e9


def _segments_intersect_batch(
    c1x: float,
    c1y: float,
    c2x: float,
    c2y: float,
    p1: np.ndarray,
    p2: np.ndarray,
) -> np.ndarray:
    """Return bool array of shape (M,) whether segment (c1->c2) intersects each segment p1[i]->p2[i]."""
    if len(p1) == 0:
        return np.zeros(0, dtype=bool)
    min_x, max_x = (c1x, c2x) if c1x <= c2x else (c2x, c1x)
    min_y, max_y = (c1y, c2y) if c1y <= c2y else (c2y, c1y)
    seg_min = np.minimum(p1, p2)
    seg_max = np.maximum(p1, p2)
    cand = (
        (seg_max[:, 0] >= min_x - 1e-3)
        & (seg_min[:, 0] <= max_x + 1e-3)
        & (seg_max[:, 1] >= min_y - 1e-3)
        & (seg_min[:, 1] <= max_y + 1e-3)
    )
    out = np.zeros(len(p1), dtype=bool)
    idx = np.flatnonzero(cand)
    if len(idx) == 0:
        return out

    rx = c2x - c1x
    ry = c2y - c1y
    q1 = p1[idx]
    q2 = p2[idx]
    dx = q2[:, 0] - q1[:, 0]
    dy = q2[:, 1] - q1[:, 1]
    wx = q1[:, 0] - c1x
    wy = q1[:, 1] - c1y
    denom = rx * dy - ry * dx
    nonzero = np.abs(denom) > 1e-9
    t = np.zeros_like(denom)
    u = np.zeros_like(denom)
    t[nonzero] = (wx[nonzero] * dy[nonzero] - wy[nonzero] * dx[nonzero]) / denom[nonzero]
    u[nonzero] = (wx[nonzero] * ry - wy[nonzero] * rx) / denom[nonzero]
    hit = nonzero & (t >= -1e-4) & (t <= 1.0 + 1e-4) & (u >= -1e-4) & (u <= 1.0 + 1e-4)
    out[idx] = hit
    return out


@dataclass
class _WadNavigationGraph:
    min_x: float
    min_y: float
    cell_size: float
    nx: int
    ny: int
    # neighbor[cell_idx] -> list of (neighbor_idx, edge_cost, lock_requirements)
    # where lock_requirements is a tuple of (sector_id, lock_code) crossed on this edge
    adj: list[list[tuple[int, float, tuple[tuple[int, int], ...]]]]
    exit_cells: tuple[int, ...]
    key_spawns: tuple[tuple[int, int], ...]  # (cell_idx, color_idx)
    locked_door_cells: dict[int, tuple[int, list[int]]]  # sector_id -> (lock_code, [cell_indices])


_NAV_CACHE: dict[tuple[str, str, float], _WadNavigationGraph | None] = {}


def _build_nav_graph(wad_path: str | Path, map_name: str = "MAP01", cell_size: float = 32.0) -> _WadNavigationGraph | None:
    resolved = str(Path(wad_path).resolve())
    cache_key = (resolved, map_name.upper(), float(cell_size))
    if cache_key in _NAV_CACHE:
        return _NAV_CACHE[cache_key]

    try:
        lumps = map_lumps(read_wad(resolved), map_name.upper())
        info: MapInfo = load_mapinfo(resolved, map_name.upper())
    except Exception:
        _NAV_CACHE[cache_key] = None
        return None

    if info.lines is None or len(info.lines.p1) == 0:
        _NAV_CACHE[cache_key] = None
        return None

    p1 = info.lines.p1
    p2 = info.lines.p2
    front = info.lines.front
    back = info.lines.back
    imp = info.lines.impassable

    one_sided = (front < 0) | (back < 0)
    solid_mask = one_sided | imp
    solid_p1 = p1[solid_mask]
    solid_p2 = p2[solid_mask]

    # Locked door boundary segments
    locked_sectors = {sid: lock for sid, lock in info.door_sectors.items() if lock > 0}
    lock_line_idx: list[tuple[int, int, int]] = []  # (line_idx, sid, lock)
    for sid, lock in locked_sectors.items():
        m = ((front == sid) | (back == sid)) & (~solid_mask)
        for li in np.flatnonzero(m).tolist():
            lock_line_idx.append((li, sid, lock))

    exit_pts: list[tuple[float, float]] = []
    key_pts: list[tuple[float, float, int]] = []
    if "TEXTMAP" in lumps:
        text = lumps["TEXTMAP"].decode("utf-8", "replace")
        blocks = parse_udmf(text)
        verts = [(float(v.get("x", 0.0)), float(v.get("y", 0.0))) for v in blocks.get("vertex", [])]
        sides = blocks.get("sidedef", [])
        sectors = blocks.get("sector", [])

        # Exits via exit linedef special or GATE3 floor sector
        for ld in blocks.get("linedef", []):
            spec = int(ld.get("special", 0) or 0)
            if spec in _EXIT_SPECIALS:
                v1, v2 = int(ld.get("v1", -1)), int(ld.get("v2", -1))
                if 0 <= v1 < len(verts) and 0 <= v2 < len(verts):
                    exit_pts.append(((verts[v1][0] + verts[v2][0]) * 0.5, (verts[v1][1] + verts[v2][1]) * 0.5))

        exit_sec_ids = {
            i for i, sec in enumerate(sectors) if str(sec.get("texturefloor", "")).upper() == "GATE3"
        }
        if exit_sec_ids:
            sec_verts: dict[int, list[tuple[float, float]]] = {}
            for ld in blocks.get("linedef", []):
                v1, v2 = int(ld.get("v1", -1)), int(ld.get("v2", -1))
                if not (0 <= v1 < len(verts) and 0 <= v2 < len(verts)):
                    continue
                for skey in ("sidefront", "sideback"):
                    si = ld.get(skey, -1)
                    if isinstance(si, int) and 0 <= si < len(sides):
                        sid = int(sides[si].get("sector", -1))
                        if sid in exit_sec_ids:
                            sec_verts.setdefault(sid, []).extend([verts[v1], verts[v2]])
            for pts in sec_verts.values():
                if pts:
                    cx = sum(pt[0] for pt in pts) / len(pts)
                    cy = sum(pt[1] for pt in pts) / len(pts)
                    exit_pts.append((cx, cy))

        for th in blocks.get("thing", []):
            t_id = int(th.get("type", 0) or 0)
            if t_id in _KEY_THING_TO_COLOR:
                key_pts.append((float(th.get("x", 0.0)), float(th.get("y", 0.0)), _KEY_THING_TO_COLOR[t_id]))

    if not exit_pts:
        _NAV_CACHE[cache_key] = None
        return None

    all_pts = np.vstack([p1, p2])
    # Offset origin by a non-rational fraction of cell_size so grid centers never sit on 64/80-unit walls
    min_x = float(all_pts[:, 0].min()) - cell_size * 0.371
    min_y = float(all_pts[:, 1].min()) - cell_size * 0.371
    max_x = float(all_pts[:, 0].max()) + cell_size * 0.5
    max_y = float(all_pts[:, 1].max()) + cell_size * 0.5

    nx = max(1, int(math.ceil((max_x - min_x) / cell_size)))
    ny = max(1, int(math.ceil((max_y - min_y) / cell_size)))
    if nx * ny > 12000:
        _NAV_CACHE[cache_key] = None
        return None

    def to_cell(x: float, y: float) -> int:
        gx = min(nx - 1, max(0, int(math.floor((x - min_x) / cell_size))))
        gy = min(ny - 1, max(0, int(math.floor((y - min_y) / cell_size))))
        return gy * nx + gx

    def cell_center(idx: int) -> tuple[float, float]:
        gy, gx = divmod(idx, nx)
        return min_x + (gx + 0.5) * cell_size, min_y + (gy + 0.5) * cell_size

    lock_p1 = p1[[x[0] for x in lock_line_idx]] if lock_line_idx else np.zeros((0, 2), dtype=np.float64)
    lock_p2 = p2[[x[0] for x in lock_line_idx]] if lock_line_idx else np.zeros((0, 2), dtype=np.float64)

    n_cells = nx * ny
    adj: list[list[tuple[int, float, tuple[tuple[int, int], ...]]]] = [[] for _ in range(n_cells)]
    card_dirs = ((1, 0, cell_size), (-1, 0, cell_size), (0, 1, cell_size), (0, -1, cell_size))

    locked_door_cells: dict[int, tuple[int, list[int]]] = {
        sid: (lock, []) for sid, lock in locked_sectors.items()
    }

    # Build cardinal edges first, then allow diagonals only when both intermediate cardinal edges exist (no corner-clipping!)
    card_map: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for gy in range(ny):
        for gx in range(nx):
            u = gy * nx + gx
            ux, uy = cell_center(u)
            for dx_i, dy_i, cost in card_dirs:
                ng_x, ng_y = gx + dx_i, gy + dy_i
                if not (0 <= ng_x < nx and 0 <= ng_y < ny):
                    continue
                v = ng_y * nx + ng_x
                if v < u:
                    continue
                vx, vy = cell_center(v)
                if np.any(_segments_intersect_batch(ux, uy, vx, vy, solid_p1, solid_p2)):
                    continue
                reqs: list[tuple[int, int]] = []
                if len(lock_p1) > 0:
                    hits = np.flatnonzero(_segments_intersect_batch(ux, uy, vx, vy, lock_p1, lock_p2))
                    for hi in hits.tolist():
                        _, sid, lock = lock_line_idx[hi]
                        reqs.append((sid, lock))
                        if u not in locked_door_cells[sid][1]:
                            locked_door_cells[sid][1].append(u)
                        if v not in locked_door_cells[sid][1]:
                            locked_door_cells[sid][1].append(v)
                req_tuple = tuple(sorted(set(reqs)))
                card_map[(u, v)] = req_tuple
                card_map[(v, u)] = req_tuple
                adj[u].append((v, cost, req_tuple))
                adj[v].append((u, cost, req_tuple))

    diag_cost = cell_size * 1.41421356
    for gy in range(ny - 1):
        for gx in range(nx - 1):
            # (gx, gy) <-> (gx+1, gy+1) via (gx+1, gy) or (gx, gy+1)
            u00 = gy * nx + gx
            u10 = gy * nx + (gx + 1)
            u01 = (gy + 1) * nx + gx
            u11 = (gy + 1) * nx + (gx + 1)
            if (u00, u10) in card_map and (u10, u11) in card_map and (u00, u01) in card_map and (u01, u11) in card_map:
                reqs = tuple(sorted(set(card_map[(u00, u10)]) | set(card_map[(u10, u11)]) | set(card_map[(u00, u01)]) | set(card_map[(u01, u11)])))
                adj[u00].append((u11, diag_cost, reqs))
                adj[u11].append((u00, diag_cost, reqs))
            if (u10, u00) in card_map and (u00, u01) in card_map and (u10, u11) in card_map and (u11, u01) in card_map:
                reqs = tuple(sorted(set(card_map[(u10, u00)]) | set(card_map[(u00, u01)]) | set(card_map[(u10, u11)]) | set(card_map[(u11, u01)])))
                adj[u10].append((u01, diag_cost, reqs))
                adj[u01].append((u10, diag_cost, reqs))

    exit_cells = tuple(sorted({to_cell(ex, ey) for ex, ey in exit_pts}))
    key_spawns = tuple((to_cell(kx, ky), col) for kx, ky, col in key_pts)
    graph = _WadNavigationGraph(
        min_x=min_x,
        min_y=min_y,
        cell_size=cell_size,
        nx=nx,
        ny=ny,
        adj=adj,
        exit_cells=exit_cells,
        key_spawns=key_spawns,
        locked_door_cells=locked_door_cells,
    )
    _NAV_CACHE[cache_key] = graph
    return graph


class GeodesicPBRS:
    """Progression-aware geodesic Potential-Based Reward Shaping for training only."""

    def __init__(
        self,
        wad_path: str | Path,
        map_name: str = "MAP01",
        *,
        scale: float = 2.0,
        gamma: float = 0.995,
        cell_size: float = 32.0,
    ) -> None:
        self.scale = float(scale)
        self.gamma = float(gamma)
        self.graph = _build_nav_graph(wad_path, map_name=map_name, cell_size=cell_size)
        self._init_ceil: np.ndarray | None = None
        self._unlocked_sectors: set[int] = set()
        self._collected_key_indices: set[int] = set()
        self._prev_stage_key: tuple[Any, ...] | None = None
        self._prev_phi: float = 0.0

    def _cell_of(self, x: float, y: float) -> int:
        assert self.graph is not None
        g = self.graph
        gx = min(g.nx - 1, max(0, int(math.floor((x - g.min_x) / g.cell_size))))
        gy = min(g.ny - 1, max(0, int(math.floor((y - g.min_y) / g.cell_size))))
        return gy * g.nx + gx

    @staticmethod
    def _held_colors(priv_state: dict[str, Any]) -> set[int]:
        keys = np.asarray(priv_state.get("keys", (0, 0, 0, 0, 0, 0)), dtype=np.float32)
        held: set[int] = set()
        if len(keys) >= 6:
            if keys[0] > 0.5 or keys[3] > 0.5:
                held.add(0)  # Blue
            if keys[1] > 0.5 or keys[4] > 0.5:
                held.add(1)  # Yellow
            if keys[2] > 0.5 or keys[5] > 0.5:
                held.add(2)  # Red
        return held

    def _can_pass_lock(self, sid: int, lock: int, held_colors: set[int]) -> bool:
        if sid in self._unlocked_sectors:
            return True
        if lock in _LOCK_TO_COLOR:
            return _LOCK_TO_COLOR[lock] in held_colors
        if lock == LOCK_ANY:
            return len(held_colors) > 0
        if lock == LOCK_ALL:
            return len(held_colors) >= 3
        return True

    def _dijkstra(self, start_cell: int, held_colors: set[int]) -> np.ndarray:
        assert self.graph is not None
        import heapq

        n = self.graph.nx * self.graph.ny
        dist = np.full(n, _INF, dtype=np.float64)
        dist[start_cell] = 0.0
        pq: list[tuple[float, int]] = [(0.0, start_cell)]
        adj = self.graph.adj
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, cost, reqs in adj[u]:
                if reqs and not all(self._can_pass_lock(sid, lk, held_colors) for sid, lk in reqs):
                    continue
                nd = d + cost
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    def _compute_stage_potential(self, priv_state: dict[str, Any]) -> tuple[tuple[Any, ...], float]:
        if self.graph is None or self.scale <= 0.0:
            return ("none",), 0.0

        # Track physically opened locked doors
        ceil = priv_state.get("sector_ceiling")
        if ceil is not None and self._init_ceil is not None and len(ceil) == len(self._init_ceil):
            raised = np.flatnonzero(np.asarray(ceil, dtype=np.float64) - self._init_ceil >= 8.0)
            for sid in raised.tolist():
                if sid in self.graph.locked_door_cells:
                    self._unlocked_sectors.add(sid)

        held_colors = self._held_colors(priv_state)
        for idx, (_cell, col) in enumerate(self.graph.key_spawns):
            if col in held_colors:
                self._collected_key_indices.add(idx)

        cur_cell = self._cell_of(float(priv_state["x"]), float(priv_state["y"]))
        dist = self._dijkstra(cur_cell, held_colors)

        # Priority 1: Exit reachable using currently unlocked + held-key doors
        exit_dists = [float(dist[c]) for c in self.graph.exit_cells if dist[c] < _INF * 0.5]
        if exit_dists:
            d_min = min(exit_dists)
            return ("exit",), -d_min / 1024.0

        # Priority 2: Nearest reachable uncollected key whose color is still needed
        key_dists: list[tuple[float, int]] = []
        for idx, (k_cell, col) in enumerate(self.graph.key_spawns):
            if idx not in self._collected_key_indices and dist[k_cell] < _INF * 0.5:
                key_dists.append((float(dist[k_cell]), idx))
        if key_dists:
            d_min, k_idx = min(key_dists)
            return ("key", tuple(sorted(held_colors))), -d_min / 1024.0

        return ("none",), 0.0

    def reset(self, priv_state: dict[str, Any]) -> None:
        self._unlocked_sectors.clear()
        self._collected_key_indices.clear()
        ceil = priv_state.get("sector_ceiling")
        self._init_ceil = None if ceil is None else np.array(ceil, dtype=np.float64, copy=True)
        if self.graph is None or self.scale <= 0.0:
            self._prev_stage_key = None
            self._prev_phi = 0.0
            return
        self._prev_stage_key, self._prev_phi = self._compute_stage_potential(priv_state)

    def step(self, priv_state: dict[str, Any]) -> float:
        if self.graph is None or self.scale <= 0.0:
            return 0.0
        if bool(priv_state.get("exit_reached")):
            # Potential at exit is 0.0
            raw = self.scale * (self.gamma * 0.0 - self._prev_phi)
            self._prev_phi = 0.0
            return float(np.clip(raw, -0.5, 0.5))
        if bool(priv_state.get("dead")):
            return 0.0

        stage_key, phi = self._compute_stage_potential(priv_state)
        if stage_key == ("none",):
            self._prev_stage_key = stage_key
            self._prev_phi = 0.0
            return 0.0

        # Reset potential cleanly when progression target switches (e.g. key -> exit)
        if self._prev_stage_key != stage_key:
            self._prev_stage_key = stage_key
            self._prev_phi = phi
            return 0.0

        raw = self.scale * (self.gamma * phi - self._prev_phi)
        self._prev_phi = phi
        return float(np.clip(raw, -0.5, 0.5))
