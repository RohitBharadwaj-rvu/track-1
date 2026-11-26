"""Training-only geodesic Potential-Based Reward Shaping (`docs/RULES.md` §2).

Parses UDMF WAD geometry + `doom_env.mapinfo` to compute exact progression-aware
shortest-path potentials from spawn -> keys -> locked doors -> exit.
Lives strictly under `training/` and is NEVER imported by `submission/`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np

from doom_env.mapinfo import (
    DOOM_MANUAL_DOORS,
    LOCK_ALL,
    LOCK_ANY,
    LOCK_BLUE,
    LOCK_RED,
    LOCK_YELLOW,
    MapInfo,
    load_mapinfo,
    parse_udmf,
)
from doom_env.sensors import MAX_STEP_UP, PLAYER_HEIGHT
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

_EXIT_SPECIALS = {11, 51, 52, 124, 197, 198, 243, 244}
_REMOTE_DOOR_LOCK = -99
_UDMF_DAMAGING_SPECIALS = {69, 71, 75, 80, 105, 115, 196}
_DOOM_DAMAGING_SPECIALS = {4, 5, 7, 11, 16}
_DOOM_REMOTE_DOOR_SPECIALS = {2, 29, 61, 63, 86, 90, 103, 105, 106, 108, 109, 111, 112, 114, 115}
_INF = 1e9
_MAX_POTENTIAL_DIST = 6.0


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
    switch_waypoints: dict[int, int] = field(default_factory=dict)  # remote door sid -> switch cell_idx


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
    solid_mask = (one_sided | imp).copy()

    n_sec = max(0, int(info.num_sectors))
    sec_hf = np.zeros(n_sec, dtype=np.float64)
    sec_hc = np.full(n_sec, 128.0, dtype=np.float64)
    sec_damaging = np.zeros(n_sec, dtype=bool)
    sec_tag = np.zeros(n_sec, dtype=np.int64)
    has_sec_geom = False

    exit_pts: list[tuple[float, float]] = []
    key_pts: list[tuple[float, float, int]] = []
    remote_switch_by_tag: dict[int, list[tuple[float, float]]] = {}
    manual_door_secs: set[int] = set()
    inward_nudge = min(16.0, float(cell_size) * 0.5)

    def _nudged_midpoint(x1: float, y1: float, x2: float, y2: float, has_back: bool) -> tuple[float, float]:
        mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        if not has_back:
            dx, dy = x2 - x1, y2 - y1
            seg_len = math.hypot(dx, dy)
            if seg_len > 1e-6:
                mx += (dy / seg_len) * inward_nudge
                my += (-dx / seg_len) * inward_nudge
        return mx, my

    if "TEXTMAP" in lumps:
        text = lumps["TEXTMAP"].decode("utf-8", "replace")
        blocks = parse_udmf(text)
        verts = [(float(v.get("x", 0.0)), float(v.get("y", 0.0))) for v in blocks.get("vertex", [])]
        sides = blocks.get("sidedef", [])
        sectors = blocks.get("sector", [])

        if len(sectors) == n_sec and n_sec > 0:
            has_sec_geom = True
            sec_hf = np.array([float(s.get("heightfloor", 0.0)) for s in sectors], dtype=np.float64)
            sec_hc = np.array([float(s.get("heightceiling", 128.0)) for s in sectors], dtype=np.float64)
            for idx, sec in enumerate(sectors):
                sec_tag[idx] = int(sec.get("id", 0) or 0)
                if int(sec.get("special", 0) or 0) in _UDMF_DAMAGING_SPECIALS:
                    sec_damaging[idx] = True

        for ld in blocks.get("linedef", []):
            spec = int(ld.get("special", 0) or 0)
            arg0 = int(ld.get("arg0", 0) or 0)
            v1, v2 = int(ld.get("v1", -1)), int(ld.get("v2", -1))
            sb = ld.get("sideback", -1)
            sf = ld.get("sidefront", -1)
            has_back = isinstance(sb, int) and int(sb) >= 0

            # Track sectors bounded by manual doors so remote-only doors can be identified cleanly.
            if spec in (11, 12, 13, 202) and arg0 == 0 and has_back and isinstance(sb, int) and 0 <= sb < len(sides):
                bsec = int(sides[sb].get("sector", -1))
                if 0 <= bsec < n_sec:
                    manual_door_secs.add(bsec)
                if isinstance(sf, int) and 0 <= sf < len(sides):
                    fsec = int(sides[sf].get("sector", -1))
                    if 0 <= fsec < n_sec:
                        manual_door_secs.add(fsec)

            if not (0 <= v1 < len(verts) and 0 <= v2 < len(verts)):
                continue
            x1, y1 = verts[v1]
            x2, y2 = verts[v2]

            if spec in _EXIT_SPECIALS:
                exit_pts.append(_nudged_midpoint(x1, y1, x2, y2, has_back))
            elif spec in (11, 12) and arg0 > 0:
                remote_switch_by_tag.setdefault(arg0, []).append(_nudged_midpoint(x1, y1, x2, y2, has_back))

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
    elif {"VERTEXES", "LINEDEFS", "SIDEDEFS", "SECTORS"}.issubset(lumps):
        # Standard Doom binary WAD lumps (e.g., freedoom2.wad MAP01..MAP32).
        v_raw = lumps["VERTEXES"]
        l_raw = lumps["LINEDEFS"]
        sd_raw = lumps["SIDEDEFS"]
        sc_raw = lumps["SECTORS"]
        t_raw = lumps.get("THINGS", b"")

        n_v = len(v_raw) // 4
        verts = [
            (float(x), float(y))
            for x, y in (struct.unpack_from("<hh", v_raw, i * 4) for i in range(n_v))
        ]
        n_sd = len(sd_raw) // 30
        side_secs = [
            int(struct.unpack_from("<hh8s8s8sh", sd_raw, i * 30)[5])
            for i in range(n_sd)
        ]
        n_sc = len(sc_raw) // 26
        if n_sc == n_sec and n_sec > 0:
            has_sec_geom = True
            for i in range(n_sc):
                hf, hc, _ft, _ct, _lt, sp, tag = struct.unpack_from("<hh8s8shhh", sc_raw, i * 26)
                sec_hf[i] = float(hf)
                sec_hc[i] = float(hc)
                sec_tag[i] = int(tag)
                if int(sp) in _DOOM_DAMAGING_SPECIALS:
                    sec_damaging[i] = True

        n_l = len(l_raw) // 14
        for i in range(n_l):
            v1, v2, _flags, spec, tag, sf_u, sb_u = struct.unpack_from("<HHHHHHH", l_raw, i * 14)
            sf = -1 if sf_u == 0xFFFF else int(sf_u)
            sb = -1 if sb_u == 0xFFFF else int(sb_u)
            has_back = 0 <= sb < n_sd
            if spec in DOOM_MANUAL_DOORS and tag == 0 and has_back:
                bsec = side_secs[sb]
                if 0 <= bsec < n_sec:
                    manual_door_secs.add(bsec)
                if 0 <= sf < n_sd and 0 <= side_secs[sf] < n_sec:
                    manual_door_secs.add(side_secs[sf])
            if not (0 <= v1 < n_v and 0 <= v2 < n_v):
                continue
            x1, y1 = verts[v1]
            x2, y2 = verts[v2]
            if spec in _EXIT_SPECIALS:
                exit_pts.append(_nudged_midpoint(x1, y1, x2, y2, has_back))
            elif spec in _DOOM_REMOTE_DOOR_SPECIALS and tag > 0:
                remote_switch_by_tag.setdefault(int(tag), []).append(_nudged_midpoint(x1, y1, x2, y2, has_back))

        n_t = len(t_raw) // 10
        for i in range(n_t):
            tx, ty, _ang, t_id, _tflags = struct.unpack_from("<hhhhh", t_raw, i * 10)
            if t_id in _KEY_THING_TO_COLOR:
                key_pts.append((float(tx), float(ty), _KEY_THING_TO_COLOR[t_id]))

    # Track directed cliff segments (allow downhill drop-offs, block uphill > MAX_STEP_UP).
    cliff_indices: list[int] = []
    cliff_low_is_front: list[bool] = []
    if has_sec_geom:
        is_door_sec = np.zeros(n_sec, dtype=bool)
        for sid in info.door_sectors:
            if 0 <= sid < n_sec:
                is_door_sec[sid] = True
        two_sided_idx = np.flatnonzero(
            (~solid_mask) & (front >= 0) & (back >= 0) & (front < n_sec) & (back < n_sec)
        )
        if len(two_sided_idx) > 0:
            sa_v, sb_v = front[two_sided_idx], back[two_sided_idx]
            non_door = ~(is_door_sec[sa_v] | is_door_sec[sb_v])
            opening = np.minimum(sec_hc[sa_v], sec_hc[sb_v]) - np.maximum(sec_hf[sa_v], sec_hf[sb_v])
            # Narrow openings (windows, grates, low ceilings) are impassable in both directions.
            too_low = two_sided_idx[non_door & (opening < PLAYER_HEIGHT)]
            solid_mask[too_low] = True
            # Vertical step-ups/drop-offs: traversable only from higher sector to lower sector.
            passable_non_door = non_door & (opening >= PLAYER_HEIGHT)
            diff_v = sec_hf[sb_v] - sec_hf[sa_v]
            for li, sa_i, sb_i, dv, nd in zip(
                two_sided_idx.tolist(), sa_v.tolist(), sb_v.tolist(), diff_v.tolist(), passable_non_door.tolist()
            ):
                if nd and abs(dv) > MAX_STEP_UP:
                    cliff_indices.append(li)
                    cliff_low_is_front.append(dv > 0.0)

    solid_p1 = p1[solid_mask]
    solid_p2 = p2[solid_mask]
    cliff_p1 = p1[cliff_indices] if cliff_indices else np.zeros((0, 2), dtype=np.float64)
    cliff_p2 = p2[cliff_indices] if cliff_indices else np.zeros((0, 2), dtype=np.float64)

    # Identify remote doors (tag > 0 matching a switch linedef, no local manual door linedefs)
    remote_doors: dict[int, tuple[float, float]] = {}
    for sid, lock in info.door_sectors.items():
        if lock == 0 and 0 <= sid < n_sec and sid not in manual_door_secs:
            tag = int(sec_tag[sid])
            if tag > 0 and tag in remote_switch_by_tag:
                pts = remote_switch_by_tag[tag]
                remote_doors[sid] = pts[0]

    # Locked door + remote door boundary segments
    locked_sectors: dict[int, int] = {sid: lock for sid, lock in info.door_sectors.items() if lock > 0}
    for sid in remote_doors:
        locked_sectors[sid] = _REMOTE_DOOR_LOCK

    lock_line_idx: list[tuple[int, int, int]] = []  # (line_idx, sid, lock)
    for sid, lock in locked_sectors.items():
        m = ((front == sid) | (back == sid)) & (~solid_mask)
        for li in np.flatnonzero(m).tolist():
            lock_line_idx.append((li, sid, lock))

    if not exit_pts:
        _NAV_CACHE[cache_key] = None
        return None

    all_pts = np.vstack([p1, p2])
    # Coarsen cell grid multiplicatively when large maps (e.g., freedoom2 MAP03..08) exceed 12,000 cells.
    eff_cell = float(cell_size)
    while True:
        # Offset origin by a non-rational fraction of eff_cell so grid centers never sit on walls
        min_x = float(all_pts[:, 0].min()) - eff_cell * 0.371
        min_y = float(all_pts[:, 1].min()) - eff_cell * 0.371
        max_x = float(all_pts[:, 0].max()) + eff_cell * 0.5
        max_y = float(all_pts[:, 1].max()) + eff_cell * 0.5
        nx = max(1, int(math.ceil((max_x - min_x) / eff_cell)))
        ny = max(1, int(math.ceil((max_y - min_y) / eff_cell)))
        if nx * ny <= 12000 or eff_cell >= 256.0:
            break
        eff_cell *= 1.5

    if nx * ny > 12000:
        _NAV_CACHE[cache_key] = None
        return None

    def to_cell(x: float, y: float) -> int:
        gx = min(nx - 1, max(0, int(math.floor((x - min_x) / eff_cell))))
        gy = min(ny - 1, max(0, int(math.floor((y - min_y) / eff_cell))))
        return gy * nx + gx

    def cell_center(idx: int) -> tuple[float, float]:
        gy, gx = divmod(idx, nx)
        return min_x + (gx + 0.5) * eff_cell, min_y + (gy + 0.5) * eff_cell

    n_cells = nx * ny
    cell_damaging = np.zeros(n_cells, dtype=bool)
    if has_sec_geom and np.any(sec_damaging):
        f_dmg = np.where((front >= 0) & (front < n_sec), sec_damaging[np.clip(front, 0, n_sec - 1)], False)
        b_dmg = np.where((back >= 0) & (back < n_sec), sec_damaging[np.clip(back, 0, n_sec - 1)], False)
        dmg_edge_mask = f_dmg != b_dmg
        if np.any(dmg_edge_mask):
            ep1 = p1[dmg_edge_mask]
            ep2 = p2[dmg_edge_mask]
            ep1_x, ep1_y = ep1[:, 0], ep1[:, 1]
            ep2_x, ep2_y = ep2[:, 0], ep2[:, 1]
            dx_e = ep2_x - ep1_x
            dy_e = ep2_y - ep1_y
            safe_dy = np.where(np.abs(dy_e) > 1e-9, dy_e, 1.0)
            gy_idx, gx_idx = np.divmod(np.arange(n_cells), nx)
            cx_arr = min_x + (gx_idx.astype(np.float64) + 0.5) * eff_cell
            cy_arr = min_y + (gy_idx.astype(np.float64) + 0.5) * eff_cell
            for start in range(0, n_cells, 1024):
                end = min(n_cells, start + 1024)
                cx_c = cx_arr[start:end, None]
                cy_c = cy_arr[start:end, None]
                cond_y = (ep1_y[None, :] > cy_c) != (ep2_y[None, :] > cy_c)
                x_int = ep1_x[None, :] + (cy_c - ep1_y[None, :]) * dx_e[None, :] / safe_dy[None, :]
                crossings = np.sum(cond_y & (cx_c < x_int), axis=1)
                cell_damaging[start:end] = (crossings % 2) == 1

    lock_p1 = p1[[x[0] for x in lock_line_idx]] if lock_line_idx else np.zeros((0, 2), dtype=np.float64)
    lock_p2 = p2[[x[0] for x in lock_line_idx]] if lock_line_idx else np.zeros((0, 2), dtype=np.float64)

    adj: list[list[tuple[int, float, tuple[tuple[int, int], ...]]]] = [[] for _ in range(n_cells)]

    locked_door_cells: dict[int, tuple[int, list[int]]] = {
        sid: (lock, []) for sid, lock in locked_sectors.items()
    }

    def _crosses_uphill_cliff(ux: float, uy: float, vx: float, vy: float) -> bool:
        if len(cliff_p1) == 0:
            return False
        hits = np.flatnonzero(_segments_intersect_batch(ux, uy, vx, vy, cliff_p1, cliff_p2))
        for hi in hits.tolist():
            x1, y1 = cliff_p1[hi]
            x2, y2 = cliff_p2[hi]
            dx, dy = x2 - x1, y2 - y1
            cross_u = (ux - x1) * dy - (uy - y1) * dx
            cross_v = (vx - x1) * dy - (vy - y1) * dx
            if cliff_low_is_front[hi]:
                if cross_u >= 0.0 and cross_v <= 0.0:
                    return True
            else:
                if cross_u <= 0.0 and cross_v >= 0.0:
                    return True
        return False

    # Build cardinal edges once per undirected pair (+X, +Y), checking uphill cliff direction separately.
    card_map: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for gy in range(ny):
        for gx in range(nx):
            u = gy * nx + gx
            ux, uy = cell_center(u)
            for dx_i, dy_i in ((1, 0), (0, 1)):
                ng_x, ng_y = gx + dx_i, gy + dy_i
                if not (0 <= ng_x < nx and 0 <= ng_y < ny):
                    continue
                v = ng_y * nx + ng_x
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
                cost = eff_cell * (3.0 if (cell_damaging[u] or cell_damaging[v]) else 1.0)
                if not _crosses_uphill_cliff(ux, uy, vx, vy):
                    card_map[(u, v)] = req_tuple
                    adj[u].append((v, cost, req_tuple))
                if not _crosses_uphill_cliff(vx, vy, ux, uy):
                    card_map[(v, u)] = req_tuple
                    adj[v].append((u, cost, req_tuple))

    diag_base = eff_cell * 1.41421356
    for gy in range(ny - 1):
        for gx in range(nx - 1):
            u00 = gy * nx + gx
            u10 = gy * nx + (gx + 1)
            u01 = (gy + 1) * nx + gx
            u11 = (gy + 1) * nx + (gx + 1)
            diag_pairs = (
                (u00, u11, (u00, u10), (u10, u11), (u00, u01), (u01, u11)),
                (u11, u00, (u11, u10), (u10, u00), (u11, u01), (u01, u00)),
                (u10, u01, (u10, u00), (u00, u01), (u10, u11), (u11, u01)),
                (u01, u10, (u01, u00), (u00, u10), (u01, u11), (u11, u10)),
            )
            for src, dst, e1a, e1b, e2a, e2b in diag_pairs:
                if e1a in card_map and e1b in card_map and e2a in card_map and e2b in card_map:
                    reqs = tuple(sorted(set(card_map[e1a]) | set(card_map[e1b]) | set(card_map[e2a]) | set(card_map[e2b])))
                    cost = diag_base * (3.0 if (cell_damaging[src] or cell_damaging[dst]) else 1.0)
                    adj[src].append((dst, cost, reqs))

    exit_cells = tuple(sorted({to_cell(ex, ey) for ex, ey in exit_pts}))
    key_spawns = tuple((to_cell(kx, ky), col) for kx, ky, col in key_pts)
    switch_waypoints = {sid: to_cell(sx, sy) for sid, (sx, sy) in remote_doors.items()}
    graph = _WadNavigationGraph(
        min_x=min_x,
        min_y=min_y,
        cell_size=eff_cell,
        nx=nx,
        ny=ny,
        adj=adj,
        exit_cells=exit_cells,
        key_spawns=key_spawns,
        locked_door_cells=locked_door_cells,
        switch_waypoints=switch_waypoints,
    )
    _NAV_CACHE[cache_key] = graph
    if len(_NAV_CACHE) > 256:
        _NAV_CACHE.pop(next(iter(_NAV_CACHE)))
    return graph


class GeodesicPBRS:
    """Progression-aware geodesic Potential-Based Reward Shaping for training only."""

    def __init__(
        self,
        wad_path: str | Path,
        map_name: str = "MAP01",
        *,
        scale: float = 2.0,
        gamma: float = 1.0,
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
        if lock == _REMOTE_DOOR_LOCK:
            return False
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

        # Track physically opened locked & remote doors (remote doors can re-close after door_close_delay).
        ceil = priv_state.get("sector_ceiling")
        if ceil is not None and self._init_ceil is not None and len(ceil) == len(self._init_ceil):
            diff = np.asarray(ceil, dtype=np.float64) - self._init_ceil
            for sid, (lk, _) in self.graph.locked_door_cells.items():
                if 0 <= sid < len(diff):
                    if diff[sid] >= 8.0:
                        self._unlocked_sectors.add(sid)
                    elif lk == _REMOTE_DOOR_LOCK:
                        self._unlocked_sectors.discard(sid)

        held_colors = self._held_colors(priv_state)
        for idx, (_cell, col) in enumerate(self.graph.key_spawns):
            if col in held_colors:
                self._collected_key_indices.add(idx)

        cur_cell = self._cell_of(float(priv_state["x"]), float(priv_state["y"]))
        dist = self._dijkstra(cur_cell, held_colors)

        # Priority 1: Exit reachable using currently open + held-key doors
        exit_dists = [float(dist[c]) for c in self.graph.exit_cells if dist[c] < _INF * 0.5]
        if exit_dists:
            d_min = min(exit_dists)
            return ("exit",), max(0.0, _MAX_POTENTIAL_DIST - d_min / 1024.0)

        # Priority 2: Nearest reachable prerequisite (uncollected key or closed remote door switch)
        key_cands = [
            float(dist[k_cell])
            for idx, (k_cell, _col) in enumerate(self.graph.key_spawns)
            if idx not in self._collected_key_indices and dist[k_cell] < _INF * 0.5
        ]
        sw_cands = [
            (float(dist[sw_cell]), sid)
            for sid, sw_cell in self.graph.switch_waypoints.items()
            if sid not in self._unlocked_sectors and dist[sw_cell] < _INF * 0.5
        ]
        best_key = min(key_cands) if key_cands else None
        best_sw = min(sw_cands) if sw_cands else None
        if best_key is not None and (best_sw is None or best_key <= best_sw[0]):
            if self.graph.switch_waypoints:
                return (
                    ("key", tuple(sorted(held_colors)), tuple(sorted(self._unlocked_sectors))),
                    max(0.0, _MAX_POTENTIAL_DIST - best_key / 1024.0),
                )
            return ("key", tuple(sorted(held_colors))), max(0.0, _MAX_POTENTIAL_DIST - best_key / 1024.0)
        if best_sw is not None:
            sw_dist, sw_sid = best_sw
            return (
                ("switch", sw_sid, tuple(sorted(held_colors)), tuple(sorted(self._unlocked_sectors))),
                max(0.0, _MAX_POTENTIAL_DIST - sw_dist / 1024.0),
            )

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
            # Potential at exit (d=0) is _MAX_POTENTIAL_DIST
            raw = self.scale * (self.gamma * _MAX_POTENTIAL_DIST - self._prev_phi)
            self._prev_phi = _MAX_POTENTIAL_DIST
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
