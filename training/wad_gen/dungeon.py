"""General randomized room-graph dungeon generator for procedural training diversity.

Unlike fixed hand-coded templates, `generate_dungeon_wad` generates connected macro
room graphs (spanning tree + optional cycles + dead-end alcoves) with variable room sizes,
cell scales (`64..112` units), wide `RAY_NONE` open arenas, 1/2/3-lock progressions
(sequential and parallel), stairs, one-way drop-offs, sight-but-no-move windows,
damaging nukage sectors (`special=71`) with `RadSuit` pickups, varied door delays,
D4 symmetry + balanced initial spawn bearings, and full monster/weapon/pickup variety.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import random

from training.interfaces import MapSpec
from training.pbrs import _build_nav_graph
from training.wad_gen.udmf import THEME_NAMES, build_grid_wad
from training.wad_gen.verify import verify_grid_solvability

_FAMILY_TIERS = {
    "combat": 1,
    "door_exit": 2,
    "multi_room": 2,
    "key_return": 3,
    "gauntlet": 4,
}

_FAMILY_TICKS = {
    "door_exit": 2100,
    "combat": 2100,
    "multi_room": 2400,
    "key_return": 3200,
    "gauntlet": 4200,
}


@dataclass(frozen=True)
class DungeonGenome:
    """Evolvable parameter vector for ACCEL procedural dungeon mutations."""

    family: str
    seed: int
    extra_rooms: int = 0
    extra_locks: int = 0
    force_window: bool = False
    force_vertical: bool = False
    force_nukage: bool = False
    force_low_ammo: bool = False
    monster_delta: int = 0
    door_delay: int | None = None
    generation: int = 0


def mutate_dungeon_genome(genome: DungeonGenome, rng: Any) -> DungeonGenome:
    """Apply 1-2 atomic ACCEL edits to produce a frontier child dungeon genome."""
    ops = [
        "extra_rooms",
        "force_window",
        "force_vertical",
        "force_nukage",
        "force_low_ammo",
        "monster_delta",
        "door_delay",
    ]
    if genome.family not in ("combat", "door_exit"):
        ops.insert(1, "extra_locks")
    n_edits = 2 if rng.random() < 0.35 else 1
    chosen = rng.choice(ops, size=n_edits, replace=False) if hasattr(rng, "choice") and hasattr(rng, "integers") else random.Random(genome.seed + genome.generation + 17).sample(ops, k=n_edits)
    g = genome
    py_rng = random.Random(int(rng.integers(0, 2**31 - 1)) if hasattr(rng, "integers") else rng.randint(0, 2**31 - 1))
    for op in list(chosen):
        if op == "extra_rooms":
            step = py_rng.choice((-1, 1)) if g.extra_rooms > 0 else 1
            g = replace(g, extra_rooms=max(0, min(2, g.extra_rooms + step)))
        elif op == "extra_locks":
            step = py_rng.choice((-1, 1)) if g.extra_locks > 0 else 1
            g = replace(g, extra_locks=max(0, min(2, g.extra_locks + step)))
        elif op == "force_window":
            g = replace(g, force_window=not g.force_window)
        elif op == "force_vertical":
            g = replace(g, force_vertical=not g.force_vertical)
        elif op == "force_nukage":
            g = replace(g, force_nukage=not g.force_nukage)
        elif op == "force_low_ammo":
            g = replace(g, force_low_ammo=not g.force_low_ammo)
        elif op == "monster_delta":
            step = py_rng.choice((-1, 1, 2))
            g = replace(g, monster_delta=max(-2, min(4, g.monster_delta + step)))
        elif op == "door_delay":
            g = replace(g, door_delay=py_rng.choice((0, 60, 100, 150, 200)))
    return replace(g, generation=g.generation + 1)


def _det_seed(family: str, seed: int, attempt: int = 0) -> int:
    h = hashlib.sha256(f"dungeon:{family}:{seed}:{attempt}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")


def _transform_cell_dict(
    data: dict[tuple[int, int], int],
    h: int,
    w: int,
    rot_k: int,
    flip_x: bool,
) -> tuple[dict[tuple[int, int], int], int, int]:
    """Apply `fliplr` (if flip_x) then `rot90` k times CCW to sparse `(r, c)` dict."""
    out: dict[tuple[int, int], int] = {}
    for (r, c), val in data.items():
        cr, cc = r, (w - 1 - c) if flip_x else c
        ch, cw = h, w
        for _ in range(rot_k % 4):
            # np.rot90 CCW: (r, c) -> (cw - 1 - c, r)
            cr, cc = cw - 1 - cc, cr
            ch, cw = cw, ch
        out[(cr, cc)] = val
    nh, nw = (w, h) if (rot_k % 2 == 1) else (h, w)
    return out, nh, nw


def _transform_grid_rows(
    grid_arr: list[list[str]],
    rot_k: int,
    flip_x: bool,
) -> list[list[str]]:
    arr = [row[::-1] if flip_x else list(row) for row in grid_arr]
    for _ in range(rot_k % 4):
        h = len(arr)
        w = len(arr[0])
        arr = [[arr[r][w - 1 - c] for r in range(h)] for c in range(w)]
    return arr


def _find_bfs_path(adj: dict[int, list[int]], start: int, goal: int) -> list[int]:
    parent: dict[int, int | None] = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        if u == goal:
            break
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                q.append(v)
    path: list[int] = []
    cur: int | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


def _subtree_nodes(
    tree_children: dict[int, list[int]],
    root: int,
    excluded: set[int] | None = None,
) -> set[int]:
    ex = excluded or set()
    res: set[int] = set()
    q = deque([root])
    while q:
        u = q.popleft()
        if u in ex or u in res:
            continue
        res.add(u)
        for ch in tree_children.get(u, ()):
            q.append(ch)
    return res


def _attempt_build_dungeon(
    family: str,
    seed: int,
    attempt: int,
    *,
    with_combat: bool,
    genome: DungeonGenome | None = None,
) -> tuple[
    str,
    dict[tuple[int, int], int],
    dict[tuple[int, int], int],
    dict[tuple[int, int], int],
    int,
    int,
    str,
    bool,
    int,
]:
    rng = random.Random(_det_seed(family, seed, attempt))
    tier = _FAMILY_TIERS[family]

    if family == "door_exit":
        mr, mc = rng.choice([(1, 3), (2, 2), (2, 3)])
    elif family == "combat":
        mr, mc = rng.choice([(2, 2), (2, 3), (3, 2)])
    elif family in ("multi_room", "key_return"):
        mr, mc = rng.choice([(2, 3), (3, 2), (3, 3)])
    else:
        mr, mc = rng.choice([(3, 3), (3, 4), (4, 3)])

    if genome is not None and genome.extra_rooms > 0:
        if mr <= mc:
            mr = min(4, mr + 1)
        else:
            mc = min(4, mc + 1)
        if genome.extra_rooms >= 2:
            mc = min(4, mc + 1)

    is_wide_arena = rng.random() < (0.42 if family in ("combat", "gauntlet", "multi_room") else 0.25)
    if is_wide_arena:
        cell_size = rng.choice((80, 96, 112))
        col_ws = [rng.randint(4, 6) for _ in range(mc)]
        row_hs = [rng.randint(4, 6) for _ in range(mr)]
        # Make one macro column/row extra spacious so rays hit RAY_NONE (>1024u)
        col_ws[rng.randrange(mc)] = rng.randint(6, 8)
        row_hs[rng.randrange(mr)] = rng.randint(6, 8)
    else:
        cell_size = rng.choice((64, 64, 80, 96))
        col_ws = [rng.randint(3, 5) for _ in range(mc)]
        row_hs = [rng.randint(3, 5) for _ in range(mr)]

    # Keep world span comfortably under PBRS's 12000-cell limit ((2600/32)^2 ~ 6600)
    total_c = 1 + sum(col_ws) + mc
    total_r = 1 + sum(row_hs) + mr
    while max(total_c, total_r) * cell_size > 2500 and cell_size > 64:
        cell_size = 80 if cell_size > 80 else 64

    grid = [["#"] * total_c for _ in range(total_r)]

    row_starts: list[int] = []
    cur_r = 1
    for rh in row_hs:
        row_starts.append(cur_r)
        cur_r += rh + 1
    col_starts: list[int] = []
    cur_c = 1
    for cw in col_ws:
        col_starts.append(cur_c)
        cur_c += cw + 1

    n_nodes = mr * mc

    def nid(r: int, c: int) -> int:
        return r * mc + c

    def ncoords(node: int) -> tuple[int, int]:
        return divmod(node, mc)

    # Carve macro room interiors
    room_rects: dict[int, tuple[int, int, int, int]] = {}
    for rr in range(mr):
        for cc in range(mc):
            r0, r1 = row_starts[rr], row_starts[rr] + row_hs[rr]
            c0, c1 = col_starts[cc], col_starts[cc] + col_ws[cc]
            room_rects[nid(rr, cc)] = (r0, r1, c0, c1)
            for r in range(r0, r1):
                for c in range(c0, c1):
                    grid[r][c] = "."

    # Build randomized spanning tree rooted at spawn_node
    spawn_node = nid(rng.randrange(mr), rng.randrange(mc))
    visited = {spawn_node}
    tree_adj: dict[int, list[int]] = {u: [] for u in range(n_nodes)}
    tree_children: dict[int, list[int]] = {u: [] for u in range(n_nodes)}
    frontier: list[tuple[int, int]] = []
    sr, sc = ncoords(spawn_node)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < mr and 0 <= nc < mc:
            frontier.append((spawn_node, nid(nr, nc)))

    while frontier:
        idx = rng.randrange(len(frontier))
        u, v = frontier.pop(idx)
        if v in visited:
            continue
        visited.add(v)
        tree_adj[u].append(v)
        tree_adj[v].append(u)
        tree_children[u].append(v)
        vr, vc = ncoords(v)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = vr + dr, vc + dc
            if 0 <= nr < mr and 0 <= nc < mc and nid(nr, nc) not in visited:
                frontier.append((v, nid(nr, nc)))

    # Pick exit_node from farthest nodes
    dists: dict[int, int] = {spawn_node: 0}
    q = deque([spawn_node])
    while q:
        u = q.popleft()
        for v in tree_adj[u]:
            if v not in dists:
                dists[v] = dists[u] + 1
                q.append(v)
    far_nodes = sorted(dists.keys(), key=lambda u: (-dists[u], rng.random()))
    exit_node = far_nodes[0]
    path_to_exit = _find_bfs_path(tree_adj, spawn_node, exit_node)

    # Plan locked edges along path_to_exit (guaranteed no soft-locks by tree partition)
    # lock_edge_map: (u, v) sorted -> lock char ('R','B','Y') or 'D'
    lock_edge_map: dict[tuple[int, int], str] = {}
    zone_id: dict[int, int] = {u: 0 for u in range(n_nodes)}
    key_placements: list[tuple[str, int]] = []  # (key char, target node)

    key_colors = [("r", "R"), ("b", "B"), ("y", "Y")]
    rng.shuffle(key_colors)

    desired_locks = 0
    parallel_locks = False
    if family == "multi_room" and rng.random() < 0.25:
        desired_locks = 1
    elif family == "key_return":
        desired_locks = 2 if (len(path_to_exit) >= 4 and rng.random() < 0.45) else 1
        parallel_locks = desired_locks == 2 and rng.random() < 0.35
    elif family == "gauntlet":
        if len(path_to_exit) >= 5 and rng.random() < 0.45:
            desired_locks = 3
            parallel_locks = rng.random() < 0.35
        elif len(path_to_exit) >= 4:
            desired_locks = 2
            parallel_locks = rng.random() < 0.35
        else:
            desired_locks = 1

    if genome is not None and genome.extra_locks > 0:
        desired_locks = min(3, desired_locks + genome.extra_locks)

    # Cap locks by path edge count
    n_path_edges = max(0, len(path_to_exit) - 1)
    actual_locks = min(desired_locks, n_path_edges)
    if actual_locks >= 1:
        # Place locks near exit end of path: edges [-actual_locks:]
        lock_edge_indices = list(range(n_path_edges - actual_locks, n_path_edges))
        gated_roots = [path_to_exit[ei + 1] for ei in lock_edge_indices]
        for z_idx, root in enumerate(gated_roots, start=1):
            for node in _subtree_nodes(tree_children, root):
                zone_id[node] = z_idx

        zone_nodes: dict[int, list[int]] = {}
        for node, z in zone_id.items():
            if node != exit_node:
                zone_nodes.setdefault(z, []).append(node)

        z0_cands = [u for u in zone_nodes.get(0, [spawn_node]) if u != spawn_node] or [spawn_node]
        if parallel_locks and len(z0_cands) >= actual_locks:
            picked_z0 = rng.sample(z0_cands, k=actual_locks)
            for li, ei in enumerate(lock_edge_indices):
                u, v = path_to_exit[ei], path_to_exit[ei + 1]
                k_ch, d_ch = key_colors[li]
                lock_edge_map[(min(u, v), max(u, v))] = d_ch
                key_placements.append((k_ch, picked_z0[li]))
        else:
            for li, ei in enumerate(lock_edge_indices):
                u, v = path_to_exit[ei], path_to_exit[ei + 1]
                k_ch, d_ch = key_colors[li]
                lock_edge_map[(min(u, v), max(u, v))] = d_ch
                # Place key li in zone li (if non-empty) else zone 0
                cands = zone_nodes.get(li) or z0_cands
                if li == 0 and len(cands) > 1 and spawn_node in cands:
                    cands = [x for x in cands if x != spawn_node]
                key_placements.append((k_ch, rng.choice(cands)))

    # Add unlocked doors 'D' on remaining path or side edges (40-70% probability)
    all_edges: list[tuple[int, int]] = []
    for u in range(n_nodes):
        for v in tree_adj[u]:
            if u < v:
                ek = (u, v)
                all_edges.append(ek)
                if ek not in lock_edge_map and rng.random() < (0.75 if family == "door_exit" else 0.45):
                    lock_edge_map[ek] = "D"

    # Extra loop edges ONLY within identical progression zones (zero lock bypass!)
    loop_budget = rng.randint(1, 3) if n_nodes >= 4 else 0
    candidate_loops: list[tuple[int, int]] = []
    existing = set(all_edges)
    for rr in range(mr):
        for cc in range(mc):
            u = nid(rr, cc)
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < mr and 0 <= nc < mc:
                    v = nid(nr, nc)
                    ek = (min(u, v), max(u, v))
                    if ek not in existing and zone_id[u] == zone_id[v]:
                        candidate_loops.append(ek)
    rng.shuffle(candidate_loops)
    loop_edges = candidate_loops[:loop_budget]
    for ek in loop_edges:
        all_edges.append(ek)
        if rng.random() < 0.30:
            lock_edge_map[ek] = "D"

    # Assign zone/room floor & ceiling heights cleanly in {0, 20, 40} so single 1-cell
    # threshold connectors (`20`) bridge 40u gaps with <=24u step-up (zero interior border clobbering!).
    room_hf: dict[int, int] = {u: 0 for u in range(n_nodes)}
    room_hc: dict[int, int] = {u: rng.choice((128, 144, 160)) for u in range(n_nodes)}
    use_verticality = (rng.random() < 0.65) or (genome is not None and genome.force_vertical)
    if use_verticality:
        comp_id = list(range(n_nodes))

        def _find(x: int) -> int:
            while comp_id[x] != x:
                comp_id[x] = comp_id[comp_id[x]]
                x = comp_id[x]
            return x

        for u, v in all_edges:
            if (u, v) in lock_edge_map:
                ru, rv = _find(u), _find(v)
                if ru != rv:
                    comp_id[rv] = ru
        comp_hf: dict[int, int] = {}
        for u in range(n_nodes):
            r = _find(u)
            if r not in comp_hf:
                comp_hf[r] = rng.choice((0, 0, 20, 40, 48))
            room_hf[u] = comp_hf[r]
            room_hc[u] = max(room_hf[u] + 96, room_hc[u])

    floor_heights: dict[tuple[int, int], int] = {}
    ceiling_heights: dict[tuple[int, int], int] = {}
    sector_specials: dict[tuple[int, int], int] = {}

    for u, (r0, r1, c0, c1) in room_rects.items():
        hf, hc = room_hf[u], room_hc[u]
        if hf != 0 or hc != 128:
            for r in range(r0, r1):
                for c in range(c0, c1):
                    floor_heights[(r, c)] = hf
                    ceiling_heights[(r, c)] = hc

    # Carve portals (open passages, doors, stairs, one-way drop-offs, windows)
    open_cells_for_connectors: set[tuple[int, int]] = set()
    edge_set = set(all_edges)

    def _carve_connector(u: int, v: int, portal_kind: str) -> None:
        ur, uc = ncoords(u)
        vr, vc = ncoords(v)
        ur0, ur1, uc0, uc1 = room_rects[u]
        vr0, vr1, vc0, vc1 = room_rects[v]
        door_ch = lock_edge_map.get((min(u, v), max(u, v)))

        if ur == vr:
            wall_c = uc1 if uc < vc else vc1
            r_lo, r_hi = max(ur0, vr0), min(ur1, vr1)
            span = r_hi - r_lo
            w_cells = 1 if door_ch is not None else min(span, rng.choice((1, 2, 2)))
            start_r = r_lo + (span - w_cells) // 2
            conn_cells = [(r, wall_c) for r in range(start_r, start_r + w_cells)]
        else:
            wall_r = ur1 if ur < vr else vr1
            c_lo, c_hi = max(uc0, vc0), min(uc1, vc1)
            span = c_hi - c_lo
            w_cells = 1 if door_ch is not None else min(span, rng.choice((1, 2, 2)))
            start_c = c_lo + (span - w_cells) // 2
            conn_cells = [(wall_r, c) for c in range(start_c, start_c + w_cells)]

        for r, c in conn_cells:
            if portal_kind in ("door", "open", "dropoff"):
                grid[r][c] = door_ch if door_ch is not None else "."
                open_cells_for_connectors.add((r, c))
                if door_ch is None:
                    if portal_kind == "dropoff":
                        # True one-way cliff dropoff: connector sits at lower floor height
                        hf_conn = min(room_hf[u], room_hf[v])
                    elif abs(room_hf[u] - room_hf[v]) > 24:
                        # Single-cell midpoint stair landing: <= 24u step from both ends!
                        hf_conn = (room_hf[u] + room_hf[v]) // 2
                    else:
                        hf_conn = min(room_hf[u], room_hf[v])
                    hc_conn = max(room_hc[u], room_hc[v], hf_conn + 80)
                    if hf_conn != 0 or hc_conn != 128:
                        floor_heights[(r, c)] = hf_conn
                        ceiling_heights[(r, c)] = hc_conn
            elif portal_kind == "window":
                grid[r][c] = "."
                base_h = max(room_hf[u], room_hf[v])
                floor_heights[(r, c)] = base_h + 32
                ceiling_heights[(r, c)] = base_h + 80  # 48u opening < 56 -> sight passes, move blocked!

    for ek in all_edges:
        u, v = ek
        if ek in lock_edge_map:
            _carve_connector(u, v, "door")
        elif ek in loop_edges and abs(room_hf[u] - room_hf[v]) > 24 and rng.random() < 0.5:
            _carve_connector(u, v, "dropoff")
        else:
            _carve_connector(u, v, "open")

    # Add windows (~40% on non-connected adjacent walls, forced if requested by genome)
    win_prob = 0.85 if (genome is not None and genome.force_window) else 0.40
    for rr in range(mr):
        for cc in range(mc):
            u = nid(rr, cc)
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < mr and 0 <= nc < mc:
                    v = nid(nr, nc)
                    ek = (min(u, v), max(u, v))
                    if ek not in edge_set and rng.random() < win_prob:
                        _carve_connector(u, v, "window")

    # Optional pillars inside spacious rooms (>= 5x5)
    pillar_cells: set[tuple[int, int]] = set()
    for u, (r0, r1, c0, c1) in room_rects.items():
        if (r1 - r0) >= 5 and (c1 - c0) >= 5 and rng.random() < 0.45:
            pr, pc = (r0 + r1) // 2, (c0 + c1) // 2
            # Leave floor clean around pillar so connectivity stays 100% intact
            grid[pr][pc] = "#"
            pillar_cells.add((pr, pc))
            floor_heights.pop((pr, pc), None)
            ceiling_heights.pop((pr, pc), None)

    def _free_room_cells(u: int) -> list[tuple[int, int]]:
        r0, r1, c0, c1 = room_rects[u]
        out = []
        for r in range(r0, r1):
            for c in range(c0, c1):
                if grid[r][c] == ".":
                    out.append((r, c))
        return out

    # Place spawn 'P' in spawn_node center/safe cell (not on step lip)
    s_r0, s_r1, s_c0, s_c1 = room_rects[spawn_node]
    p_cell = ((s_r0 + s_r1) // 2, (s_c0 + s_c1) // 2)
    if grid[p_cell[0]][p_cell[1]] != ".":
        p_cell = rng.choice(_free_room_cells(spawn_node))
    grid[p_cell[0]][p_cell[1]] = "P"
    occupied = {p_cell}

    # Place exit 'E' in exit_node corner/center
    e_r0, e_r1, e_c0, e_c1 = room_rects[exit_node]
    e_cell = ((e_r0 + e_r1) // 2, (e_c0 + e_c1) // 2)
    if grid[e_cell[0]][e_cell[1]] != "." or e_cell in occupied:
        e_cell = rng.choice([c for c in _free_room_cells(exit_node) if c not in occupied])
    grid[e_cell[0]][e_cell[1]] = "E"
    occupied.add(e_cell)

    # Place keys
    for k_ch, k_node in key_placements:
        cands = [pt for pt in _free_room_cells(k_node) if pt not in occupied]
        if not cands:
            cands = [pt for pt in _free_room_cells(spawn_node) if pt not in occupied]
        k_pt = rng.choice(cands)
        grid[k_pt[0]][k_pt[1]] = k_ch
        occupied.add(k_pt)

    # Optional nukage pools (sector special 71) + RadSuit ('Z') on ~45% of non-T1 or ~25% of T1
    rad_prob = 0.25 if tier == 1 else 0.45
    nukage_nodes = [
        u for u in range(n_nodes)
        if u not in (spawn_node, exit_node) and (room_rects[u][1] - room_rects[u][0]) >= 4 and (room_rects[u][3] - room_rects[u][2]) >= 4
    ]
    if nukage_nodes and ((genome is not None and genome.force_nukage) or rng.random() < rad_prob):
        nuk_u = rng.choice(nukage_nodes)
        r0, r1, c0, c1 = room_rects[nuk_u]
        # Inner patch leaving dry perimeter ring & no occupied key/spawn/exit cells
        has_nuk = False
        for r in range(r0 + 1, r1 - 1):
            for c in range(c0 + 1, c1 - 1):
                if grid[r][c] == "." and (r, c) not in occupied:
                    sector_specials[(r, c)] = 71
                    has_nuk = True
        if has_nuk:
            # Put RadSuit before/near nukage in spawn or adjacent safe cell
            safe_cands = [
                pt for pt in _free_room_cells(nuk_u)
                if pt not in occupied and pt not in sector_specials
            ] or [pt for pt in _free_room_cells(spawn_node) if pt not in occupied]
            if safe_cands:
                z_pt = rng.choice(safe_cands)
                grid[z_pt[0]][z_pt[1]] = "Z"
                occupied.add(z_pt)

    # Pickups, weapons, ammo & monsters by family
    if family == "door_exit":
        n_monsters = rng.choice((0, 0, 1, 2)) if (with_combat or rng.random() < 0.45) else 0
        m_pool = ["z", "i", "s"]
        w_pool = ["g"] if rng.random() < 0.4 else []
        p_pool = rng.sample(["h", "a"], k=rng.randint(0, 2))
    elif family == "combat":
        n_monsters = rng.randint(2, 5)
        m_pool = ["z", "s", "c", "i", "m", "S", "l", "k"]
        w_pool = [rng.choice(["g", "G", "W", "C"])]
        p_pool = rng.sample(["h", "H", "A", "a", "e", "x"], k=rng.randint(2, 4))
    elif family == "multi_room":
        n_monsters = rng.randint(3, 6)
        m_pool = ["z", "s", "c", "i", "m", "S", "l", "k"]
        w_pool = [rng.choice(["g", "G", "W", "C"])]
        p_pool = rng.sample(["h", "H", "A", "a", "e", "x"], k=rng.randint(2, 4))
    elif family == "key_return":
        n_monsters = rng.randint(4, 8)
        m_pool = ["z", "s", "c", "i", "m", "S", "l", "k", "n", "K"]
        w_pool = rng.sample(["g", "G", "W", "L"], k=rng.randint(1, 2))
        p_pool = rng.sample(["h", "H", "A", "M", "a", "e", "o", "x"], k=rng.randint(2, 4))
    else:
        n_monsters = rng.randint(7, 12)
        m_pool = ["z", "s", "c", "i", "m", "S", "l", "k", "n", "N", "Q", "V", "K"]
        w_pool = rng.sample(["W", "L", "F", "G", "C"], k=rng.randint(2, 3))
        low_ammo = (rng.random() < 0.35) or (genome is not None and genome.force_low_ammo)
        if low_ammo:
            p_pool = rng.sample(["h", "H", "A", "U", "x"], k=rng.randint(2, 3))
        else:
            p_pool = rng.sample(["h", "H", "A", "M", "U", "O", "e", "o", "u", "J", "x"], k=rng.randint(3, 5))

    if genome is not None and genome.monster_delta != 0:
        n_monsters = max(0, n_monsters + genome.monster_delta)

    # Place spawn-room or early weapon + pickups
    early_nodes = [spawn_node] + [v for v in tree_adj[spawn_node] if v != exit_node]
    for ch in w_pool:
        placed = False
        for node in early_nodes:
            cands = [pt for pt in _free_room_cells(node) if pt not in occupied]
            if cands:
                pt = rng.choice(cands)
                grid[pt[0]][pt[1]] = ch
                occupied.add(pt)
                placed = True
                break
        if not placed:
            break

    other_nodes = [u for u in range(n_nodes) if u != spawn_node]
    all_spawnable = other_nodes or [spawn_node]
    for ch in p_pool:
        node = rng.choice(all_spawnable)
        cands = [pt for pt in _free_room_cells(node) if pt not in occupied]
        if cands:
            pt = rng.choice(cands)
            grid[pt[0]][pt[1]] = ch
            occupied.add(pt)

    # Distribute monsters away from immediate spawn tile (unless forced)
    monster_nodes = [u for u in range(n_nodes) if u != spawn_node] or [spawn_node]
    for _ in range(n_monsters):
        node = rng.choice(monster_nodes)
        cands = [
            pt for pt in _free_room_cells(node)
            if pt not in occupied and abs(pt[0] - p_cell[0]) + abs(pt[1] - p_cell[1]) >= 3
        ]
        if not cands:
            continue
        pt = rng.choice(cands)
        grid[pt[0]][pt[1]] = rng.choice(m_pool)
        occupied.add(pt)

    # Apply D4 transform (fliplr + rot90 k times) uniformly across all 8 symmetries
    rot_k = rng.randrange(4)
    flip_x = bool(rng.randrange(2))
    h0, w0 = len(grid), len(grid[0])
    t_grid = _transform_grid_rows(grid, rot_k, flip_x)
    t_fh, nh, nw = _transform_cell_dict(floor_heights, h0, w0, rot_k, flip_x)
    t_ch, _, _ = _transform_cell_dict(ceiling_heights, h0, w0, rot_k, flip_x)
    t_sp, _, _ = _transform_cell_dict(sector_specials, h0, w0, rot_k, flip_x)

    # Choose balanced ego player_angle evenly covering ahead/left/right/behind
    pr_t = pc_t = er_t = ec_t = None
    for r in range(nh):
        for c in range(nw):
            if t_grid[r][c] == "P":
                pr_t, pc_t = r, c
            elif t_grid[r][c] == "E":
                er_t, ec_t = r, c
    assert pr_t is not None and pc_t is not None and er_t is not None and ec_t is not None
    import math
    dx = float(ec_t - pc_t)
    dy = float(pr_t - er_t)  # UDMF Y goes upward as row r decreases
    exit_deg = math.degrees(math.atan2(dy, dx)) % 360.0
    bearing_quadrant = (seed + attempt) % 4
    offset_deg = (0, 90, 180, 270)[bearing_quadrant] + rng.randint(-25, 25)
    player_angle = int(round((exit_deg + offset_deg) % 360.0))

    grid_str = "\n".join("".join(row) for row in t_grid)
    theme = rng.choice(THEME_NAMES)
    skull_keys = (actual_locks > 0) and (rng.random() < 0.45)
    door_delay = genome.door_delay if (genome is not None and genome.door_delay is not None) else rng.choice((0, 60, 100, 150, 200))
    return grid_str, t_fh, t_ch, t_sp, cell_size, player_angle, theme, skull_keys, door_delay


def generate_dungeon_from_genome(
    genome: DungeonGenome,
    out_dir: str | Path,
    *,
    split: str = "train",
    with_combat: bool = False,
    out_path: str | Path | None = None,
) -> MapSpec:
    """Synthesize a WAD from a `DungeonGenome`, verifying solvability AND valid PBRS nav graph."""
    if genome.family not in _FAMILY_TIERS:
        raise ValueError(f"Unknown family {genome.family!r}; expected one of {tuple(_FAMILY_TIERS)}")

    target_path = (
        Path(out_path)
        if out_path is not None
        else (Path(out_dir) / f"proc_accel_{genome.family}_g{genome.generation}_s{genome.seed}.wad")
    )
    last_err: tuple[str, ...] = ()
    for attempt in range(12):
        (
            grid_str,
            fh,
            ch,
            sp,
            cell_size,
            player_angle,
            theme,
            skull_keys,
            door_delay,
        ) = _attempt_build_dungeon(
            genome.family,
            genome.seed,
            attempt,
            with_combat=with_combat,
            genome=genome,
        )
        res = verify_grid_solvability(
            grid_str,
            floor_heights=fh or None,
            ceiling_heights=ch or None,
        )
        if res.ok:
            build_grid_wad(
                target_path,
                grid_str,
                cell_size=cell_size,
                player_angle=player_angle,
                theme=theme,
                skull_keys=skull_keys,
                door_close_delay=door_delay,
                floor_heights=fh or None,
                ceiling_heights=ch or None,
                sector_specials=sp or None,
            )
            graph = _build_nav_graph(target_path, "MAP01")
            if graph is not None and len(graph.exit_cells) > 0:
                return MapSpec(
                    wad=str(target_path),
                    map_name="MAP01",
                    family=genome.family,
                    tier=_FAMILY_TIERS[genome.family],
                    split=split,
                    episode_ticks=_FAMILY_TICKS[genome.family],
                    seeds=(0, 1, 2),
                )
            last_err = ("pbrs_nav_graph_invalid",)
        else:
            last_err = res.errors

    raise RuntimeError(
        f"Failed to generate PBRS-valid dungeon for genome={genome}: {last_err}"
    )


def generate_dungeon_wad(
    family: str,
    seed: int,
    out_dir: str | Path,
    *,
    split: str = "train",
    return_distance: int | None = None,
    with_combat: bool = False,
    out_path: str | Path | None = None,
) -> MapSpec:
    """Generate a randomized connected-room-graph dungeon UDMF WAD and verify zero soft-locks."""
    if family not in _FAMILY_TIERS:
        raise ValueError(f"Unknown family {family!r}; expected one of {tuple(_FAMILY_TIERS)}")

    target_path = Path(out_path) if out_path is not None else (Path(out_dir) / f"proc_dungeon_{family}_s{seed}.wad")
    last_err: tuple[str, ...] = ()
    for attempt in range(12):
        (
            grid_str,
            fh,
            ch,
            sp,
            cell_size,
            player_angle,
            theme,
            skull_keys,
            door_delay,
        ) = _attempt_build_dungeon(family, seed, attempt, with_combat=with_combat)
        res = verify_grid_solvability(
            grid_str,
            floor_heights=fh or None,
            ceiling_heights=ch or None,
        )
        if res.ok:
            build_grid_wad(
                target_path,
                grid_str,
                cell_size=cell_size,
                player_angle=player_angle,
                theme=theme,
                skull_keys=skull_keys,
                door_close_delay=door_delay,
                floor_heights=fh or None,
                ceiling_heights=ch or None,
                sector_specials=sp or None,
            )
            return MapSpec(
                wad=str(target_path),
                map_name="MAP01",
                family=family,
                tier=_FAMILY_TIERS[family],
                split=split,
                episode_ticks=_FAMILY_TICKS[family],
                seeds=(0, 1, 2),
            )
        last_err = res.errors

    raise RuntimeError(
        f"Failed to generate valid dungeon for family={family!r}, seed={seed}: {last_err}"
    )
