"""Verified Train, Dev, and Final map families across all 4 DOOM evaluation tiers (Role 3).

Strict split discipline (REVISED_PLAN.md §3):
  - `train`: 20 parameterized layouts spanning Tier 1 (combat), Tier 2 (door_exit & multi_room),
             Tier 3 (short -> long -> mirrored key_return), and Tier 4 (return-ambush & 2-lock
             gauntlet up to 4,200 ticks).
  - `dev`:   10 held-out layouts with distinct structural topologies (never in `train`),
             including 2 distinct simple door-exit layouts for the Functional MVP gate.
  - `final`: 8 untouched final layouts (never trained or tuned on):
             4 unseen grid layouts (1 per tier) + 4 independent non-grid polygonal/diagonal-wall
             layouts (1 per tier) built with `build_polygon_wad` (distinct construction method).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from training.interfaces import MapSpec
from training.wad_gen.udmf import (
    PolygonHole,
    PolygonSector,
    ThingSpec,
    build_grid_wad,
    build_polygon_wad,
    layout_hash,
)
from training.wad_gen.verify import verify_grid_solvability, verify_polygon_solvability


@dataclass(frozen=True)
class MapDefinition:
    """Complete blueprint for one verified WAD in `maps/{split}/`."""

    spec: MapSpec
    construction: Literal["grid", "polygon"]
    description: str
    tags: tuple[str, ...] = ()
    # Grid blueprint fields
    grid: str | None = None
    cell_size: int = 64
    player_angle: int = 0
    theme: str = "techbase"
    monster_angle: int = 180
    # Polygon blueprint fields
    poly_sectors: tuple[PolygonSector, ...] = field(default_factory=tuple)
    poly_things: tuple[ThingSpec, ...] = field(default_factory=tuple)
    # Expected door counts for automated verification
    min_unlocked_doors: int = 0
    min_locked_doors: int = 0

    def blueprint_hash(self) -> str:
        if self.construction == "grid":
            assert self.grid is not None
            return layout_hash(f"grid:{self.cell_size}:{self.player_angle}\n{self.grid}")
        parts = [
            f"{ps.name}:{ps.kind}:{ps.vertices}:{[h.vertices for h in ps.holes]}"
            for ps in self.poly_sectors
        ]
        tparts = [f"{t.doomednum}@{t.x},{t.y},{t.angle}" for t in self.poly_things]
        return layout_hash("poly\n" + "\n".join(parts) + "\n" + "\n".join(tparts))


# ---------------------------------------------------------------------------
# Helper constructors for the non-grid polygonal final family
# ---------------------------------------------------------------------------

def _octagon(cx: float, cy: float, r: float, a: float) -> tuple[tuple[float, float], ...]:
    """Clockwise chamfered octagon with cardinal edge length 2*a and 45-deg diagonals."""
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


def _extrude_edge(
    p1: tuple[float, float],
    p2: tuple[float, float],
    dx: float,
    dy: float,
    flare: float = 0.0,
) -> tuple[tuple[float, float], ...]:
    """Build a quad/trapezoid attached to edge (p1 -> p2) extruded by (dx, dy).

    `p1 -> p2` is a CW edge of the parent polygon; the new polygon uses `p2 -> p1`
    as its shared edge (preserving exact opposite direction for clean stitching).
    Optional `flare` widens the far edge perpendicular to the extrusion along the edge axis.
    """
    ex, ey = p2[0] - p1[0], p2[1] - p1[1]
    length = (ex * ex + ey * ey) ** 0.5
    ux, uy = ex / length, ey / length
    q1 = (round(p1[0] + dx - ux * flare, 2), round(p1[1] + dy - uy * flare, 2))
    q2 = (round(p2[0] + dx + ux * flare, 2), round(p2[1] + dy + uy * flare, 2))
    # Order: p2 -> p1 -> q1 -> q2 (shares p2->p1 opposite to parent's p1->p2)
    return (p2, p1, q1, q2)


def _poly_octagon_combat() -> tuple[tuple[PolygonSector, ...], tuple[ThingSpec, ...]]:
    """Tier 1 Final (Polygon): Octagonal arena with central diamond pillar + angled exit alcove."""
    # Central octagon: r=288, a=96 -> cardinal edges 192 units, 45-deg diagonals ~271 units
    arena = _octagon(0, 0, 288, 96)
    # East edge is arena[4]=(288, 96) -> arena[5]=(288, -96).
    # Build an angled exit vestibule on NE diagonal: arena[3]=(96, 288) -> arena[4]=(288, 96)
    p1, p2 = arena[3], arena[4]
    vestibule = _extrude_edge(p1, p2, 96, 96)
    # Far edge of vestibule is q1=(192, 384) -> q2=(384, 192) -> exit pad extruded further NE
    q1, q2 = vestibule[2], vestibule[3]
    exit_pad = _extrude_edge(q2, q1, 64, 64)

    diamond_pillar = PolygonHole(
        vertices=((0, -72), (72, 0), (0, 72), (-72, 0)),
        wall_texture="METAL1",
    )
    sectors = (
        PolygonSector(
            name="arena",
            vertices=arena,
            kind="room",
            wall_texture="BRICK1",
            floor_texture="FLOOR7_1",
            ceiling_texture="CEIL3_2",
            light_level=192,
            holes=(diamond_pillar,),
        ),
        PolygonSector(
            name="ne_vestibule",
            vertices=vestibule,
            kind="room",
            wall_texture="STONE2",
            floor_texture="FLOOR4_8",
            light_level=192,
        ),
        PolygonSector(
            name="exit_pad",
            vertices=exit_pad,
            kind="exit",
            wall_texture="GRAY1",
            light_level=208,
        ),
    )
    things = (
        ThingSpec(-192, -32, 1, angle=30),       # Player SW-W side
        ThingSpec(-128, 96, 2001, angle=0),      # Shotgun
        ThingSpec(-128, -128, 2012, angle=0),    # Medikit
        ThingSpec(-192, 64, 2049, angle=0),      # ShellBox
        ThingSpec(160, 0, 3001, angle=180),      # DoomImp East
        ThingSpec(0, 176, 9, angle=270),         # ShotgunGuy North
        ThingSpec(96, -160, 3004, angle=135),    # Zombieman SE
        ThingSpec(256, 224, 3001, angle=225),    # DoomImp guarding NE vestibule
    )
    return sectors, things


def _poly_hex_hub_explore() -> tuple[tuple[PolygonSector, ...], tuple[ThingSpec, ...]]:
    """Tier 2 Final (Polygon): Octagonal hub with 3 angled/diagonal branches (1 dead end, 2 doors, exit)."""
    hub = _octagon(0, 0, 192, 64)
    # 3 diagonal branches off NW, NE, SE diagonals!
    # NW edge: hub[1]=(-192, 64) -> hub[2]=(-64, 192), outward (-64, 64)
    nw_door = _extrude_edge(hub[1], hub[2], -48, 48)
    nw_room = _extrude_edge(nw_door[3], nw_door[2], -144, 144)

    # NE edge: hub[3]=(64, 192) -> hub[4]=(192, 64), outward (64, 64)
    ne_door = _extrude_edge(hub[3], hub[4], 48, 48)
    ne_room = _extrude_edge(ne_door[3], ne_door[2], 144, 144)
    ne_exit = _extrude_edge(ne_room[3], ne_room[2], 64, 64)

    # SE edge: hub[5]=(192, -64) -> hub[6]=(64, -192), outward (64, -64) -> open dead-end lure
    se_wing = _extrude_edge(hub[5], hub[6], 160, -160)

    # South alcove where player starts: hub[6]=(64,-192) -> hub[7]=(-64,-192), outward (0,-128)
    s_entry = _extrude_edge(hub[6], hub[7], 0, -128)

    sectors = (
        PolygonSector("hub", hub, "room", wall_texture="STONE2", floor_texture="FLOOR4_8"),
        PolygonSector("s_entry", s_entry, "room", wall_texture="STARTAN2", floor_texture="FLOOR0_1"),
        PolygonSector("nw_door", nw_door, "door"),
        PolygonSector("nw_deadend", nw_room, "room", wall_texture="BROWN1", floor_texture="FLAT14"),
        PolygonSector("ne_door", ne_door, "door"),
        PolygonSector("ne_chamber", ne_room, "room", wall_texture="METAL1", floor_texture="FLOOR5_1"),
        PolygonSector("ne_exit", ne_exit, "exit"),
        PolygonSector("se_lure", se_wing, "room", wall_texture="BRICK1", floor_texture="FLOOR7_1"),
    )
    things = (
        ThingSpec(0, -256, 1, angle=90),         # Player enters hub facing North
        ThingSpec(0, 0, 2011, angle=0),          # Stimpack in center
        ThingSpec(-224, 224, 2012, angle=0),     # Medikit in NW dead end
        ThingSpec(-192, 192, 3004, angle=315),   # Zombieman in NW dead end
        ThingSpec(208, -208, 2048, angle=0),     # ClipBox in SE lure
        ThingSpec(208, 208, 3004, angle=225),    # Guard in NE exit chamber
    )
    return sectors, things


def _poly_diamond_key_return() -> tuple[tuple[PolygonSector, ...], tuple[ThingSpec, ...]]:
    """Tier 3 Final (Polygon): Octagonal hub with Blue key in diagonal SW wing, Blue lock on NE wing."""
    hub = _octagon(0, 0, 192, 64)
    # West entry corridor: hub[0]=(-192,-64) -> hub[1]=(-192,64), outward (-128, 0)
    w_entry = _extrude_edge(hub[0], hub[1], -128, 0)
    # NW distractor branch with regular door
    nw_door = _extrude_edge(hub[1], hub[2], -48, 48)
    nw_trap = _extrude_edge(nw_door[3], nw_door[2], -128, 128)
    # SE key wing: diagonal corridor + door + key chamber
    se_hall = _extrude_edge(hub[5], hub[6], 96, -96)
    se_door = _extrude_edge(se_hall[3], se_hall[2], 48, -48)
    se_key_room = _extrude_edge(se_door[3], se_door[2], 144, -144)
    # NE locked wing: Blue locked door + vault + exit
    ne_lock = _extrude_edge(hub[3], hub[4], 48, 48)
    ne_vault = _extrude_edge(ne_lock[3], ne_lock[2], 128, 128)
    ne_exit = _extrude_edge(ne_vault[3], ne_vault[2], 64, 64)

    hub_pillar = PolygonHole(vertices=((0, -40), (40, 0), (0, 40), (-40, 0)), wall_texture="METAL1")

    sectors = (
        PolygonSector("hub", hub, "room", wall_texture="GRAY1", floor_texture="FLOOR0_3", holes=(hub_pillar,)),
        PolygonSector("w_entry", w_entry, "room", wall_texture="STARTAN2", floor_texture="FLOOR0_1"),
        PolygonSector("nw_door", nw_door, "door"),
        PolygonSector("nw_trap", nw_trap, "room", wall_texture="BROWN1", floor_texture="FLAT14"),
        PolygonSector("se_hall", se_hall, "room", wall_texture="STONE2", floor_texture="FLOOR4_8"),
        PolygonSector("se_door", se_door, "door"),
        PolygonSector("se_key_room", se_key_room, "room", wall_texture="BRICK1", floor_texture="FLOOR7_1"),
        PolygonSector("ne_blue_lock", ne_lock, "door_blue"),
        PolygonSector("ne_vault", ne_vault, "room", wall_texture="METAL1", floor_texture="FLOOR5_1"),
        PolygonSector("ne_exit", ne_exit, "exit"),
    )
    things = (
        ThingSpec(-256, 0, 1, angle=0),          # Player spawns West facing hub & NE lock
        ThingSpec(-160, 0, 2001, angle=0),       # Shotgun
        ThingSpec(336, -336, 5, angle=0),        # BlueCard deep in SE key room
        ThingSpec(288, -288, 3001, angle=135),   # DoomImp guarding BlueCard
        ThingSpec(-224, 224, 2011, angle=0),     # Stimpack in NW trap
        ThingSpec(240, 240, 3004, angle=225),    # Zombieman in NE vault
    )
    return sectors, things


def _poly_citadel_gauntlet() -> tuple[tuple[PolygonSector, ...], tuple[ThingSpec, ...]]:
    """Tier 4 Final (Polygon): Two-lock octagonal citadel (Red -> Yellow) with diagonal wings & combat."""
    hub = _octagon(0, 0, 224, 64)
    # South spawn hall: hub[6]=(64,-224) -> hub[7]=(-64,-224), outward (0, -128)
    s_spawn = _extrude_edge(hub[6], hub[7], 0, -128)
    # West wing (regular door -> Red key chamber with enemies)
    w_door = _extrude_edge(hub[0], hub[1], -48, 0)
    w_room = _extrude_edge(w_door[3], w_door[2], -192, 0)
    # East wing (Red locked door -> combat chamber -> Yellow key)
    e_red_lock = _extrude_edge(hub[4], hub[5], 48, 0)
    e_room = _extrude_edge(e_red_lock[3], e_red_lock[2], 224, 0)
    # NW diagonal distractor alcove (supplies + guard)
    nw_alcove = _extrude_edge(hub[1], hub[2], -112, 112)
    # North exit wing (Yellow locked door -> final guard room -> exit pad)
    n_yel_lock = _extrude_edge(hub[2], hub[3], 0, 48)
    n_sanctum = _extrude_edge(n_yel_lock[3], n_yel_lock[2], 0, 192)
    n_exit = _extrude_edge(n_sanctum[3], n_sanctum[2], 0, 64)

    center_diamond = PolygonHole(vertices=((0, -48), (48, 0), (0, 48), (-48, 0)), wall_texture="BRICK1")

    sectors = (
        PolygonSector("hub", hub, "room", wall_texture="STONE2", floor_texture="FLOOR4_8", holes=(center_diamond,)),
        PolygonSector("s_spawn", s_spawn, "room", wall_texture="STARTAN2", floor_texture="FLOOR0_1"),
        PolygonSector("w_door", w_door, "door"),
        PolygonSector("w_red_wing", w_room, "room", wall_texture="BROWN1", floor_texture="FLAT14"),
        PolygonSector("e_red_lock", e_red_lock, "door_red"),
        PolygonSector("e_yel_wing", e_room, "room", wall_texture="BRICK1", floor_texture="FLOOR7_1"),
        PolygonSector("nw_alcove", nw_alcove, "room", wall_texture="GRAY1", floor_texture="FLOOR0_3"),
        PolygonSector("n_yel_lock", n_yel_lock, "door_yellow"),
        PolygonSector("n_sanctum", n_sanctum, "room", wall_texture="METAL1", floor_texture="FLOOR5_1"),
        PolygonSector("n_exit", n_exit, "exit"),
    )
    things = (
        ThingSpec(0, -288, 1, angle=90),         # Player facing North toward hub & Yellow lock
        ThingSpec(0, -224, 2001, angle=0),       # Shotgun at entry
        ThingSpec(-32, -256, 2049, angle=0),     # ShellBox
        ThingSpec(-192, 192, 2018, angle=0),     # GreenArmor in NW alcove
        ThingSpec(-160, 160, 3004, angle=315),   # Guard in NW alcove
        ThingSpec(-400, 0, 13, angle=0),         # RedCard in West wing
        ThingSpec(-336, 24, 3001, angle=0),      # DoomImp in West wing
        ThingSpec(432, 0, 6, angle=180),         # YellowCard in East wing
        ThingSpec(352, -24, 9, angle=180),       # ShotgunGuy in East wing
        ThingSpec(352, 24, 2012, angle=0),       # Medikit in East wing
        ThingSpec(128, -96, 3004, angle=135),    # Hub patrol guard
        ThingSpec(0, 384, 3001, angle=270),      # Sanctum Imp before exit
    )
    return sectors, things


# ---------------------------------------------------------------------------
# TRAIN SUITE (20 diverse layouts across all 4 tiers)
# ---------------------------------------------------------------------------

TRAIN_DEFINITIONS: tuple[MapDefinition, ...] = (
    # ---- Stage 1 Connect & Simple Door-Exit Starters (Tier 2) ----
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_door_straight.wad",
            map_name="MAP01",
            family="door_exit",
            tier=2,
            split="train",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Stage 1 Connect door/exit map: two rooms separated by a manual door, no enemies.",
        tags=("connect", "mvp", "door_exit"),
        theme="techbase",
        player_angle=0,
        min_unlocked_doors=1,
        grid="""
        ###########
        #....#....#
        #.P..D..E.#
        #....#....#
        ###########
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_door_lturn.wad",
            map_name="MAP01",
            family="door_exit",
            tier=2,
            split="train",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="L-shaped two-room door traversal with north-facing doorway and stimpack.",
        tags=("mvp", "door_exit"),
        theme="stone",
        player_angle=0,
        min_unlocked_doors=1,
        grid="""
        #########
        #....E..#
        #.......#
        #####D###
        #.P..h..#
        #.......#
        #########
        """,
    ),
    # ---- Tier 1: Combat & Survival (4 layouts) ----
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_combat_pistol_zombies.wad",
            map_name="MAP01",
            family="combat",
            tier=1,
            split="train",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Basic hitscan combat arena with cover pillars, stimpacks, ammo, and exit.",
        tags=("mvp", "combat"),
        theme="techbase",
        player_angle=0,
        grid="""
        #############
        #...........#
        #.P.#...#.z.#
        #..a....h..E#
        #...#...#.z.#
        #...........#
        #############
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_combat_imp_dodge.wad",
            map_name="MAP01",
            family="combat",
            tier=1,
            split="train",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Projectile dodging hall with two Imps, central cover block, shotgun, medikit, and exit.",
        tags=("mvp", "combat"),
        theme="brown",
        cell_size=80,
        player_angle=0,
        grid="""
        ############
        #........i.#
        #.P.g.##...#
        #..e..##.H.E
        #........i.#
        ############
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_combat_mixed_alcoves.wad",
            map_name="MAP01",
            family="combat",
            tier=1,
            split="train",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Mixed hitscan/projectile/melee combat with angled spawn bearing (90 deg) and cover alcoves.",
        tags=("mvp", "combat", "post_pickup_combat"),
        theme="brick",
        cell_size=80,
        player_angle=90,
        grid="""
        ############
        #..A....s..#
        #..##..##..#
        #.P....m..E#
        #..##..##..#
        #.g.e..H.i.#
        ############
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_combat_barrel_room.wad",
            map_name="MAP01",
            family="combat",
            tier=1,
            split="train",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Back-turned spawn (180 deg) arena with chaingun, explosive barrels, Imps, and Zombiemen.",
        tags=("progression", "combat", "post_pickup_combat"),
        theme="metal",
        cell_size=80,
        player_angle=180,
        grid="""
        #############
        #..G....x.i.#
        #....#......#
        #.P..#..z..E#
        #..a.#......#
        #..H....x.z.#
        #############
        """,
    ),
    # ---- Tier 2: Multi-Room Exploration (4 layouts) ----
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_explore_three_rooms.wad",
            map_name="MAP01",
            family="multi_room",
            tier=2,
            split="train",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Three sequential rooms with two orthogonal doors and one guard Zombieman.",
        tags=("mvp", "multi_room"),
        theme="stone",
        player_angle=0,
        min_unlocked_doors=2,
        grid="""
        #############
        #...#...#...#
        #.P.D...#..E#
        #...#..zD...#
        #...#...#...#
        #############
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_explore_t_junction.wad",
            map_name="MAP01",
            family="multi_room",
            tier=2,
            split="train",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="T-junction with a health-lure dead-end branch and an exit branch behind a door.",
        tags=("mvp", "multi_room", "branch_exploration"),
        theme="gray",
        player_angle=0,
        min_unlocked_doors=2,
        grid="""
        ###########
        #####..h..#
        #####..z..#
        #####D#####
        #.P.......#
        #####D#####
        #####.....#
        #####..E..#
        ###########
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_explore_loop_courtyard.wad",
            map_name="MAP01",
            family="multi_room",
            tier=2,
            split="train",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Central loop around a courtyard block with dead-end closet, 2 doors, and far exit.",
        tags=("progression", "multi_room", "branch_exploration"),
        theme="brown",
        player_angle=0,
        min_unlocked_doors=2,
        grid="""
        ##############
        #...#.....#..#
        #.P.D.###.#h.#
        #...#.#.#.D..#
        #####.#.#.#..#
        #..z..###....#
        #.......#..E.#
        ##############
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_explore_four_chambers.wad",
            map_name="MAP01",
            family="multi_room",
            tier=2,
            split="train",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Four-chamber zigzag with N/S/E/W doorway orientations, dead-end alcove, and 2 guards.",
        tags=("progression", "multi_room", "branch_exploration"),
        theme="metal",
        player_angle=90,
        min_unlocked_doors=3,
        grid="""
        ##############
        #..h..#......#
        #..#..D..z...#
        #..#..####D###
        #..#..#......#
        #.P#..#..#...#
        #..D..#..#.E.#
        #.....#..#...#
        ##############
        """,
    ),
    # ---- Tier 3: Key & Return (6 layouts: short from Run 1 -> long -> mirrored -> combat) ----
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_key_short_red.wad",
            map_name="MAP01",
            family="key_return",
            tier=3,
            split="train",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Stage 1 Connect key/lock map & Run-1 short key-return: see Red lock ahead, Red key in short north alcove.",
        tags=("connect", "mvp", "key_return", "return_route"),
        theme="techbase",
        player_angle=0,
        min_locked_doors=1,
        grid="""
        ###########
        ###..r..###
        ###.....###
        #....#....#
        #.P..R..E.#
        #....#....#
        ###########
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_key_short_blue_right.wad",
            map_name="MAP01",
            family="key_return",
            tier=3,
            split="train",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Short Blue key-return with key on south-right wing and Blue locked exit on north-left wing.",
        tags=("mvp", "key_return", "return_route"),
        theme="stone",
        player_angle=90,
        min_locked_doors=1,
        grid="""
        ###########
        #..E..#...#
        #.....#...#
        ###B#######
        #.........#
        #...P.....#
        #######.###
        #.....#...#
        #.....#.b.#
        ###########
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_key_medium_yellow.wad",
            map_name="MAP01",
            family="key_return",
            tier=3,
            split="train",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Medium return: see Yellow lock first, traverse regular door to Yellow key, pass distractor alcove to return.",
        tags=("mvp", "progression", "key_return", "return_route", "branch_exploration"),
        theme="brown",
        player_angle=0,
        min_unlocked_doors=1,
        min_locked_doors=1,
        grid="""
        ###############
        #.....#...#...#
        #..y..D...#..h#
        #.....#...#...#
        #######...##.##
        #..P......#...#
        #.........Y.E.#
        #.........#...#
        ###############
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_key_mirrored_corridors.wad",
            map_name="MAP01",
            family="key_return",
            tier=3,
            split="train",
            episode_ticks=3150,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Mirrored corridors with repeated doorways (REVISED_PLAN §4 landmark-memory diagnostic).",
        tags=("progression", "key_return", "return_route", "mirrored"),
        theme="gray",
        player_angle=90,
        min_unlocked_doors=2,
        min_locked_doors=1,
        grid="""
        #################
        #...#.......#...#
        #.r.D.......R.E.#
        #...#.......#...#
        ###.###...###.###
        #...#.......#...#
        #.h.D...P...#...#
        #...#.......#...#
        #################
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_key_long_branch_combat.wad",
            map_name="MAP01",
            family="key_return",
            tier=3,
            split="train",
            episode_ticks=3150,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Longer Blue key branch guarded by an Imp and hub Zombieman, shotgun at start.",
        tags=("progression", "key_return", "return_route", "post_pickup_combat"),
        theme="brick",
        player_angle=0,
        min_unlocked_doors=1,
        min_locked_doors=1,
        grid="""
        #################
        #..b..#.....#...#
        #..i..D.....#...#
        #.....#####.#...#
        ###########.#...#
        #.P.g....z..B.E.#
        #..e........#...#
        #################
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_key_hidden_lock.wad",
            map_name="MAP01",
            family="key_return",
            tier=3,
            split="train",
            episode_ticks=3150,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Locked door hidden around a loop so agent discovers key or lock via exploration.",
        tags=("progression", "key_return", "branch_exploration"),
        theme="metal",
        player_angle=180,
        min_unlocked_doors=2,
        min_locked_doors=1,
        grid="""
        ################
        #....#....#....#
        #.r..D....#..E.#
        #....#.##.R....#
        ###.##.##.#....#
        #......P..D..z.#
        #..h......#....#
        ################
        """,
    ),
    # ---- Tier 4: Full Gauntlet (4 layouts, return ambushes & multi-lock up to 4,200 ticks) ----
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_gauntlet_ambush_return.wad",
            map_name="MAP01",
            family="gauntlet",
            tier=4,
            split="train",
            episode_ticks=3150,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Post-pickup return ambush gauntlet: grab Red key, fight through return guards, unlock exit.",
        tags=("progression", "full", "gauntlet", "post_pickup_combat"),
        theme="brick",
        cell_size=80,
        player_angle=0,
        min_unlocked_doors=1,
        min_locked_doors=1,
        grid="""
        ###############
        #..r..#...#...#
        #..H..D.i.#...#
        #.....#...#.E.#
        #####.##.##R###
        #.P.g....z....#
        #..e..##...s..#
        ###############
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_gauntlet_two_locks_seq.wad",
            map_name="MAP01",
            family="gauntlet",
            tier=4,
            split="train",
            episode_ticks=4200,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Sequential 2-lock gauntlet (4,200 ticks): Red key -> Red combat wing -> Blue key -> Blue exit.",
        tags=("full", "gauntlet", "return_route", "post_pickup_combat"),
        theme="metal",
        cell_size=80,
        player_angle=90,
        min_unlocked_doors=1,
        min_locked_doors=2,
        grid="""
        #################
        #..b..#...#.....#
        #..i..R...#..E..#
        #..H..#...#.....#
        #####.###.###B###
        #.....#.........#
        #..r..D..P.g.e..#
        #..z..#....s....#
        #################
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_gauntlet_two_locks_hub.wad",
            map_name="MAP01",
            family="gauntlet",
            tier=4,
            split="train",
            episode_ticks=4200,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Central hub with 2 locks (Yellow -> Red), dead-end trap, Demon + Imp + ShotgunGuy, scarce medikits.",
        tags=("full", "gauntlet", "branch_exploration", "post_pickup_combat"),
        theme="stone",
        cell_size=80,
        player_angle=0,
        min_unlocked_doors=1,
        min_locked_doors=2,
        grid="""
        #################
        #..y..#...#..r..#
        #..z..D...Y..i..#
        #.....#...#..H..#
        #####.#...#.#####
        #.P.g...m...#...#
        #..e..#.A.#.R.E.#
        #.....#...s.#...#
        #################
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/train/train_gauntlet_full_complex.wad",
            map_name="MAP01",
            family="gauntlet",
            tier=4,
            split="train",
            episode_ticks=4200,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Full 4-wing complex with regular doors, Blue -> Yellow locks, cover pillars, barrels, 4,200 ticks.",
        tags=("full", "gauntlet", "return_route", "post_pickup_combat"),
        theme="brown",
        cell_size=80,
        player_angle=0,
        min_unlocked_doors=2,
        min_locked_doors=2,
        grid="""
        ##################
        #..b..#....#..y..#
        #..z..D....B..i..#
        #.....#.##.#..H..#
        #####.#.##.#.#####
        #.P.g...x....#...#
        #..e..#....#.Y.E.#
        #####.D.s..#.#...#
        #..A..#....#.....#
        ##################
        """,
    ),
)


# ---------------------------------------------------------------------------
# DEV SUITE (10 fixed held-out layouts — NEVER in train, distinct topologies)
# ---------------------------------------------------------------------------

DEV_DEFINITIONS: tuple[MapDefinition, ...] = (
    # ---- Functional MVP Gate: 2 distinct held-out simple door-exit layouts ----
    MapDefinition(
        spec=MapSpec(
            wad="maps/dev/dev_simple_dogleg_door.wad",
            map_name="MAP01",
            family="door_exit",
            tier=2,
            split="dev",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="MVP Gate Simple Layout #1: Z-shaped dogleg corridor with South-facing door and exit.",
        tags=("mvp_gate_simple", "door_exit"),
        theme="gray",
        player_angle=0,
        min_unlocked_doors=1,
        grid="""
        ###########
        #.P.......#
        #.........#
        ######D####
        #####.....#
        #####..E..#
        ###########
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/dev/dev_simple_chamber_door.wad",
            map_name="MAP01",
            family="door_exit",
            tier=2,
            split="dev",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="MVP Gate Simple Layout #2: Wide chamber with central cover pillar, North-facing door, and exit room.",
        tags=("mvp_gate_simple", "door_exit"),
        theme="brown",
        player_angle=90,
        min_unlocked_doors=1,
        grid="""
        ###########
        #....E....#
        #.........#
        ####D######
        #....h....#
        #...##....#
        #..P......#
        ###########
        """,
    ),
    # ---- Tier 1 Dev: Combat & Survival (2 held-out layouts) ----
    MapDefinition(
        spec=MapSpec(
            wad="maps/dev/dev_combat_crossfire.wad",
            map_name="MAP01",
            family="combat",
            tier=1,
            split="dev",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Dev Combat #1: Symmetric cross-pillar arena with 2 Imps, 1 ShotgunGuy, shotgun, armor, and exit.",
        tags=("mvp_gate_combat", "combat"),
        theme="stone",
        cell_size=80,
        player_angle=0,
        grid="""
        ##############
        #..A....#..i.#
        #...#...#....#
        #.P.g...s...E#
        #..e#...#....#
        #..H....#..i.#
        ##############
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/dev/dev_combat_ambush_ring.wad",
            map_name="MAP01",
            family="combat",
            tier=1,
            split="dev",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Dev Combat #2: Ring arena with central block, angled spawn (270 deg), Demon + Imp + 2 Zombiemen.",
        tags=("mvp_gate_combat", "combat"),
        theme="brick",
        cell_size=80,
        player_angle=270,
        grid="""
        #############
        #.P.g.....z.#
        #..e.###....#
        #....###.m..#
        #..H.###...E#
        #.z.......i.#
        #############
        """,
    ),
    # ---- Tier 2 Dev: Multi-Room Exploration (2 held-out layouts) ----
    MapDefinition(
        spec=MapSpec(
            wad="maps/dev/dev_explore_double_loop.wad",
            map_name="MAP01",
            family="multi_room",
            tier=2,
            split="dev",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Dev Exploration #1: Figure-8 double loop with 3 doors, 2 dead ends, 2 guards, remote exit.",
        tags=("dev_explore", "multi_room"),
        theme="techbase",
        player_angle=0,
        min_unlocked_doors=3,
        grid="""
        ################
        #...#......#...#
        #.P.D.##...D.h.#
        #...#.##...#...#
        ###.#....###.###
        #...#.##.#.....#
        #.z.#.##.D..E..#
        #...#....#.....#
        ################
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/dev/dev_explore_spiral_doors.wad",
            map_name="MAP01",
            family="multi_room",
            tier=2,
            split="dev",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Dev Exploration #2: Inward spiral of 4 rooms with alternating E/S/W/N doors and trap alcove.",
        tags=("dev_explore", "multi_room"),
        theme="metal",
        player_angle=0,
        min_unlocked_doors=3,
        grid="""
        ##############
        #.P..#.......#
        #....D....z..#
        ######.###D###
        #..h.#.#.....#
        #....#.#..E..#
        ###D##.####.##
        #..........#.#
        ##############
        """,
    ),
    # ---- Tier 3 Dev: Key & Return (2 held-out layouts) ----
    MapDefinition(
        spec=MapSpec(
            wad="maps/dev/dev_key_short_return.wad",
            map_name="MAP01",
            family="key_return",
            tier=3,
            split="dev",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Dev Key-Return #1: Short/medium Blue key-return with visible lock at junction + distractor dead end.",
        tags=("dev_key", "key_return"),
        theme="stone",
        player_angle=0,
        min_unlocked_doors=1,
        min_locked_doors=1,
        grid="""
        ###############
        #####..h..#####
        #####.....#####
        #####D#########
        #.P......#....#
        #........B..E.#
        #####.####....#
        #####.....#####
        #####..b..#####
        ###############
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/dev/dev_key_long_mirrored.wad",
            map_name="MAP01",
            family="key_return",
            tier=3,
            split="dev",
            episode_ticks=3150,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Dev Key-Return #2: Long-return mirrored H-corridor with repeated doors, Red key NW -> Red lock SE.",
        tags=("dev_key", "key_return", "mirrored"),
        theme="gray",
        player_angle=90,
        min_unlocked_doors=3,
        min_locked_doors=1,
        grid="""
        ###################
        #...#.........#...#
        #.r.D...z.....D.h.#
        #...#.........#...#
        ###.###.....###.###
        #####....P....#####
        ###.###.....###.###
        #...#.........#...#
        #.a.D.....z...R.E.#
        #...#.........#...#
        ###################
        """,
    ),
    # ---- Tier 4 Dev: Full Gauntlet (2 held-out layouts, 4,200 ticks) ----
    MapDefinition(
        spec=MapSpec(
            wad="maps/dev/dev_gauntlet_return_ambush.wad",
            map_name="MAP01",
            family="gauntlet",
            tier=4,
            split="dev",
            episode_ticks=4200,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Dev Gauntlet #1: Yellow key deep wing + return combat ambush + dead-end trap + Yellow exit vault.",
        tags=("dev_gauntlet", "gauntlet"),
        theme="brick",
        cell_size=80,
        player_angle=0,
        min_unlocked_doors=2,
        min_locked_doors=1,
        grid="""
        #################
        #..y..#...#..h..#
        #..i..D...D..z..#
        #..H..#...#.....#
        #####.##.##.#####
        #.P.g...s...#...#
        #..e..##..i.Y.E.#
        #..A..##....#...#
        #################
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/dev/dev_gauntlet_twin_locks.wad",
            map_name="MAP01",
            family="gauntlet",
            tier=4,
            split="dev",
            episode_ticks=4200,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Dev Gauntlet #2: Twin-lock fortress (Red -> Blue) with barrels, Demon/Imp/ShotgunGuy, 4,200 ticks.",
        tags=("dev_gauntlet", "gauntlet"),
        theme="metal",
        cell_size=80,
        player_angle=90,
        min_unlocked_doors=1,
        min_locked_doors=2,
        grid="""
        ###################
        #..b..#.....#.....#
        #..i..R..x..B..E..#
        #..H..#..s..#.....#
        #####.###.###.#####
        #.....#.....#.....#
        #..r..D..P..#..A..#
        #..z..#.g.e.#..m..#
        #.....#.....#.....#
        ###################
        """,
    ),
)


# ---------------------------------------------------------------------------
# FINAL UNTOUCHED SUITE (8 layouts: 4 unseen grid + 4 independent polygonal)
# ---------------------------------------------------------------------------

_POLY_T1_SECS, _POLY_T1_THINGS = _poly_octagon_combat()
_POLY_T2_SECS, _POLY_T2_THINGS = _poly_hex_hub_explore()
_POLY_T3_SECS, _POLY_T3_THINGS = _poly_diamond_key_return()
_POLY_T4_SECS, _POLY_T4_THINGS = _poly_citadel_gauntlet()

FINAL_DEFINITIONS: tuple[MapDefinition, ...] = (
    # ---- Unseen Grid Final Layouts (1 per tier) ----
    MapDefinition(
        spec=MapSpec(
            wad="maps/final/final_combat_bunker.wad",
            map_name="MAP01",
            family="combat",
            tier=1,
            split="final",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Final Grid T1: Staggered bunker trenches with Imp, ShotgunGuy, Demon, Zombieman, shotgun & exit.",
        tags=("final_grid", "combat"),
        theme="metal",
        cell_size=80,
        player_angle=0,
        grid="""
        ###############
        #....#....#...#
        #.P..#..s.#.i.#
        #.g..#....#..E#
        #.e.A...m...H.#
        #....#..z.#...#
        ###############
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/final/final_explore_labyrinth.wad",
            map_name="MAP01",
            family="multi_room",
            tier=2,
            split="final",
            episode_ticks=3150,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Final Grid T2: Asymmetric multi-loop labyrinth with 4 doors, false-door dead ends, and 3 guards.",
        tags=("final_grid", "multi_room"),
        theme="stone",
        player_angle=90,
        min_unlocked_doors=4,
        grid="""
        #################
        #..h..#.....#...#
        #.....D..z..D.E.#
        ###.###.###.#...#
        #.....#.#...#####
        #.P.g.D.#...#...#
        #..e..#.D.z.#.i.#
        #.....#.....#...#
        #################
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/final/final_key_deep_backtrack.wad",
            map_name="MAP01",
            family="key_return",
            tier=3,
            split="final",
            episode_ticks=3150,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Final Grid T3: Deep U-wing Yellow key backtrack with distractor loop and Yellow locked exit vault.",
        tags=("final_grid", "key_return"),
        theme="brown",
        player_angle=180,
        min_unlocked_doors=2,
        min_locked_doors=1,
        grid="""
        ##################
        #..y..#.....#....#
        #..i..D.....#..E.#
        #.....#####.Y....#
        #######...#.#....#
        #..h..D...#.#....#
        #.....#..P..D..z.#
        #..g..#..e..#....#
        ##################
        """,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/final/final_gauntlet_citadel.wad",
            map_name="MAP01",
            family="gauntlet",
            tier=4,
            split="final",
            episode_ticks=4200,
            seeds=(0, 1, 2),
        ),
        construction="grid",
        description="Final Grid T4: Two-lock citadel (Blue -> Red) with return ambush, dead-end trap, scarce supplies.",
        tags=("final_grid", "gauntlet"),
        theme="brick",
        cell_size=80,
        player_angle=0,
        min_unlocked_doors=2,
        min_locked_doors=2,
        grid="""
        ###################
        #..b..#.....#..r..#
        #..z..D..s..B..i..#
        #..h..#.....#..H..#
        #####.###.###.#####
        #.P.g....x....#...#
        #..e..##...m..R.E.#
        #####.D..A..#.#...#
        #..i..#.....#.....#
        ###################
        """,
    ),
    # ---- Independent Construction Method: Non-Grid Polygonal/Diagonal-Wall Final Family (1 per tier) ----
    MapDefinition(
        spec=MapSpec(
            wad="maps/final/final_poly_octagon_combat.wad",
            map_name="MAP01",
            family="combat",
            tier=1,
            split="final",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="polygon",
        description="Final Polygon T1 (distinct non-grid construction): Octagonal arena with diamond cover pillar & angled exit.",
        tags=("final_polygon", "distinct_construction", "combat"),
        poly_sectors=_POLY_T1_SECS,
        poly_things=_POLY_T1_THINGS,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/final/final_poly_hex_hub_explore.wad",
            map_name="MAP01",
            family="multi_room",
            tier=2,
            split="final",
            episode_ticks=2100,
            seeds=(0, 1, 2),
        ),
        construction="polygon",
        description="Final Polygon T2 (distinct non-grid construction): Octagonal hub with 3 diagonal 45-deg branches & 2 angled doors.",
        tags=("final_polygon", "distinct_construction", "multi_room"),
        poly_sectors=_POLY_T2_SECS,
        poly_things=_POLY_T2_THINGS,
        min_unlocked_doors=2,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/final/final_poly_diamond_key_return.wad",
            map_name="MAP01",
            family="key_return",
            tier=3,
            split="final",
            episode_ticks=3150,
            seeds=(0, 1, 2),
        ),
        construction="polygon",
        description="Final Polygon T3 (distinct non-grid construction): Angled hub, SE diagonal Blue key wing -> NE Blue locked vault.",
        tags=("final_polygon", "distinct_construction", "key_return"),
        poly_sectors=_POLY_T3_SECS,
        poly_things=_POLY_T3_THINGS,
        min_unlocked_doors=2,
        min_locked_doors=1,
    ),
    MapDefinition(
        spec=MapSpec(
            wad="maps/final/final_poly_citadel_gauntlet.wad",
            map_name="MAP01",
            family="gauntlet",
            tier=4,
            split="final",
            episode_ticks=4200,
            seeds=(0, 1, 2),
        ),
        construction="polygon",
        description="Final Polygon T4 (distinct non-grid construction): Octagonal citadel with Red -> Yellow locks & diagonal wings.",
        tags=("final_polygon", "distinct_construction", "gauntlet"),
        poly_sectors=_POLY_T4_SECS,
        poly_things=_POLY_T4_THINGS,
        min_unlocked_doors=1,
        min_locked_doors=2,
    ),
)

ALL_DEFINITIONS: tuple[MapDefinition, ...] = (
    *TRAIN_DEFINITIONS,
    *DEV_DEFINITIONS,
    *FINAL_DEFINITIONS,
)


def materialize_definition(defn: MapDefinition, repo_root: Path) -> Path:
    """Verify static solvability and write one `MapDefinition` WAD to disk."""
    out_path = repo_root / defn.spec.wad
    if defn.construction == "grid":
        assert defn.grid is not None
        check = verify_grid_solvability(defn.grid)
        if not check.ok:
            raise ValueError(f"Static verification failed for {defn.spec.wad}: {check.errors}")
        return build_grid_wad(
            out_path,
            defn.grid,
            map_name=defn.spec.map_name,
            cell_size=defn.cell_size,
            player_angle=defn.player_angle,
            theme=defn.theme,
            monster_angle=defn.monster_angle,
        )
    ok, errs = verify_polygon_solvability(defn.poly_sectors, defn.poly_things)
    if not ok:
        raise ValueError(f"Static polygon verification failed for {defn.spec.wad}: {errs}")
    return build_polygon_wad(
        out_path,
        defn.poly_sectors,
        defn.poly_things,
        map_name=defn.spec.map_name,
    )


def materialize_all_maps(repo_root: Path) -> list[Path]:
    """Generate and statically verify every registered Train, Dev, and Final WAD."""
    return [materialize_definition(d, repo_root) for d in ALL_DEFINITIONS]
