"""Untouched Out-Of-Distribution (OOD) final test suite (`maps/final_ood/`).

Eight hand-crafted polygonal & verticality-rich test maps (2 per Tier 1..4) with
balanced initial spawn-to-exit ego bearings (2 ahead, 2 left, 2 right, 2 behind).
Never used in training, curriculum tuning, or dev validation.
"""

from __future__ import annotations

from training.interfaces import MapSpec
from training.wad_gen.families import MapDefinition
from training.wad_gen.udmf import PolygonHole, PolygonSector, ThingSpec


def _rect(
    name: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    kind: str = "room",
    wall_texture: str = "STARTAN2",
    floor_texture: str = "FLOOR0_1",
    ceiling_texture: str = "CEIL1_1",
    height_floor: int = 0,
    height_ceiling: int = 128,
    light_level: int = 192,
    holes: tuple[PolygonHole, ...] = (),
    sector_id: int = 0,
    door_switch: tuple[tuple[float, float], tuple[float, float], int] | None = None,
    blocking_edges: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = (),
    sector_special: int = 0,
) -> PolygonSector:
    return PolygonSector(
        name=name,
        vertices=((x0, y0), (x0, y1), (x1, y1), (x1, y0)),
        kind=kind,
        wall_texture=wall_texture,
        floor_texture=floor_texture,
        ceiling_texture=ceiling_texture,
        height_floor=height_floor,
        height_ceiling=height_ceiling,
        light_level=light_level,
        holes=holes,
        sector_id=sector_id,
        door_switch=door_switch,
        blocking_edges=blocking_edges,
        sector_special=sector_special,
    )


def _t1_open_crossfire() -> MapDefinition:
    """Wide >1024u octagonal crossfire arena + raised balcony stairs. Bearing: ahead."""
    pillar1 = PolygonHole(((448, 416), (448, 544), (576, 544), (576, 416)), wall_texture="STONE2")
    pillar2 = PolygonHole(((832, 736), (832, 864), (960, 864), (960, 736)), wall_texture="STONE2")
    sectors = (
        # Left angled octagonal entry hall -> merges cleanly onto central 1024-wide plaza (total x span = 1408u)
        PolygonSector(
            "west_wing",
            ((0, 384), (0, 896), (256, 1152), (256, 128)),
            wall_texture="BROWN1",
            floor_texture="FLOOR0_3",
            height_ceiling=160,
        ),
        PolygonSector(
            "main_arena",
            ((256, 128), (256, 1152), (1280, 1152), (1280, 576), (1280, 384), (1280, 128)),
            wall_texture="STARTAN2",
            floor_texture="FLOOR4_8",
            height_ceiling=160,
            holes=(pillar1, pillar2),
        ),
        # Raised East firing overlook reachable via South steps (0 -> 20 -> 40)
        PolygonSector(
            "step_1",
            ((1280, 128), (1280, 384), (1344, 384), (1408, 384), (1408, 128)),
            height_floor=20,
            height_ceiling=160,
            wall_texture="STEP1",
        ),
        PolygonSector(
            "step_2",
            ((1280, 384), (1280, 576), (1408, 576), (1408, 384), (1344, 384)),
            height_floor=40,
            height_ceiling=160,
            wall_texture="STEP1",
        ),
        PolygonSector(
            "balcony",
            ((1280, 576), (1280, 1152), (1408, 1152), (1408, 704), (1408, 576)),
            height_floor=40,
            height_ceiling=160,
            wall_texture="METAL1",
        ),
        _rect("exit", 1408, 576, 1472, 704, kind="exit", height_floor=40, height_ceiling=160),
    )
    # Wait: stitch main_arena (hf=0) directly to balcony (hf=40) across [576..1152] as a 40u cliff,
    # while step_1 -> step_2 -> balcony at [128..576] provides the <=20u staircase up!
    things = (
        ThingSpec(128, 640, 1, 0),       # Facing East directly toward exit -> ahead
        ThingSpec(320, 640, 2001, 0),    # Shotgun
        ThingSpec(320, 480, 2018, 0),    # GreenArmor
        ThingSpec(384, 800, 2049, 0),    # ShellBox
        ThingSpec(768, 256, 2011, 0),    # Stimpack
        ThingSpec(1024, 960, 2012, 0),   # Medikit
        ThingSpec(640, 640, 3004, 180),  # Zombieman
        ThingSpec(896, 448, 3001, 180),  # DoomImp
        ThingSpec(1088, 640, 9, 180),    # ShotgunGuy
        ThingSpec(1344, 896, 65, 180),   # ChaingunGuy on balcony overlook
    )
    return MapDefinition(
        MapSpec("maps/final_ood/final_ood_t1_open_crossfire.wad", family="combat", tier=1, split="final_ood", episode_ticks=2100),
        "polygon",
        "OOD T1 wide (>1024u) octagonal arena with pillars, staircase flank, and balcony crossfire.",
        poly_sectors=sectors,
        poly_things=things,
    )


def _t1_cliff_grate_arena() -> MapDefinition:
    """Center grate blocks movement between spawn (East) and Exit (West); climb north stairs. Bearing: left."""
    grate_edge = ((384, 0), (384, 384))
    sectors = (
        _rect("west_hall", 0, 0, 384, 384, wall_texture="BRICK1", floor_texture="FLOOR7_1", blocking_edges=(grate_edge,)),
        _rect("east_plaza", 384, 0, 896, 384, wall_texture="STONE2", floor_texture="FLOOR4_8"),
        _rect("step_e1", 512, 384, 768, 448, height_floor=20, height_ceiling=144, wall_texture="STEP1"),
        PolygonSector(
            "north_bridge",
            ((256, 448), (192, 640), (704, 640), (768, 448)),
            wall_texture="METAL1",
            height_floor=40,
            height_ceiling=160,
        ),
        _rect("step_w1", 128, 384, 320, 448, height_floor=20, height_ceiling=144, wall_texture="STEP1"),
        _rect("exit", -64, 128, 0, 256, kind="exit"),
    )
    # Match bridge bottom corners cleanly along y=448: step_e1 [512..768] shares y=448 (512,448)-(768,448),
    # Wait: step_w1 must share exact edge vertices with north_bridge & west_hall to stitch cleanly!
    # Fix shared edges exactly: (256, 448)-(384, 448) or bridge [256..768] along y=448!
    sectors = (
        _rect("west_hall", 0, 0, 384, 384, wall_texture="BRICK1", floor_texture="FLOOR7_1", blocking_edges=(grate_edge,)),
        _rect("east_plaza", 384, 0, 896, 384, wall_texture="STONE2", floor_texture="FLOOR4_8"),
        _rect("step_e", 512, 384, 768, 448, height_floor=20, height_ceiling=144, wall_texture="STEP1"),
        PolygonSector(
            "north_bridge",
            ((128, 448), (192, 640), (704, 640), (768, 448)),
            wall_texture="METAL1",
            height_floor=40,
            height_ceiling=160,
        ),
        _rect("step_w", 128, 384, 384, 448, height_floor=20, height_ceiling=144, wall_texture="STEP1"),
        _rect("exit", -64, 128, 0, 256, kind="exit"),
    )
    # Bridge bottom edge spans x=128..768 -> wait: (128,448)-(768,448) is ONE segment in north_bridge,
    # whereas step_w uses (128,448)-(384,448) and step_e uses (512,448)-(768,448) — T-junction unless vertices match!
    # Include intermediate vertices on north_bridge bottom edge so exact shared edges stitch 1:1!
    sectors = (
        _rect("west_hall", 0, 0, 384, 384, wall_texture="BRICK1", floor_texture="FLOOR7_1", blocking_edges=(grate_edge,)),
        _rect("east_plaza", 384, 0, 896, 384, wall_texture="STONE2", floor_texture="FLOOR4_8"),
        _rect("step_e", 512, 384, 768, 448, height_floor=20, height_ceiling=144, wall_texture="STEP1"),
        PolygonSector(
            "north_bridge",
            ((128, 448), (192, 640), (704, 640), (768, 448), (512, 448), (384, 448)),
            wall_texture="METAL1",
            height_floor=40,
            height_ceiling=160,
        ),
        _rect("step_w", 128, 384, 384, 448, height_floor=20, height_ceiling=144, wall_texture="STEP1"),
        _rect("exit", -64, 128, 0, 256, kind="exit"),
    )
    # Wait: west_hall north edge is (0,384)-(384,384), while step_w south edge is (128,384)-(384,384).
    # Give west_hall vertex (128, 384) and east_plaza vertex (512, 384) & (768, 384) for exact 1:1 stitching!
    sectors = (
        PolygonSector(
            "west_hall",
            ((0, 0), (0, 384), (128, 384), (384, 384), (384, 0)),
            wall_texture="BRICK1",
            floor_texture="FLOOR7_1",
            blocking_edges=(grate_edge,),
        ),
        PolygonSector(
            "east_plaza",
            ((384, 0), (384, 384), (512, 384), (768, 384), (896, 384), (896, 0)),
            wall_texture="STONE2",
            floor_texture="FLOOR4_8",
        ),
        _rect("step_e", 512, 384, 768, 448, height_floor=20, height_ceiling=144, wall_texture="STEP1"),
        PolygonSector(
            "north_bridge",
            ((128, 448), (192, 640), (704, 640), (768, 448), (512, 448), (384, 448)),
            wall_texture="METAL1",
            height_floor=40,
            height_ceiling=160,
        ),
        _rect("step_w", 128, 384, 384, 448, height_floor=20, height_ceiling=144, wall_texture="STEP1"),
        _rect("exit", -64, 128, 0, 256, kind="exit"),
    )
    # Also ensure (0, 128), (0, 256) exist on west_hall's left edge where exit attaches!
    sectors = (
        PolygonSector(
            "west_hall",
            ((0, 0), (0, 128), (0, 256), (0, 384), (128, 384), (384, 384), (384, 0)),
            wall_texture="BRICK1",
            floor_texture="FLOOR7_1",
            blocking_edges=(grate_edge,),
        ),
        sectors[1],
        sectors[2],
        sectors[3],
        sectors[4],
        sectors[5],
    )
    things = (
        ThingSpec(640, 192, 1, 90),     # Facing North (90), Exit at West (180) -> left
        ThingSpec(768, 128, 2005, 90),  # Chainsaw
        ThingSpec(576, 256, 2002, 90),  # Chaingun
        ThingSpec(512, 128, 2048, 0),   # ClipBox
        ThingSpec(448, 576, 2011, 0),   # Stimpack
        ThingSpec(192, 192, 3004, 0),   # Zombieman
        ThingSpec(480, 544, 9, 0),      # ShotgunGuy
        ThingSpec(256, 288, 3001, 0),   # DoomImp
        ThingSpec(128, 192, 3002, 0),   # Demon guarding exit
    )
    return MapDefinition(
        MapSpec("maps/final_ood/final_ood_t1_cliff_grate_arena.wad", family="combat", tier=1, split="final_ood", episode_ticks=2100),
        "polygon",
        "OOD T1 visible exit through central grate with elevated angled north stair flank.",
        poly_sectors=sectors,
        poly_things=things,
    )


def _t2_nukage_remote() -> MapDefinition:
    """Nukage trench hub + RadSuit + West switch opening North remote door to Exit. Bearing: right."""
    sectors = (
        # Central south courtyard
        PolygonSector(
            "south_hub",
            ((192, 0), (192, 256), (256, 256), (448, 256), (576, 256), (640, 256), (640, 0)),
            wall_texture="GRAY1",
            floor_texture="FLOOR0_3",
        ),
        # Central nukage trench (damaging floor) flanked by safe walkways!
        PolygonSector(
            "nukage_trench",
            ((256, 256), (256, 512), (576, 512), (576, 256), (448, 256)),
            wall_texture="BROWN1",
            floor_texture="NUKAGE1",
            height_floor=-16,
            sector_special=71,
        ),
        # West dry walkway + switch alcove
        PolygonSector(
            "west_walk",
            ((0, 256), (0, 512), (256, 512), (256, 256), (192, 256)),
            wall_texture="METAL1",
            floor_texture="FLOOR5_1",
            door_switch=((0.0, 256.0), (0.0, 512.0), 10),
        ),
        # East dry walkway
        PolygonSector(
            "east_walk",
            ((576, 256), (576, 512), (768, 512), (768, 256), (640, 256)),
            wall_texture="METAL1",
            floor_texture="FLOOR5_1",
        ),
        # North vestibule
        PolygonSector(
            "north_hall",
            ((192, 512), (192, 704), (352, 704), (480, 704), (640, 704), (640, 512), (576, 512), (256, 512)),
            wall_texture="STONE2",
            floor_texture="FLOOR4_8",
        ),
        _rect("remote_gate", 352, 704, 480, 768, kind="remote_door", sector_id=10),
        _rect("north_sanctum", 352, 768, 480, 896, wall_texture="BRICK1"),
        _rect("exit", 352, 896, 480, 960, kind="exit"),
    )
    things = (
        ThingSpec(416, 128, 1, 180),     # Facing West (180), Exit at North (90) -> right
        ThingSpec(288, 128, 2025, 0),    # RadSuit
        ThingSpec(544, 128, 2001, 0),    # Shotgun
        ThingSpec(672, 384, 82, 0),      # SuperShotgun
        ThingSpec(128, 384, 2019, 0),    # BlueArmor near switch
        ThingSpec(416, 384, 58, 270),    # Spectre inside nukage
        ThingSpec(416, 608, 65, 270),    # ChaingunGuy
        ThingSpec(672, 448, 3001, 180),  # DoomImp
        ThingSpec(416, 832, 3006, 270),  # LostSoul behind remote gate
    )
    return MapDefinition(
        MapSpec("maps/final_ood/final_ood_t2_nukage_remote.wad", family="multi_room", tier=2, split="final_ood", episode_ticks=2400),
        "polygon",
        "OOD T2 nukage trench + RadSuit + west switch opening north remote door to exit.",
        poly_sectors=sectors,
        poly_things=things,
    )


def _t2_stair_labyrinth() -> MapDefinition:
    """Elevated diamond hub (0 -> 20 -> 40 -> 60) with 3 manual doors & pillar hole. Bearing: behind."""
    pillar = PolygonHole(((512, 320), (512, 448), (640, 448), (640, 320)), wall_texture="BRICK1")
    sectors = (
        _rect("west_start", 0, 256, 256, 512, wall_texture="STARTAN2"),
        _rect("door_1", 256, 320, 320, 448, kind="door"),
        _rect("step_1", 320, 256, 416, 512, height_floor=20, height_ceiling=160, wall_texture="STEP1"),
        PolygonSector(
            "diamond_hub",
            ((416, 256), (416, 512), (576, 640), (768, 512), (768, 256), (576, 128)),
            wall_texture="STONE2",
            floor_texture="FLOOR4_8",
            height_floor=40,
            height_ceiling=176,
            holes=(pillar,),
        ),
        _rect("door_2", 768, 320, 832, 448, kind="door", height_floor=40, height_ceiling=176),
        _rect("east_landing", 832, 256, 992, 512, height_floor=40, height_ceiling=176, wall_texture="METAL1"),
        _rect("step_2", 992, 320, 1088, 448, height_floor=60, height_ceiling=192, wall_texture="STEP1"),
        _rect("door_3", 1088, 320, 1152, 448, kind="door", height_floor=60, height_ceiling=192),
        _rect("exit_hall", 1152, 320, 1280, 448, height_floor=60, height_ceiling=192),
        _rect("exit", 1280, 320, 1344, 448, kind="exit", height_floor=60, height_ceiling=192),
    )
    # Match exact vertex splits on east_landing, step_1, diamond_hub for door frames:
    sectors = (
        PolygonSector("west_start", ((0, 256), (0, 512), (256, 512), (256, 448), (256, 320), (256, 256))),
        sectors[1],
        PolygonSector("step_1", ((320, 256), (320, 320), (320, 448), (320, 512), (416, 512), (416, 256)), height_floor=20, height_ceiling=160, wall_texture="STEP1"),
        PolygonSector(
            "diamond_hub",
            ((416, 256), (416, 512), (576, 640), (768, 512), (768, 448), (768, 320), (768, 256), (576, 128)),
            wall_texture="STONE2",
            floor_texture="FLOOR4_8",
            height_floor=40,
            height_ceiling=176,
            holes=(pillar,),
        ),
        sectors[4],
        PolygonSector("east_landing", ((832, 256), (832, 320), (832, 448), (832, 512), (992, 512), (992, 448), (992, 320), (992, 256)), height_floor=40, height_ceiling=176, wall_texture="METAL1"),
        sectors[6],
        sectors[7],
        sectors[8],
        sectors[9],
    )
    things = (
        ThingSpec(128, 384, 1, 180),     # Facing West (180), Exit East (0) -> behind
        ThingSpec(192, 304, 2001, 0),    # Shotgun
        ThingSpec(368, 384, 2011, 0),    # Stimpack
        ThingSpec(576, 208, 2049, 0),    # ShellBox
        ThingSpec(912, 448, 2012, 0),    # Medikit
        ThingSpec(576, 560, 3004, 180),  # Zombieman
        ThingSpec(704, 384, 9, 180),     # ShotgunGuy
        ThingSpec(912, 320, 3001, 180),  # DoomImp
        ThingSpec(1216, 384, 3005, 180), # Cacodemon
    )
    return MapDefinition(
        MapSpec("maps/final_ood/final_ood_t2_stair_labyrinth.wad", family="door_exit", tier=2, split="final_ood", episode_ticks=2400),
        "polygon",
        "OOD T2 ascending stepped diamond hub with 3 manual doors and central pillar.",
        poly_sectors=sectors,
        poly_things=things,
    )


def _t3_dropoff_skull() -> MapDefinition:
    """Climb East stairs (0->20->40->64) for Red+Blue Skulls, take 64u one-way cliff drop-off to courtyard! Bearing: ahead."""
    sectors = (
        # Lower courtyard (hf=0)
        PolygonSector(
            "courtyard",
            ((0, 0), (0, 512), (192, 512), (320, 512), (448, 512), (512, 512), (512, 192), (512, 64), (512, 0)),
            wall_texture="STONE2",
            floor_texture="FLOOR4_8",
            height_ceiling=192,
        ),
        # East stairs climbing 0 -> 20 -> 40 -> 64
        _rect("stair_1", 512, 64, 576, 192, height_floor=20, height_ceiling=192, wall_texture="STEP1"),
        _rect("stair_2", 576, 64, 640, 192, height_floor=40, height_ceiling=192, wall_texture="STEP1"),
        PolygonSector(
            "rampart",
            ((640, 64), (640, 192), (640, 512), (832, 512), (832, 64)),
            height_floor=64,
            height_ceiling=192,
            wall_texture="BRICK1",
            floor_texture="FLOOR7_1",
        ),
        # One-way dropoff ledge bridging rampart (x=640, hf=64) to courtyard edge (x=512, hf=0):
        # rampart [y=192..512] -> high_ledge [512..640, y=256..448, hf=64] -> drop straight onto courtyard (hf=0)!
        PolygonSector(
            "high_overlook",
            ((512, 192), (512, 512), (640, 512), (640, 192)),
            height_floor=64,
            height_ceiling=192,
            wall_texture="METAL1",
        ),
        # Sequential locks at North of courtyard: RedDoor -> alcove -> BlueDoor -> Exit
        _rect("red_door", 192, 512, 320, 576, kind="door_red", height_ceiling=192),
        PolygonSector(
            "lock_vestibule",
            ((192, 576), (192, 704), (320, 704), (320, 576)),
            wall_texture="METAL1",
            height_ceiling=192,
        ),
        _rect("blue_door", 192, 704, 320, 768, kind="door_blue", height_ceiling=192),
        _rect("exit_sanctum", 192, 768, 320, 896, wall_texture="GRAY1", height_ceiling=192),
        _rect("exit", 192, 896, 320, 960, kind="exit", height_ceiling=192),
    )
    things = (
        ThingSpec(256, 128, 1, 90),     # Facing North (90), Exit at North (256, 928) -> ahead
        ThingSpec(128, 256, 82, 0),     # SuperShotgun
        ThingSpec(384, 256, 2003, 0),   # RocketLauncher
        ThingSpec(384, 192, 2046, 0),   # RocketBox
        ThingSpec(736, 160, 38, 0),     # RedSkull on rampart
        ThingSpec(736, 448, 40, 0),     # BlueSkull on rampart
        ThingSpec(576, 448, 2013, 0),   # Soulsphere on overlook before drop
        ThingSpec(128, 400, 2012, 0),   # Medikit
        ThingSpec(256, 384, 65, 270),   # ChaingunGuy
        ThingSpec(736, 320, 3002, 270), # Demon
        ThingSpec(576, 320, 66, 180),   # Revenant
        ThingSpec(256, 640, 69, 270),   # HellKnight
        ThingSpec(256, 832, 3001, 270), # DoomImp
    )
    return MapDefinition(
        MapSpec("maps/final_ood/final_ood_t3_dropoff_skull.wad", family="key_return", tier=3, split="final_ood", episode_ticks=3200),
        "polygon",
        "OOD T3 stairs to skull rampart with 64u one-way dropoff return to dual-locked exit.",
        poly_sectors=sectors,
        poly_things=things,
    )


def _t3_window_backtrack() -> MapDefinition:
    """Diamond/trapezoid vaults with <56u sight window, Red+Yellow backtracking, Revenant/Arachnotron. Bearing: left."""
    sectors = (
        # East spawn courtyard
        PolygonSector(
            "east_court",
            ((640, 128), (640, 384), (640, 512), (640, 768), (960, 640), (960, 256)),
            wall_texture="BROWN1",
            floor_texture="FLOOR0_1",
        ),
        # Center window bridging east_court <-> west_vault directly (hf=32, hc=80 -> 48u < 56 -> sight only!)
        _rect("window_slot", 576, 384, 640, 512, height_floor=32, height_ceiling=80, wall_texture="BRICK1"),
        # South unlocked route around window to RedCard
        PolygonSector(
            "south_corridor",
            ((320, 0), (320, 128), (640, 128), (640, 0)),
            wall_texture="STONE2",
            floor_texture="FLOOR4_8",
        ),
        # Central-west junction
        PolygonSector(
            "west_vault",
            ((320, 128), (320, 384), (320, 512), (320, 768), (576, 768), (576, 512), (576, 384), (576, 128)),
            wall_texture="METAL1",
            floor_texture="FLOOR5_1",
        ),
        # North RedDoor -> north alcove holding YellowCard
        PolygonSector("east_court_fix", ((0, 0), (0, 0))) if False else _rect("north_red_door", 400, 768, 512, 832, kind="door_red"),
        _rect("north_alcove", 320, 832, 576, 1024, wall_texture="BRICK1"),
        # West YellowDoor -> Exit hall (x=64..320)
        _rect("west_yel_door", 256, 384, 320, 512, kind="door_yellow"),
        _rect("exit_vault", 64, 384, 256, 512, wall_texture="GRAY1"),
        _rect("exit", 0, 384, 64, 512, kind="exit"),
    )
    # Fix shared vertices strictly on all shared edges!
    sectors = (
        PolygonSector(
            "east_court",
            ((640, 0), (640, 128), (640, 384), (640, 512), (640, 768), (960, 640), (960, 256), (832, 0)),
            wall_texture="BROWN1",
            floor_texture="FLOOR0_1",
        ),
        _rect("window_slot", 576, 384, 640, 512, height_floor=32, height_ceiling=80, wall_texture="BRICK1"),
        PolygonSector(
            "south_corridor",
            ((320, 0), (320, 128), (576, 128), (640, 128), (640, 0)),
            wall_texture="STONE2",
            floor_texture="FLOOR4_8",
        ),
        PolygonSector(
            "west_vault",
            ((320, 128), (320, 384), (320, 512), (320, 768), (400, 768), (512, 768), (576, 768), (576, 512), (576, 384), (576, 128)),
            wall_texture="METAL1",
            floor_texture="FLOOR5_1",
        ),
        _rect("north_red_door", 400, 768, 512, 832, kind="door_red"),
        PolygonSector(
            "north_alcove",
            ((320, 832), (320, 1024), (576, 1024), (576, 832), (512, 832), (400, 832)),
            wall_texture="BRICK1",
        ),
        _rect("west_yel_door", 256, 384, 320, 512, kind="door_yellow"),
        _rect("exit_vault", 64, 384, 256, 512, wall_texture="GRAY1"),
        _rect("exit", 0, 384, 64, 512, kind="exit"),
    )
    # Wait: east_court south edge splits against south_corridor at (640, 0)-(640, 128) cleanly!
    # Also ensure south_corridor connects to west_vault across (320,128)-(576,128). Check vertex matching!
    # In west_vault, bottom edge is (576,128)->(320,128); in south_corridor, top edge splits (320,128)->(576,128) and (576,128)->(640,128).
    # That shares (320,128)-(576,128) cleanly! And window_slot at (576,384)-(640,384) ONLY touches west_vault [384..512] and east_court [384..512]!
    # Note gap [576..640] on y=128..384 and y=512..768 acts as solid pillar wall around window!
    things = (
        ThingSpec(864, 448, 1, 90),      # Facing North (90), Exit due West (180) -> left
        ThingSpec(800, 320, 2002, 90),   # Chaingun
        ThingSpec(800, 576, 2004, 90),   # PlasmaRifle
        ThingSpec(736, 448, 2047, 0),    # Cell
        ThingSpec(448, 224, 13, 0),      # RedCard inside west_vault
        ThingSpec(448, 928, 6, 0),       # YellowCard in north alcove
        ThingSpec(448, 64, 2018, 0),     # GreenArmor
        ThingSpec(704, 224, 65, 90),     # ChaingunGuy
        ThingSpec(448, 448, 68, 0),      # Arachnotron visible through window!
        ThingSpec(448, 640, 58, 270),    # Spectre
        ThingSpec(448, 896, 67, 270),    # Fatso (Mancubus)
        ThingSpec(160, 448, 3005, 0),    # Cacodemon
    )
    return MapDefinition(
        MapSpec("maps/final_ood/final_ood_t3_window_backtrack.wad", family="key_return", tier=3, split="final_ood", episode_ticks=3200),
        "polygon",
        "OOD T3 trapezoidal vault with central sight window and Red->Yellow backtrack.",
        poly_sectors=sectors,
        poly_things=things,
    )


def _t4_three_lock_fortress() -> MapDefinition:
    """Three-lock citadel (Red -> Blue -> Yellow) with hub pillar hole, nukage moat, 9 monsters. Bearing: right."""
    pillar = PolygonHole(((576, 576), (576, 704), (704, 704), (704, 576)), wall_texture="METAL1")
    sectors = (
        # Central 512x512 octagonal hub [384..896]x[384..896] with vertices for doors N/S/E/W
        PolygonSector(
            "hub",
            (
                (384, 384), (384, 576), (384, 704), (384, 896),
                (576, 896), (704, 896), (896, 896),
                (896, 704), (896, 576), (896, 384),
                (704, 384), (576, 384),
            ),
            wall_texture="STONE2",
            floor_texture="FLOOR4_8",
            height_ceiling=192,
            holes=(pillar,),
        ),
        # West wing (unlocked): RedCard
        _rect("w_door", 320, 576, 384, 704, kind="door", height_ceiling=192),
        _rect("west_wing", 64, 448, 320, 832, wall_texture="BRICK1", floor_texture="FLOOR7_1", height_ceiling=192),
        # North wing (RedDoor): nukage center + BlueCard
        _rect("n_red_door", 576, 896, 704, 960, kind="door_red", height_ceiling=192),
        PolygonSector(
            "north_wing",
            ((384, 960), (384, 1280), (896, 1280), (896, 960), (704, 960), (576, 960)),
            wall_texture="GRAY1",
            floor_texture="FLOOR0_3",
            height_ceiling=192,
        ),
        # East wing (BlueDoor): raised step + YellowCard + Plasma/Cells
        _rect("e_blue_door", 896, 576, 960, 704, kind="door_blue", height_ceiling=192),
        PolygonSector(
            "east_wing",
            ((960, 448), (960, 576), (960, 704), (960, 832), (1280, 832), (1280, 448)),
            wall_texture="METAL1",
            floor_texture="FLOOR5_1",
            height_floor=16,
            height_ceiling=192,
        ),
        # South sanctum (YellowDoor): Exit!
        _rect("s_yel_door", 576, 320, 704, 384, kind="door_yellow", height_ceiling=192),
        PolygonSector(
            "south_sanctum",
            ((448, 128), (448, 320), (576, 320), (704, 320), (832, 320), (832, 128), (704, 128), (576, 128)),
            wall_texture="BRICK1",
            height_ceiling=192,
        ),
        _rect("exit", 576, 64, 704, 128, kind="exit", height_ceiling=192),
    )
    # Make west_wing / north_wing edges match w_door / n_red_door vertices exactly:
    sectors = (
        sectors[0],
        sectors[1],
        PolygonSector(
            "west_wing",
            ((64, 448), (64, 832), (320, 832), (320, 704), (320, 576), (320, 448)),
            wall_texture="BRICK1",
            floor_texture="FLOOR7_1",
            height_ceiling=192,
        ),
        sectors[3],
        sectors[4],
        sectors[5],
        sectors[6],
        sectors[7],
        sectors[8],
        sectors[9],
    )
    things = (
        ThingSpec(448, 640, 1, 0),       # Facing East (0), Exit South (270) -> right
        ThingSpec(448, 768, 82, 0),      # SuperShotgun
        ThingSpec(448, 512, 2002, 0),    # Chaingun
        ThingSpec(256, 768, 2018, 0),    # GreenArmor
        ThingSpec(128, 640, 13, 0),      # RedCard (West)
        ThingSpec(640, 1184, 5, 0),      # BlueCard (North)
        ThingSpec(1184, 640, 6, 0),      # YellowCard (East)
        ThingSpec(480, 1024, 2025, 0),   # RadSuit
        ThingSpec(1024, 512, 2004, 0),   # PlasmaRifle
        ThingSpec(1024, 768, 17, 0),     # CellPack
        ThingSpec(1088, 768, 2019, 0),   # BlueArmor
        ThingSpec(640, 1120, 2012, 0),   # Medikit
        ThingSpec(512, 640, 2049, 0),    # ShellBox
        # 9 diverse heavy & light enemies (reward scavenging across wings)
        ThingSpec(768, 448, 9, 180),     # ShotgunGuy
        ThingSpec(192, 512, 3002, 0),    # Demon
        ThingSpec(192, 768, 65, 0),      # ChaingunGuy
        ThingSpec(512, 1152, 68, 270),   # Arachnotron
        ThingSpec(768, 1152, 67, 270),   # Fatso
        ThingSpec(1120, 512, 3005, 180), # Cacodemon
        ThingSpec(1120, 768, 66, 180),   # Revenant
        ThingSpec(1184, 576, 3003, 180), # BaronOfHell
        ThingSpec(640, 224, 69, 90),     # HellKnight guarding Exit
    )
    return MapDefinition(
        MapSpec("maps/final_ood/final_ood_t4_three_lock_fortress.wad", family="gauntlet", tier=4, split="final_ood", episode_ticks=4200),
        "polygon",
        "OOD T4 three-lock citadel (Red -> Blue -> Yellow) with hub pillar and 9-enemy boss roster.",
        poly_sectors=sectors,
        poly_things=things,
    )


def _t4_nukage_gauntlet() -> MapDefinition:
    """Wide nukage canyon with islands, remote bridge switch, Red+Yellow skull locks, 8 monsters. Bearing: behind."""
    sectors = (
        # East spawn ledge (dry, hf=16)
        PolygonSector(
            "east_ledge",
            ((1152, 128), (1152, 384), (1152, 640), (1152, 896), (1472, 896), (1472, 128)),
            wall_texture="BROWN1",
            floor_texture="FLOOR0_3",
            height_floor=16,
            height_ceiling=192,
        ),
        # Large central nukage basin (hf=0, special=71, wide >1024 along Y too!)
        PolygonSector(
            "nukage_basin",
            (
                (512, 128), (512, 384), (512, 640), (512, 896),
                (704, 896), (960, 896), (1152, 896),
                (1152, 640), (1152, 384), (1152, 128),
            ),
            wall_texture="STONE2",
            floor_texture="NUKAGE1",
            height_floor=0,
            height_ceiling=192,
            sector_special=71,
        ),
        # Center safe island inside basin — wait to avoid T-junction/donut overlap, put North bridge tower!
        PolygonSector(
            "north_switch_tower",
            ((704, 896), (704, 1152), (960, 1152), (960, 896)),
            wall_texture="METAL1",
            floor_texture="FLOOR5_1",
            height_floor=16,
            height_ceiling=192,
            door_switch=((704.0, 1152.0), (960.0, 1152.0), 20),
        ),
        # West causeway (dry, hf=16) -> split into North RedDoor path and Direct center RemoteDoor path
        PolygonSector(
            "west_bank",
            ((256, 128), (256, 384), (256, 640), (256, 896), (512, 896), (512, 640), (512, 384), (512, 128)),
            wall_texture="BRICK1",
            floor_texture="FLOOR7_1",
            height_floor=16,
            height_ceiling=192,
        ),
        _rect("mid_remote_door", 192, 384, 256, 640, kind="remote_door", sector_id=20, height_floor=16, height_ceiling=192),
        _rect("inner_keep", 128, 384, 192, 640, wall_texture="STONE2", height_floor=16, height_ceiling=192),
        _rect("yel_lock_door", 64, 384, 128, 640, kind="door_yellow", height_floor=16, height_ceiling=192),
        _rect("exit", 0, 384, 64, 640, kind="exit", height_floor=16, height_ceiling=192),
    )
    # Wait: let's also add a RedSkull lock gating inner_keep -> yel_lock_door?
    # Keep inner_keep divided with red_lock_door + yel_lock_door so BOTH remote switch AND YellowSkull are required!
    sectors = (
        sectors[0],
        sectors[1],
        sectors[2],
        sectors[3],
        _rect("mid_remote_door", 192, 384, 256, 640, kind="remote_door", sector_id=20, height_floor=16, height_ceiling=192),
        _rect("inner_keep", 128, 384, 192, 640, wall_texture="STONE2", height_floor=16, height_ceiling=192),
        _rect("yel_lock_door", 64, 384, 128, 640, kind="door_yellow", height_floor=16, height_ceiling=192),
        _rect("exit", 0, 384, 64, 640, kind="exit", height_floor=16, height_ceiling=192),
    )
    # Also add RedCard lock guarding YellowSkull in west_bank alcove (64..256, y=640..896) -> no extra T-junction!
    # Wait: west_bank left wall is [256,128..896]; we have segments [128..384], [384..640], [640..896].
    # Attach a RedDoor on [128..256]? No: attach red_door on (256,640)-(256,896) leading to YellowSkull chamber!
    sectors = (
        sectors[0],
        sectors[1],
        sectors[2],
        sectors[3],
        sectors[4],
        sectors[5],
        sectors[6],
        sectors[7],
        _rect("nw_red_door", 192, 640, 256, 896, kind="door_red", height_floor=16, height_ceiling=192),
        _rect("nw_skull_vault", 64, 640, 192, 896, wall_texture="BRICK1", height_floor=16, height_ceiling=192),
    )
    things = (
        ThingSpec(1344, 512, 1, 0),      # Facing East (0), Exit West (0, 512) -> behind
        ThingSpec(1280, 256, 82, 180),   # SuperShotgun
        ThingSpec(1280, 768, 2003, 180), # RocketLauncher
        ThingSpec(1216, 512, 2025, 0),   # RadSuit on east ledge before nukage
        ThingSpec(1216, 320, 2049, 0),   # ShellBox
        ThingSpec(1216, 704, 2046, 0),   # RocketBox
        ThingSpec(832, 1024, 38, 0),     # RedSkull in north switch tower!
        ThingSpec(128, 768, 39, 0),      # YellowSkull in nw skull vault!
        ThingSpec(384, 256, 2013, 0),    # Soulsphere
        ThingSpec(384, 768, 2019, 0),    # BlueArmor
        ThingSpec(960, 320, 2035, 0),    # ExplosiveBarrel
        # 8 varied monsters
        ThingSpec(960, 512, 3001, 0),    # DoomImp
        ThingSpec(768, 640, 65, 0),      # ChaingunGuy
        ThingSpec(768, 320, 3002, 0),    # Demon
        ThingSpec(832, 960, 69, 270),    # HellKnight guarding switch/RedSkull
        ThingSpec(384, 512, 3005, 0),    # Cacodemon
        ThingSpec(448, 384, 67, 0),      # Fatso
        ThingSpec(128, 704, 66, 0),      # Revenant
        ThingSpec(160, 512, 3003, 0),    # BaronOfHell
    )
    return MapDefinition(
        MapSpec("maps/final_ood/final_ood_t4_nukage_gauntlet.wad", family="gauntlet", tier=4, split="final_ood", episode_ticks=4200),
        "polygon",
        "OOD T4 nukage canyon with remote gate, RedSkull+YellowSkull locks, and 8-enemy gauntlet.",
        poly_sectors=sectors,
        poly_things=things,
    )


FINAL_OOD_DEFINITIONS = (
    _t1_open_crossfire(),
    _t1_cliff_grate_arena(),
    _t2_nukage_remote(),
    _t2_stair_labyrinth(),
    _t3_dropoff_skull(),
    _t3_window_backtrack(),
    _t4_three_lock_fortress(),
    _t4_nukage_gauntlet(),
)

FINAL_OOD_MAPS: tuple[MapSpec, ...] = tuple(defn.spec for defn in FINAL_OOD_DEFINITIONS)
