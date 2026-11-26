"""Held-out matched navigation probes. Layout coordinates are in Doom units."""

from dataclasses import replace

from training.interfaces import MapSpec
from training.wad_gen.families import MapDefinition
from training.wad_gen.udmf import PolygonSector, ThingSpec


def _rect(name, x0, y0, x1, y1, **kwargs):
    return PolygonSector(name, ((x0, y0), (x0, y1), (x1, y1), (x1, y0)), **kwargs)


def _probe(name, description, sectors, things, *, locked=False):
    return MapDefinition(
        MapSpec(f"maps/stress/{name}.wad", family="key_return" if locked else "multi_room",
                tier=3 if locked else 2, split="stress", episode_ticks=4200 if locked else 2100),
        "polygon", description, poly_sectors=tuple(sectors), poly_things=tuple(things),
        min_locked_doors=int(locked),
    )


def _exit_probes():
    hall = PolygonSector("hall", ((0, 32), (0, 160), (32, 192), (384, 192),
                                  (384, 0), (32, 0)))
    pad = _rect("pad", 384, 0, 448, 192, kind="exit")
    for label, angle in (("ahead", 0), ("left", 270), ("behind", 180)):
        yield _probe(f"exit_{label}", f"Exit starts {label} of the player.",
                     (hall, pad), (ThingSpec(80, 96, 1, angle),))
    switch = replace(pad, kind="room", floor_texture="GATE3", exit_switch=((448, 0), (448, 192)))
    yield _probe("exit_switch", "Same east alcove, USE its far wall to exit.",
                 (hall, switch), (ThingSpec(80, 96, 1),))


def _key_window(window):
    sectors = [
        _rect("spawn", 0, 0, 192, 160),
        _rect("opening", 192, 0, 256, 160, height_ceiling=40 if window else 128),
        _rect("key", 256, 0, 448, 160),
        _rect("northwest", 0, 160, 192, 320),
        _rect("north", 192, 160, 256, 320),
        _rect("northeast", 256, 160, 448, 320),
        _rect("lock", -48, 0, 0, 160, kind="door_blue"),
        _rect("vestibule", -176, 0, -48, 160),
        _rect("exit", -240, 0, -176, 160, kind="exit"),
    ]
    return _probe("key_window" if window else "key_open",
                  "Retrieve a visible blue skull and return to the west lock.",
                  sectors, (ThingSpec(80, 80, 1), ThingSpec(352, 80, 40)), locked=True)


def _junction_return():
    sectors = [_rect("lock", -48, 0, 0, 128, kind="door_red"),
               _rect("vestibule", -176, 0, -48, 128),
               _rect("exit", -240, 0, -176, 128, kind="exit")]
    for i in range(12):
        x = i * 160
        sectors.append(_rect(f"hall{i}", x, 0, x + 160, 128))
        if i % 3 == 1:
            sectors.append(_rect(f"north{i}", x, 128, x + 160, 384))
            sectors.append(_rect(f"south{i}", x, -256, x + 160, 0))
    return _probe("junction_return", "Find a red key among repeated side passages and return to spawn.",
                  sectors, (ThingSpec(80, 64, 1, 180), ThingSpec(1680, 320, 13)), locked=True)


def _stairs_ledge():
    """Elevated 48-unit ledge blocks direct path (RAY_WALL) while showing key; 3-step stairs bypass it."""
    sectors = (
        _rect("spawn", 0, 0, 192, 160, height_floor=0, height_ceiling=176),
        _rect("ledge_lip", 192, 0, 256, 160, height_floor=48, height_ceiling=176, wall_texture="STONE2"),
        _rect("ledge_key", 256, 0, 448, 160, height_floor=48, height_ceiling=176, wall_texture="STONE2"),
        _rect("step1", 0, 160, 192, 224, height_floor=16, height_ceiling=176, wall_texture="BROWN1"),
        _rect("step2", 0, 224, 192, 288, height_floor=32, height_ceiling=176, wall_texture="BROWN1"),
        _rect("top_nw", 0, 288, 192, 416, height_floor=48, height_ceiling=176, wall_texture="STONE2"),
        _rect("top_mid", 192, 288, 256, 416, height_floor=48, height_ceiling=176, wall_texture="STONE2"),
        _rect("top_ne", 256, 288, 448, 416, height_floor=48, height_ceiling=176, wall_texture="STONE2"),
        _rect("east_hall", 256, 160, 448, 288, height_floor=48, height_ceiling=176, wall_texture="STONE2"),
        _rect("lock", 448, 0, 496, 160, kind="door_yellow", height_floor=48, height_ceiling=176),
        _rect("vestibule", 496, 0, 624, 160, height_floor=48, height_ceiling=176),
        _rect("exit", 624, 0, 688, 160, kind="exit", height_floor=48, height_ceiling=176),
    )
    things = (
        ThingSpec(80, 80, 1, 0),    # Player facing east toward 48u ledge & visible YellowSkull
        ThingSpec(352, 80, 39, 0),  # YellowSkull on elevated platform
    )
    return _probe(
        "stairs_ledge",
        "Climb 3-step stairs (16u each) to bypass a 48u cliff, collect YellowSkull, unlock elevated door, exit.",
        sectors,
        things,
        locked=True,
    )


def _remote_door_switch():
    """Closed remote door requires activating a wall switch in a side alcove before crossing."""
    sectors = (
        _rect("spawn", 0, 0, 256, 160),
        _rect(
            "switch_alcove",
            0,
            160,
            256,
            320,
            wall_texture="METAL1",
            door_switch=((0, 320), (256, 320), 1),
        ),
        _rect("rdoor", 256, 0, 304, 160, kind="remote_door", sector_id=1),
        _rect("after_door", 304, 0, 496, 160, wall_texture="STONE2"),
        _rect("exit", 496, 0, 560, 160, kind="exit"),
    )
    things = (ThingSpec(80, 80, 1, 0),)
    return MapDefinition(
        MapSpec("maps/stress/remote_door_switch.wad", family="multi_room", tier=2, split="stress", episode_ticks=2100),
        "polygon",
        "Activate a remote wall switch in the north alcove to open the tagged door blocking the east exit.",
        poly_sectors=sectors,
        poly_things=things,
        min_unlocked_doors=1,
    )


def _grate_combat():
    """Impassable mid-grate separates spawn from diverse monsters & exit; flank around angled north hall."""
    grate = ((256, 0), (256, 256))
    sectors = (
        _rect("west_room", 0, 0, 256, 256, wall_texture="BRICK1", blocking_edges=(grate,)),
        _rect("east_room", 256, 0, 576, 256, wall_texture="STONE2"),
        PolygonSector("flank_w", ((0, 256), (64, 416), (256, 416), (256, 256)), wall_texture="METAL1"),
        PolygonSector("flank_e", ((256, 256), (256, 416), (512, 416), (576, 256)), wall_texture="METAL1"),
        _rect("exit", 576, 0, 640, 256, kind="exit"),
    )
    things = (
        ThingSpec(80, 128, 1, 0),      # Player facing grate + visible enemies
        ThingSpec(128, 80, 82, 0),     # SuperShotgun
        ThingSpec(128, 176, 2004, 0),  # PlasmaRifle
        ThingSpec(176, 176, 2047, 0),  # CellPack
        ThingSpec(80, 208, 2019, 0),   # BlueArmor
        ThingSpec(176, 64, 2024, 0),   # BlurSphere (Powerup -> other)
        ThingSpec(384, 80, 65, 180),   # ChaingunGuy
        ThingSpec(448, 128, 3005, 180),  # Cacodemon
        ThingSpec(480, 192, 69, 180),  # HellKnight
    )
    return MapDefinition(
        MapSpec("maps/stress/grate_combat.wad", family="combat", tier=1, split="stress", episode_ticks=2100),
        "polygon",
        "Visible ChaingunGuy, Cacodemon, and HellKnight behind an impassable mid-grate with angled north flank.",
        poly_sectors=sectors,
        poly_things=things,
    )


def _nukage_hazard():
    """Damaging nukage pool (sector_special=71) with RadSuit before crossing to east exit."""
    sectors = (
        PolygonSector(
            "spawn_bank",
            ((0, 0), (0, 256), (192, 256), (192, 192), (192, 64), (192, 0)),
            wall_texture="STARTAN2",
        ),
        _rect(
            "nukage_pool",
            192,
            64,
            448,
            192,
            wall_texture="BROWN1",
            floor_texture="NUKAGE1",
            height_floor=-8,
            sector_special=71,
        ),
        _rect("dry_north_flank", 192, 192, 448, 256, wall_texture="METAL1", floor_texture="FLOOR5_1"),
        _rect("dry_south_flank", 192, 0, 448, 64, wall_texture="METAL1", floor_texture="FLOOR5_1"),
        PolygonSector(
            "east_bank",
            ((448, 0), (448, 64), (448, 192), (448, 256), (640, 256), (640, 0)),
            wall_texture="STONE2",
        ),
        _rect("exit", 640, 64, 704, 192, kind="exit"),
    )
    # Add vertices (640, 64) and (640, 192) on east_bank so exit stitches 1:1
    sectors = (
        sectors[0],
        sectors[1],
        sectors[2],
        sectors[3],
        PolygonSector(
            "east_bank",
            ((448, 0), (448, 64), (448, 192), (448, 256), (640, 256), (640, 192), (640, 64), (640, 0)),
            wall_texture="STONE2",
        ),
        sectors[5],
    )
    things = (
        ThingSpec(80, 128, 1, 0),     # Player facing nukage + east exit
        ThingSpec(144, 128, 2025, 0), # RadSuit right before nukage
        ThingSpec(544, 128, 3001, 180),
    )
    return MapDefinition(
        MapSpec("maps/stress/nukage_hazard.wad", family="multi_room", tier=2, split="stress", episode_ticks=2100),
        "polygon",
        "Damaging nukage pool (special=71) with RadSuit and narrow dry flanking catwalks.",
        poly_sectors=sectors,
        poly_things=things,
    )


def _dropoff_return():
    """Climb East 16u stairs to RedCard on +64u cliff, then take one-way 64u drop-off back to courtyard."""
    sectors = (
        PolygonSector(
            "courtyard",
            ((0, 0), (0, 256), (320, 256), (320, 192), (320, 64), (320, 0)),
            wall_texture="STONE2",
            height_ceiling=192,
        ),
        _rect("step1", 320, 0, 384, 64, height_floor=16, height_ceiling=192, wall_texture="STEP1"),
        _rect("step2", 384, 0, 448, 64, height_floor=32, height_ceiling=192, wall_texture="STEP1"),
        _rect("step3", 448, 0, 512, 64, height_floor=48, height_ceiling=192, wall_texture="STEP1"),
        PolygonSector(
            "high_ledge",
            ((320, 64), (320, 192), (512, 192), (512, 64), (448, 64)),
            height_floor=64,
            height_ceiling=192,
            wall_texture="BRICK1",
            floor_texture="FLOOR7_1",
        ),
        _rect("red_door", -48, 64, 0, 192, kind="door_red", height_ceiling=192),
        _rect("vestibule", -176, 64, -48, 192, height_ceiling=192),
        _rect("exit", -240, 64, -176, 192, kind="exit", height_ceiling=192),
    )
    # Match (0, 64), (0, 192) on courtyard left wall and (320, 64)-(512, 64) on high_ledge cleanly
    sectors = (
        PolygonSector(
            "courtyard",
            ((0, 0), (0, 64), (0, 192), (0, 256), (320, 256), (320, 192), (320, 64), (320, 0)),
            wall_texture="STONE2",
            height_ceiling=192,
        ),
        sectors[1],
        sectors[2],
        sectors[3],
        _rect("step4_top", 448, 64, 512, 192, height_floor=64, height_ceiling=192, wall_texture="BRICK1"),
        _rect("high_ledge", 320, 64, 448, 192, height_floor=64, height_ceiling=192, wall_texture="BRICK1", floor_texture="FLOOR7_1"),
        sectors[5],
        sectors[6],
        sectors[7],
    )
    things = (
        ThingSpec(160, 128, 1, 0),    # Player facing east toward high_ledge dropoff
        ThingSpec(240, 128, 2001, 0), # Shotgun on courtyard
        ThingSpec(200, 64, 2012, 0),  # Medikit
        ThingSpec(384, 128, 13, 0),   # RedCard on high_ledge (+64u cliff directly above courtyard)
        ThingSpec(432, 168, 3004, 180),
        ThingSpec(-112, 160, 9, 0),
    )
    return _probe(
        "dropoff_return",
        "Climb stairs to +64u high ledge for RedCard, then take one-way 64u drop-off back to RedDoor.",
        sectors,
        things,
        locked=True,
    )


def _open_arena_horde():
    """Wide 1280x1280 open arena (>1024u -> RAY_NONE) with dense 8-enemy Tier 4 combat horde."""
    sectors = (
        _rect(
            "arena",
            0,
            0,
            1280,
            1280,
            wall_texture="BRICK1",
            floor_texture="FLOOR7_1",
            height_ceiling=192,
        ),
        _rect("exit", 1280, 576, 1344, 704, kind="exit", height_ceiling=192),
    )
    sectors = (
        PolygonSector(
            "arena",
            ((0, 0), (0, 1280), (1280, 1280), (1280, 704), (1280, 576), (1280, 0)),
            wall_texture="BRICK1",
            floor_texture="FLOOR7_1",
            height_ceiling=192,
        ),
        sectors[1],
    )
    things = (
        ThingSpec(160, 640, 1, 0),
        ThingSpec(256, 512, 82, 0),     # SuperShotgun
        ThingSpec(256, 768, 2004, 0),   # PlasmaRifle
        ThingSpec(256, 640, 17, 0),     # CellPack
        ThingSpec(160, 448, 2018, 0),   # GreenArmor
        ThingSpec(384, 640, 2012, 0),   # Medikit
        ThingSpec(640, 320, 3004, 180),
        ThingSpec(640, 960, 9, 180),
        ThingSpec(768, 640, 65, 180),
        ThingSpec(896, 448, 3005, 180), # Cacodemon
        ThingSpec(896, 832, 69, 180),   # HellKnight
        ThingSpec(1024, 384, 68, 180),  # Arachnotron
        ThingSpec(1024, 896, 66, 180),  # Revenant
        ThingSpec(1152, 640, 3001, 180),
    )
    return MapDefinition(
        MapSpec("maps/stress/open_arena_horde.wad", family="gauntlet", tier=4, split="stress", episode_ticks=4200),
        "polygon",
        "Wide 1280x1280 open arena (>1024u rays -> RAY_NONE) with 8 diverse Tier 4 enemies.",
        poly_sectors=sectors,
        poly_things=things,
    )


def _three_lock_citadel():
    """Tier 4 three-lock citadel (Red -> Blue -> Yellow) with hub & combat."""
    sectors = (
        PolygonSector(
            "hub",
            ((256, 256), (256, 320), (256, 448), (256, 512), (320, 512), (448, 512), (512, 512), (512, 448), (512, 320), (512, 256), (448, 256), (320, 256)),
            wall_texture="STONE2",
            height_ceiling=160,
        ),
        _rect("west_wing", 0, 256, 256, 512, wall_texture="STARTAN2", height_ceiling=160),
        _rect("red_door", 512, 320, 576, 448, kind="door_red", height_ceiling=160),
        _rect("east_wing", 576, 256, 832, 512, wall_texture="METAL1", height_ceiling=160),
        _rect("blue_door", 320, 512, 448, 576, kind="door_blue", height_ceiling=160),
        _rect("north_wing", 256, 576, 512, 832, wall_texture="BRICK1", height_ceiling=160),
        _rect("yellow_door", 320, 192, 448, 256, kind="door_yellow", height_ceiling=160),
        _rect("south_sanctum", 256, 64, 512, 192, wall_texture="GRAY1", height_ceiling=160),
        _rect("exit", 320, 0, 448, 64, kind="exit", height_ceiling=160),
    )
    # Fix shared door edges strictly on west_wing/east_wing/north_wing/south_sanctum:
    sectors = (
        sectors[0],
        PolygonSector("west_wing", ((0, 256), (0, 512), (256, 512), (256, 448), (256, 320), (256, 256)), wall_texture="STARTAN2", height_ceiling=160),
        sectors[2],
        PolygonSector("east_wing", ((576, 256), (576, 320), (576, 448), (576, 512), (832, 512), (832, 256)), wall_texture="METAL1", height_ceiling=160),
        sectors[4],
        PolygonSector("north_wing", ((256, 576), (256, 832), (512, 832), (512, 576), (448, 576), (320, 576)), wall_texture="BRICK1", height_ceiling=160),
        sectors[6],
        PolygonSector("south_sanctum", ((256, 64), (256, 192), (320, 192), (448, 192), (512, 192), (512, 64), (448, 64), (320, 64)), wall_texture="GRAY1", height_ceiling=160),
        sectors[8],
    )
    things = (
        ThingSpec(384, 384, 1, 180),     # Facing West toward RedCard wing
        ThingSpec(384, 448, 82, 0),      # SuperShotgun
        ThingSpec(448, 448, 2049, 0),    # ShellBox
        ThingSpec(384, 320, 2019, 0),    # BlueArmor
        ThingSpec(320, 320, 2012, 0),    # Medikit
        ThingSpec(96, 384, 13, 0),       # RedCard in West
        ThingSpec(736, 384, 5, 0),       # BlueCard in East
        ThingSpec(640, 320, 2004, 0),    # PlasmaRifle
        ThingSpec(704, 320, 17, 0),      # CellPack
        ThingSpec(384, 736, 6, 0),       # YellowCard in North
        ThingSpec(320, 704, 2012, 0),    # Medikit
        ThingSpec(160, 320, 9, 0),
        ThingSpec(672, 448, 3002, 180),
        ThingSpec(448, 672, 66, 270),    # Revenant
        ThingSpec(448, 128, 3003, 90),   # BaronOfHell
    )
    return MapDefinition(
        MapSpec("maps/stress/three_lock_citadel.wad", family="gauntlet", tier=4, split="stress", episode_ticks=4200),
        "polygon",
        "Tier 4 three-lock citadel requiring Red -> Blue -> Yellow progression and Baron/Revenant combat.",
        poly_sectors=sectors,
        poly_things=things,
        min_locked_doors=3,
    )


STRESS_DEFINITIONS = (
    *_exit_probes(),
    _key_window(False),
    _key_window(True),
    _junction_return(),
    _stairs_ledge(),
    _remote_door_switch(),
    _grate_combat(),
    _nukage_hazard(),
    _dropoff_return(),
    _open_arena_horde(),
    _three_lock_citadel(),
)
