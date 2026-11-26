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


STRESS_DEFINITIONS = (
    *_exit_probes(),
    _key_window(False),
    _key_window(True),
    _junction_return(),
    _stairs_ledge(),
    _remote_door_switch(),
    _grate_combat(),
)
