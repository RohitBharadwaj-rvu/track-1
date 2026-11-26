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


STRESS_DEFINITIONS = (*_exit_probes(), _key_window(False), _key_window(True), _junction_return())
