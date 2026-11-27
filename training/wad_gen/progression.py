"""Branch dependencies for held-out-style progression practice in the training corpus."""

from dataclasses import replace
from functools import partial

from training.wad_gen.udmf import PolygonSector, ThingSpec, transform_polygon_map
from training.wad_gen.verify import verify_polygon_solvability


# Each tuple is one wing, listed from the shared hub outwards. A and B are
# independently shuffled key colours. Switch and gate numbers are sector tags.
WINGS = {
    "two_key_join": (
        ("key_a",), ("key_b",), ("lock_a", "lock_b", "switch_1"), ("gate_1", "exit"),
    ),
    "remote_key_relay": (
        ("switch_1",), ("gate_1", "key_a"), ("lock_a", "switch_2"), ("gate_2", "exit"),
    ),
    "revisit_wing": (
        ("key_a", "gate_1", "key_b"), ("lock_a", "switch_1"),
        ("lock_b", "switch_2"), ("gate_2", "exit"),
    ),
}


def _point(arm, distance, offset):
    dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[arm]
    return (dx * distance - dy * offset, dy * distance + dx * offset)


def build_progression(archetype, rng, seed):
    half = rng.randrange(72, 113, 8)
    sectors = [PolygonSector("hub", ((-half, -half), (half, -half), (half, half), (-half, half)))]
    things = [ThingSpec(0, 0, 1, rng.randrange(360))]
    colors = rng.sample([("red", 13, 38), ("blue", 5, 40), ("yellow", 6, 39)], 2)
    key_types = {letter: (color, rng.choice((card, skull)))
                 for letter, (color, card, skull) in zip("ab", colors)}
    arms = rng.sample(range(4), 4)
    for arm, stages in zip(arms, WINGS[archetype]):
        point = partial(_point, arm)
        start, offset = half, 0
        for index, stage in enumerate(stages):
            # A bent approach separates each interaction from the next. Leave a
            # regular sector between doors so each manual lock has its own line.
            end = start + rng.randrange(120, 241, 8)
            bend = offset + rng.randrange(-72, 73, 8)
            sectors.append(PolygonSector(f"approach_{arm}_{index}", (
                point(start, offset-half), point(start, offset+half),
                point(end, bend+half), point(end, bend-half))))
            start, offset = end, bend
            kind = "room"
            sector_id = 0
            switch = None
            is_gate = stage.startswith(("lock_", "gate_"))
            end = start + (48 if is_gate else rng.randrange(144, 225, 8))
            far = (point(end, offset-half), point(end, offset+half))
            if stage.startswith("lock_"):
                kind = "door_" + key_types[stage[-1]][0]
            elif stage.startswith("key_"):
                things.append(ThingSpec(*point((start + end) / 2, offset), key_types[stage[-1]][1]))
            elif stage.startswith("gate_"):
                kind, sector_id = "remote_door", int(stage[-1])
            elif stage.startswith("switch_"):
                switch = (*far, int(stage[-1]))
            elif stage == "exit":
                kind = "exit"
            sectors.append(PolygonSector(f"{stage}_{arm}", (
                point(start, offset-half), point(start, offset+half), far[1], far[0]),
                kind=kind, sector_id=sector_id, door_switch=switch))
            start = end

    sectors, things = transform_polygon_map(sectors, things, flip_x=bool(seed % 2),
                                            angle_deg=rng.uniform(4, 84))
    # Reject a layout if even one objective can be skipped. The caller also
    # verifies the intact map, including exact edge stitching, before writing.
    for i, thing in enumerate(things):
        if thing.doomednum != 1:
            _require_blocked(sectors, things[:i] + things[i+1:])
    for i, sector in enumerate(sectors):
        if sector.door_switch:
            disabled = replace(sector, door_switch=None)
            _require_blocked(sectors[:i] + (disabled,) + sectors[i+1:], things)
    return sectors, things, 0


def _require_blocked(sectors, things):
    ok, errors = verify_polygon_solvability(sectors, things)
    if ok or "Exit polygon sector is NOT reachable from player spawn" not in errors:
        raise ValueError(f"Required progression objective could be bypassed: {errors}")
