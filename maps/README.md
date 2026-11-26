# Maps

`two_rooms.wad` is the official starter smoke-test map (one door, one monster, an exit). It is **not** an evaluation map.

Our Role 3 map library (`training/maps.py`, generated deterministically by `training/wad_gen/`) provides strictly separated **Train**, **Dev**, and **Final (Untouched)** suites across all four evaluation tiers (`docs/EVALUATION.md`):

| Split | Directory | Count | Tiers Covered | Construction |
|---|---|---:|---|---|
| **Train** | `maps/train/` | 20 | T1 Combat (4), T2 Door-Exit & Multi-Room (6), T3 Key & Return (6), T4 Full Gauntlet (4) | Parameterized UDMF grid (+ on-the-fly procedural variations via `training.wad_gen.procedural`) |
| **Dev** | `maps/dev/` | 10 | 2 held-out simple door-exit (MVP Gate), T1 Combat (2), T2 Multi-Room (2), T3 Key & Return incl. mirrored corridors (2), T4 Gauntlet 4,200 ticks (2) | Held-out distinct UDMF grid topologies |
| **Final** | `maps/final/` | 8 | 4 unseen grid layouts (T1–T4) + **4 non-grid polygonal/diagonal-wall layouts (T1–T4)** | Unseen UDMF grid + independent polygonal/angled-wall UDMF builder |

## Regenerate & Verify

Every WAD is statically verified for reachability (no soft-locks, key reachable before lock, exit reachable) and engine-verified in `doom_env` (`freedoom2.wad`, skill 3, `< 2,048` sectors, exact `wad` geometry, verified `RAY_DOOR` / `RAY_LOCKED` linedef specials):

```bash
uv run python -m training.maps --generate --verify
uv run python -m training.maps --list --split dev
```

To dry-run the submitted agent on any registered map:

```bash
uv run python validate.py --map maps/dev/dev_simple_dogleg_door.wad:MAP01
```
