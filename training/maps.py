"""Registered Train, Dev, and Final map suites + Curriculum API (Role 3).

Frozen contract consumed by:
  - Role 2 (`training/rollout.py`): `get_maps(split="train")`, `TRAIN_MAPS`, `get_curriculum_maps(...)`
  - Role 4 (`training/eval.py`): `get_maps(split="dev" | "final")`, `DEV_MAPS`, `FINAL_MAPS`, `ALL_MAPS`

CLI usage:
    uv run python -m training.maps --list
    uv run python -m training.maps --generate --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from training.interfaces import MapSpec
from training.wad_gen.families import (
    ALL_DEFINITIONS,
    DEV_DEFINITIONS,
    FINAL_DEFINITIONS,
    TRAIN_DEFINITIONS,
    MapDefinition,
    materialize_definition,
)
from training.wad_gen.final_ood import FINAL_OOD_DEFINITIONS, FINAL_OOD_MAPS
from training.wad_gen.procedural import generate_procedural_wad
from training.wad_gen.stress import STRESS_DEFINITIONS
from training.wad_gen.verify import WadVerificationResult, verify_wad_in_engine

REGISTERED_DEFINITIONS = (*ALL_DEFINITIONS, *STRESS_DEFINITIONS, *FINAL_OOD_DEFINITIONS)
STRESS_MAPS = tuple(d.spec for d in STRESS_DEFINITIONS)

ROOT = Path(__file__).resolve().parent.parent

TRAIN_MAPS: tuple[MapSpec, ...] = tuple(d.spec for d in TRAIN_DEFINITIONS)
DEV_MAPS: tuple[MapSpec, ...] = tuple(d.spec for d in DEV_DEFINITIONS)
FINAL_MAPS: tuple[MapSpec, ...] = tuple(d.spec for d in FINAL_DEFINITIONS)
ALL_MAPS: tuple[MapSpec, ...] = (*TRAIN_MAPS, *DEV_MAPS, *FINAL_MAPS)

# Stage 1 (Connect) exit-gate pair: 1 verified door/exit map + 1 verified key/lock map
CONNECT_MAPS: tuple[MapSpec, ...] = tuple(
    d.spec for d in TRAIN_DEFINITIONS if "connect" in d.tags
)

# Stage 2 (Functional MVP Gate) held-out Dev subsets (REVISED_PLAN.md §2)
MVP_GATE_SIMPLE_MAPS: tuple[MapSpec, ...] = tuple(
    d.spec for d in DEV_DEFINITIONS if "mvp_gate_simple" in d.tags
)
MVP_GATE_COMBAT_MAPS: tuple[MapSpec, ...] = tuple(
    d.spec for d in DEV_DEFINITIONS if "mvp_gate_combat" in d.tags
)

# Distinct-construction Final family (non-grid polygonal/diagonal-wall maps)
FINAL_POLYGON_MAPS: tuple[MapSpec, ...] = tuple(
    d.spec for d in FINAL_DEFINITIONS if d.construction == "polygon"
)

_DEFINITIONS_BY_WAD: dict[str, MapDefinition] = {d.spec.wad: d for d in REGISTERED_DEFINITIONS}

CURRICULUM_STAGES: tuple[str, ...] = (
    "connect",
    "mvp",
    "progression",
    "full",
    "branch_exploration",
    "return_route",
    "post_pickup_combat",
)


def get_maps(
    split: str = "dev",
    *,
    tiers: Sequence[int] | None = None,
    families: Sequence[str] | None = None,
    stage: str | None = None,
) -> list[MapSpec]:
    """Return registered `MapSpec` entries for `split` (`train`, `dev`, `final`, `stress`, or legacy `all`).

    Optional filters:
      - `tiers`: keep only specs with `spec.tier in tiers`.
      - `families`: keep only specs with `spec.family in families`.
      - `stage`: when `split == "train"`, select a curriculum stage via `get_curriculum_maps`.
    """
    if stage is not None:
        if split != "train":
            raise ValueError("Curriculum `stage` filtering is only valid for split='train'")
        specs = get_curriculum_maps(stage)
    else:
        by_split = {
            "train": TRAIN_MAPS,
            "dev": DEV_MAPS,
            "final": FINAL_MAPS,
            "final_ood": FINAL_OOD_MAPS,
            "stress": STRESS_MAPS,
            "all": ALL_MAPS,
        }
        if split not in by_split:
            raise ValueError(f"Unknown split {split!r}; expected one of {tuple(by_split)}")
        specs = list(by_split[split])

    if tiers is not None:
        tier_set = {int(t) for t in tiers}
        specs = [s for s in specs if s.tier in tier_set]
    if families is not None:
        fam_set = set(families)
        specs = [s for s in specs if s.family in fam_set]
    return specs


def get_curriculum_maps(stage: str = "mvp") -> list[MapSpec]:
    """Return the training `MapSpec` subset for a named curriculum or failure-focus stage.

    Always retains mastered shorter tasks while introducing harder mechanics (REVISED_PLAN.md §3-§4):
      - `connect`: 1 door/exit + 1 short key-return (Stage 1 Connect)
      - `mvp`: combat + simple door-exit + multi-room + short key-return (Run 1 mixture)
      - `progression`: retains MVP + adds longer/mirrored key-return and return-ambush gauntlet
      - `full`: entire training suite including 2-lock 4,200-tick gauntlets
      - `branch_exploration`: retains MVP + over-samples multi-room branches, loops, dead ends
      - `return_route`: retains MVP + over-samples short/medium/long/mirrored key-return
      - `post_pickup_combat`: retains MVP + over-samples combat & return-ambush encounters
    """
    if stage not in CURRICULUM_STAGES:
        raise ValueError(f"Unknown curriculum stage {stage!r}; expected one of {CURRICULUM_STAGES}")
    if stage == "connect":
        return list(CONNECT_MAPS)
    if stage == "mvp":
        return [d.spec for d in TRAIN_DEFINITIONS if "mvp" in d.tags]
    if stage == "progression":
        return [d.spec for d in TRAIN_DEFINITIONS if ("mvp" in d.tags or "progression" in d.tags)]
    if stage == "full":
        return list(TRAIN_MAPS)
    # Failure-driven focus presets: keep full suite or MVP + focus tags (use weights to emphasize)
    return [d.spec for d in TRAIN_DEFINITIONS if ("mvp" in d.tags or stage in d.tags)]


def get_curriculum_weights(stage: str = "mvp") -> list[float]:
    """Return normalized sampling weights aligned 1:1 with `get_curriculum_maps(stage)`."""
    specs = get_curriculum_maps(stage)
    raw: list[float] = []
    for spec in specs:
        defn = _DEFINITIONS_BY_WAD[spec.wad]
        w = 1.0
        if stage == "mvp":
            # Give extra early signal to door traversal and short key-return alongside combat
            if spec.family in ("door_exit", "key_return"):
                w = 1.5
        elif stage == "progression":
            if spec.tier >= 3:
                w = 2.0
        elif stage == "full":
            w = {1: 1.0, 2: 1.2, 3: 2.0, 4: 2.2}.get(spec.tier, 1.0)
        elif stage in ("branch_exploration", "return_route", "post_pickup_combat"):
            # Failure-driven focus: 3x weight on targeted failure transition while retaining mastered tasks
            w = 3.0 if stage in defn.tags else 1.0
        raw.append(w)
    total = sum(raw)
    return [w / total for w in raw]


def get_reserve_seeds(spec: MapSpec, count: int = 3) -> tuple[int, ...]:
    """Return non-overlapping reserve evaluation seeds for single-map gain checks (REVISED_PLAN §5)."""
    start = (max(spec.seeds) + 1) if spec.seeds else 0
    return tuple(range(start, start + int(count)))


def get_definition(wad: str) -> MapDefinition:
    """Return the full `MapDefinition` metadata (description, construction method, tags) for a WAD."""
    return _DEFINITIONS_BY_WAD[wad]


def generate_training_variations(
    out_dir: str | Path,
    *,
    count_per_family: int = 2,
    base_seed: int = 1000,
    families: Sequence[str] = ("door_exit", "combat", "multi_room", "key_return", "gauntlet"),
) -> list[MapSpec]:
    """Generate fresh procedural training variations on demand (not committed to git)."""
    specs: list[MapSpec] = []
    seed = int(base_seed)
    for fam in families:
        for _ in range(int(count_per_family)):
            specs.append(generate_procedural_wad(fam, seed=seed, out_dir=out_dir, split="train"))
            seed += 1
    return specs


def verify_all_maps(
    repo_root: Path = ROOT,
    *,
    splits: Sequence[str] = ("train", "dev", "final"),
) -> list[WadVerificationResult]:
    """Verify every registered WAD in real `doom_env` (freedoom2.wad, skill 3)."""
    results: list[WadVerificationResult] = []
    for defn in REGISTERED_DEFINITIONS:
        if defn.spec.split not in splits:
            continue
        wad_path = repo_root / defn.spec.wad
        res = verify_wad_in_engine(
            wad_path,
            map_name=defn.spec.map_name,
            episode_ticks=defn.spec.episode_ticks,
            min_unlocked_doors=defn.min_unlocked_doors,
            min_locked_doors=defn.min_locked_doors,
        )
        results.append(res)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Role 3 map registry, generator, and verifier.")
    parser.add_argument("--generate", action="store_true", help="Regenerate all WADs for the selected split.")
    parser.add_argument("--verify", action="store_true", help="Verify all registered WADs in doom_env.")
    parser.add_argument("--list", action="store_true", help="List registered maps by split and tier.")
    parser.add_argument(
        "--split",
        choices=("train", "dev", "final", "final_ood", "stress", "all"),
        default="all",
        help="Split to generate, list, or verify. The legacy all suite contains Train, Dev, and Final.",
    )
    args = parser.parse_args(argv)

    if args.generate:
        paths = [materialize_definition(get_definition(spec.wad), ROOT) for spec in get_maps(args.split)]
        print(f"Generated {len(paths)} verified WADs in {ROOT / 'maps'}.")

    if args.list or (not args.generate and not args.verify):
        for spec in get_maps(split=args.split):
            defn = _DEFINITIONS_BY_WAD[spec.wad]
            print(
                f"[{spec.split:5s}] T{spec.tier} {spec.family:10s} ({defn.construction:7s}) "
                f"{spec.episode_ticks:4d}t  {spec.wad}"
            )

    if args.verify:
        splits = ("train", "dev", "final") if args.split == "all" else (args.split,)
        results = verify_all_maps(ROOT, splits=splits)
        failed = [r for r in results if not r.ok]
        for r in results:
            status = "OK  " if r.ok else "FAIL"
            print(
                f"{status} {Path(r.wad).name:36s} sectors={r.num_sectors:3d} "
                f"doors={r.num_unlocked_doors} locked={r.num_locked_doors} geom={r.geometry_source}"
                + (f" ERRORS={r.errors}" if r.errors else "")
            )
        print(f"\nVerified {len(results) - len(failed)}/{len(results)} WADs cleanly.")
        return 1 if failed else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
