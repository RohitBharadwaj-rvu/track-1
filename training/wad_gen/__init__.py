"""UDMF WAD generators, map families, procedural variation, and verification (Role 3)."""

from __future__ import annotations

from training.wad_gen.udmf import (
    DOOR_CHARS,
    KEY_CHARS,
    LOCK_CODES,
    MONSTER_CHARS,
    PolygonHole,
    PolygonSector,
    ThingSpec,
    build_grid_textmap,
    build_grid_wad,
    build_polygon_textmap,
    build_polygon_wad,
    layout_hash,
)
from training.wad_gen.verify import (
    GridVerificationResult,
    WadVerificationResult,
    verify_grid_solvability,
    verify_polygon_solvability,
    verify_wad_in_engine,
)

__all__ = [
    "DOOR_CHARS",
    "KEY_CHARS",
    "LOCK_CODES",
    "MONSTER_CHARS",
    "PolygonHole",
    "PolygonSector",
    "ThingSpec",
    "build_grid_textmap",
    "build_grid_wad",
    "build_polygon_textmap",
    "build_polygon_wad",
    "layout_hash",
    "GridVerificationResult",
    "WadVerificationResult",
    "verify_grid_solvability",
    "verify_polygon_solvability",
    "verify_wad_in_engine",
]
