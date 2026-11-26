"""Unit & integration tests for continuous rotation, extreme archetypes, and 5k-style corpus generation."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from training.generate_corpus import CorpusPoolSampler, generate_corpus
from training.pbrs import compute_pbrs_geodesic_summary
from training.train import TrainConfig, run_training
from training.wad_gen.extreme import EXTREME_ARCHETYPES, generate_extreme_wad
from training.wad_gen.udmf import PolygonSector, ThingSpec, build_polygon_textmap, transform_polygon_map
from training.wad_gen.verify import verify_polygon_solvability, verify_wad_in_engine


def test_continuous_polygon_rotation_preserves_exact_stitching_and_clean_udmf():
    s0 = PolygonSector("a", ((0.0, 0.0), (0.0, 128.0), (160.0, 128.0), (160.0, 0.0)))
    s1 = PolygonSector("b", ((160.0, 0.0), (160.0, 128.0), (320.0, 128.0), (320.0, 0.0)), kind="exit")
    things = (ThingSpec(64.0, 64.0, 1, 0),)

    for angle in (17.3, 63.7, 141.2, 289.5):
        rs, rt = transform_polygon_map((s0, s1), things, rot_k=1, flip_x=True, angle_deg=angle)
        ok, errs = verify_polygon_solvability(rs, rt, check_t_junctions=True)
        assert ok, f"Rotation {angle} deg failed check_t_junctions: {errs}"
        textmap = build_polygon_textmap(rs, rt)
        assert "vertex" in textmap and "linedef" in textmap


def test_all_extreme_archetypes_valid_pbrs_non_axis_and_engine_runnable(tmp_path):
    for idx, arch in enumerate(EXTREME_ARCHETYPES):
        spec = generate_extreme_wad(arch, seed=100 + idx, out_dir=tmp_path)
        pbrs_sum = compute_pbrs_geodesic_summary(spec.wad, spec.map_name)
        assert pbrs_sum["pbrs_valid"] is True, f"{arch} failed PBRS reachability"
        assert pbrs_sum["geodesic_horizon_u"] > 500.0
        eng = verify_wad_in_engine(spec.wad, map_name=spec.map_name, episode_ticks=spec.episode_ticks, step_count=2)
        assert eng.ok, f"{arch} failed engine verification: {eng.errors}"


def test_generate_corpus_end_to_end_sampler_and_trainer_streaming(tmp_path):
    corpus_dir = tmp_path / "mini_corpus"
    audit = generate_corpus(
        count=24,
        out_dir=corpus_dir,
        seed=42,
        workers=1,
        include_freedoom=True,
        engine_spotcheck=6,
    )
    assert audit["count_generated"] == 24
    assert audit["count_disjoint_freedoom"] >= 8
    integ = audit["integrity"]
    assert integ["held_out_d4_collisions"] == 0
    assert integ["pbrs_valid_rate"] == 1.0
    assert integ["unique_d4_ratio"] == 1.0
    assert integ["engine_spotcheck_passed"] == 6

    diff = audit["difficulty_summary"]
    assert diff["very_high_plus_count"] >= 8
    var = audit["objective_variety"]
    assert var["frac_non_axis"] >= 0.45
    assert var["pooled_wall_angle_entropy_bits"] >= 2.5

    # Verify CorpusPoolSampler rotation + elite retention
    manifest_path = corpus_dir / "manifest.jsonl"
    assert manifest_path.is_file()
    sampler = CorpusPoolSampler(manifest_path, include_freedoom=True)
    rng = np.random.default_rng(7)
    pool1 = sampler.sample_pool(pool_size=12, high_diff_frac=0.5, rng=rng)
    assert len(pool1) == 12
    elite_target = pool1[2].wad
    pool2 = sampler.sample_pool(pool_size=12, high_diff_frac=0.5, rng=rng, scores_by_wad={elite_target: 9.9})
    assert elite_target in [s.wad for s in pool2]

    # Verify trainer streams smoothly from corpus manifest across a pool refresh
    run_dir = tmp_path / "train_stream"
    res = run_training(
        TrainConfig(
            corpus_manifest=str(manifest_path),
            corpus_pool_size=10,
            corpus_high_diff_frac=0.5,
            procedural_refresh_updates=1,
            num_envs=2,
            vec_mode="sync",
            rollout_steps=8,
            chunk_len=4,
            num_minibatches=2,
            update_epochs=1,
            total_steps=32,
            max_cached_envs=12,
            seed=0,
            device="cpu",
            checkpoint_dir=str(run_dir),
            export_path=str(tmp_path / "stream_policy.pt"),
            save_interval=0,
        )
    )
    assert res["optimizer_updates"] == 2
    assert res["total_decisions"] == 32
