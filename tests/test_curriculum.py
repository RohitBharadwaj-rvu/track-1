"""TDD tests for staged curriculum and tier/family sampling (`training/curriculum.py`)."""

from __future__ import annotations

import pytest

from training.curriculum import CurriculumConfig, CurriculumScheduler
from training.interfaces import EpisodeMetrics, MapSpec


def _sample_specs() -> list[MapSpec]:
    return [
        MapSpec(wad="t1_combat.wad", family="combat", tier=1, split="train"),
        MapSpec(wad="t2_explore.wad", family="multi_room", tier=2, split="train"),
        MapSpec(wad="t3_key.wad", family="key_return", tier=3, split="train"),
        MapSpec(wad="t4_gauntlet.wad", family="gauntlet", tier=4, split="train"),
    ]


def _ep(tier: int, *, exit_ok: bool = False, unlocked: int = 0) -> EpisodeMetrics:
    return EpisodeMetrics(
        wad=f"m_t{tier}.wad",
        map_name="MAP01",
        family="key_return" if tier == 3 else "combat",
        tier=tier,
        seed=0,
        outcome="exit" if exit_ok else "timeout",
        locked_doors_unlocked=unlocked,
    )


def test_curriculum_warmup_weights_and_stage_promotion():
    specs = _sample_specs()
    sched = CurriculumScheduler(CurriculumConfig(auto_curriculum=True, window_size=16))
    assert sched.phase == 0
    w0 = sched.compute_map_weights(specs)
    assert sum(w0) == pytest.approx(1.0)
    # Warmup prioritizes Tier 1 & Tier 2 over Tier 3 & Tier 4 while keeping all > 0
    assert all(w > 0.0 for w in w0)
    assert w0[0] > w0[2] > w0[3]

    # Feed 16 Tier 1/2 episodes with 50% exit rate (>= 0.40 threshold) -> promotes to phase 1
    eps_p1 = [_ep(1 if i % 2 == 0 else 2, exit_ok=(i % 2 == 0)) for i in range(16)]
    promoted_1 = sched.observe_episodes(eps_p1)
    assert promoted_1 is True
    assert sched.phase == 1
    w1 = sched.compute_map_weights(specs)
    assert sum(w1) == pytest.approx(1.0)
    # Phase 1 shifts highest weight to Tier 3 while keeping Tier 1-2 replay slice
    assert w1[2] > w1[0] > 0.0

    # Feed 16 Tier 3 episodes with 37.5% unlock rate (>= 0.30 threshold) -> promotes to phase 2
    eps_p2 = [_ep(3, unlocked=(1 if i < 6 else 0)) for i in range(16)]
    promoted_2 = sched.observe_episodes(eps_p2)
    assert promoted_2 is True
    assert sched.phase == 2
    w2 = sched.compute_map_weights(specs)
    assert sum(w2) == pytest.approx(1.0)
    assert w2[3] > w0[3]


def test_explicit_tier_and_family_weights_disable_auto_shift():
    specs = _sample_specs()
    sched = CurriculumScheduler(
        CurriculumConfig(
            auto_curriculum=False,
            tier_weights={1: 0.1, 2: 0.2, 3: 0.7},
            family_weights={"gauntlet": 2.0},
        )
    )
    w = sched.compute_map_weights(specs)
    assert sum(w) == pytest.approx(1.0)
    assert w[2] > w[1] > w[0]
    assert sched.observe_episodes([_ep(1, exit_ok=True) for _ in range(20)]) is False
    assert sched.phase == 0


def test_adaptive_weights_upweight_unmastered_maps_and_tier3_require_exit():
    specs = [
        MapSpec(wad="t3_easy.wad", family="key_return", tier=3, split="train"),
        MapSpec(wad="t3_hard.wad", family="key_return", tier=3, split="train"),
    ]
    sched = CurriculumScheduler(
        CurriculumConfig(
            auto_curriculum=True,
            window_size=16,
            min_episodes_to_promote=8,
            tier12_exit_promote_threshold=0.5,
            tier3_progress_promote_threshold=0.5,
            tier3_require_exit=True,
            adaptive_weights=True,
        )
    )
    sched.phase = 1
    # Unlocking without exiting must NOT promote when tier3_require_exit=True
    unlock_only = [
        EpisodeMetrics(wad="t3_easy.wad", map_name="MAP01", family="key_return", tier=3, seed=0, outcome="timeout", locked_doors_unlocked=1)
        for _ in range(8)
    ]
    assert sched.observe_episodes(unlock_only) is False

    # Easy map exits 16/16 (flushes earlier 8 timeouts), hard map fails 0/8 -> adaptive weights strongly favor t3_hard.wad
    mixed = [
        EpisodeMetrics(wad="t3_easy.wad", map_name="MAP01", family="key_return", tier=3, seed=0, outcome="exit")
        for _ in range(16)
    ] + [
        EpisodeMetrics(wad="t3_hard.wad", map_name="MAP01", family="key_return", tier=3, seed=0, outcome="timeout")
        for _ in range(8)
    ]
    assert sched.observe_episodes(mixed) is True  # 16/24 >= 0.50 promotes to phase 2
    w = sched.compute_map_weights(specs)
    assert w[1] > w[0] * 3.0 and w[0] > 0.0
