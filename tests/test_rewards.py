"""TDD tests for Role 2 reward shaping (`training/rewards.py`)."""

from __future__ import annotations

import numpy as np
import pytest

from training.rewards import RewardConfig, RewardShaper


def _priv(
    *,
    tic: int = 0,
    episode_ticks: int = 1000,
    x: float = 0.0,
    y: float = 0.0,
    kills: int = 0,
    damage_taken: float = 0.0,
    keys: list[int] | None = None,
    sector_ceiling: list[float] | None = None,
    door_lock: list[int] | None = None,
    exit_reached: bool = False,
    dead: bool = False,
    timeout: bool = False,
) -> dict:
    return {
        "tic": tic,
        "episode_ticks": episode_ticks,
        "x": x,
        "y": y,
        "z": 0.0,
        "angle": 0.0,
        "health": 0.0 if dead else 100.0,
        "kills": kills,
        "items": 0,
        "damage_taken": damage_taken,
        "keys": np.array(keys if keys is not None else [0, 0, 0, 0, 0, 0], dtype=np.float32),
        "dead": dead,
        "timeout": timeout,
        "exit_reached": exit_reached,
        "finished": exit_reached or dead or timeout,
        "sector_floor": np.zeros(3, dtype=np.float64),
        "sector_ceiling": np.array(sector_ceiling if sector_ceiling is not None else [0.0, 0.0, 128.0]),
        "door_lock": np.array(door_lock if door_lock is not None else [0, 1, -1]),
    }


def test_progression_and_exit_rewards_fire_once_with_efficiency():
    cfg = RewardConfig(
        exit_reward=10.0,
        efficiency_bonus=2.0,
        key_pickup_reward=3.0,
        locked_door_unlock_reward=4.0,
        door_open_reward=1.5,
        new_cell_reward=0.15,
        step_penalty=-0.002,
    )
    shaper = RewardShaper(cfg)
    shaper.reset(_priv(tic=0, x=0.0, y=0.0))

    # Step 1: pick up red key + open unlocked door (sector 0), stay in same cell
    r1, ev1, c1 = shaper.step(_priv(tic=4, keys=[0, 0, 1, 0, 0, 0], sector_ceiling=[64.0, 0.0, 128.0]))
    assert ev1.keys_picked_up == 1
    assert ev1.doors_opened == 1
    assert c1["key"] == pytest.approx(3.0)
    assert c1["door"] == pytest.approx(1.5)
    assert c1["explore"] == pytest.approx(0.0)
    assert c1["time"] == pytest.approx(-0.002)
    assert r1 == pytest.approx(sum(c1.values()))

    # Step 2: unlock locked door (sector 1); sector 0 oscillates slightly and must NOT double-count
    r2, ev2, c2 = shaper.step(_priv(tic=8, keys=[0, 0, 1, 0, 0, 0], sector_ceiling=[48.0, 64.0, 128.0]))
    assert ev2.doors_opened == 0
    assert ev2.locked_doors_unlocked == 1
    assert c2["door"] == pytest.approx(0.0)
    assert c2["unlock"] == pytest.approx(4.0)
    assert r2 == pytest.approx(sum(c2.values()))

    # Step 3: reach exit at tic=250 out of 1000 -> efficiency fraction = 0.75
    r3, ev3, c3 = shaper.step(
        _priv(
            tic=250,
            episode_ticks=1000,
            keys=[0, 0, 1, 0, 0, 0],
            sector_ceiling=[64.0, 64.0, 128.0],
            exit_reached=True,
        )
    )
    assert ev3.exit_reached is True
    assert c3["exit"] == pytest.approx(10.0)
    assert c3["efficiency"] == pytest.approx(2.0 * 0.75)
    assert r3 == pytest.approx(sum(c3.values()))


def test_exploration_cap_combat_and_reward_hierarchy():
    cfg = RewardConfig()
    shaper = RewardShaper(cfg)

    # 1. Completed run (minimal exploration, reaches exit halfway through episode)
    shaper.reset(_priv(tic=0, x=0.0))
    completed_return = 0.0
    for step_idx in range(1, 11):
        r, _, _ = shaper.step(
            _priv(
                tic=step_idx * 40,
                episode_ticks=1000,
                x=float(step_idx * 64),
                exit_reached=(step_idx == 10),
            )
        )
        completed_return += r

    # 2. Stalled max-exploration + moderate combat run (visits 60 cells > cap 40, 2 kills, times out)
    shaper.reset(_priv(tic=0, x=0.0))
    explore_sum = 0.0
    stalled_return = 0.0
    for step_idx in range(1, 61):
        r, _, comp = shaper.step(
            _priv(
                tic=step_idx * 16,
                episode_ticks=1000,
                x=float(step_idx * 64),
                kills=2 if step_idx >= 20 else 0,
                timeout=(step_idx == 60),
            )
        )
        explore_sum += comp["explore"]
        stalled_return += r

    assert shaper.unique_cells_visited == 61
    assert explore_sum == pytest.approx(cfg.max_exploration_cells * cfg.new_cell_reward)

    # 3. Early death run with damage
    shaper.reset(_priv(tic=0, x=0.0))
    death_r, ev_death, comp_death = shaper.step(
        _priv(tic=40, x=0.0, damage_taken=100.0, dead=True)
    )
    assert ev_death.dead is True
    assert comp_death["damage"] == pytest.approx(100.0 * cfg.damage_penalty)
    assert comp_death["death"] == pytest.approx(cfg.death_penalty)

    assert completed_return > stalled_return > death_r


def test_reward_shaper_on_real_key_door_exit_doom_env():
    from doom_env import make_env
    from doom_env import spaces as S
    from fixtures import fixture_wad

    wad = fixture_wad("############\n#P.r..R...E#\n############")
    env = make_env(wad, "MAP01", episode_ticks=1000, render_buffers=False)
    shaper = RewardShaper()
    fwd = np.zeros(len(S.BUTTONS), dtype=np.int8)
    fwd[S.BUTTONS.index("MOVE_FORWARD")] = 1
    use = np.zeros(len(S.BUTTONS), dtype=np.int8)
    use[S.BUTTONS.index("USE")] = 1

    try:
        env.reset(seed=0)
        shaper.reset(env.privileged_state())
        totals = {"key": 0.0, "unlock": 0.0, "exit": 0.0, "explore": 0.0}
        last_x = None
        for i in range(240):
            st = env.privileged_state()
            stuck = last_x is not None and abs(st["x"] - last_x) < 0.5
            last_x = st["x"]
            action = use if (stuck and i % 6 == 0) else fwd
            _, _, term, trunc, _ = env.step(action)
            _, _, comp = shaper.step(env.privileged_state())
            for k in totals:
                totals[k] += comp[k]
            if term or trunc:
                break
        assert totals["key"] == pytest.approx(shaper.config.key_pickup_reward)
        assert totals["unlock"] == pytest.approx(shaper.config.locked_door_unlock_reward)
        assert totals["exit"] == pytest.approx(shaper.config.exit_reward)
        assert totals["explore"] > 0.0
    finally:
        env.close()
