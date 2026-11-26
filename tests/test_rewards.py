"""TDD tests for upgraded Role 2 reward shaping (`training/rewards.py`)."""

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
    health: float | None = None,
    armor: float = 0.0,
    ammo_total: float = 50.0,
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
        "health": (0.0 if dead else 100.0) if health is None else health,
        "armor": armor,
        "ammo_total": ammo_total,
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


def _obs(*, bullets: float = 50.0, enemy_visible: bool = False) -> dict:
    vitals = np.zeros(17, dtype=np.float32)
    vitals[4] = bullets
    mask = np.zeros(32, dtype=np.int8)
    etype = np.zeros(32, dtype=np.int64)
    if enemy_visible:
        mask[0] = 1
        etype[0] = 2  # enemy_hitscan
    return {
        "vitals": vitals,
        "visible_entities": {
            "features": np.zeros((32, 5), dtype=np.float32),
            "entity_type": etype,
            "mask": mask,
        },
    }


def test_progression_and_exit_time_plus_health_efficiency():
    cfg = RewardConfig()
    shaper = RewardShaper(cfg)
    shaper.reset(_priv(tic=0, x=0.0, y=0.0, health=100.0))

    r1, ev1, c1 = shaper.step(_priv(tic=4, keys=[0, 0, 1, 0, 0, 0], sector_ceiling=[64.0, 0.0, 128.0]))
    assert ev1.keys_picked_up == 1
    assert ev1.doors_opened == 1
    assert c1["key"] == pytest.approx(cfg.key_pickup_reward)
    assert c1["door"] == pytest.approx(cfg.door_open_reward)
    assert r1 == pytest.approx(sum(c1.values()))

    # Locked door unlock; sector 0 oscillates within margin without door-bounce threshold or double-count
    r2, ev2, c2 = shaper.step(_priv(tic=8, keys=[0, 0, 1, 0, 0, 0], sector_ceiling=[64.0, 64.0, 128.0]))
    assert ev2.doors_opened == 0
    assert ev2.locked_doors_unlocked == 1
    assert c2["unlock"] == pytest.approx(cfg.locked_door_unlock_reward)
    assert r2 == pytest.approx(sum(c2.values()))

    # Exit at tic=250/1000 (0.75 time remaining) and health=80 (0.80 health frac)
    r3, ev3, c3 = shaper.step(
        _priv(
            tic=250,
            episode_ticks=1000,
            health=80.0,
            keys=[0, 0, 1, 0, 0, 0],
            sector_ceiling=[64.0, 64.0, 128.0],
            exit_reached=True,
        )
    )
    assert ev3.exit_reached is True
    assert c3["exit"] == pytest.approx(cfg.exit_reward)
    expected_eff = cfg.efficiency_bonus * 0.75 + cfg.health_efficiency_bonus * 0.80
    assert c3["efficiency"] == pytest.approx(expected_eff)
    assert r3 == pytest.approx(sum(c3.values()))


def test_diminishing_exploration_250_cells_and_post_key_reset():
    cfg = RewardConfig()
    shaper = RewardShaper(cfg)
    shaper.reset(_priv(tic=0, x=0.0))

    # Visit 280 cells: 250 should earn diminishing reward, >250 earns 0, total < exit_reward
    explore_total = 0.0
    for step_idx in range(1, 281):
        _, _, comp = shaper.step(_priv(tic=step_idx * 4, x=float(step_idx * 64)))
        explore_total += comp["explore"]

    assert cfg.max_exploration_cells == 250
    assert 0.0 < explore_total < cfg.exit_reward
    _, _, comp_extra = shaper.step(_priv(tic=1200, x=float(281 * 64)))
    assert comp_extra["explore"] == pytest.approx(0.0)

    # Pick up key at cell x=281*64, then backtrack through already-visited cells x=280*64, 279*64
    _, ev_k, comp_k = shaper.step(_priv(tic=1204, x=float(281 * 64), keys=[1, 0, 0, 0, 0, 0]))
    assert ev_k.keys_picked_up == 1
    assert comp_k["post_key_explore"] == pytest.approx(0.0)

    _, _, comp_back1 = shaper.step(_priv(tic=1208, x=float(280 * 64), keys=[1, 0, 0, 0, 0, 0]))
    _, _, comp_back2 = shaper.step(_priv(tic=1212, x=float(279 * 64), keys=[1, 0, 0, 0, 0, 0]))
    assert comp_back1["explore"] == pytest.approx(0.0)
    assert comp_back1["post_key_explore"] == pytest.approx(cfg.post_key_cell_reward)
    assert comp_back2["post_key_explore"] == pytest.approx(cfg.post_key_cell_reward)


def test_use_waste_door_bounce_and_combat_resource_rewards():
    cfg = RewardConfig()
    shaper = RewardShaper(cfg)
    shaper.reset(_priv(tic=0, health=50.0, armor=10.0, sector_ceiling=[0.0, 0.0, 128.0]))

    prev_act = np.zeros(9, dtype=np.int8)
    use_act = np.zeros(9, dtype=np.int8)
    use_act[7] = 1

    # 1. Wasted USE pulse (no door opens, no door sector starts rising)
    _, _, c_waste = shaper.step(
        _priv(tic=4, health=50.0, sector_ceiling=[0.0, 0.0, 128.0]),
        action=use_act,
        prev_action=prev_act,
        obs=_obs(bullets=50.0, enemy_visible=False),
    )
    assert c_waste["use_penalty"] == pytest.approx(cfg.use_waste_penalty)

    # 2. Door sector 0 starts rising (+12.0), then bounces back down (-8.0) -> door bounce penalty
    shaper.step(_priv(tic=8, health=50.0, sector_ceiling=[12.0, 0.0, 128.0]), action=use_act, prev_action=prev_act)
    _, _, c_bounce = shaper.step(
        _priv(tic=12, health=50.0, sector_ceiling=[4.0, 0.0, 128.0]),
        action=use_act,
        prev_action=prev_act,
    )
    assert c_bounce["use_penalty"] == pytest.approx(cfg.door_bounce_penalty)

    # 3. Wasted ammo (ammo drops 50 -> 49, no kill, no enemy visible) + low-health pickup bonus
    _, _, c_ammo = shaper.step(
        _priv(tic=16, health=70.0, armor=30.0, sector_ceiling=[4.0, 0.0, 128.0]),
        obs=_obs(bullets=49.0, enemy_visible=False),
    )
    assert c_ammo["ammo_waste"] == pytest.approx(cfg.wasted_ammo_penalty)
    assert c_ammo["pickup"] > 0.0

    # 4. Rebalanced combat: killing an enemy while taking one 20 HP hit stays positive (no combat cowardice),
    # while dying after heavy damage remains strongly negative.
    _, _, c_trade = shaper.step(
        _priv(tic=20, health=50.0, armor=30.0, kills=1, damage_taken=20.0, sector_ceiling=[4.0, 0.0, 128.0]),
        obs=_obs(bullets=48.0, enemy_visible=True),
    )
    assert c_trade["ammo_waste"] == pytest.approx(0.0)
    assert c_trade["kill"] + c_trade["damage"] > 0.0
    _, _, c_dead = shaper.step(
        _priv(tic=24, health=0.0, dead=True, damage_taken=50.0, sector_ceiling=[4.0, 0.0, 128.0]),
        obs=_obs(bullets=48.0, enemy_visible=True),
    )
    assert c_dead["death"] + c_dead["damage"] < -1.0


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
        obs, _ = env.reset(seed=0)
        shaper.reset(env.privileged_state())
        prev_action = np.zeros(len(S.BUTTONS), dtype=np.int8)
        totals = {"key": 0.0, "unlock": 0.0, "exit": 0.0, "explore": 0.0}
        last_x = None
        for i in range(240):
            st = env.privileged_state()
            stuck = last_x is not None and abs(st["x"] - last_x) < 0.5
            last_x = st["x"]
            action = use if (stuck and i % 6 == 0) else fwd
            obs, _, term, trunc, _ = env.step(action)
            _, _, comp = shaper.step(env.privileged_state(), action=action, prev_action=prev_action, obs=obs)
            prev_action = action
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


def test_use_hold_attack_press_and_weapon_switch_penalties():
    cfg = RewardConfig()
    shaper = RewardShaper(cfg)
    shaper.reset(_priv(tic=0, health=100.0, sector_ceiling=[0.0, 0.0, 128.0]))

    prev_use = np.zeros(9, dtype=np.int8)
    prev_use[7] = 1
    hold_use = np.zeros(9, dtype=np.int8)
    hold_use[7] = 1
    _, _, c_hold = shaper.step(
        _priv(tic=4, health=100.0, sector_ceiling=[0.0, 0.0, 128.0]),
        action=hold_use,
        prev_action=prev_use,
        obs=_obs(bullets=50.0, enemy_visible=False),
    )
    assert c_hold["use_penalty"] == pytest.approx(cfg.use_hold_penalty)

    atk_act = np.zeros(9, dtype=np.int8)
    atk_act[6] = 1
    _, _, c_atk = shaper.step(
        _priv(tic=8, health=100.0, sector_ceiling=[0.0, 0.0, 128.0]),
        action=atk_act,
        prev_action=np.zeros(9, dtype=np.int8),
        obs=_obs(bullets=50.0, enemy_visible=False),
    )
    assert c_atk["ammo_waste"] == pytest.approx(cfg.wasted_ammo_penalty)

    wpn_act = np.zeros(9, dtype=np.int8)
    wpn_act[8] = 1
    obs_loaded = _obs(bullets=50.0, enemy_visible=False)
    obs_loaded["vitals"][2] = 2.0  # Pistol
    obs_loaded["vitals"][3] = 50.0
    _, _, c_sw_loaded = shaper.step(
        _priv(tic=12, health=100.0, sector_ceiling=[0.0, 0.0, 128.0]),
        action=wpn_act,
        prev_action=np.zeros(9, dtype=np.int8),
        obs=obs_loaded,
    )
    assert c_sw_loaded["weapon_switch_penalty"] == pytest.approx(cfg.weapon_switch_waste_penalty)

    obs_empty = _obs(bullets=0.0, enemy_visible=False)
    obs_empty["vitals"][2] = 2.0
    obs_empty["vitals"][3] = 0.0
    _, _, c_sw_empty = shaper.step(
        _priv(tic=16, health=100.0, sector_ceiling=[0.0, 0.0, 128.0]),
        action=wpn_act,
        prev_action=np.zeros(9, dtype=np.int8),
        obs=obs_empty,
    )
    assert c_sw_empty["weapon_switch_penalty"] == pytest.approx(0.0)

    obs_fist = _obs(bullets=50.0, enemy_visible=False)
    obs_fist["vitals"][2] = 1.0  # Fist
    obs_fist["vitals"][3] = 50.0
    _, _, c_sw_fist = shaper.step(
        _priv(tic=20, health=100.0, sector_ceiling=[0.0, 0.0, 128.0]),
        action=wpn_act,
        prev_action=np.zeros(9, dtype=np.int8),
        obs=obs_fist,
    )
    assert c_sw_fist["weapon_switch_penalty"] == pytest.approx(0.0)
