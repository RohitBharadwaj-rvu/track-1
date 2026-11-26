"""Integration tests for Role 2 training loop, checkpoint resume, and two-map export."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
doom_env = pytest.importorskip("doom_env")

from training.interfaces import MapSpec
from training.train import TrainConfig, run_training, verify_checkpoint_two_maps


def test_run_training_saves_resumes_and_plays_two_maps(tmp_path):
    run_dir = tmp_path / "run1"
    export_pt = tmp_path / "weights" / "policy.pt"
    cfg = TrainConfig(
        arch="stub",
        hidden_size=32,
        maps=("maps/two_rooms.wad:MAP01:door_exit:1:120",),
        num_envs=2,
        vec_mode="sync",
        rollout_steps=16,
        chunk_len=8,
        num_minibatches=2,
        update_epochs=1,
        total_steps=64,  # 2 updates * (16 * 2 = 32 steps)
        seed=0,
        device="cpu",
        checkpoint_dir=str(run_dir),
        export_path=str(export_pt),
        save_interval=1,
    )
    res1 = run_training(cfg)
    assert res1["optimizer_updates"] == 2
    assert res1["total_decisions"] == 64
    assert export_pt.is_file()
    assert (run_dir / "latest_train_state.pt").is_file()
    assert (run_dir / "metrics.jsonl").is_file()

    # Resume for 2 more updates
    cfg_resume = TrainConfig(
        arch="stub",
        hidden_size=32,
        maps=("maps/two_rooms.wad:MAP01:door_exit:1:120",),
        num_envs=2,
        vec_mode="sync",
        rollout_steps=16,
        chunk_len=8,
        num_minibatches=2,
        update_epochs=1,
        total_steps=64,
        seed=1,
        device="cpu",
        checkpoint_dir=str(run_dir),
        export_path=str(export_pt),
        resume=res1["resume_path"],
        save_interval=1,
    )
    res2 = run_training(cfg_resume)
    assert res2["optimizer_updates"] == 4
    assert res2["total_decisions"] == 128

    # Verify exported weights play 2 consecutive maps with clean memory resets
    eps = verify_checkpoint_two_maps(
        export_pt,
        [MapSpec(wad="maps/two_rooms.wad", map_name="MAP01", family="door_exit", tier=1, episode_ticks=80)],
        device="cpu",
    )
    assert len(eps) == 2
    for ep in eps:
        assert ep.outcome in {"exit", "death", "timeout"}
        assert ep.steps > 0
