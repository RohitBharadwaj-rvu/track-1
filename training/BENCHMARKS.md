# Role 2 Verification & Benchmarks

Recorded from end-to-end verification on branch `role2/ppo-pipeline` (`telemetry_gru_v1`, `687,098` params).

## 1. Unified Linux + CUDA Test Suite (Milestone 1)

- **Environment:** Google Colab `Tesla T4` (`2` CPUs), Linux (`Python 3.13.16`), `vizdoom==1.3.3`, `gymnasium>=1.0`, `pytest`
- **Command:** `python -m pytest -v`
- **Result:** **`77 passed` in `93.31s` (`0 failed`, `0 skipped`)** across Role 1 (`test_policy_and_agent.py`), Role 2 (`test_contracts.py`, `test_rewards.py`, `test_rollout_env.py`, `test_ppo.py`, `test_train.py`), and official template checks (`test_env.py`, `test_remote.py`, `test_validate.py`).

## 2. Throughput Sweep (`telemetry_gru_v1`, CUDA)

Measured on Colab `Tesla T4` (`2` CPUs), `maps/two_rooms.wad`, `rollout_steps=128`, `SubprocVecDoomEnv` (`render_buffers=False`):

| Workers (`--num-envs`) | Decisions | Elapsed (s) | Throughput (`decisions_per_sec`) |
|---|---|---|---|
| `1` | `256` | `1.600` | **`160.0 dec/s`** |
| `2` | `512` | `1.986` | **`257.8 dec/s`** |
| `4` | `1024` | `3.145` | **`325.6 dec/s`** |

*(During training on 4 workers / 2 CPUs, per-update throughput peaked at **`417 dec/s`**; on a 12-CPU A100/L4 runtime, sweep `--bench-workers 1,4,8,12` first and pick the fastest `--num-envs`.)*

## 3. Solvable Door/Exit Overfit Proof (`#P..D...E#`)

- **Map:** `tests/fixtures/9a5147dc56a65733.wad` (`episode_ticks=600`), `arch=telemetry_gru_v1`, `num_envs=4`, `rollout_steps=64`, `chunk_len=32`, `lr=5e-4`, `ent_coef=0.02`, `device=cuda`
- **Learning progression:**
  - Update `1` (`256` steps): `exit_rate=0.00`, `door_open_rate=0.00`, `mean_reward=0.00`
  - Update `11` (`2,816` steps): `exit_rate=0.06`, `door_open_rate=0.24`, `mean_reward=1.11`
  - Update `22` (`5,632` steps): `exit_rate=0.55`, `door_open_rate=1.00`, `mean_reward=8.66`
  - Update `313` (`80,128` steps, `1,932` episodes): **`exit_rate=1.00`, `door_open_rate=1.00`, `mean_reward=14.494`**, `explained_variance=0.558`
- **Resume check (`--resume latest_train_state.pt`, updates `318–321`):**
  - Resumed to `82,176` total decisions (`2,005` episodes): **`exit_rate=1.00`, `door_open_rate=1.00`, `mean_reward=14.496`**, `explained_variance=0.808`.
- **Two-map `DoomAgent` check (`--verify-two-maps`, reloaded via `submission.agent.DoomAgent` with `agent.reset()`):**
  - Map 1 (`seed=0`): `outcome="exit"`, `24` steps, `doors_opened=1`, `unique_cells_visited=8`, `total_reward=14.502`, `agent_time_s=0.115s`
  - Map 2 (`seed=1`): `outcome="exit"`, `24` steps, `doors_opened=1`, `unique_cells_visited=8`, `total_reward=14.502`, `agent_time_s=0.111s`

## 4. Official `validate.py` on Linux + CUDA (Trained `telemetry_gru_v1` Weights)

Command: `python validate.py --map maps/two_rooms.wad --map maps/two_rooms.wad`

- **Repo size:** `3.1 MB` of `1.0 GB`
- **CPU pinning:** Pinned to 1 CPU (`core 1`)
- **Initialization:** `2.4s` (limit `60s`) on `cuda`
- **Episode 1 (`525` steps, tick cap):** `2.2s` agent time of `21.0s` budget (`p50 4.0 ms`, `p99 6.9 ms`, `max 20 ms`, `first 20 ms`)
- **Episode 2 (`479` steps, died):** `2.5s` agent time of `21.0s` budget (`p50 4.8 ms`, `p99 10.8 ms`, `max 18 ms`, `first 5 ms`)
- **Peak RAM:** `1,380 MB` resident (limit `8,192 MB`)
- **Peak GPU VRAM:** `156 MB` (limit `2,048 MB`)

## 5. Launching Full MVP Training on Colab A100

When A100 capacity is available (set `$env:COLAB_CLI_ASSIGN_RETRIES=0` if you want instant fallback when testing GPU pools):

```bash
# 1. Sweep workers on the 12-CPU A100 VM
python -m training.train --arch telemetry_gru_v1 --bench-workers 1,4,8,12 --rollout-steps 128 --device cuda

# 2. Launch training on Role 3's train split using the winning --num-envs
python -m training.train --arch telemetry_gru_v1 --split train --num-envs 12 --rollout-steps 128 --chunk-len 32 --num-minibatches 4 --update-epochs 4 --total-steps 2000000 --device cuda --export-path submission/weights/policy.pt --verify-two-maps
```
