# Training & Parallel Workshare

This folder holds our training pipeline, curriculum maps, reward shaping, and independent evaluator (reviewed by judges). Follow [REVISED_PLAN.md](../REVISED_PLAN.md).

## 4-Person Parallel Split (Zero-Conflict File Ownership)

| Role | Branch prefix | Owns | Depends on (frozen contracts) |
|---|---|---|---|
| **1. Policy & Exported Agent** | `role1/` | `submission/policy.py` (real `TelemetryPolicy`), `submission/agent.py`, `submission/obs.py` | `PolicyConfig`, `act_step`, `evaluate_chunks`, `save/load_checkpoint` |
| **2. PPO & Rollout Collection** | `role2/` | `training/rollout.py`, `training/ppo.py`, `training/rewards.py`, `training/train.py` | `submission/obs.py`, `submission/policy.py` (can run on `arch="stub"` immediately), `training/interfaces.py` |
| **3. Playable Maps & Curriculum** | `role3/` | `training/maps.py`, `maps/` (verified WADs & generators) | `MapSpec` in `training/interfaces.py` |
| **4. Evaluation & Release** | `role4/` | `training/eval.py`, Docker/LFS release checks, 2-page PDF brief | `EpisodeMetrics` & `PrivilegedEventTracker` in `training/interfaces.py`, `validate.py` |

## Frozen Shared Contracts

1. **Observation preprocessing (`submission/obs.py`):**
   - `preprocess_obs(obs)` -> `ray_dist (64,)`, `ray_type (64,)`, `ent_feat (32, 5)`, `ent_type (32,)`, `ent_mask (32,)`, `vitals (17,)`
   - `batch_preprocessed_obs(list_of_obs)` -> stacked `(B, ...)`
   - `obs_to_torch(obs_batch, device)` -> PyTorch tensors on `device`
2. **Policy & checkpoints (`submission/policy.py`):**
   - `model = build_policy(PolicyConfig(arch="stub"))` works right away for Role 2.
   - `hidden = model.init_hidden(B, device)` -> `(1, B, H)`
   - `logits, value, next_hidden = model.act_step(obs_batch, prev_action, hidden)`
   - `log_prob, entropy, value = model.evaluate_chunks(obs_seq, prev_action_seq, action_seq, init_hidden, episode_starts)`
   - `save_checkpoint(path, model, config, meta)` / `load_checkpoint(path, device)`
3. **Maps & Event Tracking (`training/interfaces.py`):**
   - `MapSpec` registers train/dev/final maps with tier, family, tick cap, and fixed seeds.
   - `PrivilegedEventTracker` extracts key pickups, door opens, locked-door unlocks, kills, damage, unique cells, and episode outcome (`exit` / `death` / `timeout`) from `env.privileged_state()`.

## Stage 1 (Connect) Exit Gate

Merge Roles 1–4 when:
1. Role 2 collects real trajectories, runs a recurrent PPO update, and saves a `.pt` checkpoint to `submission/weights/`.
2. Role 1 reloads that checkpoint in `submission/agent.py` (`DoomAgent`), plays **two consecutive maps with memory reset**, and passes `uv run python validate.py`.
3. Role 3 provides one verified door/exit map and one verified key/lock map.
4. Role 4 runs the evaluator on both maps and distinguishes **pickup, unlock, exit, death, and timeout**.
