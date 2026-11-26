# Track 1 improvements: generalization, partial observability, long-horizon memory, and combat

*Uploaded: 2026-11-26 17:42 IST (`improvements/2026-11-26_17-42/`)*

Audit of the repository state against [REVISED_PLAN.md](../../REVISED_PLAN.md), the official Track 1 specification (`Yugaantar DevForge Tracks.docx`), and the three Core Challenge Objectives:

1. **Partial observability:** Navigating 3D spaces without a god-view radar, relying only on what is inside the player's field of view and local sensory range.
2. **Long-horizon progression:** Solving multi-step room dependencies such as finding a keycard on a branch, remembering where a locked door was sighted earlier, and backtracking to unlock the exit.
3. **Tactical combat and survival:** Managing health, ammo, and movement momentum while dodging incoming projectiles and prioritizing multiple threats.

---

## 1. Current GitHub status vs. Plan A

All four role branches now have real implementations on GitHub, but **none of them are merged into `master`**, and `role2/ppo-pipeline` has not yet merged Role 3's maps or Role 4's evaluator.

| Branch | Latest commit | Current status |
|---|---|---|
| `origin/master` | `a9fd328` | Frozen contracts and `StubRecurrentPolicy` only. The scorer evaluates the last commit on the default branch at the deadline ([docs/EVALUATION.md](../../docs/EVALUATION.md#L20-L23)). |
| `origin/role1/telemetry-policy` | `d8f89c2` | `TelemetryPolicy` (`telemetry_gru_v1`) in [submission/policy.py](../../submission/policy.py#L342-L472), `DoomAgent` in [submission/agent.py](../../submission/agent.py#L49-L106), Colab A100 setup, and untrained smoke weights. |
| `origin/role2/ppo-pipeline` | `691cf3f` | Merged Role 1 and added vectorized rollouts ([training/rollout.py](../../training/rollout.py)), recurrent PPO ([training/ppo.py](../../training/ppo.py)), reward shaping ([training/rewards.py](../../training/rewards.py)), and CLI training ([training/train.py](../../training/train.py)). |
| `origin/role3/maps-curriculum` | `c79708d` | Added UDMF WAD generators (`training/wad_gen/`), `training/maps.py`, and verified Train (20 WADs), Dev (10 WADs), and Final (8 WADs, including polygon layouts) suites. Not merged yet. |
| `origin/role4/eval-release` | `d132506` | Added `training/eval.py`, `training/release_check.py`, placeholder `docs/BRIEF.md`, and `updates.md`. Not merged yet. |

---

## 2. Generalization across unseen maps

### 2.1 Merge all four roles onto `master` and wire Role 3's procedural maps into training (All roles)
- `role2/ppo-pipeline` currently falls back to `maps/two_rooms.wad` because `training/maps.py` is still only on `role3/maps-curriculum`.
- Merge `role1`, `role2`, `role3`, and `role4` into `master` (or an integration branch first, then `master`) so Role 2 trains across Role 3's 20 train WADs plus on-the-fly variants from `training/wad_gen/procedural.py`, and Role 4 evaluates on Role 3's held-out `dev` and `final` suites.
- Keep generating procedural training variations (room sizes, mirrored left/right key branches, all three keycard colors, loops, dead ends, enemy placements) so the agent cannot memorize the 20 static train WADs.

### 2.2 Replace uniform map sampling with staged curriculum sampling (Role 2 + Role 3)
- `_DoomWorkerSlot.reset()` in [training/rollout.py](../../training/rollout.py#L151-L153) samples `map_specs` uniformly forever.
- Add per-tier and per-family sampling weights to `TrainConfig` so early updates focus on fast, dense tasks (short door exits, combat arenas, short key returns) and shift weight toward long Tier 3 and Tier 4 maps as exit rates rise, while keeping a replay slice of earlier tiers to prevent forgetting.

### 2.3 Use allowed training-time curricula (Role 2 + Role 3)
- Section 6 of the Track 1 spec and [docs/RULES.md](../../docs/RULES.md#L19-L25) allow engine and map modifications during training (auto-opening doors during early combat pretraining, shorter return branches, extra ammo) provided final fine-tuning and evaluation run on vanilla ViZDoom rules.

---

## 3. Partial observability

### 3.1 Fix the exploration reward cap (Role 2)
- In [training/rewards.py](../../training/rewards.py#L25) and [training/rewards.py](../../training/rewards.py#L55-L57), `max_exploration_cells = 40` with `cell_size = 64.0`.
- A single medium DOOM room is easily 40+ cells of size 64. On a multi-room Tier 2, 3, or 4 map, exploration reward drops to zero before the agent leaves the second room.
- Raise `max_exploration_cells` to 200+ (or track sector coverage via `env.sector_polygons()` plus a higher cell cap with diminishing per-cell reward so total exploration stays strictly below the exit reward).

### 3.2 Account for the exit and switch telemetry blind spot (Roles 1 and 3)
- In [env/doom_env/sensors.py](../../env/doom_env/sensors.py#L25) and [env/doom_env/spaces.py](../../env/doom_env/spaces.py#L16-L34), `RAY_HIT_TYPES` only distinguishes `none, wall, door, locked_door, enemy, item`, and `ENTITY_TYPES` has no `exit` or `switch` class.
- Wall-mounted exit switches appear identical to plain `wall` hits in rays, and `sensors.py` lines [134-135](../../env/doom_env/sensors.py#L134-L135) collapse walls, ledges > 24 units, and openings < 56 units into `RAY_WALL`.
- Verify whether hidden evaluation maps can use wall exit switches or vertical ledges. If switches or drops appear on Tier 2 to 4 maps, test adding a tiny downsampled `depth` or `rgb` side branch within the 40 ms budget.

### 3.3 Replace 9 independent Bernoullis with structured factorized action heads (Role 1 + Role 2)
- [submission/policy.py](../../submission/policy.py#L360-L364) currently outputs 9 independent binary logits (`MultiBinary(9)`). That lets the policy press `MOVE_FORWARD + MOVE_BACKWARD`, `MOVE_LEFT + MOVE_RIGHT`, and `TURN_LEFT + TURN_RIGHT` at the same time, canceling motion and wasting entropy.
- Switch the actor to:
  - Three 3-way categorical heads: `(forward / backward / none)`, `(strafe_left / strafe_right / none)`, `(turn_left / turn_right / none)`.
  - Three independent Bernoulli heads: `ATTACK`, `USE`, `SELECT_NEXT_WEAPON`.
- Map those outputs deterministically to the required `(9,)` binary array. All decisions still come from trained neural weights ([docs/RULES.md](../../docs/RULES.md#L6-L18)).

### 3.4 Add explicit motion-delta and wall-snag features (Role 1)
- In `VitalsActionEncoder` ([submission/policy.py](../../submission/policy.py#L235-L249)), add scalar horizontal speed `hypot(vel_forward, vel_right)` and optional 1-step ray deltas (`ray_dist_t - ray_dist_{t-1}` or `min_front_ray`).
- Seeing commanded forward motion paired with zero horizontal speed lets the policy detect wall-snags in one step instead of grinding against corners.

### 3.5 Add training-only intrinsic curiosity for Tier 2 exploration (Role 2)
- Add a lightweight training-only intrinsic bonus such as Random Network Distillation (RND) or next-step telemetry prediction error `(ray_t, vitals_t, action_t) -> (ray_{t+1}, vitals_{t+1})`.
- It adds zero evaluation latency, helps break dead-end traps on unseen maps, and directly supports the Round 2/3 innovation rubric.

---

## 4. Long-horizon progression (keycard, locked door, backtracking)

### 4.1 Extend BPTT horizon and discount factor (Role 2)
- Current defaults in [training/ppo.py](../../training/ppo.py#L24-L31) and [training/train.py](../../training/train.py#L41-L50) are `chunk_len = 32`, `rollout_steps = 128`, and `gamma = 0.99`.
- At `FRAME_SKIP = 4` (8.75 decisions/sec), `chunk_len = 32` gives only 3.6 seconds of gradient flow. A Tier 3 key-and-return episode spans 200 to 600 decisions (25 to 70 seconds), and `gamma = 0.99` shrinks a reward 250 steps away to `0.08x`.
- Update defaults to:
  - `gamma`: `0.995` to `0.997`
  - `rollout_steps`: `256` to `512`
  - `chunk_len`: `64` to `128`

### 4.2 Implement odometry-updated neural landmark memory (Role 1, Challenger 1)
- `PolicyConfig.num_memory_slots` ([submission/policy.py](../../submission/policy.py#L63)) is currently a dead placeholder (`0`). Also note line [412](../../submission/policy.py#L412): the `gru_out + fused` residual shortcut makes it easy for PPO to lean on reactive inputs early and under-train GRU memory.
- Once the GRU MVP baseline is frozen, build **odometry-updated neural landmark memory** (8 to 16 persistent slots, or a compact differentiable egocentric latent grid):
  - Each slot stores a learned descriptor vector plus an estimated ego-frame relative offset `(dx, dy)`.
  - On each step, translate and rotate stored `(dx, dy)` vectors using ego velocity `(vel_forward, vel_right)` from `vitals` and action-conditioned yaw turn rate.
  - Learned write gates commit sightings (such as `hit_type == 3` locked doors, keys, or branch junctions); learned attention read gates query slots conditioned on key inventory and current GRU state.
- Uses zero global map coordinates, zero `A*` or Dijkstra, 100% learned parameters, stays under 1 ms per step, and provides a strong architecture story for Round 2/3 judging.

### 4.3 Add a training-only asymmetric critic and auxiliary memory supervision (Role 1 + Role 2)
- `TelemetryPolicy.critic` ([submission/policy.py](../../submission/policy.py#L365-L369)) currently receives only blind `gru_out`.
- **Asymmetric critic:** Feed privileged training features from `env.privileged_state()` (unique cells visited, keys held, doors opened/unlocked, remaining tick fraction, progress proxies) into the critic during PPO updates. Export only the blind actor at submission.
- **Auxiliary supervision heads (training-only):** Train small auxiliary heads on top of the GRU/memory state to predict (a) whether a locked door has been seen this episode, and (b) the ego-frame unit direction `(sin, cos)` back to the sighted locked door or nearest unvisited area. This gives direct gradient signal to memory from update 1 rather than waiting for sparse unlock rewards to propagate 400 steps.

### 4.4 Fix Tier 3 reward traps (Role 2)
- **Post-key exploration reset:** After picking up a keycard, every cell along the return route is already in `_visited_cells`, so backtracking earns zero exploration reward and only `-0.002/step`. Reset a secondary post-key visit set when `events.keys_picked_up > 0` so revisiting corridors while holding a new key gives bounded exploration signal.
- **`USE` door-bounce penalty:** Pressing `USE` on an already-moving door reverses it ([docs/CONTRACT.md](../../docs/CONTRACT.md#L95-L98)). Add a small penalty (for example `-0.02`) when `USE` pulses `0 -> 1` without opening a door or unlocking a lock, or when a rising door sector drops back down.
- **Health bonus on exit:** [training/rewards.py](../../training/rewards.py#L60-L63) rewards only remaining time on exit; [docs/EVALUATION.md](../../docs/EVALUATION.md#L44) rewards both remaining health and time. Add a remaining-health efficiency term on exit.

---

## 5. Tactical combat and survival

### 5.1 Upgrade `MaskedEntityAttentionEncoder` with combat kinematics and threat counts (Role 1)
Look at [submission/policy.py](../../submission/policy.py#L251-L340):
- `entity_num_queries = 2` is small, and softmax attention averages entities together so the network cannot distinguish 1 enemy from 5 enemies.
- `query_proj` ([submission/policy.py](../../submission/policy.py#L272-L275)) conditions only on `[vitals_feat, ray_feat]`, not on `h_{t-1}`.
- The 5 raw entity fields force a 2-layer MLP to infer projectile collision math from scratch.

Concrete upgrades (all deterministic observation encoding allowed by [docs/RULES.md](../../docs/RULES.md#L16-L17)):
- Derive per-entity ego kinematics:
  - **Closing speed:** `-(rel_vel_fwd * cos_a + rel_vel_right * sin_a)`
  - **Lateral speed:** `(-rel_vel_fwd * sin_a + rel_vel_right * cos_a)`
  - **Inverse time-to-impact:** `clamp(closing_speed, 0) / (dist + 0.05)`
  - **Crosshair alignment:** `exp(-|sin_a| / 0.1) * (cos_a > 0)`
- Increase `entity_num_queries` to 4 or 6 (or add 1 masked self-attention layer before cross-attention).
- Concatenate explicit per-category counts `log1p([n_melee, n_hitscan, n_proj_enemy, n_projectiles, n_health, n_ammo, n_keys])` plus nearest-threat distance so the actor can prioritize threats and recognize when it is outnumbered.

### 5.2 Fix the ammo-empty blind spot in `VitalsActionEncoder` (Role 1)
- [submission/obs.py](../../submission/obs.py#L49-L53) divides ammo by `[100, 200, 50, 50, 300]`, and [submission/policy.py](../../submission/policy.py#L243) then applies `torch.log1p(...)` to those decimals. That compresses `0` vs `2` shells into `0.000` vs `0.039`, while `weapon_switch_init_bias = -2.0` discourages switching.
- Add explicit binary flags in `VitalsActionEncoder`:
  - `selected_ammo_empty = (selected_weapon_ammo <= 0)`
  - `has_any_other_ammo`
  - `low_health = (health < 0.35)`

### 5.3 Rebalance combat and resource rewards (Role 2)
- In [training/rewards.py](../../training/rewards.py#L26-L28), `kill_reward = 0.5` and `damage_penalty = -0.01`. Taking a 20 HP fireball costs `-0.20`, so standing still and trading hits is net positive. There is also no penalty for shooting walls and no reward for picking up health or ammo when low.
- Raise `damage_penalty` to `-0.03` or `-0.04` per HP so projectile dodging pays off.
- Add a small wasted-ammo penalty (for example `-0.005` when ammo drops with no enemy in FOV) to preserve ammo on Tier 4 gauntlets.
- Add a small bounded bonus for health/armor gain when hurt (`health < 80`) or weapon pickups, keeping total combat/pickup rewards strictly below progression and exit rewards.

### 5.4 Benchmark evaluation determinism (Role 1 + Role 4)
- [submission/policy.py](../../submission/policy.py#L75) defaults to `eval_deterministic: bool = False`, causing independent Bernoulli sampling on every step in [submission/agent.py](../../submission/agent.py#L100-L103). That introduces noisy `USE` double-taps and aim jitter.
- Once factorized action heads are in place, compare deterministic argmax vs. low-temperature sampling on Role 4's dev suite and store the winning configuration in `PolicyConfig`.

---

## 6. Judging and submission checklist

- **Innovation rubric (Round 2 desk review and Round 3 finals):** Custom neural modules (odometry-updated landmark memory, asymmetric critic, auxiliary door-bearing supervision, curiosity) are explicitly rewarded over plain recurrent PPO. Follow [REVISED_PLAN.md](../../REVISED_PLAN.md): freeze a working GRU MVP first, then promote upgrades only when `python -m training.eval --compare` shows gains on held-out Tier 3 and 4 seeds.
- **2-page Project Brief (PDF, max 2 pages):** Replace the TBD tables in `docs/BRIEF.md` (currently on `origin/role4/eval-release`) with measured baseline vs. GRU MVP vs. memory-upgrade numbers, sandbox latency numbers, and one concrete failure replay analysis.
- **Default branch and sandbox limits:** Merge final code and Git LFS weights to `master` before the deadline, keep total repo + LFS size `<= 1 GB`, do not rely on custom `Dockerfile` edits, keep `submission/requirements.txt` minimal or empty, and pass `python validate.py --docker` from a fresh clone.

---

## 7. Recommended execution order

1. **Merge Roles 1, 2, 3, and 4 into `master` first** so training uses Role 3's Train/procedural maps and evaluation runs on Role 3's Dev/Final suites via Role 4's harness.
2. **Land low-risk, high-ROI pre-MVP upgrades before spending Colab A100 hours:**
   - Factorized 3-way movement/strafe/turn + 3-Bernoulli action heads in [submission/policy.py](../../submission/policy.py).
   - Entity combat kinematics, category counts, `entity_num_queries = 4..6`, horizontal speed, and `selected_ammo_empty` flag in [submission/policy.py](../../submission/policy.py).
   - Longer BPTT (`chunk_len = 64`, `rollout_steps = 256`, `gamma = 0.995`) in [training/ppo.py](../../training/ppo.py) and [training/train.py](../../training/train.py).
   - Reward shaper updates in [training/rewards.py](../../training/rewards.py) (higher exploration cap, post-key cell reset, higher `damage_penalty`, `USE` bounce penalty, wasted-ammo penalty, health-on-exit bonus).
3. **Train and freeze the GRU MVP checkpoint on Colab A100**, verify `validate.py`, and record Role 4 dev baselines.
4. **Run Challenger 1 (odometry landmark memory + asymmetric critic + auxiliary door-bearing loss)** against the frozen GRU MVP on identical held-out Tier 3 and 4 seeds via `python -m training.eval --compare`. Promote only if unlocks and exits improve.
