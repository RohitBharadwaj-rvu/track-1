# Track 1 Hackathon Strategy: "Can It Play DOOM?"
**Template:** [`Reinforce-SST/devforge-doom-template`](https://github.com/Reinforce-SST/devforge-doom-template) (Contract `1.0.0-draft`)
**Architecture:** Telemetry-Based World Model (RSSM) + Differentiable Spatial Memory + Privileged Teacher Distillation
**Target:** Maximize weighted score across the 4-Tier Zero-Shot Evaluation within **2 GB VRAM / 8 GB RAM / 1 CPU thread / 40 ms per decision step** (`limits.py`).

---

## 1. The Core Bet: Why We Are Doing This
Standard RL (PPO) fails because it memorizes maps. Pixel-based World Models (DreamerV3) waste compute rendering wall textures and risk blowing past the **2 GB VRAM / 40 ms per-step** budget on a shared RTX 3090 with **1 CPU thread**.

Instead, we build a **Telemetry-Based World Model** on top of `doom_env`:
*   During training, `make_env(..., render_buffers=False)` skips `rgb` and `depth` rendering so the environment steps at thousands of steps/sec on CPU.
*   By feeding `rays`, `visible_entities`, and `vitals` into a Recurrent State-Space Model (RSSM) + Differentiable Spatial Memory, the agent learns Doom physics, projectile dodging, object permanence, and keycard/door backtracking—using **only trained neural parameters** (compliant with `docs/RULES.md`).

---

## 2. System Architecture (Aligned with `doom_env.spaces` & `docs/CONTRACT.md`)

### A. Ego-Centric Inputs (Zero Pixels)
We ignore `rgb` `(120, 160, 3)` and `depth` `(120, 160)` and encode only the three structured observation streams:
*   **`rays` (`NUM_RAYS = 64`, `RAY_RANGE = 1024.0`):**
    *   `distance`: `(64,) float32` in `[0, 1]` (`1.0` = nothing hit), spread evenly over **360°** (index 0 = straight ahead, counter-clockwise; index 16 = 90° left; index 32 = behind).
    *   `hit_type`: `MultiDiscrete([6] * 64)` (`0=none, 1=wall, 2=door, 3=locked_door, 4=enemy, 5=item`). Note: corpses, decorations, and projectiles do **not** stop rays.
    *   *Encoder:* Circular 1D Convolutions (periodic padding so 0° and 354° wrap seamlessly) over `[distance, one_hot(hit_type)]`.
*   **`visible_entities` (`MAX_ENTITIES = 32`, 90° FOV, LOS up to 4096 units):**
    *   `features`: `(32, 5) float32` = `[distance/1024, sin(rel_angle), cos(rel_angle), rel_vel_forward, rel_vel_right]`.
    *   `entity_type`: `MultiDiscrete([12] * 32)` (`0=pad, 1=enemy_melee, 2=enemy_hitscan, 3=enemy_projectile, 4=projectile, 5=health, 6=armor, 7=ammo, 8=weapon, 9=key, 10=barrel, 11=other`).
    *   `mask`: `MultiBinary(32)` (`1` = valid entity, `0` = padding).
    *   *Encoder:* Masked Set Transformer / Cross-Attention module. Because projectiles (`enemy_projectile`) do not stop rays, this stream is critical for **Tier 1 & Tier 4 projectile dodging**.
*   **`vitals` (`17, float32`, `VITALS_INDEX`):**
    *   `health` (0–200), `armor` (0–200), `selected_weapon` (0–7), `selected_weapon_ammo`, `ammo_{bullets,shells,rockets,cells}`, 6 sticky `0/1` key flags (`key_{blue,yellow,red}_{card,skull}`), and ego-frame velocity `(vel_forward, vel_right, vel_up)`.
    *   *Encoder:* Normalized MLP.

### B. The World Model (RSSM Core)
*   **Mechanism:** Encoders fuse `[rays, visible_entities, vitals]` into a latent embedding. A GRU updates the deterministic recurrent state `h_t` and stochastic state `z_t`.
*   **Forward Prediction (Training Only):** Predicts next-step ray distances/hit-types, entity shifts, vitals deltas, and reward.
*   **Curiosity (Tier 2 Exploration Driver):** World-model prediction error (surprise) serves as an intrinsic exploration reward when entering unseen rooms/sectors.

### C. Differentiable Spatial Memory (Tier 3 Long-Horizon Driver)
*   **The Problem:** An RSSM alone forgets where a `locked_door` (`hit_type=3`) or `key` (`entity_type=9`) was after 1,000+ steps (`episode_ticks` up to 4,200 = 1,050 decision steps).
*   **The Fix:** Maintain an egocentric 2D latent grid in hidden memory, shifted/rotated using `vel_forward`, `vel_right`, and inferred/action-conditioned yaw, with neural read/write heads that pin `locked_door`, `door`, `key`, and visited-region embeddings to spatial cells for learned backtracking.

### D. Action Head (`MultiBinary(9)`) & Doom Mechanics
*   **Action Space:** `[MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT, ATTACK, USE, SELECT_NEXT_WEAPON]` (`FRAME_SKIP = 4`, 35 tics/s).
*   **Critical `USE` / Door Rule (`CONTRACT.md §3`):**
    *   `USE` and `SELECT_NEXT_WEAPON` trigger only on the **button-down edge** (holding `USE=1` presses it once; must output `USE=0` for >= 1 step before pressing again).
    *   **Pressing `USE` on an already-moving door reverses it.** Spamming `USE` in a doorway bounces the door shut. We feed `prev_action` into the RSSM so the policy learns to pulse `USE` once and wait ~1–2s (9–18 steps) for doors to open.
*   **Rules Compliance (`RULES.md §1`):** All 9 binary outputs come strictly from trained neural network parameters (no `if/else` action trees, no `A*`/Dijkstra, no `atan2` aimbots, no unmodified SB3).

---

## 3. The Training Pipeline (`training/` on Cloud GPUs e.g. Colab A100)

1.  **Procedural WAD Generation (Adversarial Curriculum):**
    *   Train across thousands of procedural WADs (e.g., Oblige/Obsidian or custom Python WAD generator) rather than `maps/two_rooms.wad` alone.
    *   *Template Constraints:* Must use `freedoom2.wad` IWAD, `skill 3` ("Hurt me plenty"), stay **under 2,048 sectors** per map (ViZDoom limit in `CONTRACT.md §7`), and use standard Doom/Boom/Hexen/UDMF door and locked-door linedef specials so `doom_env` emits `door` / `locked_door` ray types.
2.  **Reward Shaping via `env.privileged_state()` & `env.sector_polygons()`:**
    *   `doom_env`'s default reward (`+1` exit, `-1` death, `+0.1` kill) is too sparse. During training, use `env.privileged_state()` (`x, y, z, angle, health, kills, items, damage_taken, keys, door_lock`) and `env.sector_polygons()` to shape rewards matching all 4 `EVALUATION.md` scoring categories:
        *   **Exit & Efficiency:** Large exit bonus + remaining health/time bonus.
        *   **Progression:** Keycard pickup, door opening, locked-door unlocking.
        *   **Exploration:** Unique sector coverage (via `sector_polygons()`) + RSSM curiosity.
        *   **Combat & Survival:** Kills, damage dealt, penalties for `damage_taken` and early death.
3.  **Privileged Teacher Distillation:**
    *   Train a "God-mode" Teacher that receives `env.privileged_state()` (global `(x, y, angle)`, `door_lock`, sector layout) alongside observations.
    *   Distill into the "Blind" Student (`DoomAgent`) via PPO + KL-Divergence loss, while the Student only sees ego-centric `doom_env` observations.

---

## 4. Sandbox, Submission & Validation Compliance (`docs/SUBMISSION.md` & `limits.py`)

*   **Repository Structure:** Built from `Reinforce-SST/devforge-doom-template`:
    *   `submission/agent.py`: Implements `DoomAgent(__init__, reset, act)`. Loads weights relative to `Path(__file__).resolve().parent / "weights"`.
    *   `submission/weights/`: Tracked with **Git LFS** (`.gitattributes` already tracks `*.pt`, `*.pth`, `*.safetensors`, etc.). Total repo + LFS size **<= 1 GB**.
    *   `submission/requirements.txt`: Keep empty or minimal (`package==version` only, max 50 lines). Base image (`docker/requirements.lock`) already includes Python 3.13, `vizdoom`, `gymnasium`, `torch`, `jax`, `flax`, `numpy`, `scipy`, `opencv-python-headless`, and `doom_env`.
    *   `training/`: Full training, reward-shaping, and WAD-generation code (reviewed by judges).
*   **Runtime & Timeout Budget (`env/doom_env/limits.py`):**
    *   **Hardware:** 2 GB VRAM cap (`torch.cuda.set_per_process_memory_fraction`), 8 GB RAM cap (including `/tmp` and `/dev/shm`), **1 CPU thread** (`OMP_NUM_THREADS=1`), read-only repo, no network.
    *   **No Latent Rollouts at Test Time:** `act()` runs only `Encoder -> Posterior Update -> Spatial Memory Read -> Actor Head` (~2–5 ms, well under the **40 ms/step** budget; e.g., 21 s total for 2,100-tick maps, 42 s for 4,200-tick maps).
    *   **Stateless Resets:** `reset(self)` completely zeros GRU hidden states, spatial memory grids, and `prev_action` buffers (`RESET_TIMEOUT_S = 10.0`).
    *   **Fast Warm-Up:** `__init__(self)` (`INIT_TIMEOUT_S = 60.0`) loads weights onto `device` and runs a dummy forward pass so CUDA/JIT initialization finishes before the `FIRST_ACT_TIMEOUT_S = 10.0` (and `ACT_TIMEOUT_S = 5.0` thereafter).
*   **Pre-Submission Verification & Deliverables:**
    *   Run `uv run python validate.py` locally and `python validate.py --docker` (against `Dockerfile` / official image) before every major milestone and before final submission.
    *   Do **not** rely on a custom `Dockerfile` (evaluation always runs in the organizers' official base image + `submission/requirements.txt`).
    *   Prepare the **2-Page Project Brief (PDF, max 2 pages)** summarizing our USP, neural architecture, spatial memory design, and training/exploration pipeline (required alongside the public GitHub repo URL).

---

## 5. Implementation Phases

1.  **Phase 1: Repo Sync, Validation & Data Pipeline**
    *   Sync template scaffolding (`env/`, `submission/`, `validate.py`, `tests/`, `Dockerfile`, `.gitattributes` LFS) and verify `uv run python validate.py` passes on the baseline.
    *   Build `doom_env` tensorizer (normalizing `rays`, `visible_entities`, `vitals`) and procedural WAD generator (< 2,048 sectors, `freedoom2.wad`, `skill 3`).
2.  **Phase 2: Core Network (Encoders + RSSM)**
    *   Implement Circular 1D Conv Raycast Encoder, Masked Set-Transformer Entity Encoder, and Vitals MLP.
    *   Implement RSSM (GRU + stochastic latent + next-step prediction heads for training).
3.  **Phase 3: Spatial Memory & Reward Shaping**
    *   Implement egocentric velocity-shifted Differentiable Spatial Memory grid.
    *   Implement `privileged_state()` + `sector_polygons()` reward shaper and RSSM curiosity bonus.
4.  **Phase 4: RL + Privileged Distillation & Docker Validation**
    *   Train Teacher (privileged) -> Student (`DoomAgent`) with PPO + KL distillation on Colab A100 across Tier 1 -> Tier 4 procedural WADs.
    *   Export weights via Git LFS to `submission/weights/`, implement `submission/agent.py`, and verify `python validate.py --docker` passes all VRAM/RAM/CPU/latency checks.
