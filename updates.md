# Track 1 Technical Updates: Rule-Compliant Accuracy & Performance Improvements

This guide explains five legal changes that improve how accurately your Reinforcement Learning (RL) agent plays unseen DOOM levels. Other teams using the same starter template will likely train a basic policy on a few maps. These five changes target the exact failure modes of the official environment while following [docs/RULES.md](docs/RULES.md) and [docs/CONTRACT.md](docs/CONTRACT.md).

---

## 1. Increase Training Map Diversity Before Adding Model Complexity

**Confirmed behavior:** The evaluation server tests every agent on unseen maps ([docs/EVALUATION.md](docs/EVALUATION.md)). An agent trained on a small set of fixed maps memorizes fixed turns and stalls when room layouts change.

**Recommendation:** Role 3 should generate many varied WAD (DOOM map) files for training. Vary the following properties across maps:
* Room sizes and corridor shapes (loops, dead ends, and branching paths)
* Left-versus-right placement of keys and locked doors
* Door orientations and enemy placements
* Keycard colors (blue, yellow, and red)

**Effect:** When layouts change every episode, the neural network must rely on its live sensors (`rays` and `visible_entities`) rather than memorized paths.

---

## 2. Prevent Invalid and Self-Canceling Button Combinations

**Confirmed behavior ([docs/CONTRACT.md](docs/CONTRACT.md#L74-L100)):** The environment expects a 9-element binary action vector (`MultiBinary(9)`). Two game rules cause common failures:
1. Pressing opposite movement buttons together (such as `MOVE_FORWARD` and `MOVE_BACKWARD`, or `TURN_LEFT` and `TURN_RIGHT`) cancels movement and wastes decision steps.
2. The `USE` button triggers only when its value changes from `0` to `1`. Pressing `USE` while a door is already opening reverses the door and closes it.

**Recommendations:**
* **Structured action heads (Role 1):** Instead of 9 independent binary outputs, have the neural network output three 3-way choices (`forward / backward / none`, `strafe left / strafe right / none`, `turn left / turn right / none`) plus three independent binary outputs (`ATTACK`, `USE`, `SELECT_NEXT_WEAPON`). Convert those outputs into the required 9-element binary array. All decisions still come from trained neural parameters, which satisfies [docs/RULES.md](docs/RULES.md) §1.
* **Button-edge context and penalty (Roles 1 and 2):** Pass the previous action (`prev_action`) into the Gated Recurrent Unit (GRU, a recurrent neural layer that keeps short-term state across steps). In `training/rewards.py`, apply a small penalty when the agent presses `USE` repeatedly without opening a door or triggering a switch.

**Effect:** The agent stops wasting steps standing still and stops closing doors that are already opening.

---

## 3. Train with an Asymmetric Critic

**Confirmed behavior ([docs/RULES.md](docs/RULES.md) §2 and [docs/CONTRACT.md](docs/CONTRACT.md) §7):** During training, `env.privileged_state()` provides exact ground-truth information such as player coordinates, kills, and door heights. During evaluation, the agent receives only ego-centric sensor observations.

**Recommendation (Role 2):** In Proximal Policy Optimization (PPO, the RL training algorithm), split the inputs between the two networks:
* **Actor network (chooses actions):** Receives only the official ego-centric observations (`rays`, `visible_entities`, `vitals`, and `prev_action`).
* **Critic network (estimates future reward during training):** Receives the same observations plus privileged progress signals from `env.privileged_state()` (such as unique grid cells visited, doors opened, and key status).

At submission time, export only the Actor network.

**Effect:** The Critic produces more accurate reward estimates on long maps, which helps the Actor learn faster during training without breaking evaluation rules.

---

## 4. Add Persistent Memory Only After Measuring a Memory Failure

**Confirmed behavior ([docs/RULES.md](docs/RULES.md) §5):** Final judging rewards custom architectures such as spatial memory or curiosity modules.

**Assumption:** A standard GRU often loses track of a locked door after several hundred steps of searching another branch for a keycard.

**Recommendation (Roles 1 and 4):**
1. Train and save a working Minimum Viable Product (MVP) agent that uses a GRU alone.
2. Evaluate the GRU agent on held-out key-and-return maps using [training/eval.py](training/eval.py).
3. If the agent picks up keys (`keys_picked_up > 0`) but fails to unlock doors (`locked_doors_unlocked == 0`), add **learned landmark memory slots** (a small set of persistent neural memory vectors that the network writes to and reads from using attention).
4. Compare the GRU agent and the memory-slot agent on the exact same maps and seeds using `python -m training.eval --compare`.

**Effect:** You keep the memory module only if it improves measured Tier 3–4 unlocks or exits, and you gain verified before-and-after results for [docs/BRIEF.md](docs/BRIEF.md).

---

## 5. Diagnose Failures by Event Counts Instead of Total Reward

**Confirmed behavior:** Total training reward combines exploration, combat, progression, and exit bonuses into one number. A flat reward curve does not show which specific skill failed.

**Recommendation (Role 4):** After each training run, run [training/eval.py](training/eval.py) and inspect the event counts in order:
1. **`doors_opened == 0`:** The agent is not opening doors. Check `USE` timing, door-bounce penalties, and door rewards.
2. **`unique_cells_visited` is low:** The agent is staying in one area. Increase first-visit exploration rewards or add branching training maps.
3. **`keys_picked_up > 0` and `locked_doors_unlocked == 0`:** The agent finds keys but cannot return to the locked door. Add key-return maps or test landmark memory slots.
4. **`deaths` increase after key pickup:** Enemies on the return route kill the agent. Add combat encounters on the return path during training.

**Effect:** Each training iteration fixes one verified bottleneck instead of changing multiple hyperparameters at once.
