# Track 1 | Can It Play DOOM?

**PLANNED.** Build a submission-ready recurrent agent, then spend the remaining competition budget improving its weakest high-value skill. The target is a policy that can see a locked door, find the key on another branch, return under pressure, and reach the exit on unfamiliar layouts. This document assumes four people and covers the DOOM problem statement supplied by the team.

## 1. Build one agent and one way to judge it

Work in two pairs. One person owns the policy and exported agent, one owns PPO and rollout collection, one owns playable maps and curriculum, and one owns evaluation and release. The first two share one training run. The other two give that run playable scenarios, independent measurements, and a working submission path. Every checkpoint goes through the same observation encoder and action selection used by the submitted agent.

| Stage | Training pair | Maps and evaluation pair | Advance when |
|---|---|---|---|
| Connect | Collect real trajectories, perform an optimizer update, save weights, reload through the official entry point. | Pin the event template revision. Play through one door/exit map and one key/lock map, check event counts, run the starter validator, and time a full evaluation cycle. | Exported weights play two consecutive maps with memory reset. The evaluator distinguishes a pickup, unlock, exit, death, and timeout. |
| Train the MVP | Train a custom telemetry policy on a mixture of combat, door traversal, and short key-return tasks. Save resumable checkpoints. | Run each candidate on fixed development layouts, collect full episodes, and keep one success replay and one common failure. Start the two-page brief with the real baseline architecture. | The functional MVP gate below passes. Preserve that exact checkpoint and config. |
| Compete | Run one challenger against the best saved agent. Select it from the failure diagnosis below. | Compare both on identical maps and seeds. Spend spare evaluation capacity on longer returns, different layout families, and combat during navigation. | Promote or discard the challenger. Repeat with the next bottleneck while a complete comparison and release still fit. |
| Release | Freeze the chosen weights and inference settings. | Evaluate the untouched suite once, finish the brief, validate a fresh clone with LFS, and verify the final default-branch commit. | The public repository, trained weights, training pipeline, validator log, and two-page PDF are available. |

**Budget rule.** Measure end-to-end decisions per second, checkpoint export, development evaluation, and a fresh-clone release rehearsal. Before each challenger, subtract the measured time for final evaluation, packaging, and one retry from the time remaining. Launch only if the candidate can train and be compared inside that budget. Otherwise use the remaining capacity to evaluate the strongest checkpoint across more difficult cases and finish the release. This permits several improvements in succession without putting several untested features into one run.

## 2. Competitive MVP

- **Perception.** Apply circular convolutions to ray distances and hit types and preserve directional bins through fusion so the actor can distinguish left from right. Embed each visible entity's class, bearing, range, and relative motion. Pool with a masked attention query conditioned on vitals and ray features. Supply a defined empty-set output for rooms with no visible entities.
- **Memory and action.** Combine both encoders, normalized vitals, and the previous executed action in a GRU. Produce nine learned binary button decisions. Previous action matters for releasing and pressing USE again. Clear it and recurrent state on each map reset.
- **Learning.** Use recurrent PPO with a value head and generalized advantage estimation. Keep rollout chunks contiguous, carry their initial hidden states, mask episode boundaries, and distinguish a time-limit truncation from death when bootstrapping. Begin with a tiny overfit run on a solvable map to catch broken rewards or recurrence. Then train on mixed layouts.
- **Throughput.** Train on telemetry with rendering disabled where supported. Sweep a small number of CPU worker counts and retain the fastest end-to-end learner configuration. Count policy decisions, completed episodes, and optimizer updates. An A100 helps only to the extent that the CPU workers can supply experience.
- **Reward.** Give the largest bonus for exit, then first pickup, first successful unlock/open per door, and bounded first-visit exploration. Keep kill rewards and damage/death penalties small enough that a reachable exit remains preferable to lingering in combat. Verify this by comparing event totals on completed and stalled runs. Training-side event labels can use documented privileged signals. Ship an actor that accepts only official observations.

**Functional MVP gate.** Before training, register development maps, seeds, and success criteria. Suggested starter gate: on two distinct held-out simple layouts, open a required door and exit on at least two of three seeds per layout. On the combat set, exceed the starter's mean kills with no increase in early deaths. Complete every development episode with valid actions, reload the checkpoint in a clean process, clear memory between maps, and pass the official validator within its sandbox limits. Report key pickups, unlocks and exits on harder tiers even when the counts are zero. These are team acceptance targets, not measured results. A passing checkpoint remains available throughout later experiments.

## 3. Train for the hidden tiers from the first run

Start with a small library of verified map families. Generate variations from each playable family as the trainer consumes them. Confirm door direction, key reachability, and a traversable exit before adding a family. Keep development and final families separate from training by layout structure. Reserve at least one final family from a different map source or construction method where available.

| Family | What to vary | What to record |
|---|---|---|
| Combat and survival | Enemy mix, cover, spawn bearing, resources | Kills, damage, early death; survival alone can reward hiding |
| Multi-room exploration | Loops, dead ends, doorway orientation, exit location | Unique areas, first door crossings, exits, decisions to exit |
| Key and return | Side containing the key, whether the door is seen first, return distance, distractor branches | Key pickup, unlock given pickup, exit given unlock, results by return distance |
| Full gauntlet | Combine familiar mechanics in new arrangements with encounters on the return route | Complete progression funnel, deaths after pickup, exits, remaining health |

Use short tasks to produce early learning signals. Keep sampling mastered tasks while increasing return distance, adding enemies and extending episodes to the official tick caps. Give extra training examples to the first transition that consistently fails on development replays. Use the private training events only to diagnose or teach. The actor must choose its own actions at evaluation.

## 4. Choose the next experiment from the first broken step

| Observed failure | Single next experiment | Evidence for keeping it |
|---|---|---|
| Agent sees the lock and gets the key but loses the return route | **Learned landmark memory**, the preferred architectural upgrade. Keep a small set of persistent slots. Learn writes from ray/entity embeddings and recurrent context, then read slots by attention conditioned on key inventory and current observations. Carry the GRU alongside the slots. | Compare with GRU alone on short and long returns at equal training experience. Measure unlocks per pickup and actual exits. Disable the memory read for a diagnostic replay. |
| Agent rarely sees a key or repeatedly explores the same rooms | Change the curriculum toward branches, loops and first visits. Hold architecture and reward settings fixed for the first comparison. | More keys and exits on unseen branches, with retained performance on previously mastered tasks. |
| Agent collects the key but dies on the return | Add encounters after pickup to the training mixture while keeping navigation layouts comparable. | Fewer post-pickup deaths and more exits, including in the mixed gauntlet. |
| Long-horizon learning stalls despite frequent relevant trajectories | Try a training-only asymmetric critic using documented progression and coverage signals. Keep actor inputs and network fixed. | Faster improvement per environment decision and a stronger exported actor on the full development suite. |

Persistent slots are a proposed alternative to both plans' velocity-warped grid. A landmark can stay in a slot after the agent leaves the local area. The main risk is confusing visually similar branches, so include mirrored corridors and repeated doorways in the comparison. If replays specifically expose directional confusion and another run fits, test the original egocentric grid against the best agent. A full RSSM, surprise reward, or teacher/student run earns a slot only with a distinct failure it could resolve and enough budget for a fair comparison.

## 5. Select and ship

Prioritize exits in the harder tiers, then progression, with exploration and combat as additional signals. Publish per-tier counts and denominators. Before comparing candidates, agree on a local selection rule: first require a clean full run, then compare combined Tier 3–4 exit counts on the same balanced suite, then successful unlocks, then Tier 1–2 exits and combat. Review losses on individual maps. If a gain comes from a single map, repeat on reserve development seeds before promotion. Keep the incumbent when the comparison is inconclusive. The organizers keep exact weights private, so label local scores as proxies. [Official evaluation](https://github.com/Reinforce-SST/devforge-doom-template/blob/main/docs/EVALUATION.md)

For each candidate, save the code revision, checkpoint hash, training decisions, map/seed manifest, per-episode outcomes, and runtime. Run the chosen agent in the official image with one CPU thread, 8 GB RAM, 2 GB VRAM, and the map compute budget based on 40 ms per decision. Record the hardware actually used. Verify initialization, reset, first action and subsequent actions against the pinned limits. From a fresh clone, fetch LFS weights, run the official validator, check the repository including LFS is at most 1 GB, and confirm inference works offline with a read-only repository. Page one of the brief should explain the shipped policy and memory. Page two should show baseline versus final outcomes, runtime, and one informative failure. [Starter and validation commands](https://github.com/Reinforce-SST/devforge-doom-template) · [Agent contract](https://github.com/Reinforce-SST/devforge-doom-template/blob/main/docs/CONTRACT.md)

## 6. What changed and how this differs from existing solutions

| Reference | What we keep | What changes |
|---|---|---|
| Earlier MVP-first execution plan | Four owners, recurrent PPO, explicit fallback, official-image validation | Two integrated workstreams. Short key-return tasks enter the first curriculum. Measured throughput and release cost determine how many successive experiments fit. |
| Earlier world-model strategy | Direction-aware telemetry, partial observability, long-horizon memory as a design goal | Each added module must fix a measured failure. Landmark slots are the first memory candidate and can be tested independently of prediction, curiosity, or distillation. |
| Previous conservative synthesis | A fully functional MVP before feature work | Stronger attention encoder and hard-tier diagnostics from the beginning. Multiple successive promotions are possible. Easy exits qualify the MVP; difficult exits drive subsequent selection. |
| Official starter | Its wrapper, sensor contract, action interface, and validator | The supplied agent samples random buttons. Our deliverables are a trained recurrent policy, task-specific curriculum, cross-layout evaluation, and evidence for the chosen upgrade. [Starter policy](https://github.com/Reinforce-SST/devforge-doom-template/blob/main/submission/agent.py) |
| Sample Factory ViZDoom example | Parallel experience collection and measured learner throughput are relevant engineering ideas | Its published quickstart trains a pixel-based DoomBasic policy. Our target is the official telemetry contract and a multi-room progression task. Returning to a previously sighted door provides a specific behavioral comparison. Sample Factory also supports custom models. This comparison is with its example, and relative performance would require a matched run. [Sample Factory quickstart](https://github.com/alex-petrenko/sample-factory#quickstart) |
