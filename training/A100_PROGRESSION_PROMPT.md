# Copy-paste prompt for the Sprint 6B A100 hill climb

Run the following experiment on an A100 in the Track 1 repository. Execute it, monitor it through completion, and save the commands, decisions, and results under `training/runs/a100_progression`. Run four hill-climb rounds. Each round starts from the current accepted incumbent, trains one challenger for about 20 minutes of rollout and PPO updates, and evaluates it for promotion. The budget excludes validation. Finish the active PPO update when the timer expires.

1. Check out `hill-climb/sprint-6-ood` from `https://github.com/RohitBharadwaj-rvu/track-1` and pull Git LFS. Record the commit, GPU, available RAM and disk, and hashes of checkpoint files. Use `uv sync`. Confirm CUDA works and use the checked-in Sprint 6B policy and optimizer state as the initial incumbent: `training/checkpoints/s6b_policy.pt` and `training/checkpoints/s6b_train_state.pt`. If those files differ in architecture or model tensors, stop and report the mismatch instead of starting from random weights.
2. From the repo root, generate and split the corpus if needed. Keep its manifest, split seed, and audit as run inputs. Use the same manifests across all epochs.

   ```sh
   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
   CORPUS=training/runs/corpus_5k_progression
   RUN=training/runs/a100_progression
   mkdir -p "$RUN"
   BASELINE_POLICY=training/checkpoints/s6b_policy.pt
   INCUMBENT_POLICY="$BASELINE_POLICY"
   INCUMBENT_STATE=training/checkpoints/s6b_train_state.pt
   INCUMBENT_REPORT="$RUN/baseline_dev_panel.json"
   HIGH=0.45
   POOL=60
   uv run python -m training.generate_corpus --count 5000 --out "$CORPUS" --seed 2026 --workers 12 --engine-spotcheck 30
   uv run python -m training.split_corpus "$CORPUS/manifest.jsonl" --seed 2026
   ```

   Check that all four difficulty bands and all three progression archetypes occur in every split, and that map identities and geometry signatures are disjoint. For rollouts use `train.jsonl` and the trainer's registered train maps. The unsplit corpus and unsplit progression manifest contain holdout maps. Use `dev_panel.jsonl` for every epoch's comparison, `dev.jsonl` to confirm the selected checkpoint, and reserve `eval.jsonl` for final evaluation.
3. Evaluate the incumbent before training. Use deterministic CPU inference with seed 0 for each map, and keep the same device, mode, maps, and seed for later comparisons.

   ```sh
   DOOM_AGENT_WEIGHTS="$PWD/$INCUMBENT_POLICY" uv run python -m training.eval --corpus-manifest "$CORPUS/dev_panel.jsonl" --seeds 0 --device cpu --eval-mode deterministic --out "$RUN/baseline_dev_panel.json"
   ```

4. Train in 20-minute epochs. Start with four subprocess environments, a cache of two environments per worker, a 60-map rotating pool, and a 45% high-difficulty target. Keep automatic curriculum and adaptive sampling enabled. Rotate the working pool every 20 PPO updates so each epoch covers a varied subset. The scheduler balances learning potential and value error and the pool retains some promising maps. Give each epoch a fresh seed. Write exports to an epoch-specific run directory. For the first round, set `EPOCH=01` and `SEED=202601`. Increment both for each new challenger. Resume each round from `INCUMBENT_STATE`. Keep the restored optimizer learning rate (Sprint 6B starts at 0.00005) with `--no-anneal-lr`.

   ```sh
   uv run python -m training.train --split train --device cuda --seed "$SEED" \
     --corpus-manifest "$CORPUS/train.jsonl" --corpus-pool-size "$POOL" \
     --corpus-high-diff-frac "$HIGH" --procedural-refresh-updates 20 \
     --auto-curriculum --adaptive-weights --adaptive-mode sfl_pvl \
     --num-envs 4 --vec-mode subproc --max-cached-envs 2 \
     --rollout-steps 128 --chunk-len 32 --num-minibatches 4 --update-epochs 4 \
     --total-steps 10000000 --max-train-seconds 1200 --save-every-seconds 1200 \
     --save-interval 10 --no-anneal-lr --resume "$INCUMBENT_STATE" \
     --checkpoint-dir "$RUN/epoch_$EPOCH" --export-path "$RUN/epoch_$EPOCH/policy.pt"
   ```

   Retain the timed `update_*` snapshot and the epoch-end `policy.pt` and `latest_train_state.pt`. The periodic update saves provide recovery points within the 20-minute window. If workers run out of memory, preserve the latest readable state and retry that epoch with two workers and one cached environment per worker. Keep the same data split and record the retry.
5. At the end of each epoch, evaluate the export on the frozen dev panel and compare it with the current incumbent's report.

   ```sh
   DOOM_AGENT_WEIGHTS="$PWD/$RUN/epoch_$EPOCH/policy.pt" uv run python -m training.eval --corpus-manifest "$CORPUS/dev_panel.jsonl" --seeds 0 --device cpu --eval-mode deterministic --out "$RUN/epoch_$EPOCH/dev_panel.json"
   uv run python -m training.eval --compare "$INCUMBENT_REPORT" "$RUN/epoch_$EPOCH/dev_panel.json" > "$RUN/epoch_$EPOCH/comparison.json"
   uv run python -m training.eval --compare-nash "$INCUMBENT_REPORT" "$RUN/epoch_$EPOCH/dev_panel.json" > "$RUN/epoch_$EPOCH/nash_comparison.txt"
   ```

   Join the per-episode results to the Dev manifest by WAD and map name. Track pickups, unlocks,
   exits after unlock, and deaths by difficulty and archetype. Reject a challenger with runtime
   errors or compute-budget violations. Require a strict improvement using the following order:
   Tier 3?4 exits, unlock count, total exits, then fewer deaths. Leave the incumbent in place on
   ties. Use Nash/minimax-regret to identify the maps behind the change, and CVaR-25 and the worst
   rubric weighting to check tail risk. If the gain is only one episode or the robustness metrics
   suggest a trade-off, compare both policies on seed 1 and apply the same ranking to the combined
   results. Promote only if the combined improvement holds. Save the seed-1 reports separately so
   the next round still has a matched seed-0 comparison.

   On promotion, set `INCUMBENT_POLICY`, `INCUMBENT_STATE`, and `INCUMBENT_REPORT` to that round's
   `policy.pt`, `latest_train_state.pt`, and seed-0 `dev_panel.json`. On rejection, keep all three
   incumbent paths and start the next challenger from that accepted optimizer state. Record the
   decision, metric deltas, and checkpoint hashes.

   Between rounds, adjust one sampling setting based on the Dev results. If easier routes or
   survival regress, decrease `HIGH` by 0.10. If easier routes hold and hard routes gain pickups or
   unlocks, increase `HIGH` by up to 0.10. Keep `HIGH` between 0.25 and 0.60. If the agent repeatedly
   unlocks and then times out, hold `HIGH` steady and set `POOL=48` to revisit training maps more
   often. Use Dev outcomes for promotion and these sampling decisions. The training rewards come
   from rollouts on Train.
6. After round four, evaluate the accepted incumbent and the original Sprint 6B policy on full Dev. Apply the same promotion criteria to confirm the gain. If the gain fails confirmation, select Sprint 6B. Run held-out Eval once on the selected policy using seed 0 and save the report. Summarize outcomes by difficulty, tier, and progression pattern. Report wall time, decisions, checkpoint hashes, and the complete path to the winning export and resumable state. Run `uv run python validate.py` with the selected checkpoint's absolute path supplied through `DOOM_AGENT_WEIGHTS`. Keep the experiment artifacts and selection decision for review.
