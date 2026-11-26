# Copy-paste prompt for the A100 training CLI

Run the following experiment on an A100 in the Track 1 repository. Execute it, monitor it through completion, and save the commands, decisions, and results under `training/runs/a100_progression`. Start with four training epochs. Here an epoch is about 20 minutes of rollout and PPO updates followed by validation. The budget excludes validation. Finish the active PPO update when the timer expires.

1. Check out `hill-climb/sprint-6-ood` from `https://github.com/RohitBharadwaj-rvu/track-1` and pull Git LFS. Record the commit, GPU, available RAM and disk, and hashes of checkpoint files. Use `uv sync`. Confirm CUDA works and use the checked-in Sprint 5A policy and matching resumable state as epoch-zero incumbent: `submission/weights/policy.pt` and `training/checkpoints/s5a_train_state.pt`. If those files differ in architecture or model tensors, stop and report the mismatch instead of starting from random weights.
2. From the repo root, generate and split the corpus if needed. Keep its manifest, split seed, and audit as run inputs. Use the same manifests across all epochs.

   ```sh
   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
   CORPUS=training/runs/corpus_5k_progression
   RUN=training/runs/a100_progression
   mkdir -p "$RUN"
   uv run python -m training.generate_corpus --count 5000 --out "$CORPUS" --seed 2026 --workers 12 --engine-spotcheck 30
   uv run python -m training.split_corpus "$CORPUS/manifest.jsonl" --seed 2026
   ```

   Check that all four difficulty bands and all three progression archetypes occur in every split, and that map identities and geometry signatures are disjoint. For rollouts use `train.jsonl` and the trainer's registered train maps. The unsplit corpus and unsplit progression manifest contain holdout maps. Use `dev_panel.jsonl` for every epoch's comparison, `dev.jsonl` to confirm the selected checkpoint, and reserve `eval.jsonl` for final evaluation.
3. Evaluate the incumbent before training. Use deterministic CPU inference with seed 0 for each map, and keep the same device, mode, maps, and seed for later comparisons.

   ```sh
   DOOM_AGENT_WEIGHTS="$PWD/submission/weights/policy.pt" uv run python -m training.eval --corpus-manifest "$CORPUS/dev_panel.jsonl" --seeds 0 --device cpu --eval-mode deterministic --out "$RUN/baseline_dev_panel.json"
   ```

4. Train in 20-minute epochs. Start with four subprocess environments, a cache of two environments per worker, a 60-map rotating pool, and a 45% high-difficulty target. Keep automatic curriculum and adaptive sampling enabled. Rotate the working pool every 20 PPO updates so each epoch covers a varied subset. The scheduler balances learning potential and value error and the pool retains some promising maps. Give each epoch a fresh seed. Write exports to an epoch-specific run directory. Start with this command, setting `EPOCH=01`, `SEED=202601`, `HIGH=0.45`, and `RESUME=training/checkpoints/s5a_train_state.pt` for the first epoch.

   ```sh
   uv run python -m training.train --split train --device cuda --seed "$SEED" \
     --corpus-manifest "$CORPUS/train.jsonl" --corpus-pool-size 60 \
     --corpus-high-diff-frac "$HIGH" --procedural-refresh-updates 20 \
     --auto-curriculum --adaptive-weights --adaptive-mode sfl_pvl \
     --num-envs 4 --vec-mode subproc --max-cached-envs 2 \
     --rollout-steps 128 --chunk-len 32 --num-minibatches 4 --update-epochs 4 \
     --total-steps 10000000 --max-train-seconds 1200 --save-every-seconds 1200 \
     --save-interval 10 --no-anneal-lr --lr 0.0001 --resume "$RESUME" \
     --checkpoint-dir "$RUN/epoch_$EPOCH" --export-path "$RUN/epoch_$EPOCH/policy.pt"
   ```

   Retain the timed `update_*` snapshot and the epoch-end `policy.pt` and `latest_train_state.pt`. The periodic update saves provide recovery points within the 20-minute window. If workers run out of memory, preserve the latest readable state and retry that epoch with two workers and one cached environment per worker. Keep the same data split and record the retry.
5. At the end of each epoch, evaluate the export on the frozen dev panel and compare it with the current incumbent's report.

   ```sh
   DOOM_AGENT_WEIGHTS="$PWD/$RUN/epoch_$EPOCH/policy.pt" uv run python -m training.eval --corpus-manifest "$CORPUS/dev_panel.jsonl" --seeds 0 --device cpu --eval-mode deterministic --out "$RUN/epoch_$EPOCH/dev_panel.json"
   uv run python -m training.eval --compare "$INCUMBENT_REPORT" "$RUN/epoch_$EPOCH/dev_panel.json"
   ```

   Start `INCUMBENT_REPORT` at the baseline report. Join the per-episode results to the dev manifest by WAD and map name to track exits, pickups, unlocks, exits after unlock, and deaths by difficulty and archetype. Reject any candidate with runtime errors or compute-budget violations. Rank eligible candidates by Tier 3–4 exits, unlock count, total exits, then fewer deaths. If the advantage is only one episode, confirm on the same panel with seed 1 for both policies before promotion. Update the incumbent report and save its matching optimizer state when promoted.

   Between epochs, change one sampling setting at a time. If easier routes or survival regress, decrease `HIGH` by 0.10 to restore easier exposure. If easier routes hold and the agent gains pickups or unlocks on the harder routes, increase `HIGH` by at most 0.10 to build on that progress. Clamp `HIGH` to 0.25–0.60. If the agent repeatedly unlocks and times out, keep hard exposure steady and increase revisit opportunities by reducing pool size from 60 to 48 for the next epoch. Record the evidence and adjustment. Continue from the latest healthy training state during a plateau, keeping the incumbent available. Restart from the incumbent state after a clear degradation in exits or survival.
6. After epoch four, compare the incumbent and starting baseline on full Dev. Confirm the selection before running it on the held-out eval set. Run final held-out eval once using seed 0 and save the report. Summarize outcomes by difficulty, tier, and progression pattern. Report wall time, decisions, checkpoint hashes, and the complete path to the winning export and resumable state. Run `uv run python validate.py` with the winning checkpoint supplied through `DOOM_AGENT_WEIGHTS`. Keep the experiment artifacts and selection decision for review.
