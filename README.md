# Track 1: Telemetry-Driven Recurrent DOOM Agent

Submission for **Yugaantar 2026 · DevForge — Track 1: "Can It Play DOOM?"**

Our agent plays unseen `freedoom2.wad` (`skill 3`) levels zero-shot using `doom_env`'s ego-centric telemetry (`rays`, `visible_entities`, `vitals`) paired with a pure-NumPy odometric 2D spatial memory (`EgocentricSpatialMap`) and a custom Recurrent PPO pipeline.

- **2-Page Project Brief:** [docs/BRIEF.pdf](docs/BRIEF.pdf) (source: [docs/BRIEF.md](docs/BRIEF.md))
- **Checkpoint Promotion Playbook:** [docs/CHECKPOINT_PROMOTION.md](docs/CHECKPOINT_PROMOTION.md)
- **Failure Analysis:** [improvements/2026-11-27_04-48/SPRINT6B_FAILURE_ANALYSIS.md](improvements/2026-11-27_04-48/SPRINT6B_FAILURE_ANALYSIS.md)

---

## Current Submission Checkpoint

- **Checkpoint:** `Sprint 6B` (`training/checkpoints/s6b_policy.pt` -> `submission/weights/policy.pt`, tracked via Git LFS)
- **Architecture:** `telemetry_gru_v1` + `EgocentricSpatialMap` (`884,133` parameters, `3.55 MB`)
- **Inference config:** `eval_deterministic=True`, `rocket_safety_mask=True`, `eval_mirror_ensemble=False`
- **Checksums:** source `cbe1ca42c08dd378`, promoted `06761f9a68f11ea8`

### Held-Out Proxy Benchmarks (Sprint 6B, Deterministic)

| Suite | Episodes | Exits | Keys / Unlocks | Kills | Mean Cells | Proxy Score | Report |
|---|---:|---:|---:|---:|---:|---:|---|
| `dev` (10 maps) | 20 | 19 (95.0%) | 10 / 10 | 16 | 42.35 | 0.9792 | [eval_s6b_dev_det.json](training/results/sprint6b-2026-11-27/eval_s6b_dev_det.json) |
| `stress` (14 probes) | 28 | 23 (82.1%) | 16 / 16 | 27 | 37.04 | 0.8750 | [eval_s6b_stress.json](training/results/sprint6b-2026-11-27/eval_s6b_stress.json) |
| `final_ood` (8 maps) | 16 | 1 (6.2%) | 4 / 2 | 32 | 54.88 | 0.0312 | [eval_s6b_final_ood.json](training/results/sprint6b-2026-11-27/eval_s6b_final_ood.json) |
| `freedoom` (`MAP01–08`) | 8 | 0 (0.0%) | 0 / 0 | 31 | 89.38 | 0.0000 | [eval_s6b_freedoom.json](training/results/sprint6b-2026-11-27/eval_s6b_freedoom.json) |

---

## Architecture Overview

1. **Circular Raycast Encoder (`submission/policy.py`):** Circular 1D convolutions over 64 ego-centric rays (normalized distance + 6-class hit type) preserving 360-degree directional structure.
2. **Combat-Aware Entity Attention (`submission/policy.py`, `submission/obs.py`):** 12-class entity embeddings + ego-frame kinematics (closing/lateral speed, inverse TTI, crosshair alignment) + category counts, pooled via 4-head / 4-query masked attention with deterministic empty-set fallback.
3. **Odometric 2D Spatial Memory (`submission/ego_map.py`):** Pure-NumPy `128x128` backing grid updated strictly from legal observations (`rays`, `visible_entities`, `vitals`) and `prev_action` (zero privileged state, zero pathfinding). Renders an ego-aligned `(8, 25, 25)` map crop with off-crop landmark beacons into a compact CNN.
4. **Residual GRU & Factorized Actor (`submission/policy.py`, `submission/agent.py`):** 256-dim residual GRU with explicit ammo/health/speed and `prev_action` inputs, outputting 3x3-way categorical movement/turn heads + 3 Bernoulli heads (`ATTACK`, `USE`, `SELECT_NEXT_WEAPON`) mapped to `MultiBinary(9)` with an observation-based close-range rocket safety mask.

---

## Quickstart & Validation

```bash
uv sync
uv run pytest
uv run python validate.py
```

Official Linux sandbox validation (matches evaluation 1-CPU, RAM, read-only, offline constraints):

```bash
docker build -t devforge-doom .
uv run python validate.py --docker --image devforge-doom
```

Fresh-clone LFS & release pre-flight check:

```bash
uv run python -m training.release_check --fresh-clone
```

---

## Training, Evaluation & Promoting New Checkpoints

Generate the procedural progression corpus and split into train/dev/eval:

```bash
uv run python -m training.generate_corpus --count 5000 --out training/runs/corpus_5k_progression --workers 12 --engine-spotcheck 30
uv run python -m training.split_corpus training/runs/corpus_5k_progression/manifest.jsonl
```

Evaluate the current submission weights on held-out suites:

```bash
uv run python -m training.eval --suite dev       --eval-mode default --out eval_dev.json
uv run python -m training.eval --suite stress    --eval-mode default --out eval_stress.json
uv run python -m training.eval --suite final_ood --eval-mode default --out eval_final_ood.json
uv run python -m training.eval --suite freedoom  --eval-mode default --out eval_freedoom.json
```

When a newer checkpoint or model soup beats the incumbent, follow [docs/CHECKPOINT_PROMOTION.md](docs/CHECKPOINT_PROMOTION.md):

```bash
uv run python -m training.promote_checkpoint training/checkpoints/<new_policy>.pt --label "<Run Name>"
uv run python -m training.brief_tools --export-html docs/BRIEF.html --export-pdf docs/BRIEF.pdf
```

---

## Repository Layout

```text
submission/           DoomAgent entry point, TelemetryPolicy, EgocentricSpatialMap, LFS weights
training/             Custom Recurrent PPO, PBRS rewards, curiosity, procedural WAD generator, eval, promotion tools
env/                  Official ViZDoom Gymnasium wrapper (doom_env, unmodified)
maps/                 Synthetic train, dev, stress, final, and final_ood WAD suites
tests/                Unit, contract, env, eval, PPO, and validator test suites
docs/                 2-page brief (BRIEF.md/html/pdf), promotion guide, official contract/rules/submission docs
validate.py           Official local & Docker submission validator
```

## Documentation

| Doc | Contents |
| --- | --- |
| [docs/BRIEF.pdf](docs/BRIEF.pdf) | 2-page Project Brief (architecture, USP, training, benchmarks, failure analysis) |
| [docs/CHECKPOINT_PROMOTION.md](docs/CHECKPOINT_PROMOTION.md) | Step-by-step guide to swap submission checkpoints and refresh brief/README |
| [docs/CONTRACT.md](docs/CONTRACT.md) | Official observation/action contract and `DoomAgent` interface |
| [docs/RULES.md](docs/RULES.md) | Official ground rules and fair-play requirements |
| [docs/SUBMISSION.md](docs/SUBMISSION.md) | Official submission checklist, Git LFS, 1 GB limit, `requirements.txt` |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Hidden map tiers, scoring categories, hardware limits, timeouts |
