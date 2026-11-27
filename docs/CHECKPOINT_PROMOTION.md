# Checkpoint Promotion & Submission Update Guide

Follow this checklist whenever a new checkpoint (or model soup) replaces the current `submission/weights/policy.pt`. Every step keeps the default branch (`master`), Git LFS weights, `README.md`, and the 2-page `docs/BRIEF.pdf` in sync.

## 1. Promote the checkpoint to `submission/weights/policy.pt`

Use `training/promote_checkpoint.py` so the weights stay tracked by Git LFS and have the intended inference flags baked into `PolicyConfig`:

```bash
uv run python -m training.promote_checkpoint training/checkpoints/<new_policy>.pt \
  --label "<Sprint / Run Name>" \
  --deterministic \
  --rocket-safety-mask \
  --no-mirror-ensemble
```

Verify Git LFS picked it up:

```bash
git lfs ls-files
```

`submission/weights/policy.pt` must appear in the output.

## 2. Re-evaluate the promoted checkpoint

Run the standard held-out suites against `submission/weights/policy.pt` (using the baked-in `--eval-mode default` so you test the exact behavior the grader sees):

```bash
OUT_DIR="training/results/<run-id>"
mkdir -p "$OUT_DIR"

uv run python -m training.eval --suite dev       --eval-mode default --out "$OUT_DIR/eval_dev.json"
uv run python -m training.eval --suite stress    --eval-mode default --out "$OUT_DIR/eval_stress.json"
uv run python -m training.eval --suite final_ood --eval-mode default --out "$OUT_DIR/eval_final_ood.json"
uv run python -m training.eval --suite freedoom  --eval-mode default --out "$OUT_DIR/eval_freedoom.json"
```

Add `--device cuda` when running on a GPU machine.

## 3. Update the 2-page brief (`docs/BRIEF.md`, `docs/BRIEF.html`, `docs/BRIEF.pdf`)

1. Update the Outcomes table in `docs/BRIEF.md` (either edit directly or run `training.brief_tools`):

   ```bash
   uv run python -m training.brief_tools \
     --update-row "Final Shipped Agent (<label>)" \
     --report "$OUT_DIR/eval_dev.json" \
     --suite-label "10 dev maps (20 eps)"
   ```

2. Update any changed numbers in `docs/BRIEF.md`:
   - Checkpoint label, parameter count, and architecture changes on Page 1 (if architecture changed).
   - Stress, `final_ood`, and `freedoom` summary metrics on Page 2.
   - `validate.py` / `validate.py --docker` timing, RAM, and VRAM numbers on Page 2.
   - Failure analysis section if a previously documented failure mode was resolved.
3. Re-export the HTML and 2-page PDF:

   ```bash
   uv run python -m training.brief_tools --export-html docs/BRIEF.html --export-pdf docs/BRIEF.pdf
   ```

4. Open `docs/BRIEF.pdf` and confirm it stays strictly **2 pages** (no spillover onto page 3).

## 4. Update `README.md`

Update the **Current Submission Checkpoint** block and benchmark table in `README.md` with the new checkpoint label, source path, SHA-256 prefix, inference flags, and eval results.

## 5. Run tests and official validation

```bash
uv run pytest
uv run python validate.py
uv run python -m training.release_check --fresh-clone
```

On a Linux machine or with Docker Desktop running, run the sandbox check that matches the grader (1 CPU thread, Linux `/proc` RAM, read-only repo, offline):

```bash
docker build -t devforge-doom .
uv run python validate.py --docker --image devforge-doom
```

## 6. Commit and push to `master`

The automated grader evaluates the **last commit on `master` at the deadline** (`docs/EVALUATION.md`).

```bash
git checkout master
git add submission/weights/policy.pt docs/BRIEF.md docs/BRIEF.html docs/BRIEF.pdf README.md
git status
git commit -m "Promote <label> to submission checkpoint and refresh brief"
git push origin master
```

After pushing, confirm:
- `git status` is clean under `submission/`.
- Total tracked size reported by `validate.py` stays well under `1.0 GB`.
- If the portal accepts a separate PDF upload, re-upload the refreshed `docs/BRIEF.pdf`.
