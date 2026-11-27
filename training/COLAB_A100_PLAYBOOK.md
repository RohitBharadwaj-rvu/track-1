# Colab A100 Allocation & Session-Survival Playbook (`colab-cli`)

> Operational rules for acquiring and keeping a Google Colab A100 session alive during Track 1 training runs. Adhere strictly to every rule below throughout the entire run.

---

## 1. Getting an A100 (A100-Only, No Fallbacks, Zero Token Waste)

- **Never fall back to L4/T4/CPU.** A100 `503 Service Unavailable` comes in waves and clears quickly once a slot opens (we saw 14 consecutive 503s, then succeeded in 4 and 6 attempts shortly after).
- **Disable `colab-cli`'s internal retries and run your own infinite 45s loop** in a single background process with **no time cap** and **no periodic agent polling/timers** (so zero tokens are spent while waiting; the task wakes you only when `colab new` exits `0`):
  - Set `COLAB_CLI_ASSIGN_RETRIES=0` and `COLAB_CLI_RETRY_BACKOFF=0`.
  - Loop `colab new -s sprint-a100 --gpu A100` every **45 seconds** until exit code `0`.
- **Critical traceback bug to avoid:** Do **not** grep the full output of `colab new` for `TooManyAssignmentsError` to abort — `colab-cli`'s Rich traceback prints the source line `except TooManyAssignmentsError:` even on a normal `503 Service Unavailable`. Only check the **last 4 non-empty lines**, and keep retrying whenever `Service Unavailable` appears (only stop on real auth errors like `invalid_grant`, `Unauthorized`, or `401`).
- **Syntax gotcha:** `colab exec` requires `-f <script.py>` (passing inline code as a positional arg fails).

---

## 2. Keeping the Allocated A100 Alive No Matter What

1. **NEVER call `colab stop`** (not between runs, not in a `finally` block — this overrides any generic skill default). Reuse the exact same named session (`-s sprint-a100`) for every command, and update code in-place inside the VM (`git fetch && git reset --hard ...`) instead of recreating the runtime.
2. **Always keep an active streaming `colab exec --timeout 0` websocket attached.** If the local websocket drops (e.g., local process dies/restarts), Google Colab treats the runtime as a detached headless session and terminates the VM **~20–30 minutes later**, even while GPU training is still actively running.
3. **Run work in 20–35 minute chunks with frequent `flush=True` output** (at least every ~30–60s) so the websocket stays live and never hits an inactivity drop.
4. **NEVER queue a second `colab exec` while the kernel is `BUSY`.** A queued `colab exec` times out and throws `RuntimeError: Connection was lost.` after ~6 minutes. To check progress on a running job without touching the kernel, use the **HTTP-only commands**:
   - `colab status -s sprint-a100`
   - `colab ls -s sprint-a100 /path/on/vm`
   - `colab download -s sprint-a100 /path/on/vm/file ./local_file`
5. **Do not leave the session completely idle when paused.** Colab reclaims completely idle runtimes after ~30–90 minutes. Keep turnaround between runs short (<15 min), or if you must pause longer, run a tiny keepalive script via `colab exec --timeout 0 -s sprint-a100 -f keepalive.py` that prints a heartbeat every ~60s (and kill just that keepalive before starting the next run — never `colab stop`).
6. **Always copy artifacts off `/content` immediately after every chunk.** `/content` is ephemeral — if Google ever preempts the VM unexpectedly, everything on `/content` vanishes instantly. Chain `colab download` right after every `colab exec` (and/or mount Google Drive via `colab drivemount`) so you never lose progress even if Google recycles the machine.
