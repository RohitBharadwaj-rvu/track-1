"""Role 4 release pre-flight check (Git LFS, submission cleanliness, fresh clone, and validate.py).

Usage:
    python -m training.release_check
    python -m training.release_check --fresh-clone
    python -m training.release_check --docker
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
WEIGHT_EXTS = {".pt", ".pth", ".safetensors", ".ckpt", ".npz", ".msgpack", ".onnx"}


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd or ROOT), capture_output=True, text=True)


def _rmtree_force(path: Path) -> None:
    def _handle_readonly(func, target, exc_info):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError:
            pass

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_handle_readonly)
    else:
        shutil.rmtree(path, onerror=_handle_readonly)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def check_lfs_and_submission(repo: Path = ROOT) -> tuple[bool, list[str]]:
    """Verify Git LFS setup, LFS tracking for weights, clean submission/ tree, and brief artifacts."""
    notes: list[str] = []
    ok = True

    lfs_ver = _run(["git", "lfs", "version"], cwd=repo)
    if lfs_ver.returncode != 0:
        ok = False
        notes.append("FAIL: git-lfs is not available on PATH.")
    else:
        notes.append(f"OK: {lfs_ver.stdout.strip().splitlines()[0]}")

    lfs_files_res = _run(["git", "-C", str(repo), "lfs", "ls-files", "--name-only"], cwd=repo)
    lfs_tracked = {line.strip().replace("\\", "/") for line in lfs_files_res.stdout.splitlines() if line.strip()}

    weights_dir = repo / "submission" / "weights"
    if weights_dir.is_dir():
        weight_files = [
            p for p in weights_dir.rglob("*")
            if p.is_file() and p.name != ".gitkeep"
        ]
        if not weight_files:
            notes.append("INFO: submission/weights/ has no weight files yet (Stage 1 baseline).")
        for wf in weight_files:
            rel = str(wf.relative_to(repo)).replace("\\", "/")
            if wf.suffix.lower() in WEIGHT_EXTS and rel not in lfs_tracked:
                ok = False
                notes.append(f"FAIL: Weight file '{rel}' is not tracked by Git LFS.")
            else:
                notes.append(f"OK: Weight file '{rel}' checked (LFS tracked: {rel in lfs_tracked}).")

    status_res = _run(["git", "-C", str(repo), "status", "--porcelain", "submission/"], cwd=repo)
    dirty = [ln for ln in status_res.stdout.splitlines() if ln.strip()]
    if dirty:
        notes.append("WARN: Uncommitted/untracked files in submission/ (commit before final release):")
        notes.extend(f"  {ln}" for ln in dirty)
    else:
        notes.append("OK: submission/ working tree is clean.")

    brief_md = repo / "docs" / "BRIEF.md"
    brief_pdf = repo / "docs" / "BRIEF.pdf"
    if brief_md.is_file():
        notes.append("OK: docs/BRIEF.md exists.")
    else:
        notes.append("WARN: docs/BRIEF.md is missing.")
    if brief_pdf.is_file():
        notes.append("OK: docs/BRIEF.pdf exists.")
    else:
        notes.append("INFO: docs/BRIEF.pdf not generated yet (run `python -m training.brief_tools --export-pdf` before Stage 4 release).")

    return ok, notes


def check_docker_daemon() -> tuple[bool, str]:
    """Verify the Docker daemon is reachable before running `validate.py --docker`."""
    res = _run(["docker", "info", "--format", "{{.ServerVersion}}"])
    if res.returncode != 0:
        err = (res.stderr or res.stdout).strip().splitlines()
        detail = err[-1] if err else "daemon not reachable"
        return False, f"FAIL: Docker daemon is not running ({detail}). Start Docker Desktop and retry."
    return True, f"OK: Docker daemon ready (server {res.stdout.strip()})."


def run_fresh_clone_rehearsal(args: argparse.Namespace) -> int:
    """Clone the local repository into a temp directory and run LFS + validate.py checks."""
    t0 = time.perf_counter()
    tmp_dir = Path(tempfile.mkdtemp(prefix="track1_fresh_"))
    clone_dir = tmp_dir / "repo"
    try:
        print(f"\n=== Fresh-Clone Rehearsal ({clone_dir}) ===")
        clone_res = _run(["git", "clone", "--no-hardlinks", str(ROOT), str(clone_dir)], cwd=tmp_dir)
        if clone_res.returncode != 0:
            print(f"FAIL: git clone failed:\n{clone_res.stderr}")
            return 1

        lfs_ok, notes = check_lfs_and_submission(clone_dir)
        for note in notes:
            print(note)

        val_cmd = [sys.executable, str(clone_dir / "validate.py")]
        if args.docker:
            val_cmd.append("--docker")
        if args.image:
            val_cmd.extend(["--image", args.image])
        for m in args.map or []:
            val_cmd.extend(["--map", m])

        print(f"Running validate.py in fresh clone: {' '.join(val_cmd)}")
        val_res = subprocess.run(val_cmd, cwd=str(clone_dir))
        elapsed = time.perf_counter() - t0
        print(f"Fresh-clone rehearsal finished in {elapsed:.1f}s (exit={val_res.returncode}).")
        if not lfs_ok:
            return 1
        return val_res.returncode
    finally:
        _rmtree_force(tmp_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Role 4 release pre-flight check.")
    parser.add_argument("--docker", action="store_true", help="Run validate.py --docker")
    parser.add_argument("--image", default=None, help="Docker image name passed to validate.py --image")
    parser.add_argument("--map", action="append", default=None, help="Extra map(s) passed to validate.py --map")
    parser.add_argument(
        "--fresh-clone",
        action="store_true",
        help="Rehearse LFS + validate.py inside a fresh temporary git clone",
    )
    args = parser.parse_args(argv)

    print("=== Role 4 Release Pre-Flight ===")
    if args.docker:
        docker_ok, docker_msg = check_docker_daemon()
        print(docker_msg)
        if not docker_ok:
            return 1

    if args.fresh_clone:
        return run_fresh_clone_rehearsal(args)

    lfs_ok, notes = check_lfs_and_submission(ROOT)
    for note in notes:
        print(note)

    val_cmd = [sys.executable, str(ROOT / "validate.py")]
    if args.docker:
        val_cmd.append("--docker")
    if args.image:
        val_cmd.extend(["--image", args.image])
    for m in args.map or []:
        val_cmd.extend(["--map", m])

    print(f"\n=== Running: {' '.join(val_cmd)} ===")
    val_res = subprocess.run(val_cmd, cwd=str(ROOT))
    if not lfs_ok:
        return 1
    return val_res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
