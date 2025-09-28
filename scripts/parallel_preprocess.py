#!/usr/bin/env python3
"""
Run LipBengal preprocessing in parallel shards with live progress bars.

- Spawns N shard workers of scripts/prepare_lipbengal.py (with --num_shards/--shard)
- Limits per-worker threads to keep CPU headroom (OMP/MKL/OPENBLAS/NUMEXPR=1)
- Shows per-shard tqdm bars and a combined total bar (fast and visible)

Usage examples:
  python scripts/parallel_preprocess.py --split val --shards 3 --overwrite
  python scripts/parallel_preprocess.py --split test --shards 4 --overwrite
  python scripts/parallel_preprocess.py --split train --shards 3 --overwrite

Notes:
  - Default shards ~= 80% of CPU cores but capped to 4 to avoid GPU pressure.
  - Uses logs/parallel_{split}_shard{i}.log to parse progress (non-blocking).
  - Requires tqdm and torch to be installed (torch for reading indices count).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import math
import subprocess
from pathlib import Path
from typing import List, Tuple

try:
    import torch
except Exception:
    torch = None  # type: ignore

try:
    from tqdm import tqdm
except Exception as e:
    print("tqdm is required for live progress. Please install it (pip install tqdm).", file=sys.stderr)
    sys.exit(1)


def cpu_shards(default_cap: int = 4) -> int:
    cores = os.cpu_count() or 2
    target = max(2, int(math.ceil(cores * 0.8)))
    return max(2, min(default_cap, target))


def load_split_counts(root: Path, split: str, shards: int) -> List[int]:
    """Load total item count and compute per-shard counts by index modulo assignment.
    Relies on datasets/LipBengal/indices/{split}.pt saved as a list of dicts.
    """
    idx_path = root / "indices" / f"{split}.pt"
    assert idx_path.exists(), f"Indices not found: {idx_path}"
    assert torch is not None, "torch is required to read index .pt files"
    items = torch.load(idx_path)
    totals = [0 for _ in range(shards)]
    for i, _ in enumerate(items):
        totals[i % shards] += 1
    return totals


def spawn_workers(
    split: str,
    shards: int,
    root: Path,
    overwrite: bool,
    drop_on_fail: bool,
    fail_list_dir: str,
) -> Tuple[List[subprocess.Popen], List[Path]]:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    procs: List[subprocess.Popen] = []
    log_paths: List[Path] = []
    for i in range(shards):
        log_path = logs_dir / f"parallel_{split}_shard{i}.log"
        # Truncate log
        log_path.write_text("")
        env = os.environ.copy()
        # Keep total CPU usage reasonable by limiting per-process threads
        env.update({
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        })
        args = [
            sys.executable,
            str(Path(__file__).with_name("prepare_lipbengal.py")),
            "--root", str(root),
            "--align",
            "--landmarks", "lipcrop",
            "--splits", split,
            "--split_mode", os.environ.get("LB_SPLIT_MODE", "by_speaker"),
            "--split_seed", os.environ.get("LB_SPLIT_SEED", "42"),
            "--num_shards", str(shards),
            "--shard", str(i),
        ]
        if overwrite:
            args.append("--overwrite")
        else:
            args.append("--update_only")
        if drop_on_fail:
            args.append("--drop_on_fail")
        if fail_list_dir:
            args.extend(["--fail_list_dir", fail_list_dir])
        # Launch with lower priority to keep system responsive
        cmd = ["nice", "-n", "10"] + args
        with open(log_path, "wb") as lf:
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        procs.append(p)
        log_paths.append(log_path)
        time.sleep(0.2)
    return procs, log_paths


PROG_RE = re.compile(r"(\d+)\s*/\s*(\d+).*(?:,\s*([0-9]+(?:\.[0-9]+)?)clip/s)?")


def parse_progress(path: Path) -> Tuple[int | None, int | None, float | None]:
    if not path.exists():
        return None, None, None
    try:
        txt = path.read_text(errors="ignore")
    except Exception:
        return None, None, None
    cur = tot = None
    spd = None
    for ln in reversed(txt.splitlines()[-800:]):
        m = PROG_RE.search(ln)
        if m:
            cur = int(m.group(1))
            tot = int(m.group(2))
            if m.group(3):
                try:
                    spd = float(m.group(3))
                except Exception:
                    spd = None
            break
    return cur, tot, spd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--root", default="datasets/LipBengal")
    ap.add_argument("--shards", type=int, default=0, help="Number of parallel shards (default: ~80% CPU, cap 4)")
    ap.add_argument("--overwrite", action="store_true", help="Force recompute even if prepared exists")
    ap.add_argument("--drop_on_fail", action="store_true", help="Drop items where lip detection failed (center fallback)")
    ap.add_argument("--fail_list_dir", default="logs", help="Directory to write failure lists")
    ap.add_argument("--split_mode", choices=["by_speaker", "by_clip"], default="by_clip", help="Split mode to pass to worker")
    ap.add_argument("--split_seed", type=int, default=42, help="Random seed for by-clip split")
    ap.add_argument("--clean", action="store_true", help="Delete existing prepared crops and indices for this split before running")
    args = ap.parse_args()

    root = Path(args.root)
    shards = args.shards or cpu_shards()
    # Export split mode to worker spawn
    os.environ["LB_SPLIT_MODE"] = args.split_mode
    os.environ["LB_SPLIT_SEED"] = str(args.split_seed)
    print(f"Split={args.split} | shards={shards} | mode={'overwrite' if args.overwrite else 'update_only'} | drop_on_fail={args.drop_on_fail} | split_mode={args.split_mode} | seed={args.split_seed} | root={root}")
    # Optionally clean artifacts for a fresh run
    if args.clean:
        try:
            prep_dir = Path(root) / "prepared" / args.split
            idx_dir = Path(root) / "indices"
            # Remove split prepared crops
            if prep_dir.exists():
                import shutil
                shutil.rmtree(prep_dir)
                print(f"Removed {prep_dir}")
            # Remove shard indices for this split (keep others)
            for p in idx_dir.glob(f"{args.split}_shard*_of_*.pt"):
                try:
                    p.unlink()
                except Exception:
                    pass
            # Do not remove the merged {split}.pt yet; it will be overwritten at merge time
        except Exception as e:
            print(f"Warning: clean step failed: {e}")
    per_totals = load_split_counts(root, args.split, shards)
    total_all = sum(per_totals)

    procs, logs = spawn_workers(args.split, shards, root, args.overwrite, args.drop_on_fail, args.fail_list_dir)

    # Setup per-shard and overall progress bars
    bars = []
    for i in range(shards):
        bars.append(tqdm(total=per_totals[i], position=i, desc=f"{args.split}[{i}/{shards}]", leave=True))
    overall = tqdm(total=total_all, position=shards, desc=f"{args.split} total", leave=True)

    prev_vals = [0] * shards
    try:
        while True:
            done = 0
            overall_n = 0
            for i, log in enumerate(logs):
                cur, tot, spd = parse_progress(log)
                if tot is not None and bars[i].total != tot:
                    bars[i].total = tot
                if cur is not None:
                    delta = max(0, cur - prev_vals[i])
                    if delta:
                        bars[i].update(delta)
                        prev_vals[i] = cur
                overall_n += prev_vals[i]
                if prev_vals[i] >= bars[i].total:
                    done += 1
            overall.n = overall_n
            overall.refresh()
            if done == shards and all(p.poll() is not None for p in procs):
                break
            # If any process exited early with non-zero code, show a hint
            for p in procs:
                if p.poll() is not None and p.returncode != 0:
                    print(f"Warning: a shard exited with code {p.returncode}. Check logs.")
            time.sleep(1.5)
    except KeyboardInterrupt:
        print("\nInterrupted. Terminating shard workers...")
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
    finally:
        for p in procs:
            if p.poll() is None:
                try:
                    p.wait(timeout=2)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
        for b in bars:
            try:
                b.close()
            except Exception:
                pass
        try:
            overall.close()
        except Exception:
            pass
    # Merge shard indices into a canonical indices/{split}.pt so training sees dropped items removed
    try:
        if torch is not None:
            idx_dir = root / "indices"
            merged: list = []
            for i in range(shards):
                shard_p = idx_dir / f"{args.split}_shard{i}_of_{shards}.pt"
                if shard_p.exists():
                    part = torch.load(shard_p)
                    merged.extend(part)
            if merged:
                out = idx_dir / f"{args.split}.pt"
                torch.save(merged, out)
                print(f"Merged {len(merged)} items into {out}")
        else:
            print("Warning: torch not available; skipping index merge.")
    except Exception as e:
        print(f"Warning: failed to merge shard indices: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
