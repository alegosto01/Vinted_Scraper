#!/usr/bin/env python3
"""Backfill historical experiment runs into experiments.db.

Walks existing experiment directories and inserts past runs from manifest.json
and metrics_long.csv files. Idempotent: skips runs already in DB.

Usage:
    cd /home/ale/Desktop/vinted/Vinted_New_Version
    python experiments/old/tracking/backfill.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.old.tracking.db import init_db, log_run, log_metrics, run_exists

ROOT = Path(__file__).resolve().parents[3]
OLD_EXPERIMENTS_ROOT = ROOT / "experiments" / "old"


def _read_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def backfill_deal_finder() -> int:
    count = 0
    base = OLD_EXPERIMENTS_ROOT / "deal_finder" / "data" / "offline_runs"
    if not base.exists():
        return 0
    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        if run_exists(run_id):
            continue
        manifest = _read_json(run_dir / "manifest.json", {})
        metrics_csv = run_dir / "metrics_long.csv"
        if not metrics_csv.exists():
            continue
        try:
            metrics_df = pd.read_csv(metrics_csv, low_memory=False)
        except Exception:
            continue
        extra = manifest.get("extra", {})
        git_info = manifest.get("git", {})
        log_run({
            "run_id": run_id,
            "family": "deal_finder",
            "command": extra.get("command", "model_sweep"),
            "status": "completed",
            "created_at": manifest.get("created_at"),
            "finished_at": manifest.get("created_at"),
            "run_dir": str(run_dir),
            "git_branch": git_info.get("branch"),
            "git_head": git_info.get("head"),
            "git_dirty": bool(git_info.get("status_short")),
            "promotion_count": extra.get("promotion_candidate_count", 0),
        })
        log_metrics(run_id, metrics_df)
        count += 1
        print(f"  backfilled deal_finder/{run_id} ({len(metrics_df)} rows)")
    return count


def backfill_full_scrape_model() -> int:
    count = 0
    base = OLD_EXPERIMENTS_ROOT / "full_scrape_model" / "data" / "offline_runs"
    if not base.exists():
        return 0
    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        if run_exists(run_id):
            continue
        manifest = _read_json(run_dir / "manifest.json", {})
        metrics_csv = run_dir / "metrics_long.csv"
        if not metrics_csv.exists():
            continue
        try:
            metrics_df = pd.read_csv(metrics_csv, low_memory=False)
        except Exception:
            continue
        extra = manifest.get("extra", {})
        git_info = manifest.get("git", {})
        log_run({
            "run_id": run_id,
            "family": "full_scrape_model",
            "command": extra.get("command", "model_sweep"),
            "status": "completed",
            "created_at": manifest.get("created_at"),
            "finished_at": manifest.get("created_at"),
            "run_dir": str(run_dir),
            "git_branch": git_info.get("branch"),
            "git_head": git_info.get("head"),
            "git_dirty": bool(git_info.get("status_short")),
            "promotion_count": extra.get("promotion_candidate_count", 0),
        })
        log_metrics(run_id, metrics_df)
        count += 1
        print(f"  backfilled full_scrape_model/{run_id} ({len(metrics_df)} rows)")
    return count


def backfill_benchmark_basic_to_full() -> int:
    count = 0
    base = OLD_EXPERIMENTS_ROOT / "benchmark_basic_to_full" / "data" / "live_runs"
    if not base.exists():
        return 0
    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        if run_exists(run_id):
            continue
        manifest = _read_json(run_dir / "manifest.json", {})
        if not manifest:
            continue
        extra = manifest.get("extra", {})
        git_info = manifest.get("git", {})
        log_run({
            "run_id": run_id,
            "family": "benchmark_basic_to_full",
            "command": extra.get("command", "cascade_run"),
            "status": "completed",
            "created_at": manifest.get("created_at"),
            "finished_at": manifest.get("created_at"),
            "run_dir": str(run_dir),
            "git_branch": git_info.get("branch"),
            "git_head": git_info.get("head"),
            "git_dirty": bool(git_info.get("status_short")),
            "promotion_count": 0,
        })
        count += 1
        print(f"  backfilled benchmark_basic_to_full/{run_id}")
    return count


def backfill_teacher_student() -> int:
    count = 0
    base = OLD_EXPERIMENTS_ROOT / "teacher_student_basic_filter" / "data" / "offline_runs"
    if not base.exists():
        return 0
    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        if run_exists(run_id):
            continue
        # teacher-student doesn't have manifest.json; infer from CSVs
        metrics_csv = run_dir / "metrics_long.csv"
        if not metrics_csv.exists():
            continue
        try:
            metrics_df = pd.read_csv(metrics_csv, low_memory=False)
        except Exception:
            continue
        log_run({
            "run_id": run_id,
            "family": "teacher_student",
            "command": "train_student",
            "status": "completed",
            "created_at": None,
            "finished_at": None,
            "run_dir": str(run_dir),
            "git_branch": None,
            "git_head": None,
            "git_dirty": None,
            "promotion_count": 0,
        })
        log_metrics(run_id, metrics_df)
        count += 1
        print(f"  backfilled teacher_student/{run_id} ({len(metrics_df)} rows)")
    return count


def backfill_photo_arbitrage() -> int:
    count = 0
    base = OLD_EXPERIMENTS_ROOT / "photo_arbitrage" / "data" / "reports"
    if not base.exists():
        return 0
    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        if run_exists(run_id):
            continue
        manifest = _read_json(run_dir / "manifest.json", {})
        if not manifest:
            continue
        extra = manifest.get("extra", {})
        git_info = manifest.get("git", {})
        log_run({
            "run_id": run_id,
            "family": "photo_arbitrage",
            "command": extra.get("command", "report"),
            "status": "completed",
            "created_at": manifest.get("created_at"),
            "finished_at": manifest.get("created_at"),
            "run_dir": str(run_dir),
            "git_branch": git_info.get("branch"),
            "git_head": git_info.get("head"),
            "git_dirty": bool(git_info.get("status_short")),
            "promotion_count": 0,
        })
        count += 1
        print(f"  backfilled photo_arbitrage/{run_id}")
    return count


def main() -> int:
    init_db()
    print("Backfilling historical experiment runs...")
    total = 0
    total += backfill_deal_finder()
    total += backfill_full_scrape_model()
    total += backfill_benchmark_basic_to_full()
    total += backfill_teacher_student()
    total += backfill_photo_arbitrage()
    print(f"Done. Backfilled {total} runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
