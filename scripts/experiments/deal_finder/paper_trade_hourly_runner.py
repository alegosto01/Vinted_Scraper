#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.deal_finder.paper_trading import collect_snapshot, recheck_due
from experiments.deal_finder.paths import (
    LIVE_RUNS_DIR,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    utc_now_iso,
    write_json,
    write_manifest,
)


def append_jsonl(path: Path, payload: dict) -> None:
    path = assert_experiment_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_hourly(
    *,
    offline_run: Path,
    searches: list[str],
    max_searches: int,
    interval_hours: float,
    iterations: int | None,
    out_dir: Path | None,
    dry_run: bool,
    recheck_after_collect: bool = True,
    recheck_due_hours: float = 1.0,
    recheck_above_threshold_only: bool = True,
) -> dict:
    ensure_experiment_dirs()
    if out_dir is None:
        out_dir = LIVE_RUNS_DIR / run_id("hourly")
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_dir": str(out_dir),
        "offline_run": str(offline_run),
        "searches": searches,
        "max_searches": int(max_searches),
        "interval_hours": float(interval_hours),
        "iterations": iterations,
        "dry_run": bool(dry_run),
        "recheck_after_collect": bool(recheck_after_collect),
        "recheck_due_hours": float(recheck_due_hours),
        "recheck_above_threshold_only": bool(recheck_above_threshold_only),
        "started_at": utc_now_iso(),
    }
    write_manifest(out_dir / "manifest.json", command="paper_trade_hourly_runner", extra=config)
    event_log = out_dir / "events.jsonl"

    iteration = 0
    while iterations is None or iteration < iterations:
        iteration += 1
        event = {
            "iteration": iteration,
            "started_at": utc_now_iso(),
            "status": "started",
        }
        append_jsonl(event_log, event)
        try:
            result = collect_snapshot(
                max_searches=max_searches,
                searches=searches,
                offline_run=offline_run,
                out_dir=out_dir,
                dry_run=dry_run,
            )
            if recheck_after_collect:
                result["recheck"] = recheck_due(
                    live_run=out_dir,
                    due_hours=recheck_due_hours,
                    above_threshold_only=recheck_above_threshold_only,
                    dry_run=dry_run,
                )
            event = {
                "iteration": iteration,
                "finished_at": utc_now_iso(),
                "status": "ok",
                "result": result,
            }
        except Exception as exc:
            event = {
                "iteration": iteration,
                "finished_at": utc_now_iso(),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        append_jsonl(event_log, event)
        write_json(out_dir / "latest_status.json", event)
        print(json.dumps(event, sort_keys=True), flush=True)

        if iterations is not None and iteration >= iterations:
            break
        time.sleep(max(1.0, interval_hours * 3600.0))

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hourly paper-trading snapshot collection for selected searches.")
    parser.add_argument("--offline-run", required=True)
    parser.add_argument("--search", action="append", required=True)
    parser.add_argument("--max-searches", type=int, default=2)
    parser.add_argument("--interval-hours", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=0, help="0 means run until the process is stopped.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-recheck-after-collect", action="store_true")
    parser.add_argument("--recheck-due-hours", type=float, default=1.0)
    parser.add_argument("--recheck-all-tracked", action="store_true")
    args = parser.parse_args()
    run_hourly(
        offline_run=Path(args.offline_run),
        searches=args.search,
        max_searches=args.max_searches,
        interval_hours=args.interval_hours,
        iterations=args.iterations or None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        dry_run=args.dry_run,
        recheck_after_collect=not args.no_recheck_after_collect,
        recheck_due_hours=args.recheck_due_hours,
        recheck_above_threshold_only=not args.recheck_all_tracked,
    )


if __name__ == "__main__":
    main()
