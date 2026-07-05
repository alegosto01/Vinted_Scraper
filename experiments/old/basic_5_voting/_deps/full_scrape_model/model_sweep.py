#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.old.basic_5_voting._deps.deal_finder import model_sweep as base_sweep
from experiments.old.basic_5_voting._deps.full_scrape_model.dataset import TASK_CHOICES, build_datasets
from experiments.old.basic_5_voting._deps.full_scrape_model.paths import (
    EXPERIMENT_ROOT,
    MODELS_DIR,
    OFFLINE_RUNS_DIR,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_json,
    write_manifest,
)


def patch_base_sweep_paths() -> None:
    base_sweep.MODELS_DIR = MODELS_DIR
    base_sweep.assert_experiment_path = assert_experiment_path
    base_sweep.write_json = write_json
    base_sweep.write_manifest = write_manifest


def metadata_path_for_artifact(artifact_path: str) -> Path:
    artifact = Path(artifact_path)
    return artifact.with_name(artifact.stem + "_metadata.json")


def rewrite_metadata_for_task(row: dict, *, task: str) -> None:
    if row.get("status") != "trained":
        return
    metadata = dict(row.get("model_metadata", {}))
    if not metadata:
        return
    metadata["experiment_family"] = "full_scrape_model"
    metadata["task"] = task
    metadata["task_positive_label"] = "score_high" if task == "extreme_score" else "sold"
    metadata["task_negative_label"] = "score_low" if task == "extreme_score" else "not_sold"
    metadata["live_success_label"] = metadata["task_positive_label"]
    row["model_metadata"] = metadata
    metadata_path = metadata_path_for_artifact(str(metadata.get("artifact_path", "")))
    if str(metadata.get("artifact_path", "")):
        write_json(metadata_path, base_sweep.to_builtin(metadata))


def run_sweep(
    *,
    task: str,
    searches: list[str] | None = None,
    limit_rows: int | None = None,
    out_dir: Path | None = None,
    approach_names: list[str] | None = None,
    seeds: list[int] | None = None,
) -> dict[str, object]:
    ensure_experiment_dirs()
    patch_base_sweep_paths()
    if out_dir is None:
        out_dir = OFFLINE_RUNS_DIR / run_id(f"{task}_sweep")
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = seeds or [base_sweep.DEFAULT_SEED]
    selected_names = set(approach_names or [spec.name for spec in base_sweep.APPROACHES])
    selected_approaches = [spec for spec in base_sweep.APPROACHES if spec.name in selected_names]
    dataset_results = build_datasets(task=task, searches=searches, out_dir=out_dir, limit_rows=limit_rows)

    metrics: list[dict[str, object]] = []
    for dataset_result in dataset_results:
        df = pd.read_csv(dataset_result.path, low_memory=False)
        frame = base_sweep.prepare_sweep_frame(df)
        print(
            f"[full_scrape_model] task={task} search={dataset_result.search_name} eligible_rows={len(frame)}",
            flush=True,
        )
        if len(frame) < 50 or frame[base_sweep.TARGET_COL].nunique() < 2:
            for spec in selected_approaches:
                metrics.append(
                    {
                        "task": task,
                        "search_name": dataset_result.search_name,
                        "approach": spec.name,
                        "status": "skipped",
                        "seed": seeds[0],
                        "reason": "not enough eligible rows or only one label class",
                    }
                )
            continue
        for seed in seeds:
            for spec in selected_approaches:
                step_started = time.perf_counter()
                print(
                    f"[full_scrape_model] train task={task} search={dataset_result.search_name} approach={spec.name} seed={seed}",
                    flush=True,
                )
                try:
                    row = base_sweep.train_approach(
                        frame,
                        search_name=dataset_result.search_name,
                        spec=spec,
                        seed=seed,
                        model_prefix=out_dir.name,
                    )
                    row["task"] = task
                    rewrite_metadata_for_task(row, task=task)
                    metrics.append(row)
                    print(
                        f"[full_scrape_model] done task={task} search={dataset_result.search_name} approach={spec.name} "
                        f"status={row.get('status')} seconds={time.perf_counter() - step_started:.1f}",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[full_scrape_model] error task={task} search={dataset_result.search_name} approach={spec.name} "
                        f"seconds={time.perf_counter() - step_started:.1f}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    metrics.append(
                        {
                            "task": task,
                            "search_name": dataset_result.search_name,
                            "approach": spec.name,
                            "status": "error",
                            "seed": seed,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )

    metrics_path = out_dir / "metrics.json"
    write_json(metrics_path, base_sweep.to_builtin({"task": task, "metrics": metrics}))
    metrics_df = pd.DataFrame([{**base_sweep.flatten_metrics(row), "task": row.get("task", task)} for row in metrics])
    metrics_long_path = assert_experiment_path(out_dir / "metrics_long.csv")
    metrics_df.to_csv(metrics_long_path, index=False)
    by_search, by_approach = base_sweep.best_tables(metrics_df)
    by_search_path = assert_experiment_path(out_dir / "best_by_search.csv")
    by_approach_path = assert_experiment_path(out_dir / "best_by_approach.csv")
    by_search.to_csv(by_search_path, index=False)
    by_approach.to_csv(by_approach_path, index=False)
    candidates_df = metrics_df[metrics_df.get("qualified_for_paper_trading", False) == True].copy() if not metrics_df.empty else pd.DataFrame()
    candidate_records = candidates_df.replace({np.nan: None}).to_dict("records") if not candidates_df.empty else []
    write_json(out_dir / "promotion_candidates.json", base_sweep.to_builtin({"task": task, "promotion_candidates": candidate_records}))
    report_path = base_sweep.write_sweep_report(out_dir, metrics_df, by_search, candidate_records)
    summary = {
        "task": task,
        "run_dir": str(out_dir),
        "dataset_count": len(dataset_results),
        "approaches": [spec.name for spec in selected_approaches],
        "seeds": seeds,
        "model_rows": len(metrics),
        "trained_rows": int((metrics_df["status"] == "trained").sum()) if not metrics_df.empty else 0,
        "promotion_candidate_count": len(candidate_records),
        "outputs": {
            "metrics_long": str(metrics_long_path),
            "best_by_search": str(by_search_path),
            "best_by_approach": str(by_approach_path),
            "promotion_candidates": str(out_dir / "promotion_candidates.json"),
            "sweep_report": str(report_path),
        },
        "auto_paper_trading": False,
    }
    manifest_path = out_dir / "manifest.json"
    write_manifest(manifest_path, command="full_scrape_model.model_sweep", extra=base_sweep.to_builtin(summary))

    # ── Experiment tracker ──────────────────────────────────────────────────
    try:
        from experiments.old.basic_5_voting._deps.tracking.db import init_db, log_run, log_metrics
        init_db()
        manifest = base_sweep.read_json(manifest_path, {})
        git_info = manifest.get("git", {})
        log_run({
            "run_id": out_dir.name,
            "family": "full_scrape_model",
            "command": "model_sweep",
            "status": "completed",
            "created_at": manifest.get("created_at"),
            "finished_at": manifest.get("created_at"),
            "run_dir": str(out_dir),
            "git_branch": git_info.get("branch"),
            "git_head": git_info.get("head"),
            "git_dirty": bool(git_info.get("status_short")),
            "promotion_count": len(candidate_records),
        })
        log_metrics(out_dir.name, metrics_df)
    except Exception as exc:
        print(f"[tracking] warning: failed to log run: {exc}", flush=True)
    # ─────────────────────────────────────────────────────────────────────────

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline full-scrape-model sweeps with the same approaches as deal_finder.")
    parser.add_argument("--task", choices=[*TASK_CHOICES, "all"], default="all")
    parser.add_argument("--all-searches", action="store_true")
    parser.add_argument("--search", action="append", default=[])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--approach", action="append", default=[])
    parser.add_argument("--robustness", action="store_true", help="Run seeds 42, 123, and 2026 instead of only seed 42.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all_searches and not args.search:
        raise SystemExit("Use --all-searches or at least one --search.")
    tasks = list(TASK_CHOICES) if args.task == "all" else [args.task]
    out_root = Path(args.out_dir) if args.out_dir else None
    summaries = []
    for task in tasks:
        task_out_dir = (out_root / task) if out_root else None
        summary = run_sweep(
            task=task,
            searches=None if args.all_searches else args.search,
            limit_rows=args.limit_rows,
            out_dir=task_out_dir,
            approach_names=args.approach or None,
            seeds=list(base_sweep.ROBUSTNESS_SEEDS) if args.robustness else [base_sweep.DEFAULT_SEED],
        )
        summaries.append({k: summary[k] for k in ("task", "run_dir", "trained_rows", "promotion_candidate_count")})
    print(json.dumps(summaries, indent=2))
    print(f"Output root: {out_root or EXPERIMENT_ROOT}")


if __name__ == "__main__":
    main()
