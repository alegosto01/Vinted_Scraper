#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
ANALYSIS_DIR = SCRIPTS_DIR / "analysis_pipeline"
for path in (SCRIPTS_DIR, ANALYSIS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd

from analysis_pipeline.evaluation.analyze_blur_signal import (
    bootstrap_mean_diff_ci,
    common_language_effect,
    cohens_d,
    permutation_pvalue,
)
from analysis_pipeline.scoring.foreground_blur import compute_foreground_blur_metrics
from analysis_pipeline.scoring.visual_rerank import normalize_image_sources

DEFAULT_DATASETS = {
    "ps4": ROOT / "data/simple_scrape/ps4/image_cache/balanced_raw_eval/balanced_raw_with_local_images.csv",
    "gucci": ROOT / "data/simple_scrape/gucci/image_cache/balanced_raw_eval/balanced_raw_with_local_images.csv",
    "prada": ROOT / "data/simple_scrape/prada/image_cache/balanced_raw_eval/balanced_raw_with_local_images.csv",
}

METRICS = [
    "whole_gradient_sharpness",
    "whole_laplacian_variance",
    "whole_tenengrad",
    "foreground_laplacian_variance",
    "foreground_tenengrad",
    "foreground_sharpness_score",
    "foreground_relative_sharpness",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare foreground-aware blur metrics against the old whole-image sharpness metric on local cached images."
    )
    ap.add_argument("--dataset", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument(
        "--out_dir",
        default=str(ROOT / "data/simple_scrape/tuning_reports/foreground_blur_signal"),
    )
    ap.add_argument("--backend", default="auto", choices=["auto", "heuristic", "sam"])
    ap.add_argument("--permutations", type=int, default=300)
    ap.add_argument("--bootstrap_samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--preview_examples", type=int, default=0)
    ap.add_argument("--flush_every", type=int, default=10)
    ap.add_argument("--progress_every", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def parse_dataset_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise SystemExit(f"Invalid --dataset value '{raw}'. Expected NAME=PATH.")
    name, path = raw.split("=", 1)
    return name.strip(), Path(path).expanduser().resolve()


def resolve_datasets(dataset_args: list[str]) -> list[tuple[str, Path]]:
    if not dataset_args:
        return [(name, path.resolve()) for name, path in DEFAULT_DATASETS.items()]
    return [parse_dataset_arg(item) for item in dataset_args]


def get_row_image_source(row: pd.Series) -> str | None:
    local_paths = normalize_image_sources(row.get("LocalImagePaths"))
    if local_paths:
        return local_paths[0]
    primary = row.get("LocalPrimaryImagePath")
    if isinstance(primary, str) and primary.strip():
        return primary.strip()
    remote_paths = normalize_image_sources(row.get("Images"))
    return remote_paths[0] if remote_paths else None


def summarize_metric(values_sold, values_unsold, permutations: int, bootstrap_samples: int, seed: int) -> dict[str, float | int | None]:
    sold = pd.Series(values_sold, dtype=float).dropna().to_numpy()
    unsold = pd.Series(values_unsold, dtype=float).dropna().to_numpy()
    if sold.size == 0 or unsold.size == 0:
        return {
            "sold_n": int(sold.size),
            "unsold_n": int(unsold.size),
            "sold_mean": None,
            "unsold_mean": None,
            "mean_diff_sold_minus_unsold": None,
            "mean_diff_ci_low": None,
            "mean_diff_ci_high": None,
            "cohens_d": None,
            "common_language_effect": None,
            "permutation_pvalue": None,
        }
    ci_low, ci_high = bootstrap_mean_diff_ci(sold, unsold, bootstrap_samples, seed)
    return {
        "sold_n": int(sold.size),
        "unsold_n": int(unsold.size),
        "sold_mean": float(sold.mean()),
        "unsold_mean": float(unsold.mean()),
        "mean_diff_sold_minus_unsold": float(sold.mean() - unsold.mean()),
        "mean_diff_ci_low": ci_low,
        "mean_diff_ci_high": ci_high,
        "cohens_d": cohens_d(sold, unsold),
        "common_language_effect": common_language_effect(sold, unsold),
        "permutation_pvalue": permutation_pvalue(sold, unsold, permutations, seed),
    }


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--:--"
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    ratio = max(0.0, min(1.0, done / total))
    filled = int(round(width * ratio))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def write_progress_files(progress_json_path: Path, progress_text_path: Path, payload: dict[str, object]) -> None:
    progress_json_path.write_text(json.dumps(payload, indent=2))
    line = (
        f"{payload['progress_bar']} {payload['completed_rows']}/{payload['total_rows']} "
        f"({payload['progress_pct']:.1f}%) | dataset={payload['current_dataset']} "
        f"| ok={payload['successful_rows']} fail={payload['failed_rows']} "
        f"| elapsed={payload['elapsed']} eta={payload['eta']}\n"
    )
    progress_text_path.write_text(line)


def build_progress_payload(
    *,
    completed_rows: int,
    total_rows: int,
    successful_rows: int,
    failed_rows: int,
    current_dataset: str,
    dataset_done: int,
    dataset_total: int,
    dataset_scored: int,
    dataset_failures: int,
    started_at: float,
) -> dict[str, object]:
    elapsed_seconds = time.time() - started_at
    rate = completed_rows / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining = max(total_rows - completed_rows, 0)
    eta_seconds = (remaining / rate) if rate > 0 else None
    return {
        "started_at_unix": started_at,
        "updated_at_unix": time.time(),
        "completed_rows": int(completed_rows),
        "total_rows": int(total_rows),
        "progress_pct": (100.0 * completed_rows / total_rows) if total_rows else 100.0,
        "progress_bar": progress_bar(completed_rows, total_rows),
        "successful_rows": int(successful_rows),
        "failed_rows": int(failed_rows),
        "current_dataset": current_dataset,
        "current_dataset_done": int(dataset_done),
        "current_dataset_total": int(dataset_total),
        "current_dataset_scored": int(dataset_scored),
        "current_dataset_failures": int(dataset_failures),
        "elapsed": format_duration(elapsed_seconds),
        "eta": format_duration(eta_seconds),
        "rows_per_second": rate,
    }


def emit_progress(payload: dict[str, object], *, tty_mode: bool) -> None:
    line = (
        f"{payload['progress_bar']} {payload['completed_rows']}/{payload['total_rows']} "
        f"({payload['progress_pct']:.1f}%) | dataset={payload['current_dataset']} "
        f"{payload['current_dataset_done']}/{payload['current_dataset_total']} "
        f"| ok={payload['successful_rows']} fail={payload['failed_rows']} "
        f"| elapsed={payload['elapsed']} eta={payload['eta']}"
    )
    if tty_mode:
        print(f"\r{line}", end="", flush=True)
    else:
        print(line, flush=True)


def save_metrics(rows: list[dict[str, object]], metrics_path: Path) -> None:
    pd.DataFrame(rows).to_csv(metrics_path, index=False)


def build_summary_rows(
    metrics_df: pd.DataFrame,
    *,
    permutations: int,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for dataset_name, subset in list(metrics_df.groupby("Dataset")) + [("combined", metrics_df)]:
        for metric in METRICS:
            summary = summarize_metric(
                subset.loc[subset["SoldLabel"] == 1, metric],
                subset.loc[subset["SoldLabel"] == 0, metric],
                permutations,
                bootstrap_samples,
                seed,
            )
            summary.update(
                {
                    "Dataset": dataset_name,
                    "Metric": metric,
                    "HigherMeansSharper": True,
                }
            )
            summary_rows.append(summary)
    return summary_rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "previews"
    metrics_path = out_dir / "foreground_blur_metrics.csv"
    summary_path = out_dir / "foreground_blur_signal_summary.csv"
    report_path = out_dir / "foreground_blur_signal_report.json"
    progress_json_path = out_dir / "foreground_blur_progress.json"
    progress_text_path = out_dir / "foreground_blur_progress.txt"
    log_path = out_dir / "foreground_blur_run.log"
    tty_mode = sys.stdout.isatty()

    existing_rows: list[dict[str, object]] = []
    processed_keys: set[tuple[str, str]] = set()
    if args.resume and metrics_path.exists():
        existing_df = pd.read_csv(metrics_path)
        if not existing_df.empty:
            existing_rows = existing_df.to_dict(orient="records")
            for record in existing_rows:
                processed_keys.add((str(record.get("Dataset")), str(record.get("Dataid"))))

    rows: list[dict[str, object]] = list(existing_rows)
    dataset_info: dict[str, dict[str, int]] = {}
    started_at = time.time()
    datasets = resolve_datasets(args.dataset)
    total_rows = 0
    dataset_frames: list[tuple[str, pd.DataFrame]] = []
    for dataset_name, dataset_path in datasets:
        df = pd.read_csv(dataset_path)
        if args.max_rows > 0:
            df = df.head(args.max_rows).copy()
        dataset_frames.append((dataset_name, df))
        total_rows += int(len(df))

    completed_rows = len(processed_keys)
    successful_rows = len(processed_keys)
    failed_rows = 0
    initial_payload = build_progress_payload(
        completed_rows=completed_rows,
        total_rows=total_rows,
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        current_dataset="starting",
        dataset_done=0,
        dataset_total=0,
        dataset_scored=0,
        dataset_failures=0,
        started_at=started_at,
    )
    write_progress_files(progress_json_path, progress_text_path, initial_payload)
    emit_progress(initial_payload, tty_mode=tty_mode)
    log_path.write_text("")

    for dataset_name, df in dataset_frames:
        preview_budget = args.preview_examples
        scored = sum(1 for dataset, _ in processed_keys if dataset == dataset_name)
        failures = 0
        dataset_done = scored
        for idx, row in enumerate(df.itertuples(index=False), start=1):
            row_dict = row._asdict()
            row_key = (dataset_name, str(row_dict.get("Dataid")))
            if row_key in processed_keys:
                continue
            source = get_row_image_source(pd.Series(row_dict))
            if not source:
                failures += 1
                failed_rows += 1
                completed_rows += 1
                dataset_done += 1
                continue
            save_preview_to = None
            if preview_budget > 0:
                save_preview_to = preview_dir / dataset_name / f"{row_dict.get('Dataid')}.png"
            try:
                metrics = compute_foreground_blur_metrics(
                    source,
                    backend=args.backend,
                    save_preview_to=save_preview_to,
                )
                if save_preview_to is not None:
                    preview_budget -= 1
            except Exception as exc:
                failures += 1
                failed_rows += 1
                completed_rows += 1
                dataset_done += 1
                if idx % max(args.progress_every, 1) == 0:
                    payload = build_progress_payload(
                        completed_rows=completed_rows,
                        total_rows=total_rows,
                        successful_rows=successful_rows,
                        failed_rows=failed_rows,
                        current_dataset=dataset_name,
                        dataset_done=dataset_done,
                        dataset_total=len(df),
                        dataset_scored=scored,
                        dataset_failures=failures,
                        started_at=started_at,
                    )
                    write_progress_files(progress_json_path, progress_text_path, payload)
                    emit_progress(payload, tty_mode=tty_mode)
                    with log_path.open("a") as fh:
                        fh.write(f"FAIL dataset={dataset_name} dataid={row_dict.get('Dataid')} error={type(exc).__name__}: {exc}\n")
                continue
            scored += 1
            successful_rows += 1
            completed_rows += 1
            dataset_done += 1
            record = {
                "Dataset": dataset_name,
                "Dataid": row_dict.get("Dataid"),
                "SoldLabel": int(row_dict.get("SoldLabel", 0)),
                "Title": row_dict.get("Title"),
                "ImagePath": source,
            }
            record.update(metrics.to_dict())
            rows.append(record)
            processed_keys.add(row_key)

            if scored % max(args.flush_every, 1) == 0:
                save_metrics(rows, metrics_path)
                payload = build_progress_payload(
                    completed_rows=completed_rows,
                    total_rows=total_rows,
                    successful_rows=successful_rows,
                    failed_rows=failed_rows,
                    current_dataset=dataset_name,
                    dataset_done=dataset_done,
                    dataset_total=len(df),
                    dataset_scored=scored,
                    dataset_failures=failures,
                    started_at=started_at,
                )
                write_progress_files(progress_json_path, progress_text_path, payload)
                emit_progress(payload, tty_mode=tty_mode)

            elif idx % max(args.progress_every, 1) == 0:
                payload = build_progress_payload(
                    completed_rows=completed_rows,
                    total_rows=total_rows,
                    successful_rows=successful_rows,
                    failed_rows=failed_rows,
                    current_dataset=dataset_name,
                    dataset_done=dataset_done,
                    dataset_total=len(df),
                    dataset_scored=scored,
                    dataset_failures=failures,
                    started_at=started_at,
                )
                write_progress_files(progress_json_path, progress_text_path, payload)
                emit_progress(payload, tty_mode=tty_mode)

        dataset_info[dataset_name] = {
            "dataset_rows": int(len(df)),
            "rows_scored": scored,
            "failures": failures,
        }
        save_metrics(rows, metrics_path)
        payload = build_progress_payload(
            completed_rows=completed_rows,
            total_rows=total_rows,
            successful_rows=successful_rows,
            failed_rows=failed_rows,
            current_dataset=dataset_name,
            dataset_done=dataset_done,
            dataset_total=len(df),
            dataset_scored=scored,
            dataset_failures=failures,
            started_at=started_at,
        )
        write_progress_files(progress_json_path, progress_text_path, payload)
        emit_progress(payload, tty_mode=tty_mode)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(metrics_path, index=False)

    summary_rows = build_summary_rows(
        metrics_df,
        permutations=args.permutations,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    best_rows = []
    if not summary_df.empty:
        tmp = summary_df.copy()
        tmp["abs_cohens_d"] = tmp["cohens_d"].abs()
        best_rows = (
            tmp.sort_values(["Dataset", "abs_cohens_d", "common_language_effect"], ascending=[True, False, False])
            .groupby("Dataset", as_index=False)
            .head(1)
            .drop(columns=["abs_cohens_d"])
            .to_dict(orient="records")
        )

    report = {
        "datasets": dataset_info,
        "backend": args.backend,
        "metric_note": "All metrics are directional: higher means sharper.",
        "files": {
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            "progress_json": str(progress_json_path),
            "progress_text": str(progress_text_path),
            "run_log": str(log_path),
            "preview_dir": str(preview_dir) if args.preview_examples > 0 else None,
        },
        "best_metric_per_dataset": best_rows,
    }
    report_path.write_text(json.dumps(report, indent=2))
    final_payload = build_progress_payload(
        completed_rows=total_rows,
        total_rows=total_rows,
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        current_dataset="complete",
        dataset_done=total_rows,
        dataset_total=total_rows,
        dataset_scored=successful_rows,
        dataset_failures=failed_rows,
        started_at=started_at,
    )
    write_progress_files(progress_json_path, progress_text_path, final_payload)
    if tty_mode:
        print()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
