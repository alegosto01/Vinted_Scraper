#!/usr/bin/env python3
"""Tune per-search thresholds for a trained full_scrape_visual giant-model run.

Near-verbatim port of experiments/current/basic_5_giant_model/report_per_search_thresholds.py,
repointed at full_scrape_giant_model's paths and validation/test score CSVs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.current.full_scrape_giant_model.paths import (  # noqa: E402
    OFFLINE_RUNS_DIR,
    assert_experiment_path,
    write_json,
)
from experiments.current.full_scrape_giant_model._deps.deal_finder import model_sweep as base_sweep  # noqa: E402
from experiments.current.full_scrape_giant_model._deps.deal_finder.modeling import TARGET_COL, choose_threshold  # noqa: E402


SCORE_PREFIX = "score__"


def score_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(col for col in frame.columns if col.startswith(SCORE_PREFIX))


def approach_from_score_col(column: str) -> str:
    if not column.startswith(SCORE_PREFIX):
        raise ValueError(f"Not a score column: {column}")
    return column[len(SCORE_PREFIX) :]


def threshold_is_qualified(row: dict[str, Any], *, min_precision: float, min_count: int) -> bool:
    precision = row.get("precision")
    count = int(row.get("count", 0))
    return count >= min_count and pd.notna(precision) and float(precision) >= min_precision


def evaluate_group(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    evaluated = base_sweep.evaluate_scores(frame, scores, threshold)
    return {
        "rows": evaluated["rows"],
        "positive_rows": evaluated["positive_rows"],
        "base_rate": evaluated["base_rate"],
        "threshold_count": evaluated["threshold"]["count"],
        "threshold_precision": evaluated["threshold"]["precision"],
        "threshold_recall": evaluated["threshold"]["recall"],
        "positive_count": evaluated["threshold"]["positive_count"],
        "precision_at_10": evaluated["precision_at"]["p@10"]["precision"],
        "precision_at_25": evaluated["precision_at"]["p@25"]["precision"],
        "precision_at_50": evaluated["precision_at"]["p@50"]["precision"],
        "roc_auc": evaluated["roc_auc"],
        "pr_auc": evaluated["pr_auc"],
    }


def per_search_threshold_rows(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    min_precision: float,
    min_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in score_columns(validation):
        if column not in test.columns:
            continue
        approach = approach_from_score_col(column)
        for search, validation_group in validation.groupby("SearchName", sort=True):
            test_group = test[test["SearchName"].astype(str).eq(str(search))].copy()
            if validation_group.empty or test_group.empty:
                continue
            val_labels = validation_group[TARGET_COL].astype(int).to_numpy()
            val_scores = validation_group[column].to_numpy(dtype=float)
            threshold_row = choose_threshold(
                val_labels,
                val_scores,
                min_precision=min_precision,
                min_count=min_count,
            )
            threshold = float(threshold_row["threshold"])
            test_eval = evaluate_group(test_group, test_group[column].to_numpy(dtype=float), threshold)
            rows.append(
                {
                    "search_name": str(search),
                    "approach": approach,
                    "per_search_threshold": threshold,
                    "validation_threshold_count": int(threshold_row["count"]),
                    "validation_threshold_precision": threshold_row["precision"],
                    "validation_positive_count": int(threshold_row["positive_count"]),
                    "validation_threshold_qualified": threshold_is_qualified(
                        threshold_row,
                        min_precision=min_precision,
                        min_count=min_count,
                    ),
                    **test_eval,
                }
            )
    return rows


def comparison_rows(per_search: pd.DataFrame, global_thresholds: pd.DataFrame) -> pd.DataFrame:
    if per_search.empty or global_thresholds.empty:
        return pd.DataFrame()
    # full_scrape_giant_model's per_search_metrics.csv (from metrics.summarize) uses
    # selected_count/precision/recall rather than basic5's threshold_count/
    # threshold_precision/threshold_recall naming; normalize before merging.
    global_renamed = global_thresholds.rename(
        columns={
            "selected_count": "threshold_count",
            "precision": "threshold_precision",
            "recall": "threshold_recall",
        }
    )
    global_cols = [
        "search_name",
        "approach",
        "threshold",
        "threshold_count",
        "threshold_precision",
        "threshold_recall",
        "precision_at_25",
        "roc_auc",
        "pr_auc",
    ]
    available = [col for col in global_cols if col in global_renamed.columns]
    merged = per_search.merge(
        global_renamed[available],
        on=["search_name", "approach"],
        how="left",
        suffixes=("_per_search", "_global"),
    )
    if "threshold_count_global" in merged:
        merged["selected_count_delta"] = merged["threshold_count_per_search"] - merged["threshold_count_global"]
    if "threshold_precision_global" in merged:
        merged["precision_delta"] = merged["threshold_precision_per_search"] - merged["threshold_precision_global"]
    if "threshold_recall_global" in merged:
        merged["recall_delta"] = merged["threshold_recall_per_search"] - merged["threshold_recall_global"]
    return merged


def latest_run_dir() -> Path:
    base = OFFLINE_RUNS_DIR
    candidates = sorted(
        (path for path in base.glob("full_scrape_giant_visual_*") if path.is_dir() and (path / "validation_scores.csv").exists()),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No full_scrape_giant_visual run with validation scores found under {base}")
    return candidates[0]


def pick_focus_approach(per_search: pd.DataFrame) -> str:
    """Pick the approach with the most qualified per-search thresholds, tie-broken by mean test precision."""
    if per_search.empty:
        return ""
    grouped = per_search.groupby("approach").agg(
        qualified=("validation_threshold_qualified", "sum"),
        mean_precision=("threshold_precision", "mean"),
    )
    grouped = grouped.sort_values(["qualified", "mean_precision"], ascending=[False, False])
    return str(grouped.index[0])


def write_report(
    run_dir: Path,
    per_search: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    min_precision: float,
    min_count: int,
) -> Path:
    path = assert_experiment_path(run_dir / "per_search_threshold_report.md")

    def fmt(value: object, decimals: int = 3) -> str:
        try:
            if value is None or pd.isna(value):
                return "-"
            return f"{float(value):.{decimals}f}"
        except Exception:
            return str(value)

    lines = [
        "# Full-Scrape Visual Giant Model - Per-Search Thresholds",
        "",
        f"Run folder: `{run_dir}`",
        "",
        f"Thresholds are selected independently for each `SearchName` on validation scores with min precision `{min_precision:.2f}` and min count `{min_count}`.",
        "The already-trained giant models are reused; no models are retrained.",
        "",
    ]
    if per_search.empty:
        lines.append("No per-search threshold rows were produced.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    top = (
        per_search.sort_values(
            [
                "search_name",
                "validation_threshold_qualified",
                "threshold_precision",
                "threshold_count",
                "precision_at_25",
                "roc_auc",
            ],
            ascending=[True, False, False, False, False, False],
            na_position="last",
        )
        .drop_duplicates("search_name", keep="first")
        .reset_index(drop=True)
    )
    lines.extend(
        [
            "## Best Per Search",
            "",
            "| search | approach | threshold | val qualified | test count | test precision | test recall | P@25 | AUC |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in top.iterrows():
        lines.append(
            f"| {row['search_name']} | {row['approach']} | {fmt(row['per_search_threshold'], 4)} | "
            f"{bool(row['validation_threshold_qualified'])} | {int(row['threshold_count'])} | "
            f"{fmt(row['threshold_precision'])} | {fmt(row['threshold_recall'])} | "
            f"{fmt(row['precision_at_25'])} | {fmt(row['roc_auc'])} |"
        )

    focus_approach = pick_focus_approach(per_search)
    focus = per_search[per_search["approach"].eq(focus_approach)].sort_values("search_name")
    if not focus.empty:
        lines.extend(
            [
                "",
                f"## {focus_approach} Per-Search Thresholds",
                "",
                "| search | threshold | val count | val precision | test count | test precision | test recall | P@25 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in focus.iterrows():
            lines.append(
                f"| {row['search_name']} | {fmt(row['per_search_threshold'], 4)} | "
                f"{int(row['validation_threshold_count'])} | {fmt(row['validation_threshold_precision'])} | "
                f"{int(row['threshold_count'])} | {fmt(row['threshold_precision'])} | "
                f"{fmt(row['threshold_recall'])} | {fmt(row['precision_at_25'])} |"
            )

    if not comparison.empty and focus_approach:
        focus_cmp = comparison[comparison["approach"].eq(focus_approach)].sort_values("search_name")
        if not focus_cmp.empty:
            lines.extend(
                [
                    "",
                    f"## {focus_approach} Versus Global Threshold",
                    "",
                    "| search | global count | per-search count | count delta | global precision | per-search precision | recall delta |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for _, row in focus_cmp.iterrows():
                lines.append(
                    f"| {row['search_name']} | {int(row['threshold_count_global']) if pd.notna(row['threshold_count_global']) else 0} | "
                    f"{int(row['threshold_count_per_search'])} | {int(row['selected_count_delta']) if pd.notna(row['selected_count_delta']) else 0} | "
                    f"{fmt(row['threshold_precision_global'])} | {fmt(row['threshold_precision_per_search'])} | "
                    f"{fmt(row['recall_delta'])} |"
                )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `per_search_threshold_metrics.csv`: threshold-tuned held-out metrics by search and approach.",
            "- `per_search_threshold_comparison.csv`: comparison against the original single global threshold.",
            "",
            "## Notes",
            "",
            "- A per-search validation threshold can increase coverage for searches that were under-selected by the global threshold.",
            "- Very high test precision on small selected counts is unstable; count and recall matter alongside precision.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    run_dir: Path,
    *,
    min_precision: float = base_sweep.PROMOTION_PRECISION,
    min_count: int = base_sweep.VALIDATION_MIN_COUNT,
) -> dict[str, Any]:
    run_dir = assert_experiment_path(run_dir)
    validation_path = run_dir / "validation_scores.csv"
    test_path = run_dir / "test_scores.csv"
    if not validation_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing validation/test score files in {run_dir}")
    validation = pd.read_csv(validation_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    per_search = pd.DataFrame(
        per_search_threshold_rows(
            validation,
            test,
            min_precision=min_precision,
            min_count=min_count,
        )
    )
    per_search_path = assert_experiment_path(run_dir / "per_search_threshold_metrics.csv")
    per_search.to_csv(per_search_path, index=False)

    global_path = run_dir / "per_search_metrics.csv"
    global_thresholds = pd.read_csv(global_path, low_memory=False) if global_path.exists() else pd.DataFrame()
    comparison = comparison_rows(per_search, global_thresholds)
    comparison_path = assert_experiment_path(run_dir / "per_search_threshold_comparison.csv")
    comparison.to_csv(comparison_path, index=False)

    report_path = write_report(
        run_dir,
        per_search,
        comparison,
        min_precision=min_precision,
        min_count=min_count,
    )
    summary = {
        "run_dir": str(run_dir),
        "min_precision": float(min_precision),
        "min_count": int(min_count),
        "rows": int(len(per_search)),
        "outputs": {
            "metrics": str(per_search_path),
            "comparison": str(comparison_path),
            "report": str(report_path),
        },
    }
    write_json(run_dir / "per_search_threshold_summary.json", base_sweep.to_builtin(summary))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune per-search thresholds for a trained full_scrape_visual giant-model run.")
    parser.add_argument("--run-dir", default=None, help="Existing full_scrape_giant_visual run dir. Defaults to latest.")
    parser.add_argument("--min-precision", type=float, default=base_sweep.PROMOTION_PRECISION)
    parser.add_argument("--min-count", type=int, default=base_sweep.VALIDATION_MIN_COUNT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    summary = run_report(run_dir, min_precision=args.min_precision, min_count=args.min_count)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
