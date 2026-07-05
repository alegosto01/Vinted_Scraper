#!/usr/bin/env python3
"""Per-search threshold calibration for giant_basic_visual offline runs.

Mirrors basic_5_giant_model/report_per_search_thresholds.py: thresholds are
chosen independently per SearchName on validation scores (min precision,
min count via choose_threshold), then evaluated on the held-out test split.
This lets basic_5_control be recompared against main_image_simple /
main_image_scores using the same per-search-threshold methodology that
produced basic5_giant's "xgboost_basic_v1 per-search" headline number.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "scripts"))

from experiments.current.full_scrape_giant_model._deps.giant_basic_visual.features import FEATURE_MODES  # noqa: E402
from experiments.current.full_scrape_giant_model._deps.giant_basic_visual.paths import (  # noqa: E402
    OFFLINE_RUNS_DIR,
    assert_experiment_path,
    write_json,
)
from experiments.current.full_scrape_giant_model._deps.deal_finder import model_sweep as base_sweep  # noqa: E402
from experiments.current.full_scrape_giant_model._deps.deal_finder.modeling import TARGET_COL, choose_threshold  # noqa: E402

SCORE_PREFIX = "score__"


def score_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(col for col in frame.columns if col.startswith(SCORE_PREFIX))


def parse_score_col(column: str) -> tuple[str, str]:
    """Split 'score__<feature_mode>__<approach>' into (feature_mode, approach)."""
    if not column.startswith(SCORE_PREFIX):
        raise ValueError(f"Not a score column: {column}")
    rest = column[len(SCORE_PREFIX):]
    for mode in FEATURE_MODES:
        prefix = f"{mode}__"
        if rest.startswith(prefix):
            return mode, rest[len(prefix):]
    raise ValueError(f"Cannot determine feature_mode for score column: {column}")


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
        "precision_at_25": evaluated["precision_at"]["p@25"]["precision"],
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
        feature_mode, approach = parse_score_col(column)
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
                    "feature_mode": feature_mode,
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
    global_cols = [
        "search_name",
        "feature_mode",
        "approach",
        "threshold",
        "threshold_count",
        "threshold_precision",
        "threshold_recall",
        "precision_at_25",
        "roc_auc",
        "pr_auc",
    ]
    available = [col for col in global_cols if col in global_thresholds.columns]
    merged = per_search.merge(
        global_thresholds[available],
        on=["search_name", "feature_mode", "approach"],
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


def aggregate_rows(per_search: pd.DataFrame) -> pd.DataFrame:
    if per_search.empty:
        return pd.DataFrame()
    grouped = per_search.groupby(["feature_mode", "approach"], as_index=False).agg(
        n_searches=("search_name", "nunique"),
        n_qualified=("validation_threshold_qualified", "sum"),
        total_rows=("rows", "sum"),
        total_positive_rows=("positive_rows", "sum"),
        total_selected=("threshold_count", "sum"),
        total_tp=("positive_count", "sum"),
    )
    grouped["overall_precision"] = grouped["total_tp"] / grouped["total_selected"].replace(0, np.nan)
    grouped["overall_recall"] = grouped["total_tp"] / grouped["total_positive_rows"].replace(0, np.nan)
    return grouped


def latest_run_dir() -> Path:
    candidates = sorted(
        (
            path
            for path in OFFLINE_RUNS_DIR.glob("giant_basic_visual_*")
            if path.is_dir() and (path / "validation_scores.csv").exists()
        ),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No giant_basic_visual run with validation scores found under {OFFLINE_RUNS_DIR}")
    return candidates[0]


def write_report(
    run_dir: Path,
    aggregate: pd.DataFrame,
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
        "# Giant Basic Visual - Per-Search Thresholds",
        "",
        f"Run folder: `{run_dir}`",
        "",
        f"Thresholds are selected independently for each `SearchName` on validation scores with "
        f"min precision `{min_precision:.2f}` and min count `{min_count}` (same `choose_threshold` "
        "policy as basic_5_giant_model's per-search report). No models are retrained.",
        "",
    ]

    if aggregate.empty:
        lines.append("No per-search threshold rows were produced.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    lines.extend(
        [
            "## Aggregate (sum across searches)",
            "",
            "| feature mode | approach | qualified | total selected | total TP | overall precision | overall recall |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in aggregate.sort_values(["approach", "feature_mode"]).iterrows():
        lines.append(
            f"| {row['feature_mode']} | {row['approach']} | {int(row['n_qualified'])}/{int(row['n_searches'])} | "
            f"{int(row['total_selected'])} | {int(row['total_tp'])} | "
            f"{fmt(row['overall_precision'])} | {fmt(row['overall_recall'])} |"
        )

    focus_approaches = ["hist_gradient_basic_numeric_v1", "xgboost_basic_v1"]
    for approach in focus_approaches:
        detail = per_search[per_search["approach"].eq(approach)].copy()
        if detail.empty:
            continue
        lines.extend(
            [
                "",
                f"## Per-Search Detail: `{approach}`",
                "",
                "| search | mode | threshold | val count | val precision | test selected | test TP | test precision | test recall |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in detail.sort_values(["search_name", "feature_mode"]).iterrows():
            lines.append(
                f"| {row['search_name']} | {row['feature_mode']} | {fmt(row['per_search_threshold'], 4)} | "
                f"{int(row['validation_threshold_count'])} | {fmt(row['validation_threshold_precision'])} | "
                f"{int(row['threshold_count'])} | {int(row['positive_count'])} | "
                f"{fmt(row['threshold_precision'])} | {fmt(row['threshold_recall'])} |"
            )

    if not comparison.empty:
        focus = comparison[comparison["approach"].isin(focus_approaches)].copy()
        if not focus.empty:
            lines.extend(
                [
                    "",
                    "## Per-Search Threshold Versus Global Single Threshold",
                    "",
                    "| search | mode | approach | global selected | global precision | per-search selected | per-search precision | selected delta |",
                    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for _, row in focus.sort_values(["approach", "feature_mode", "search_name"]).iterrows():
                lines.append(
                    f"| {row['search_name']} | {row['feature_mode']} | {row['approach']} | "
                    f"{int(row['threshold_count_global']) if pd.notna(row['threshold_count_global']) else 0} | "
                    f"{fmt(row['threshold_precision_global'])} | "
                    f"{int(row['threshold_count_per_search'])} | {fmt(row['threshold_precision_per_search'])} | "
                    f"{int(row['selected_count_delta']) if pd.notna(row['selected_count_delta']) else 0} |"
                )

    lines.extend(
        [
            "",
            "## Versus basic5_giant_model",
            "",
            "`basic_5_giant_model`'s per-search report (own offline run/dataset, 7 searches, no "
            "main-image filter) picked `xgboost_basic_v1` per-search thresholds: 290 selected at "
            "test precision 0.952 (see `docs/BASIC_5_GIANT_MODEL.md`). The aggregate rows above use "
            "the same `choose_threshold` policy on `giant_basic_visual`'s rows, which are filtered to "
            "items with a readable `LocalPrimaryImagePath` - row counts are not directly comparable, "
            "but the `basic_5_control` vs `main_image_*` rows here are apples-to-apples (same rows, "
            "same split).",
            "",
            "## Files",
            "",
            "- `per_search_threshold_metrics.csv`: per-search per-mode/approach threshold metrics.",
            "- `per_search_threshold_aggregate.csv`: summed across searches per mode/approach.",
            "- `per_search_threshold_comparison.csv`: per-search threshold vs the original global threshold.",
            "",
            "## Notes",
            "",
            "- A per-search validation threshold can increase coverage for searches that were "
            "under-selected by the global threshold.",
            "- Very high test precision on small selected counts is unstable; count and recall "
            "matter alongside precision.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(run_dir: Path, *, min_precision: float, min_count: int) -> dict[str, Any]:
    run_dir = assert_experiment_path(run_dir)
    validation_path = run_dir / "validation_scores.csv"
    test_path = run_dir / "test_scores.csv"
    if not validation_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing validation/test score files in {run_dir}")
    validation = pd.read_csv(validation_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)

    per_search = pd.DataFrame(
        per_search_threshold_rows(validation, test, min_precision=min_precision, min_count=min_count)
    )
    per_search_path = assert_experiment_path(run_dir / "per_search_threshold_metrics.csv")
    per_search.to_csv(per_search_path, index=False)

    aggregate = aggregate_rows(per_search)
    aggregate_path = assert_experiment_path(run_dir / "per_search_threshold_aggregate.csv")
    aggregate.to_csv(aggregate_path, index=False)

    global_path = run_dir / "per_search_metrics.csv"
    global_thresholds = pd.read_csv(global_path, low_memory=False) if global_path.exists() else pd.DataFrame()
    comparison = comparison_rows(per_search, global_thresholds)
    comparison_path = assert_experiment_path(run_dir / "per_search_threshold_comparison.csv")
    comparison.to_csv(comparison_path, index=False)

    report_path = write_report(
        run_dir,
        aggregate,
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
            "aggregate": str(aggregate_path),
            "comparison": str(comparison_path),
            "report": str(report_path),
        },
    }
    write_json(run_dir / "per_search_threshold_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune per-search thresholds for a giant_basic_visual offline run.")
    parser.add_argument("--run-dir", default=None, help="Existing giant_basic_visual run dir. Defaults to latest.")
    parser.add_argument("--min-precision", type=float, default=base_sweep.PROMOTION_PRECISION)
    parser.add_argument("--min-count", type=int, default=base_sweep.VALIDATION_MIN_COUNT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    summary = run_report(run_dir, min_precision=args.min_precision, min_count=args.min_count)
    print(f"Per-search threshold report written for {summary['run_dir']}")
    for key, value in summary["outputs"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
