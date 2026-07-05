#!/usr/bin/env python3
"""Retrospective threshold sweep on a cascade benchmark run.

Given an existing run folder with tracked_items.csv, recompute precision
across a grid of stage-1 and stage-2 threshold increases.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.old.benchmark_basic_to_full.paths import (
    LIVE_RUNS_DIR,
    assert_experiment_path,
    write_json,
)

WINDOW_HOURS = [*range(1, 25), *range(27, 49, 3), *range(60, 169, 12)]


def window_label(hours: int | float) -> str:
    value = float(hours)
    if value.is_integer():
        return f"{int(value)}h"
    return f"{value:g}h"


def outcome_col(hours: int | float) -> str:
    return f"sold_within_{window_label(hours)}"


def evaluated_col(hours: int | float) -> str:
    return f"evaluated_at_{window_label(hours)}"


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def compute_precision(
    tracked: pd.DataFrame,
    stage1_delta: float,
    stage2_delta: float,
) -> dict[str, Any]:
    """Recompute cascade precision for given threshold deltas."""
    out = tracked.copy()

    # Recompute stage-1 pass with new threshold
    out["_s1_new_pass"] = out["Stage1Score"] >= (out["Stage1Threshold"] + stage1_delta)

    # Recompute stage-2 pass with new threshold
    # Items without Stage2Score cannot pass stage-2
    out["_s2_new_pass"] = False
    has_stage2 = out["Stage2Score"].notna()
    out.loc[has_stage2, "_s2_new_pass"] = (
        out.loc[has_stage2, "Stage2Score"] >= (out.loc[has_stage2, "Stage2Threshold"] + stage2_delta)
    )

    # Final pass = passed stage-1 AND passed stage-2
    out["_final_pass"] = out["_s1_new_pass"] & out["_s2_new_pass"]

    # Subset that passed stage-1 (candidates that would have been tracked)
    s1_pass = out[out["_s1_new_pass"]].copy()

    # Subset that passed both stages
    final_pass = out[out["_final_pass"]].copy()

    # all_stage1_pass should include ALL tracked items, not just current s1 passers,
    # because tracked items are historical stage-1 passes.
    all_tracked = out.copy()

    # Stats independent of windows
    stats = {
        "stage1_delta": float(stage1_delta),
        "stage2_delta": float(stage2_delta),
        "stage1_pass_count": int(s1_pass.shape[0]),
        "stage2_pass_count": int(final_pass.shape[0]),
        "stage2_scored_total": int(has_stage2.sum()),
    }

    # Per-window precision
    rows: list[dict[str, Any]] = []
    for hours in WINDOW_HOURS:
        oc = outcome_col(hours)
        ec = evaluated_col(hours)

        # --- final_stage2_pass precision ---
        scope = final_pass[final_pass[ec].notna()].copy()
        if scope.empty:
            rows.append({
                "stage1_delta": float(stage1_delta),
                "stage2_delta": float(stage2_delta),
                "window": window_label(hours),
                "group": "final_stage2_pass",
                "evaluated_count": 0,
                "sold_count": 0,
                "precision": np.nan,
            })
        else:
            sold = bool_series(scope[oc])
            rows.append({
                "stage1_delta": float(stage1_delta),
                "stage2_delta": float(stage2_delta),
                "window": window_label(hours),
                "group": "final_stage2_pass",
                "evaluated_count": int(len(scope)),
                "sold_count": int(sold.sum()),
                "precision": float(sold.mean()),
            })

        # --- stage1_pass_stage2_reject precision (false negative check) ---
        rejected = s1_pass[~s1_pass["_s2_new_pass"]].copy()
        scope = rejected[rejected[ec].notna()].copy()
        if scope.empty:
            rows.append({
                "stage1_delta": float(stage1_delta),
                "stage2_delta": float(stage2_delta),
                "window": window_label(hours),
                "group": "stage1_pass_stage2_reject",
                "evaluated_count": 0,
                "sold_count": 0,
                "precision": np.nan,
            })
        else:
            sold = bool_series(scope[oc])
            rows.append({
                "stage1_delta": float(stage1_delta),
                "stage2_delta": float(stage2_delta),
                "window": window_label(hours),
                "group": "stage1_pass_stage2_reject",
                "evaluated_count": int(len(scope)),
                "sold_count": int(sold.sum()),
                "precision": float(sold.mean()),
            })

        # --- all_stage1_pass precision (baseline) ---
        # Use all tracked items because they all historically passed stage 1.
        scope = all_tracked[all_tracked[ec].notna()].copy()
        if scope.empty:
            rows.append({
                "stage1_delta": float(stage1_delta),
                "stage2_delta": float(stage2_delta),
                "window": window_label(hours),
                "group": "all_stage1_pass",
                "evaluated_count": 0,
                "sold_count": 0,
                "precision": np.nan,
            })
        else:
            sold = bool_series(scope[oc])
            rows.append({
                "stage1_delta": float(stage1_delta),
                "stage2_delta": float(stage2_delta),
                "window": window_label(hours),
                "group": "all_stage1_pass",
                "evaluated_count": int(len(scope)),
                "sold_count": int(sold.sum()),
                "precision": float(sold.mean()),
            })

    precision_df = pd.DataFrame(rows)
    return {"stats": stats, "precision": precision_df}


def pivot_summary(precision_df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Pivot precision table to wide format for a specific group."""
    sub = precision_df[precision_df["group"] == group].copy()
    if sub.empty:
        return pd.DataFrame()
    # Keep only windows with at least some evaluated items
    sub = sub[sub["evaluated_count"] > 0].copy()
    if sub.empty:
        return pd.DataFrame()
    return sub[["window", "evaluated_count", "sold_count", "precision"]].copy()


def run_sweep(
    out_dir: Path,
    stage1_deltas: list[float],
    stage2_deltas: list[float],
) -> Path:
    out_dir = assert_experiment_path(out_dir)
    tracked_path = out_dir / "tracked_items.csv"
    tracked = read_csv_or_empty(tracked_path)
    if tracked.empty:
        print(f"No tracked items found: {tracked_path}")
        return out_dir

    all_stats: list[dict[str, Any]] = []
    all_precision_rows: list[pd.DataFrame] = []

    for s1_delta in stage1_deltas:
        for s2_delta in stage2_deltas:
            result = compute_precision(tracked, s1_delta, s2_delta)
            all_stats.append(result["stats"])
            all_precision_rows.append(result["precision"])

    stats_df = pd.DataFrame(all_stats)
    precision_all = pd.concat(all_precision_rows, ignore_index=True)

    # Merge stats into precision rows for convenience
    stats_renamed = stats_df.rename(columns={
        "stage1_pass_count": "Stage1PassCount",
        "stage2_pass_count": "Stage2PassCount",
        "stage2_scored_total": "Stage2ScoredTotal",
    })
    precision_all = precision_all.merge(
        stats_renamed,
        on=["stage1_delta", "stage2_delta"],
        how="left",
    )

    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    precision_path = reports_dir / "threshold_sweep_precision.csv"
    precision_all.to_csv(precision_path, index=False)
    stats_path = reports_dir / "threshold_sweep_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    # Markdown report focused on precision changes
    lines = [
        "# Cascade Threshold Sweep Report",
        "",
        f"Run folder: `{out_dir}`",
        "",
        f"Tracked items: `{len(tracked)}`",
        f"Items with Stage2 score: `{int(tracked['Stage2Score'].notna().sum())}`",
        f"Items evaluated at 24h: `{int(tracked['evaluated_at_24h'].notna().sum())}`",
        "",
        "## Config",
        "",
        f"- Stage-1 deltas: `{stage1_deltas}`",
        f"- Stage-2 deltas: `{stage2_deltas}`",
        "",
        "## Summary: Final Stage-2 Pass Precision",
        "",
    ]

    # Build a compact table: one row per delta combo, columns for key windows
    key_windows = ["24h", "48h", "168h"]
    summary_rows: list[dict[str, Any]] = []
    for _, row in stats_df.iterrows():
        s1 = row["stage1_delta"]
        s2 = row["stage2_delta"]
        sub = precision_all[
            (precision_all["stage1_delta"] == s1)
            & (precision_all["stage2_delta"] == s2)
            & (precision_all["group"] == "final_stage2_pass")
        ].copy()
        out_row: dict[str, Any] = {
            "stage1_delta": s1,
            "stage2_delta": s2,
            "s1_pass": int(row["stage1_pass_count"]),
            "s2_pass": int(row["stage2_pass_count"]),
        }
        for w in key_windows:
            match = sub[sub["window"] == w]
            if not match.empty:
                out_row[f"eval_{w}"] = int(match.iloc[0]["evaluated_count"])
                out_row[f"sold_{w}"] = int(match.iloc[0]["sold_count"])
                out_row[f"prec_{w}"] = float(match.iloc[0]["precision"]) if pd.notna(match.iloc[0]["precision"]) else np.nan
            else:
                out_row[f"eval_{w}"] = 0
                out_row[f"sold_{w}"] = 0
                out_row[f"prec_{w}"] = np.nan
        summary_rows.append(out_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = reports_dir / "threshold_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    header = [
        "| s1_delta | s2_delta | s1_pass | s2_pass | eval_24h | sold_24h | prec_24h | eval_48h | sold_48h | prec_48h | eval_168h | sold_168h | prec_168h |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(header)
    for _, row in summary_df.iterrows():
        def fmt(val):
            if pd.isna(val):
                return ""
            if isinstance(val, float):
                return f"{val:.3f}"
            return str(int(val))
        lines.append(
            f"| {fmt(row['stage1_delta'])} | {fmt(row['stage2_delta'])} | {fmt(row['s1_pass'])} | {fmt(row['s2_pass'])} | "
            f"{fmt(row['eval_24h'])} | {fmt(row['sold_24h'])} | {fmt(row['prec_24h'])} | "
            f"{fmt(row['eval_48h'])} | {fmt(row['sold_48h'])} | {fmt(row['prec_48h'])} | "
            f"{fmt(row['eval_168h'])} | {fmt(row['sold_168h'])} | {fmt(row['prec_168h'])} |"
        )

    # Stage-1 only sweep section
    lines.extend(["", "## Stage-1 Only Sweep (holding Stage-2 delta = 0)", ""])
    s1_only = summary_df[summary_df["stage2_delta"] == 0.0].copy().sort_values("stage1_delta")
    if not s1_only.empty:
        lines.extend(header)
        for _, row in s1_only.iterrows():
            def fmt(val):
                if pd.isna(val):
                    return ""
                if isinstance(val, float):
                    return f"{val:.3f}"
                return str(int(val))
            lines.append(
                f"| {fmt(row['stage1_delta'])} | {fmt(row['stage2_delta'])} | {fmt(row['s1_pass'])} | {fmt(row['s2_pass'])} | "
                f"{fmt(row['eval_24h'])} | {fmt(row['sold_24h'])} | {fmt(row['prec_24h'])} | "
                f"{fmt(row['eval_48h'])} | {fmt(row['sold_48h'])} | {fmt(row['prec_48h'])} | "
                f"{fmt(row['eval_168h'])} | {fmt(row['sold_168h'])} | {fmt(row['prec_168h'])} |"
            )

    # Stage-2 only sweep section
    lines.extend(["", "## Stage-2 Only Sweep (holding Stage-1 delta = 0)", ""])
    s2_only = summary_df[summary_df["stage1_delta"] == 0.0].copy().sort_values("stage2_delta")
    if not s2_only.empty:
        lines.extend(header)
        for _, row in s2_only.iterrows():
            def fmt(val):
                if pd.isna(val):
                    return ""
                if isinstance(val, float):
                    return f"{val:.3f}"
                return str(int(val))
            lines.append(
                f"| {fmt(row['stage1_delta'])} | {fmt(row['stage2_delta'])} | {fmt(row['s1_pass'])} | {fmt(row['s2_pass'])} | "
                f"{fmt(row['eval_24h'])} | {fmt(row['sold_24h'])} | {fmt(row['prec_24h'])} | "
                f"{fmt(row['eval_48h'])} | {fmt(row['sold_48h'])} | {fmt(row['prec_48h'])} | "
                f"{fmt(row['eval_168h'])} | {fmt(row['sold_168h'])} | {fmt(row['prec_168h'])} |"
            )

    lines.append("")
    report_path = reports_dir / "threshold_sweep_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # JSON manifest
    write_json(
        reports_dir / "threshold_sweep_manifest.json",
        {
            "run_dir": str(out_dir),
            "tracked_rows": int(len(tracked)),
            "stage2_scored_rows": int(tracked["Stage2Score"].notna().sum()),
            "stage1_deltas": stage1_deltas,
            "stage2_deltas": stage2_deltas,
            "outputs": {
                "precision": str(precision_path),
                "stats": str(stats_path),
                "summary": str(summary_path),
                "report": str(report_path),
            },
        },
    )

    print(f"Report written to: {report_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Precision CSV: {precision_path}")
    return report_path


def latest_run() -> Path | None:
    if not LIVE_RUNS_DIR.exists():
        return None
    runs = [p for p in LIVE_RUNS_DIR.iterdir() if p.is_dir()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrospective threshold sweep on cascade benchmark.")
    parser.add_argument("--out-dir", default=None, help="Cascade run directory. Defaults to latest run.")
    parser.add_argument(
        "--stage1-delta",
        action="append",
        type=float,
        default=[],
        help="Delta to add to stage-1 threshold. Can specify multiple.",
    )
    parser.add_argument(
        "--stage2-delta",
        action="append",
        type=float,
        default=[],
        help="Delta to add to stage-2 threshold. Can specify multiple.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        current = latest_run()
        if current is None:
            print("No runs found. Use --out-dir.")
            return 1
        out_dir = current

    stage1_deltas = args.stage1_delta if args.stage1_delta else [0.0]
    stage2_deltas = args.stage2_delta if args.stage2_delta else [0.0]

    # Always include baseline (0,0)
    if 0.0 not in stage1_deltas:
        stage1_deltas = [0.0] + stage1_deltas
    if 0.0 not in stage2_deltas:
        stage2_deltas = [0.0] + stage2_deltas

    run_sweep(out_dir, sorted(stage1_deltas), sorted(stage2_deltas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
