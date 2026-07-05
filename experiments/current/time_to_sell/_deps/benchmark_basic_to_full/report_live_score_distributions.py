#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.current.time_to_sell._deps.benchmark_basic_to_full.paths import (
    LIVE_RUNS_DIR,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_manifest,
)


STATE_FILE = "tracked_items.csv"
PLOT_FILE = "live_score_distribution_sold_vs_unsold.png"
STAGE_SPECS = (
    {
        "label": "Stage 1 basic student",
        "model_col": "Stage1Model",
        "score_col": "Stage1Score",
        "threshold_col": "Stage1Threshold",
    },
    {
        "label": "Stage 2 full + visual",
        "model_col": "Stage2Model",
        "score_col": "Stage2Score",
        "threshold_col": "Stage2Threshold",
    },
)
SOLD_COLOR = "#007a5e"
UNSOLD_COLOR = "#59636e"
THRESHOLD_COLOR = "#b42318"


def window_label(hours: float) -> str:
    return f"{int(hours)}h" if float(hours).is_integer() else f"{float(hours):g}h"


def evaluated_col(hours: float) -> str:
    return f"evaluated_at_{window_label(hours)}"


def outcome_col(hours: float) -> str:
    return f"sold_within_{window_label(hours)}"


def bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype("string").fillna("").str.strip().str.lower()
    return numeric.fillna(0).ne(0) | text.isin({"true", "t", "yes", "y"})


def numeric_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def first_text(values: pd.Series) -> str:
    text = values.dropna().astype(str).str.strip()
    text = text[text.ne("")]
    return text.iloc[0] if not text.empty else ""


def read_tracked(run_dir: Path) -> pd.DataFrame:
    path = run_dir / STATE_FILE
    if not path.exists():
        raise FileNotFoundError(f"Cascade tracked state not found: {path}")
    tracked = pd.read_csv(path, low_memory=False)
    if tracked.empty:
        raise ValueError(f"Cascade tracked state is empty: {path}")
    return tracked


def latest_run_with_state() -> Path:
    candidates = [path.parent for path in LIVE_RUNS_DIR.glob(f"*/{STATE_FILE}")]
    if not candidates:
        raise FileNotFoundError(f"No live cascade runs with {STATE_FILE} found under {LIVE_RUNS_DIR}")
    return max(candidates, key=lambda path: (path / STATE_FILE).stat().st_mtime)


def item_key(frame: pd.DataFrame) -> pd.Series:
    index = frame.index
    if "tracking_key" in frame:
        stored = frame["tracking_key"].fillna("").astype(str).str.strip()
    else:
        stored = pd.Series("", index=index, dtype=object)
    search = frame.get("SearchName", pd.Series("", index=index)).fillna("").astype(str).str.strip().str.lower()
    if "item_id" in frame:
        item_id = frame["item_id"]
    else:
        item_id = frame.get("Dataid", pd.Series("", index=index))
    fallback = search + "::" + item_id.fillna("").astype(str).str.strip()
    return stored.where(stored.ne(""), fallback)


def unique_tracked_items(tracked: pd.DataFrame) -> pd.DataFrame:
    unique = tracked.copy()
    unique["_tracking_key"] = item_key(unique)
    if unique["_tracking_key"].eq("::").any():
        missing = int(unique["_tracking_key"].eq("::").sum())
        raise ValueError(f"{missing} tracked rows have neither tracking_key nor SearchName/item_id identity")
    if "last_seen_at" in unique:
        unique["_last_seen_at"] = pd.to_datetime(unique["last_seen_at"], errors="coerce", utc=True)
    else:
        unique["_last_seen_at"] = pd.NaT
    unique["_row_order"] = np.arange(len(unique))
    unique = unique.sort_values(["_last_seen_at", "_row_order"], kind="stable")
    unique = unique.drop_duplicates("_tracking_key", keep="last")
    return unique.sort_values("_row_order", kind="stable").reset_index(drop=True)


def label_live_cohort(tracked: pd.DataFrame, *, window_hours: float | None) -> tuple[pd.DataFrame, dict[str, str]]:
    labeled = tracked.copy()
    if window_hours is None:
        if "sold_at" not in labeled:
            raise ValueError("tracked state is missing sold_at")
        sold = pd.to_datetime(labeled["sold_at"], errors="coerce", utc=True).notna()
        cohort_text = "all unique tracked stage-1 pass items labeled by detected sale so far"
        sold_name = "detected sold"
        unsold_name = "not sold yet"
    else:
        evaluated = evaluated_col(window_hours)
        outcome = outcome_col(window_hours)
        missing = [col for col in (evaluated, outcome) if col not in labeled]
        if missing:
            raise ValueError(f"tracked state is missing checkpoint columns for {window_label(window_hours)}: {missing}")
        labeled = labeled[labeled[evaluated].notna()].copy()
        sold = bool_series(labeled[outcome])
        cohort_text = f"unique tracked items evaluated at the {window_label(window_hours)} checkpoint"
        sold_name = f"sold within {window_label(window_hours)}"
        unsold_name = f"not sold by {window_label(window_hours)}"
    if labeled.empty:
        raise ValueError("No tracked items match the requested live score distribution cohort")
    labeled["SoldLabel"] = sold.astype(int).to_numpy()
    labeled["SoldLabelName"] = np.where(sold.to_numpy(), sold_name, unsold_name)
    return labeled, {
        "cohort": cohort_text,
        "sold_name": sold_name,
        "unsold_name": unsold_name,
    }


def normalize_searches(values: Iterable[str]) -> list[str]:
    searches: list[str] = []
    for value in values:
        search = str(value).strip().lower()
        if search and search not in searches:
            searches.append(search)
    return searches


def validate_scores(frame: pd.DataFrame) -> None:
    required = {"SearchName", "Stage1Score", "Stage1Threshold", "Stage2Score", "Stage2Threshold"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"tracked state is missing score distribution columns: {sorted(missing)}")


def threshold_values(values: pd.Series) -> list[float]:
    thresholds = numeric_series(values).dropna().round(8).unique().tolist()
    return sorted(float(value) for value in thresholds)


def threshold_text(values: list[float]) -> str:
    if not values:
        return "threshold unavailable"
    if len(values) == 1:
        return f"thr={values[0]:.4f}"
    return f"thr={min(values):.4f}-{max(values):.4f} ({len(values)} values)"


def weighted_hist(ax: plt.Axes, scores: pd.Series, *, color: str, label: str, fill: bool) -> None:
    if scores.empty:
        return
    weights = np.full(len(scores), 1.0 / len(scores))
    ax.hist(
        scores,
        bins=np.linspace(0.0, 1.0, 31),
        weights=weights,
        color=color,
        alpha=0.32 if fill else 1.0,
        histtype="stepfilled" if fill else "step",
        linewidth=1.6,
        label=label,
    )


def plot_stage_cell(
    ax: plt.Axes,
    group: pd.DataFrame,
    *,
    search: str,
    spec: dict[str, str],
    sold_name: str,
    unsold_name: str,
) -> None:
    scores = numeric_series(group[spec["score_col"]])
    scored = group[scores.notna()].copy()
    scored["_score"] = scores[scores.notna()].clip(0.0, 1.0)
    sold_scores = scored.loc[scored["SoldLabel"].eq(1), "_score"]
    unsold_scores = scored.loc[scored["SoldLabel"].eq(0), "_score"]
    weighted_hist(ax, unsold_scores, color=UNSOLD_COLOR, label=unsold_name, fill=True)
    weighted_hist(ax, sold_scores, color=SOLD_COLOR, label=sold_name, fill=False)

    thresholds = threshold_values(group[spec["threshold_col"]])
    for idx, threshold in enumerate(thresholds):
        ax.axvline(threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.25, alpha=0.9)
        if len(thresholds) <= 2:
            ax.text(
                threshold,
                0.98 - (0.10 * idx),
                f" {threshold:.4f}",
                color="#7a271a",
                fontsize=7,
                ha="left",
                va="top",
                rotation=90,
                transform=ax.get_xaxis_transform(),
            )

    model = first_text(group.get(spec["model_col"], pd.Series(dtype=object)))
    model_text = model or "model unavailable"
    ax.set_title(
        f"{search} | {spec['label']}\n{model_text} | {threshold_text(thresholds)} | "
        f"scored={len(scored)} | sold={len(sold_scores)} | unsold={len(unsold_scores)}",
        fontsize=8.4,
    )
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.18)
    ax.set_ylabel("Share of class")


def plot_distributions(cohort: pd.DataFrame, *, searches: list[str], label_meta: dict[str, str], out_path: Path) -> None:
    fig, axes = plt.subplots(
        len(searches),
        len(STAGE_SPECS),
        figsize=(18, max(3.2 * len(searches), 6.4)),
        sharex=True,
        squeeze=False,
    )
    for row_idx, search in enumerate(searches):
        group = cohort[cohort["SearchName"].astype(str).str.lower().eq(search)].copy()
        for col_idx, spec in enumerate(STAGE_SPECS):
            plot_stage_cell(
                axes[row_idx, col_idx],
                group,
                search=search,
                spec=spec,
                sold_name=label_meta["sold_name"],
                unsold_name=label_meta["unsold_name"],
            )
    for ax in axes[-1, :]:
        ax.set_xlabel("Model score")
    fig.legend(
        [
            Patch(facecolor=UNSOLD_COLOR, edgecolor=UNSOLD_COLOR, alpha=0.32),
            Line2D([0], [0], color=SOLD_COLOR, linewidth=1.6),
        ],
        [label_meta["unsold_name"], label_meta["sold_name"]],
        loc="upper right",
        frameon=False,
    )
    fig.suptitle("Live cascade score distributions by sold outcome", fontsize=16, y=0.998)
    fig.tight_layout(rect=(0, 0, 0.985, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def summary_rows(cohort: pd.DataFrame, searches: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for search in searches:
        group = cohort[cohort["SearchName"].astype(str).str.lower().eq(search)].copy()
        for spec in STAGE_SPECS:
            scores = numeric_series(group[spec["score_col"]])
            scored = group[scores.notna()]
            thresholds = threshold_values(group[spec["threshold_col"]])
            sold = int(scored["SoldLabel"].eq(1).sum())
            rows.append(
                {
                    "search_name": search,
                    "stage": spec["label"],
                    "model": first_text(group.get(spec["model_col"], pd.Series(dtype=object))),
                    "thresholds": ";".join(f"{threshold:.8g}" for threshold in thresholds),
                    "cohort_items": int(len(group)),
                    "scored_items": int(len(scored)),
                    "sold_items": sold,
                    "unsold_items": int(len(scored) - sold),
                    "score_min": float(scores.dropna().min()) if scores.notna().any() else np.nan,
                    "score_median": float(scores.dropna().median()) if scores.notna().any() else np.nan,
                    "score_max": float(scores.dropna().max()) if scores.notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def compact_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "_tracking_key",
        "tracking_key",
        "SearchName",
        "item_id",
        "Dataid",
        "Title",
        "Link",
        "first_stage1_pass_at",
        "last_seen_at",
        "sold_at",
        "SoldLabel",
        "SoldLabelName",
        "Stage1Model",
        "Stage1Score",
        "Stage1Threshold",
        "Stage2Model",
        "Stage2Score",
        "Stage2Threshold",
        "Stage2Passed",
    ]
    return cohort[[col for col in keep if col in cohort.columns]].copy()


def write_report(
    out_dir: Path,
    *,
    run_dir: Path,
    searches: list[str],
    label_meta: dict[str, str],
    summary: pd.DataFrame,
) -> Path:
    path = assert_experiment_path(out_dir / "live_score_distribution_report.md")
    stage1_rows = int(summary.loc[summary["stage"].eq(STAGE_SPECS[0]["label"]), "scored_items"].sum())
    stage2_rows = int(summary.loc[summary["stage"].eq(STAGE_SPECS[1]["label"]), "scored_items"].sum())
    lines = [
        "# Live Cascade Score Distributions",
        "",
        f"Run folder: `{run_dir}`",
        f"Cohort: {label_meta['cohort']}.",
        f"Searches: `{', '.join(searches)}`.",
        "",
        "The plot compares sold and unsold outcome shapes for the two cascade score columns.",
        "Each histogram is normalized within its sold or unsold class so uneven class counts do not hide the score shape.",
        "The red dashed lines are the effective per-search thresholds stored on the tracked rows.",
        "",
        "The live state tracks threshold-passing stage-1 items, so this plot does not include online stage-1 rejects.",
        "Stage-2 panels use only tracked rows with a `Stage2Score`.",
        "",
        "## Counts",
        "",
        f"- Unique plotted cohort items: `{int(summary.groupby('search_name')['cohort_items'].first().sum())}`",
        f"- Stage-1 scored rows in the cohort: `{stage1_rows}`",
        f"- Stage-2 scored rows in the cohort: `{stage2_rows}`",
        "",
        "## Files",
        "",
        f"- `{PLOT_FILE}`: stage-1 and stage-2 sold/unsold score distributions.",
        "- `live_score_distribution_items.csv`: deduplicated live rows used for the plot.",
        "- `live_score_distribution_summary.csv`: per-search model, threshold, count, and score summaries.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot live basic-to-full cascade scores by sold checkpoint outcome.")
    parser.add_argument("--run-dir", default=None, help="Live cascade run. Defaults to the newest run with tracked state.")
    parser.add_argument("--search", action="append", default=[], help="Search to include. Repeat for multiple searches.")
    parser.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help="Evaluated checkpoint used for sold/not-sold labels. Default: 24.",
    )
    parser.add_argument(
        "--all-observed",
        action="store_true",
        help="Label every unique tracked item as detected sold or not sold yet instead of using a matured checkpoint.",
    )
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_experiment_dirs()
    run_dir = assert_experiment_path(Path(args.run_dir) if args.run_dir else latest_run_with_state())
    tracked = unique_tracked_items(read_tracked(run_dir))
    validate_scores(tracked)
    tracked["SearchName"] = tracked["SearchName"].fillna("").astype(str).str.strip().str.lower()
    wanted_searches = normalize_searches(args.search)
    if wanted_searches:
        tracked = tracked[tracked["SearchName"].isin(wanted_searches)].copy()
    cohort, label_meta = label_live_cohort(tracked, window_hours=None if args.all_observed else args.window_hours)
    searches = wanted_searches or sorted(cohort["SearchName"].dropna().astype(str).unique().tolist())
    searches = [search for search in searches if not cohort[cohort["SearchName"].eq(search)].empty]
    if not searches:
        raise ValueError("No searches have live score distribution rows after filtering")

    out_dir = assert_experiment_path(
        Path(args.out_dir) if args.out_dir else run_dir / "reports" / run_id("live_score_distributions")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / PLOT_FILE
    plot_distributions(cohort, searches=searches, label_meta=label_meta, out_path=plot_path)
    compact_cohort(cohort).to_csv(out_dir / "live_score_distribution_items.csv", index=False)
    summary = summary_rows(cohort, searches)
    summary.to_csv(out_dir / "live_score_distribution_summary.csv", index=False)
    report_path = write_report(out_dir, run_dir=run_dir, searches=searches, label_meta=label_meta, summary=summary)
    write_manifest(
        out_dir / "manifest.json",
        command="benchmark_basic_to_full.report_live_score_distributions",
        extra={
            "source_run_dir": str(run_dir),
            "searches": searches,
            "window_hours": None if args.all_observed else float(args.window_hours),
            "label_cohort": label_meta["cohort"],
            "plot_path": str(plot_path),
            "report_path": str(report_path),
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "plot_path": str(plot_path), "report_path": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
