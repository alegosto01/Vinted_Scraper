#!/usr/bin/env python3
from __future__ import annotations

import argparse
import __main__
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.deal_finder.model_sweep import (
    RulePriceScorer,
    add_engineered_snapshot_features,
    add_image_features,
    prepare_sweep_frame,
)
from experiments.deal_finder.modeling import load_pickle, score_with_model
from experiments.full_scrape_model.paths import (
    MODELS_DIR,
    OFFLINE_RUNS_DIR,
    REPORTS_DIR,
    ROOT,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_manifest,
)


setattr(__main__, "RulePriceScorer", RulePriceScorer)


DEFAULT_SWEEP_RUN = "sold_status_sweep_20260515_101018"
DEFAULT_EXCLUDED_SEARCHES = {"Borse_Griffate", "Scarpe_Griffate"}
ID_COLUMNS = ["item_id", "Dataid", "Title", "Brand", "Size", "Price", "Likes", "Link"]
PLOT_ORDER = [
    "logistic_v1_baseline",
    "logistic_snapshot_v2",
    "sgd_text_numeric_v1",
    "linear_svm_calibrated_v1",
    "numeric_tree_v1",
    "rules_price_v1",
    "visual_basic_v1",
]


def sweep_run_dir(run_name: str) -> Path:
    return OFFLINE_RUNS_DIR / run_name


def normalize_searches(searches: list[str]) -> list[str]:
    normalized: list[str] = []
    for search in searches:
        value = str(search).strip()
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metadata_rows(run_name: str, searches: list[str]) -> list[dict[str, Any]]:
    wanted = {search.lower() for search in searches}
    rows: list[dict[str, Any]] = []
    for path in sorted(MODELS_DIR.glob(f"{run_name}_*_metadata.json")):
        metadata = read_json(path)
        search = str(metadata.get("search_name", "")).strip()
        artifact = Path(str(metadata.get("artifact_path", "")))
        if search.lower() not in wanted or not artifact.exists():
            continue
        metadata["metadata_path"] = str(path)
        metadata["artifact_path"] = str(artifact)
        rows.append(metadata)
    if not rows:
        raise FileNotFoundError(f"No saved model metadata found for run {run_name}")
    return rows


def available_searches(run_dir: Path, *, include_excluded: bool) -> list[str]:
    datasets_dir = run_dir / "datasets"
    searches = [path.stem for path in sorted(datasets_dir.glob("*.csv"))]
    if not include_excluded:
        searches = [search for search in searches if search not in DEFAULT_EXCLUDED_SEARCHES]
    return searches


def score_frame(dataset_path: Path, *, with_images: bool) -> pd.DataFrame:
    raw = pd.read_csv(dataset_path, low_memory=False)
    frame = prepare_sweep_frame(raw)
    frame = add_engineered_snapshot_features(frame)
    if with_images:
        frame = add_image_features(frame)
    return frame


def sort_approaches(values: list[str]) -> list[str]:
    order = {approach: idx for idx, approach in enumerate(PLOT_ORDER)}
    return sorted(values, key=lambda value: (order.get(value, len(order)), value))


def prepare_search_frames(dataset_path: Path, model_rows: list[dict[str, Any]]) -> dict[bool, pd.DataFrame]:
    needs_images = any(bool(row.get("requires_images")) for row in model_rows)
    base = score_frame(dataset_path, with_images=False)
    frames = {False: base}
    if needs_images:
        frames[True] = add_image_features(base)
    return frames


def numeric(value: object) -> float:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(converted) if pd.notna(converted) else float("nan")


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def score_one_model(frame: pd.DataFrame, metadata: dict[str, Any], best_approach: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    model = load_pickle(Path(metadata["artifact_path"]))
    scores = np.asarray(score_with_model(model, frame), dtype=float)
    scores = np.clip(scores, 0.0, 1.0)
    threshold = numeric(metadata.get("threshold"))
    above = scores >= threshold if pd.notna(threshold) else np.zeros(len(frame), dtype=bool)
    sold = pd.to_numeric(frame["offline_sold_label"], errors="coerce").fillna(0).astype(int)
    approach = str(metadata.get("approach", ""))
    search_name = str(metadata.get("search_name", ""))
    keep = [col for col in ID_COLUMNS if col in frame.columns]
    item_scores = frame[keep].copy()
    item_scores["search_name"] = search_name
    item_scores["approach"] = approach
    item_scores["feature_policy"] = str(metadata.get("feature_policy", ""))
    item_scores["score"] = scores
    item_scores["threshold"] = threshold
    item_scores["above_threshold"] = above
    item_scores["offline_sold_label"] = sold.to_numpy()
    item_scores["is_best_by_search"] = approach == best_approach
    summary = {
        "search_name": search_name,
        "approach": approach,
        "feature_policy": str(metadata.get("feature_policy", "")),
        "requires_images": bool(metadata.get("requires_images")),
        "sampled_for_visual_during_sweep": bool(metadata.get("sampled_for_visual")),
        "sweep_input_rows": int(numeric(metadata.get("input_rows"))) if pd.notna(numeric(metadata.get("input_rows"))) else np.nan,
        "sweep_rows": int(numeric(metadata.get("sweep_rows"))) if pd.notna(numeric(metadata.get("sweep_rows"))) else np.nan,
        "fit_train_rows": int(numeric(metadata.get("fit_train_rows"))) if pd.notna(numeric(metadata.get("fit_train_rows"))) else np.nan,
        "all_offline_rows_scored": int(len(frame)),
        "offline_sold_rows": int(sold.sum()),
        "offline_sold_base_rate": safe_divide(int(sold.sum()), len(frame)),
        "threshold": threshold,
        "above_threshold_count": int(above.sum()),
        "all_data_precision_above_threshold": safe_divide(int(sold[above].sum()), int(above.sum())),
        "all_data_recall_above_threshold": safe_divide(int(sold[above].sum()), int(sold.sum())),
        "score_min": float(np.nanmin(scores)) if len(scores) else np.nan,
        "score_p10": float(np.nanquantile(scores, 0.10)) if len(scores) else np.nan,
        "score_median": float(np.nanmedian(scores)) if len(scores) else np.nan,
        "score_p90": float(np.nanquantile(scores, 0.90)) if len(scores) else np.nan,
        "score_max": float(np.nanmax(scores)) if len(scores) else np.nan,
        "is_best_by_search": approach == best_approach,
        "metadata_path": str(metadata.get("metadata_path", "")),
    }
    return item_scores, summary


def score_all_models(run_dir: Path, model_rows: list[dict[str, Any]], searches: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_path = run_dir / "best_by_search.csv"
    best = pd.read_csv(best_path, low_memory=False) if best_path.exists() else pd.DataFrame()
    best_map = (
        best.drop_duplicates("search_name", keep="first").set_index("search_name")["approach"].astype(str).to_dict()
        if not best.empty and {"search_name", "approach"}.issubset(best.columns)
        else {}
    )
    score_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for search in searches:
        search_models = [row for row in model_rows if str(row.get("search_name", "")) == search]
        if not search_models:
            continue
        dataset_path = run_dir / "datasets" / f"{search}.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found for search {search}: {dataset_path}")
        frames = prepare_search_frames(dataset_path, search_models)
        best_approach = best_map.get(search, "")
        for metadata in sorted(search_models, key=lambda row: sort_approaches([str(row.get("approach", ""))])[0]):
            frame = frames[bool(metadata.get("requires_images"))]
            scored, summary = score_one_model(frame, metadata, best_approach)
            score_parts.append(scored)
            summaries.append(summary)
            print(
                f"[sweep_score_distribution] search={search} approach={summary['approach']} rows={len(scored)}",
                flush=True,
            )
    if not score_parts:
        raise ValueError("No score rows were generated.")
    summary_frame = pd.DataFrame(summaries)
    summary_frame["_approach_order"] = summary_frame["approach"].map(
        {approach: idx for idx, approach in enumerate(PLOT_ORDER)}
    ).fillna(len(PLOT_ORDER))
    summary_frame = summary_frame.sort_values(["search_name", "_approach_order", "approach"]).drop(columns="_approach_order")
    return pd.concat(score_parts, ignore_index=True), summary_frame.reset_index(drop=True)


def plot_search(search_scores: pd.DataFrame, search: str, out_path: Path) -> None:
    approaches = sort_approaches(search_scores["approach"].dropna().astype(str).unique().tolist())
    cols = 2
    rows = int(np.ceil(len(approaches) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, max(4.4 * rows, 8)), sharex=True)
    axes_flat = np.asarray(axes).reshape(-1).tolist()
    bins = np.linspace(0.0, 1.0, 31)
    for ax, approach in zip(axes_flat, approaches):
        group = search_scores[search_scores["approach"].astype(str) == approach].copy()
        sold = pd.to_numeric(group["offline_sold_label"], errors="coerce").fillna(0).astype(int).eq(1)
        not_sold_scores = pd.to_numeric(group.loc[~sold, "score"], errors="coerce").dropna()
        sold_scores = pd.to_numeric(group.loc[sold, "score"], errors="coerce").dropna()
        ax.hist(
            [not_sold_scores, sold_scores],
            bins=bins,
            stacked=True,
            color=["#59636e", "#1f9d72"],
            edgecolor="#f7f3ed",
            linewidth=0.35,
            label=["offline not sold", "offline sold"],
        )
        threshold = numeric(group["threshold"].iloc[0]) if not group.empty else np.nan
        if pd.notna(threshold):
            ax.axvline(threshold, color="#b42318", linestyle="--", linewidth=1.4)
            ax.text(
                threshold,
                0.98,
                f" threshold {threshold:.4f}",
                color="#7a271a",
                fontsize=8,
                ha="left",
                va="top",
                rotation=90,
                transform=ax.get_xaxis_transform(),
            )
        best = bool(group["is_best_by_search"].fillna(False).astype(bool).any())
        best_text = " | best by search" if best else ""
        ax.set_title(f"{approach}{best_text}\nn={len(group)}", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylabel("Offline rows")
        ax.grid(axis="y", alpha=0.18)
    for ax in axes_flat[len(approaches) :]:
        ax.axis("off")
    for ax in axes_flat[-cols:]:
        if ax.axison:
            ax.set_xlabel("Score")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(f"All offline sold-status scores by model: {search}", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 0.985, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def pct(value: object) -> str:
    number = numeric(value)
    return "" if pd.isna(number) else f"{100.0 * number:.1f}%"


def num(value: object, digits: int = 4) -> str:
    number = numeric(value)
    return "" if pd.isna(number) else f"{number:.{digits}f}"


def integer(value: object) -> str:
    number = numeric(value)
    return "" if pd.isna(number) else str(int(number))


def report_table(summary: pd.DataFrame) -> list[str]:
    cols = [
        ("approach", "Model"),
        ("threshold", "Threshold"),
        ("all_offline_rows_scored", "Rows"),
        ("offline_sold_base_rate", "Sold base"),
        ("above_threshold_count", "Above thr"),
        ("all_data_precision_above_threshold", "Precision"),
        ("all_data_recall_above_threshold", "Recall"),
        ("fit_train_rows", "Fit train"),
    ]
    lines = [
        "| " + " | ".join(label for _col, label in cols) + " |",
        "| " + " | ".join("---" if col == "approach" else "---:" for col, _label in cols) + " |",
    ]
    for _idx, row in summary.iterrows():
        model = str(row["approach"])
        if bool(row.get("is_best_by_search")):
            model += " (best)"
        formatters = {
            "approach": lambda _value: f"`{model}`",
            "threshold": num,
            "all_offline_rows_scored": integer,
            "offline_sold_base_rate": pct,
            "above_threshold_count": integer,
            "all_data_precision_above_threshold": pct,
            "all_data_recall_above_threshold": pct,
            "fit_train_rows": integer,
        }
        lines.append("| " + " | ".join(str(formatters[col](row.get(col))) for col, _label in cols) + " |")
    return lines


def write_report(out_dir: Path, *, run_name: str, searches: list[str], summary: pd.DataFrame) -> Path:
    path = assert_experiment_path(out_dir / "all_model_score_distribution_report.md")
    lines = [
        "# Sold-Status Sweep Score Distributions",
        "",
        f"Source sweep: `{run_name}`.",
        "",
        "The plots score every saved model for a search on all eligible deduplicated offline sold/not-sold rows in that search dataset.",
        "That population includes rows used for fitting, validation, and offline test selection in the original sweep.",
        "Use these distributions to inspect score shape; use held-out test metrics for performance claims.",
        "",
        "## Search Plots",
        "",
    ]
    for search in searches:
        plot_name = f"{search}_all_model_score_distributions.png"
        if (out_dir / "plots" / plot_name).exists():
            lines.append(f"- `{search}`: `plots/{plot_name}`")
    lines.extend(
        [
            "",
            "## Model Summaries",
            "",
        ]
    )
    for search in searches:
        search_summary = summary[summary["search_name"].astype(str) == search].copy()
        if search_summary.empty:
            continue
        lines.extend([f"### {search}", "", *report_table(search_summary), ""])
    lines.extend(
        [
            "## Files",
            "",
            "- `all_model_scores.csv`: long row-level scores for every model/search pair.",
            "- `all_model_score_summary.csv`: thresholds, all-data precision/recall, and score quantiles by model/search.",
            "- `plots/`: one distribution figure per search; sold and not-sold labels are stacked in each subplot.",
            "",
            "Visual models were originally trained with a capped visual sweep sample when the dataset exceeded the cap.",
            "The summary keeps `sweep_rows` and `fit_train_rows` so those model distributions can be read with that context.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot all-data score distributions for every saved model in a sold-status sweep.")
    parser.add_argument("--run", default=DEFAULT_SWEEP_RUN)
    parser.add_argument("--search", action="append", default=[])
    parser.add_argument("--include-excluded", action="store_true")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_experiment_dirs()
    run_dir = sweep_run_dir(args.run)
    if not run_dir.exists():
        raise FileNotFoundError(f"Sweep run not found: {run_dir}")
    searches = normalize_searches(args.search) or available_searches(run_dir, include_excluded=args.include_excluded)
    models = metadata_rows(args.run, searches)
    scored, summary = score_all_models(run_dir, models, searches)
    out_dir = assert_experiment_path(
        Path(args.out_dir) if args.out_dir else REPORTS_DIR / run_id(f"{args.run}_score_distributions")
    )
    plots_dir = assert_experiment_path(out_dir / "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_dir / "all_model_scores.csv", index=False)
    summary.to_csv(out_dir / "all_model_score_summary.csv", index=False)
    plotted_searches: list[str] = []
    for search in searches:
        search_scores = scored[scored["search_name"].astype(str) == search]
        if search_scores.empty:
            continue
        plot_search(search_scores, search, plots_dir / f"{search}_all_model_score_distributions.png")
        plotted_searches.append(search)
    report_path = write_report(out_dir, run_name=args.run, searches=plotted_searches, summary=summary)
    write_manifest(
        out_dir / "manifest.json",
        command="full_scrape_model.report_sweep_score_distributions",
        extra={
            "source_run": args.run,
            "source_run_dir": str(run_dir),
            "searches": plotted_searches,
            "report_path": str(report_path),
            "score_population": "all_eligible_deduplicated_offline_rows",
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "report_path": str(report_path), "searches": plotted_searches}, indent=2))


if __name__ == "__main__":
    main()
