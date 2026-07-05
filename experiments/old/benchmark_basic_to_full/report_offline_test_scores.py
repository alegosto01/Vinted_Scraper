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

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.old.benchmark_basic_to_full.paths import (
    REPORTS_DIR,
    ROOT,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_manifest,
)


DEFAULT_STAGE1_RUN = "student_binary_topk_20260518_085123"
DEFAULT_STAGE2_RUN = "sold_status_feature_modalities_20260518_082505"
DEFAULT_SEARCHES = (
    "nike",
    "ps4",
    "gucci",
    "prada",
    "griffati_uomo_all",
    "griffati_donna_all",
)
SEARCH_ALIASES = {
    "griffati_donna": "griffati_donna_all",
    "griffati_uomo": "griffati_uomo_all",
}
STAGE1_REQUIRED_COLUMNS = {
    "SearchName",
    "offline_sold_label",
    "StudentModel",
    "StudentScore",
    "StudentThreshold",
    "TeacherModel",
    "TeacherScore",
    "TeacherThreshold",
}


def bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.fillna("").astype(str).str.strip().str.lower()
    return numeric.fillna(0).ne(0) | text.isin({"true", "t", "yes", "y"})


def numeric_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def normalize_searches(searches: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for search in searches:
        name = str(search).strip().lower()
        if not name:
            continue
        name = SEARCH_ALIASES.get(name, name)
        if name not in normalized:
            normalized.append(name)
    return normalized


def stage1_run_dir(run_name: str) -> Path:
    return ROOT / "experiments" / "old" / "teacher_student_basic_filter" / "data" / "offline_runs" / run_name


def stage2_run_dir(run_name: str) -> Path:
    return ROOT / "experiments" / "old" / "full_scrape_model" / "data" / "offline_runs" / run_name


def read_offline_scores(run_dir: Path, searches: list[str]) -> pd.DataFrame:
    path = run_dir / "test_scored_items.csv"
    if not path.exists():
        raise FileNotFoundError(f"Stage-1 offline test scores not found: {path}")
    scored = pd.read_csv(path, low_memory=False)
    missing = STAGE1_REQUIRED_COLUMNS.difference(scored.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    scored["SearchName"] = scored["SearchName"].fillna("").astype(str).str.strip().str.lower()
    scored = scored[scored["SearchName"].isin(searches)].copy()
    scored["offline_sold_label"] = numeric_series(scored["offline_sold_label"]).fillna(0).astype(int)
    for score_col in ("StudentScore", "StudentThreshold", "TeacherScore", "TeacherThreshold"):
        scored[score_col] = numeric_series(scored[score_col])
    scored["StudentPass"] = scored["StudentScore"] >= scored["StudentThreshold"]
    scored["TeacherPass"] = scored["TeacherScore"] >= scored["TeacherThreshold"]
    return scored.reset_index(drop=True)


def top_precision(frame: pd.DataFrame, *, score_col: str, target_col: str, k: int = 10) -> float:
    top = frame.sort_values(score_col, ascending=False, kind="stable").head(min(k, len(frame)))
    return safe_divide(int(bool_series(top[target_col]).sum()), len(top))


def stage1_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for search, group in scored.groupby("SearchName", sort=True):
        sold = group["offline_sold_label"].eq(1)
        stage1_pass = bool_series(group["StudentPass"])
        stage2_pass = bool_series(group["TeacherPass"])
        selected = group.loc[stage1_pass]
        rows.append(
            {
                "search_name": search,
                "test_items": int(len(group)),
                "offline_sold_base_rate": safe_divide(int(sold.sum()), len(group)),
                "stage1_model": first_text(group["StudentModel"]),
                "stage1_threshold": first_number(group["StudentThreshold"]),
                "stage1_pass_count": int(stage1_pass.sum()),
                "stage1_pass_rate": safe_divide(int(stage1_pass.sum()), len(group)),
                "stage1_sold_precision": safe_divide(
                    int(selected["offline_sold_label"].eq(1).sum()), len(selected)
                ),
                "stage1_stage2_precision": safe_divide(int((stage1_pass & stage2_pass).sum()), int(stage1_pass.sum())),
                "stage1_stage2_recall": safe_divide(int((stage1_pass & stage2_pass).sum()), int(stage2_pass.sum())),
                "stage1_top10_sold_precision": top_precision(
                    group.assign(_sold=sold), score_col="StudentScore", target_col="_sold"
                ),
                "stage1_top10_stage2_precision": top_precision(
                    group.assign(_stage2=stage2_pass), score_col="StudentScore", target_col="_stage2"
                ),
            }
        )
    return pd.DataFrame(rows)


def stage2_summary(scored: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    metric_lookup = (
        metrics[metrics["feature_mode"].astype(str) == "full_scrape_plus_visual"]
        .drop_duplicates("search_name", keep="first")
        .set_index("search_name")
    )
    rows: list[dict[str, object]] = []
    for search, group in scored.groupby("SearchName", sort=True):
        sold = group["offline_sold_label"].eq(1)
        stage2_pass = bool_series(group["TeacherPass"])
        selected = group.loc[stage2_pass]
        metric_row = metric_lookup.loc[search] if search in metric_lookup.index else pd.Series(dtype=object)
        rows.append(
            {
                "search_name": search,
                "test_items": int(len(group)),
                "offline_sold_base_rate": safe_divide(int(sold.sum()), len(group)),
                "stage2_model": first_text(group["TeacherModel"]),
                "stage2_threshold": first_number(group["TeacherThreshold"]),
                "stage2_pass_count": int(stage2_pass.sum()),
                "stage2_pass_rate": safe_divide(int(stage2_pass.sum()), len(group)),
                "stage2_sold_precision": safe_divide(
                    int(selected["offline_sold_label"].eq(1).sum()), len(selected)
                ),
                "stage2_sold_recall": safe_divide(int((stage2_pass & sold).sum()), int(sold.sum())),
                "stage2_top10_sold_precision": top_precision(
                    group.assign(_sold=sold), score_col="TeacherScore", target_col="_sold"
                ),
                "stage2_test_roc_auc": numeric_value(metric_row.get("test_roc_auc")),
                "stage2_test_pr_auc": numeric_value(metric_row.get("test_pr_auc")),
            }
        )
    return pd.DataFrame(rows)


def first_text(values: pd.Series) -> str:
    text = values.dropna().astype(str)
    return text.iloc[0] if not text.empty else ""


def first_number(values: pd.Series) -> float:
    numeric = numeric_series(values).dropna()
    return float(numeric.iloc[0]) if not numeric.empty else float("nan")


def numeric_value(value: object) -> float:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(converted) if pd.notna(converted) else float("nan")


def plot_stage_distribution(
    scored: pd.DataFrame,
    *,
    searches: list[str],
    score_col: str,
    threshold_col: str,
    model_col: str,
    title: str,
    threshold_label: str,
    out_path: Path,
) -> None:
    bins = np.linspace(0.0, 1.0, 31)
    fig, axes = plt.subplots(3, 2, figsize=(16, 13), sharex=True)
    axes_flat = list(axes.flat)
    for ax, search in zip(axes_flat, searches):
        group = scored[scored["SearchName"] == search].copy()
        sold_scores = numeric_series(group.loc[group["offline_sold_label"].eq(1), score_col]).dropna()
        unsold_scores = numeric_series(group.loc[group["offline_sold_label"].eq(0), score_col]).dropna()
        ax.hist(
            [unsold_scores, sold_scores],
            bins=bins,
            stacked=True,
            color=["#59636e", "#1f9d72"],
            edgecolor="#f7f3ed",
            linewidth=0.35,
            label=["offline not sold", "offline sold"],
        )
        threshold = first_number(group[threshold_col])
        if pd.notna(threshold):
            ax.axvline(threshold, color="#b42318", linestyle="--", linewidth=1.5)
            ax.text(
                threshold,
                0.98,
                f" {threshold_label} {threshold:.4f}",
                color="#7a271a",
                fontsize=8,
                ha="left",
                va="top",
                rotation=90,
                transform=ax.get_xaxis_transform(),
            )
        model = first_text(group[model_col])
        ax.set_title(f"{search}\n{model} | n={len(group)}", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.18)
        ax.set_ylabel("Offline test items")
    for ax in axes_flat[len(searches) :]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Score")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 0.98, 0.965))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def pct(value: object) -> str:
    number = numeric_value(value)
    return "" if pd.isna(number) else f"{100.0 * number:.1f}%"


def num(value: object, digits: int = 4) -> str:
    number = numeric_value(value)
    return "" if pd.isna(number) else f"{number:.{digits}f}"


def integer(value: object) -> str:
    number = numeric_value(value)
    return "" if pd.isna(number) else str(int(number))


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]], formatters: dict[str, object]) -> list[str]:
    lines = [
        "| " + " | ".join(label for _col, label in columns) + " |",
        "| " + " | ".join("---:" if col != "search_name" else "---" for col, _label in columns) + " |",
    ]
    for _idx, row in frame.iterrows():
        values: list[str] = []
        for col, _label in columns:
            formatter = formatters.get(col, str)
            values.append(str(formatter(row.get(col, ""))))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    out_dir: Path,
    *,
    stage1_run: str,
    stage2_run: str,
    stage1: pd.DataFrame,
    stage2: pd.DataFrame,
) -> Path:
    path = assert_experiment_path(out_dir / "offline_cascade_test_report.md")
    stage1_cols = [
        ("search_name", "Search"),
        ("stage1_threshold", "S1 thr"),
        ("test_items", "Test items"),
        ("offline_sold_base_rate", "Sold base"),
        ("stage1_pass_count", "S1 pass"),
        ("stage1_sold_precision", "Sold precision"),
        ("stage1_stage2_precision", "S2 precision"),
        ("stage1_top10_stage2_precision", "Top-10 S2 precision"),
    ]
    stage2_cols = [
        ("search_name", "Search"),
        ("stage2_threshold", "S2 thr"),
        ("test_items", "Test items"),
        ("offline_sold_base_rate", "Sold base"),
        ("stage2_pass_count", "S2 pass"),
        ("stage2_sold_precision", "Sold precision"),
        ("stage2_sold_recall", "Sold recall"),
        ("stage2_test_pr_auc", "PR AUC"),
    ]
    formatters = {
        "search_name": lambda value: f"`{value}`",
        "stage1_threshold": num,
        "stage2_threshold": num,
        "test_items": integer,
        "stage1_pass_count": integer,
        "stage2_pass_count": integer,
        "offline_sold_base_rate": pct,
        "stage1_sold_precision": pct,
        "stage1_stage2_precision": pct,
        "stage1_top10_stage2_precision": pct,
        "stage2_sold_precision": pct,
        "stage2_sold_recall": pct,
        "stage2_test_pr_auc": lambda value: num(value, 3),
    }
    lines = [
        "# Offline Cascade Test Scores",
        "",
        "This report uses the offline held-out test rows saved by the teacher-student cascade run.",
        "It does not use live paper-trading rows.",
        "",
        f"- Stage 1 run: `{stage1_run}`",
        f"- Stage 2 full+visual teacher run: `{stage2_run}`",
        "- Stage 1 score: `StudentScore`.",
        "- Stage 2 score: `TeacherScore` saved on the same offline test rows.",
        "- Labels in the score histograms are the offline `sold` vs `not sold` labels.",
        "",
        "## Stage 1 Student",
        "",
        "Stage 1 is trained to forward items that the stage-2 teacher would approve.",
        "`S2 precision` below means the share of threshold-passing stage-1 rows also passed stage 2.",
        "",
        *markdown_table(stage1, stage1_cols, formatters),
        "",
        "## Stage 2 Full + Visual",
        "",
        "Stage 2 is the offline sold-status teacher used by the cascade.",
        "",
        *markdown_table(stage2, stage2_cols, formatters),
        "",
        "## Files",
        "",
        "- `offline_test_scores.csv`: row-level held-out scores for both cascade stages.",
        "- `stage1_offline_test_summary.csv`: stage-1 summary by search.",
        "- `stage2_offline_test_summary.csv`: stage-2 summary by search.",
        "- `stage1_offline_score_distribution.png`: stage-1 score histograms by search.",
        "- `stage2_offline_score_distribution.png`: stage-2 score histograms by search.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def compact_scores(scored: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "SearchName",
        "item_id",
        "Dataid",
        "Title",
        "Brand",
        "Size",
        "Price",
        "Likes",
        "Link",
        "offline_sold_label",
        "StudentModel",
        "StudentScore",
        "StudentThreshold",
        "StudentPass",
        "TeacherModel",
        "TeacherScore",
        "TeacherThreshold",
        "TeacherPass",
    ]
    out = scored[[col for col in keep if col in scored.columns]].copy()
    return out.rename(
        columns={
            "SearchName": "search_name",
            "StudentModel": "stage1_model",
            "StudentScore": "stage1_score",
            "StudentThreshold": "stage1_threshold",
            "StudentPass": "stage1_pass",
            "TeacherModel": "stage2_model",
            "TeacherScore": "stage2_score",
            "TeacherThreshold": "stage2_threshold",
            "TeacherPass": "stage2_pass",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report offline held-out score distributions for the basic-to-full cascade.")
    parser.add_argument("--stage1-run", default=DEFAULT_STAGE1_RUN)
    parser.add_argument("--stage2-run", default=DEFAULT_STAGE2_RUN)
    parser.add_argument("--search", action="append", default=[], help="Search to include. Repeat for multiple searches.")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_experiment_dirs()
    searches = normalize_searches(args.search or DEFAULT_SEARCHES)
    run_dir = stage1_run_dir(args.stage1_run)
    full_run_dir = stage2_run_dir(args.stage2_run)
    metrics_path = full_run_dir / "best_by_search_mode.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Stage-2 best-by-mode metrics not found: {metrics_path}")
    out_dir = assert_experiment_path(
        Path(args.out_dir) if args.out_dir else REPORTS_DIR / run_id("offline_cascade_test_scores")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = read_offline_scores(run_dir, searches)
    missing_searches = [search for search in searches if search not in set(scored["SearchName"])]
    if missing_searches:
        raise ValueError(f"No offline test scores found for searches: {missing_searches}")
    stage2_metrics = pd.read_csv(metrics_path, low_memory=False)
    stage1 = stage1_summary(scored)
    stage2 = stage2_summary(scored, stage2_metrics)

    compact_scores(scored).to_csv(out_dir / "offline_test_scores.csv", index=False)
    stage1.to_csv(out_dir / "stage1_offline_test_summary.csv", index=False)
    stage2.to_csv(out_dir / "stage2_offline_test_summary.csv", index=False)
    plot_stage_distribution(
        scored,
        searches=searches,
        score_col="StudentScore",
        threshold_col="StudentThreshold",
        model_col="StudentModel",
        title="Stage 1 offline held-out score distribution by search",
        threshold_label="S1",
        out_path=out_dir / "stage1_offline_score_distribution.png",
    )
    plot_stage_distribution(
        scored,
        searches=searches,
        score_col="TeacherScore",
        threshold_col="TeacherThreshold",
        model_col="TeacherModel",
        title="Stage 2 offline held-out score distribution by search",
        threshold_label="S2",
        out_path=out_dir / "stage2_offline_score_distribution.png",
    )
    report_path = write_report(
        out_dir,
        stage1_run=args.stage1_run,
        stage2_run=args.stage2_run,
        stage1=stage1,
        stage2=stage2,
    )
    write_manifest(
        out_dir / "manifest.json",
        command="report_offline_test_scores",
        extra={
            "stage1_run": args.stage1_run,
            "stage2_run": args.stage2_run,
            "searches": searches,
            "source_test_scores": str(run_dir / "test_scored_items.csv"),
            "source_stage2_best_by_mode": str(metrics_path),
            "report_path": str(report_path),
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "report_path": str(report_path), "searches": searches}, indent=2))


if __name__ == "__main__":
    main()
