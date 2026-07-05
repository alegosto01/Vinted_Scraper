#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
for _path in (ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.current.basic_5_giant_model.paths import assert_experiment_path, write_json  # noqa: E402
from experiments.current.basic_5_giant_model.report_per_search_thresholds import (  # noqa: E402
    SCORE_PREFIX,
    approach_from_score_col,
    evaluate_group,
    latest_run_dir,
    score_columns,
    threshold_is_qualified,
)
from experiments.current.basic_5_giant_model._deps.deal_finder import model_sweep as base_sweep  # noqa: E402
from experiments.current.basic_5_giant_model._deps.deal_finder.modeling import TARGET_COL, choose_threshold  # noqa: E402


ENSEMBLE_PREFIX = "ensemble__"


@dataclass(frozen=True)
class Scheme:
    name: str
    vote_mode: str
    weight_metric: str


SCHEMES = (
    Scheme("uniform_soft_mean_9", "soft", "uniform"),
    Scheme("auc_soft_weighted", "soft", "auc"),
    Scheme("pr_auc_soft_weighted", "soft", "pr_auc"),
    Scheme("p25_soft_weighted", "soft", "precision_at_25"),
    Scheme("qualified_precision_soft_weighted", "soft", "qualified_precision_count"),
    Scheme("uniform_hard_vote_9", "hard", "uniform"),
    Scheme("auc_hard_weighted_vote", "hard", "auc"),
    Scheme("qualified_precision_hard_weighted_vote", "hard", "qualified_precision_count"),
)


def normalize_weights(values: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(float(value), 0.0) for key, value in values.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        equal = 1.0 / len(clipped) if clipped else 0.0
        return {key: equal for key in clipped}
    return {key: value / total for key, value in clipped.items()}


def validation_metric_rows(
    validation: pd.DataFrame,
    *,
    min_precision: float,
    min_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in score_columns(validation):
        approach = approach_from_score_col(column)
        for search, group in validation.groupby("SearchName", sort=True):
            labels = group[TARGET_COL].astype(int).to_numpy()
            scores = group[column].to_numpy(dtype=float)
            evaluated = base_sweep.evaluate_scores(group, scores, threshold=0.5)
            threshold = choose_threshold(labels, scores, min_precision=min_precision, min_count=min_count)
            p25 = evaluated["precision_at"]["p@25"]["precision"]
            base_rate = evaluated["base_rate"]
            rows.append(
                {
                    "search_name": str(search),
                    "approach": approach,
                    "score_col": column,
                    "base_rate": base_rate,
                    "roc_auc": evaluated["roc_auc"],
                    "pr_auc": evaluated["pr_auc"],
                    "precision_at_25": p25,
                    "threshold": float(threshold["threshold"]),
                    "threshold_count": int(threshold["count"]),
                    "threshold_precision": threshold["precision"],
                    "threshold_positive_count": int(threshold["positive_count"]),
                    "threshold_qualified": threshold_is_qualified(
                        threshold,
                        min_precision=min_precision,
                        min_count=min_count,
                    ),
                }
            )
    return pd.DataFrame(rows)


def raw_weight(row: pd.Series, metric: str) -> float:
    if metric == "uniform":
        return 1.0
    if metric == "auc":
        return max(float(row.get("roc_auc", np.nan)) - 0.5, 0.0)
    if metric == "pr_auc":
        return max(float(row.get("pr_auc", np.nan)) - float(row.get("base_rate", 0.0)), 0.0)
    if metric == "precision_at_25":
        return max(float(row.get("precision_at_25", np.nan)) - float(row.get("base_rate", 0.0)), 0.0)
    if metric == "qualified_precision_count":
        precision = row.get("threshold_precision")
        count = int(row.get("threshold_count", 0))
        if bool(row.get("threshold_qualified", False)) and pd.notna(precision):
            return float(precision) * np.log1p(count)
        return 0.0
    raise ValueError(f"Unsupported weight metric: {metric}")


def weights_for_search(validation_metrics: pd.DataFrame, search: str, metric: str) -> dict[str, float]:
    group = validation_metrics[validation_metrics["search_name"].eq(search)]
    raw = {str(row["approach"]): raw_weight(row, metric) for _, row in group.iterrows()}
    return normalize_weights(raw)


def threshold_map_for_search(validation_metrics: pd.DataFrame, search: str) -> dict[str, float]:
    group = validation_metrics[validation_metrics["search_name"].eq(search)]
    return {str(row["approach"]): float(row["threshold"]) for _, row in group.iterrows()}


def ensemble_scores_for_frame(
    frame: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    scheme: Scheme,
) -> np.ndarray:
    out = np.full(len(frame), np.nan, dtype=float)
    approaches = [approach_from_score_col(col) for col in score_columns(frame)]
    for search, group in frame.groupby("SearchName", sort=True):
        weights = weights_for_search(validation_metrics, str(search), scheme.weight_metric)
        thresholds = threshold_map_for_search(validation_metrics, str(search))
        values = np.zeros(len(group), dtype=float)
        weight_total = 0.0
        for approach in approaches:
            score_col = f"{SCORE_PREFIX}{approach}"
            if score_col not in group.columns:
                continue
            weight = float(weights.get(approach, 0.0))
            if weight <= 0.0:
                continue
            raw_scores = group[score_col].to_numpy(dtype=float)
            if scheme.vote_mode == "soft":
                component = raw_scores
            elif scheme.vote_mode == "hard":
                component = (raw_scores >= float(thresholds.get(approach, 1.0))).astype(float)
            else:
                raise ValueError(f"Unsupported vote mode: {scheme.vote_mode}")
            values += weight * component
            weight_total += weight
        if weight_total > 0.0:
            values = values / weight_total
        out[group.index.to_numpy(dtype=int)] = values
    return out


def add_ensemble_scores(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    schemes: tuple[Scheme, ...] = SCHEMES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_out = validation.copy()
    test_out = test.copy()
    for scheme in schemes:
        col = f"{ENSEMBLE_PREFIX}{scheme.name}"
        validation_out[col] = ensemble_scores_for_frame(validation_out, validation_metrics, scheme)
        test_out[col] = ensemble_scores_for_frame(test_out, validation_metrics, scheme)
    return validation_out, test_out


def ensemble_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(col for col in frame.columns if col.startswith(ENSEMBLE_PREFIX))


def scheme_from_col(column: str) -> str:
    if not column.startswith(ENSEMBLE_PREFIX):
        raise ValueError(f"Not an ensemble column: {column}")
    return column[len(ENSEMBLE_PREFIX) :]


def global_threshold_rows(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    min_precision: float,
    min_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = validation[TARGET_COL].astype(int).to_numpy()
    for column in ensemble_columns(validation):
        scheme = scheme_from_col(column)
        threshold = choose_threshold(
            labels,
            validation[column].to_numpy(dtype=float),
            min_precision=min_precision,
            min_count=min_count,
        )
        test_eval = evaluate_group(test, test[column].to_numpy(dtype=float), float(threshold["threshold"]))
        rows.append(
            {
                "scheme": scheme,
                "scope": "global_threshold",
                "threshold": float(threshold["threshold"]),
                "validation_threshold_count": int(threshold["count"]),
                "validation_threshold_precision": threshold["precision"],
                "validation_threshold_qualified": threshold_is_qualified(
                    threshold,
                    min_precision=min_precision,
                    min_count=min_count,
                ),
                **test_eval,
            }
        )
    return rows


def per_search_threshold_rows(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    min_precision: float,
    min_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    total_positives = int(test[TARGET_COL].astype(int).sum())
    for column in ensemble_columns(validation):
        scheme = scheme_from_col(column)
        for search, validation_group in validation.groupby("SearchName", sort=True):
            test_group = test[test["SearchName"].astype(str).eq(str(search))].copy()
            threshold = choose_threshold(
                validation_group[TARGET_COL].astype(int).to_numpy(),
                validation_group[column].to_numpy(dtype=float),
                min_precision=min_precision,
                min_count=min_count,
            )
            test_eval = evaluate_group(test_group, test_group[column].to_numpy(dtype=float), float(threshold["threshold"]))
            detail_rows.append(
                {
                    "scheme": scheme,
                    "search_name": str(search),
                    "threshold": float(threshold["threshold"]),
                    "validation_threshold_count": int(threshold["count"]),
                    "validation_threshold_precision": threshold["precision"],
                    "validation_threshold_qualified": threshold_is_qualified(
                        threshold,
                        min_precision=min_precision,
                        min_count=min_count,
                    ),
                    **test_eval,
                }
            )
        detail = pd.DataFrame([row for row in detail_rows if row["scheme"] == scheme])
        selected = int(detail["threshold_count"].sum()) if not detail.empty else 0
        positives = int(detail["positive_count"].sum()) if not detail.empty else 0
        summary_rows.append(
            {
                "scheme": scheme,
                "scope": "per_search_thresholds",
                "threshold": np.nan,
                "validation_threshold_count": int(detail["validation_threshold_count"].sum()) if not detail.empty else 0,
                "validation_threshold_precision": np.nan,
                "validation_threshold_qualified": bool(detail["validation_threshold_qualified"].all()) if not detail.empty else False,
                "rows": int(len(test)),
                "positive_rows": total_positives,
                "base_rate": float(total_positives / len(test)) if len(test) else np.nan,
                "threshold_count": selected,
                "threshold_precision": float(positives / selected) if selected else np.nan,
                "threshold_recall": float(positives / total_positives) if total_positives else np.nan,
                "positive_count": positives,
                "precision_at_10": np.nan,
                "precision_at_25": np.nan,
                "precision_at_50": np.nan,
                "roc_auc": base_sweep.evaluate_scores(test, test[column].to_numpy(dtype=float), 0.5)["roc_auc"],
                "pr_auc": base_sweep.evaluate_scores(test, test[column].to_numpy(dtype=float), 0.5)["pr_auc"],
            }
        )
    return detail_rows, summary_rows


def baseline_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "per_search_threshold_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    per = pd.read_csv(path, low_memory=False)
    rows = []
    for approach, group in per.groupby("approach", sort=True):
        selected = int(group["threshold_count"].sum())
        positives = int(group["positive_count"].sum())
        rows.append(
            {
                "scheme": f"single_model__{approach}",
                "scope": "per_search_thresholds",
                "threshold_count": selected,
                "positive_count": positives,
                "threshold_precision": float(positives / selected) if selected else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    run_dir: Path,
    summary: pd.DataFrame,
    detail: pd.DataFrame,
    weights: pd.DataFrame,
    baselines: pd.DataFrame,
    *,
    min_precision: float,
    min_count: int,
) -> Path:
    path = assert_experiment_path(run_dir / "weighted_voting_report.md")

    def fmt(value: object, decimals: int = 3) -> str:
        try:
            if value is None or pd.isna(value):
                return "-"
            return f"{float(value):.{decimals}f}"
        except Exception:
            return str(value)

    per_search = summary[summary["scope"].eq("per_search_thresholds")].copy()
    per_search = per_search.sort_values(
        ["threshold_precision", "threshold_count", "pr_auc"],
        ascending=[False, False, False],
        na_position="last",
    )
    lines = [
        "# Basic 5 Giant Model - Weighted Voting",
        "",
        f"Run folder: `{run_dir}`",
        "",
        f"Weighted ensembles are fit from validation metrics only. Thresholds use min precision `{min_precision:.2f}` and min count `{min_count}`.",
        "",
        "## Weighted Voting With Per-Search Thresholds",
        "",
        "| scheme | selected | positives | precision | recall | AUC | PR AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in per_search.iterrows():
        lines.append(
            f"| {row['scheme']} | {int(row['threshold_count'])} | {int(row['positive_count'])} | "
            f"{fmt(row['threshold_precision'])} | {fmt(row['threshold_recall'])} | "
            f"{fmt(row['roc_auc'])} | {fmt(row['pr_auc'])} |"
        )

    if not baselines.empty:
        xgb = baselines[baselines["scheme"].eq("single_model__xgboost_basic_v1")]
        if not xgb.empty:
            row = xgb.iloc[0]
            lines.extend(
                [
                    "",
                    "## Baseline",
                    "",
                    f"- XGBoost single-model per-search thresholds selected `{int(row['threshold_count'])}` rows with precision `{fmt(row['threshold_precision'])}`.",
                ]
            )
    best = per_search.head(1)
    if not best.empty:
        row = best.iloc[0]
        lines.extend(
            [
                "",
                "## Takeaway",
                "",
                f"- Best weighted voting row: `{row['scheme']}` selected `{int(row['threshold_count'])}` rows at precision `{fmt(row['threshold_precision'])}`.",
            ]
        )
        if not baselines.empty:
            xgb = baselines[baselines["scheme"].eq("single_model__xgboost_basic_v1")]
            if not xgb.empty:
                b = xgb.iloc[0]
                delta_count = int(row["threshold_count"]) - int(b["threshold_count"])
                delta_precision = float(row["threshold_precision"]) - float(b["threshold_precision"])
                lines.append(
                    f"- Versus XGBoost per-search thresholds: count delta `{delta_count:+d}`, precision delta `{delta_precision:+.3f}`."
                )

    if not detail.empty:
        top_scheme = str(per_search.iloc[0]["scheme"]) if not per_search.empty else ""
        top_detail = detail[detail["scheme"].eq(top_scheme)].sort_values("search_name")
        if not top_detail.empty:
            lines.extend(
                [
                    "",
                    f"## Best Scheme By Search: {top_scheme}",
                    "",
                    "| search | threshold | selected | precision | recall |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for _, row in top_detail.iterrows():
                lines.append(
                    f"| {row['search_name']} | {fmt(row['threshold'], 4)} | {int(row['threshold_count'])} | "
                    f"{fmt(row['threshold_precision'])} | {fmt(row['threshold_recall'])} |"
                )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `weighted_voting_summary.csv`: global and per-search-threshold weighted ensemble metrics.",
            "- `weighted_voting_per_search.csv`: per-search rows for each weighted ensemble.",
            "- `weighted_voting_weights.csv`: validation-derived weights per search/model/scheme.",
            "- `weighted_voting_scores_validation.csv` / `weighted_voting_scores_test.csv`: generated ensemble scores.",
            "",
            "## Notes",
            "",
            "- Soft voting averages model scores. Hard voting averages thresholded model votes.",
            "- Weights are derived only from validation metrics, then evaluated on held-out test rows.",
            "- Better precision can come at lower coverage; selected count and recall should be read with precision.",
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
    validation = pd.read_csv(run_dir / "validation_scores.csv", low_memory=False)
    test = pd.read_csv(run_dir / "test_scores.csv", low_memory=False)
    validation_metrics = validation_metric_rows(validation, min_precision=min_precision, min_count=min_count)
    validation_scored, test_scored = add_ensemble_scores(validation, test, validation_metrics)

    global_rows = global_threshold_rows(
        validation_scored,
        test_scored,
        min_precision=min_precision,
        min_count=min_count,
    )
    detail_rows, per_search_rows = per_search_threshold_rows(
        validation_scored,
        test_scored,
        min_precision=min_precision,
        min_count=min_count,
    )
    summary = pd.DataFrame([*global_rows, *per_search_rows])
    detail = pd.DataFrame(detail_rows)

    weight_rows: list[dict[str, Any]] = []
    for scheme in SCHEMES:
        for search in sorted(validation["SearchName"].dropna().astype(str).unique()):
            weights = weights_for_search(validation_metrics, search, scheme.weight_metric)
            for approach, weight in weights.items():
                weight_rows.append(
                    {
                        "scheme": scheme.name,
                        "vote_mode": scheme.vote_mode,
                        "weight_metric": scheme.weight_metric,
                        "search_name": search,
                        "approach": approach,
                        "weight": weight,
                    }
                )
    weights = pd.DataFrame(weight_rows)
    baselines = baseline_summary(run_dir)

    summary_path = assert_experiment_path(run_dir / "weighted_voting_summary.csv")
    detail_path = assert_experiment_path(run_dir / "weighted_voting_per_search.csv")
    weights_path = assert_experiment_path(run_dir / "weighted_voting_weights.csv")
    val_scores_path = assert_experiment_path(run_dir / "weighted_voting_scores_validation.csv")
    test_scores_path = assert_experiment_path(run_dir / "weighted_voting_scores_test.csv")
    summary.to_csv(summary_path, index=False)
    detail.to_csv(detail_path, index=False)
    weights.to_csv(weights_path, index=False)
    validation_scored[[*validation.columns, *ensemble_columns(validation_scored)]].to_csv(val_scores_path, index=False)
    test_scored[[*test.columns, *ensemble_columns(test_scored)]].to_csv(test_scores_path, index=False)

    report_path = write_report(
        run_dir,
        summary,
        detail,
        weights,
        baselines,
        min_precision=min_precision,
        min_count=min_count,
    )
    result = {
        "run_dir": str(run_dir),
        "min_precision": float(min_precision),
        "min_count": int(min_count),
        "schemes": [scheme.name for scheme in SCHEMES],
        "outputs": {
            "summary": str(summary_path),
            "per_search": str(detail_path),
            "weights": str(weights_path),
            "validation_scores": str(val_scores_path),
            "test_scores": str(test_scores_path),
            "report": str(report_path),
        },
    }
    write_json(run_dir / "weighted_voting_summary.json", base_sweep.to_builtin(result))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate weighted voting over the nine basic_5 giant model scores.")
    parser.add_argument("--run-dir", default=None, help="Existing basic_5_giant run dir. Defaults to latest full run.")
    parser.add_argument("--min-precision", type=float, default=base_sweep.PROMOTION_PRECISION)
    parser.add_argument("--min-count", type=int, default=base_sweep.VALIDATION_MIN_COUNT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    print(json.dumps(run_report(run_dir, min_precision=args.min_precision, min_count=args.min_count), indent=2))


if __name__ == "__main__":
    main()
