#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.basic_5_stacking.paths import (
    EXPERIMENT_ROOT,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_manifest,
)
from experiments.basic_5_voting import run as voting
from experiments.deal_finder import model_sweep as base_sweep
from experiments.deal_finder.modeling import (
    TARGET_COL,
    choose_threshold,
    load_pickle,
    precision_at_k,
    score_with_model,
    threshold_metrics,
)


ALL_APPROACHES = (*voting.ORIGINAL_APPROACHES, *voting.NEW_APPROACHES)
NON_DUPLICATE_APPROACHES = tuple(
    approach for approach in ALL_APPROACHES if approach != "logistic_snapshot_v2"
)
LEAN_APPROACHES = tuple(approach for approach in NON_DUPLICATE_APPROACHES if approach != "rules_price_v1")
ENSEMBLE_SCORE_PREFIX = "ensemble_score__"
META_STACKER_SETS = (
    ("all_9", ALL_APPROACHES),
    ("nonduplicate_8", NON_DUPLICATE_APPROACHES),
    ("lean_7", LEAN_APPROACHES),
)
MAIN_METHODS = (
    "best_single_validation_auc",
    "sum_9_scores",
    "mean_9_scores",
    "mean_nonduplicate_8_scores",
    "mean_lean_7_scores",
    "validation_top3_mean",
    "validation_top5_mean",
    "per_search_logistic_stacker_all_9",
    "per_search_logistic_stacker_nonduplicate_8",
    "per_search_logistic_stacker_lean_7",
    "global_logistic_stacker_all_9_with_search",
    "global_logistic_stacker_nonduplicate_8_with_search",
    "global_logistic_stacker_lean_7_with_search",
)


def approach_score_col(approach: str) -> str:
    return f"score__{approach}"


def ensemble_score_col(method: str) -> str:
    return f"{ENSEMBLE_SCORE_PREFIX}{method}"


def sold_labels(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(frame[TARGET_COL], errors="coerce").fillna(0).astype(int).to_numpy()


def init_item_scores(frame: pd.DataFrame, *, search: str, split: str) -> pd.DataFrame:
    keep = [col for col in voting.ID_COLUMNS if col in frame.columns]
    scores = frame[keep].copy().reset_index(drop=True)
    scores["search_name"] = search
    scores["split_name"] = split
    scores[TARGET_COL] = sold_labels(frame)
    return scores


def score_search_base_models(
    search: str,
    *,
    seed: int,
    original_run: str,
    new_run: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_frame, test_frame = voting.common_split_frames(search, seed)
    if validation_frame.empty or test_frame.empty:
        raise ValueError(f"Stacking needs non-empty validation and test rows for {search}")

    validation = init_item_scores(validation_frame, search=search, split="validation")
    test = init_item_scores(test_frame, search=search, split="test")
    metadata = [
        *voting.load_metadata(original_run, search, voting.ORIGINAL_APPROACHES, seed),
        *voting.load_metadata(new_run, search, voting.NEW_APPROACHES, seed),
    ]
    for model_metadata in metadata:
        approach = str(model_metadata["approach"])
        model = load_pickle(Path(model_metadata["artifact_path"]))
        validation[approach_score_col(approach)] = np.clip(
            np.asarray(score_with_model(model, validation_frame), dtype=float),
            0.0,
            1.0,
        )
        test[approach_score_col(approach)] = np.clip(
            np.asarray(score_with_model(model, test_frame), dtype=float),
            0.0,
            1.0,
        )
    return validation, test


def component_score(frame: pd.DataFrame, approaches: tuple[str, ...], *, reducer: str) -> np.ndarray:
    columns = [approach_score_col(approach) for approach in approaches]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing component score columns: {missing}")
    values = frame[columns].to_numpy(dtype=float)
    if reducer == "mean":
        return np.nanmean(values, axis=1)
    if reducer == "sum":
        return np.nansum(values, axis=1)
    raise ValueError(f"Unsupported reducer: {reducer}")


def select_top_approaches(frame: pd.DataFrame, approaches: tuple[str, ...], count: int) -> tuple[str, ...]:
    labels = frame[TARGET_COL].astype(int).to_numpy()
    rows = []
    for approach in approaches:
        roc_auc, _pr_auc = voting.safe_auc(labels, frame[approach_score_col(approach)].to_numpy(dtype=float))
        rows.append({"approach": approach, "roc_auc": roc_auc})
    rows.sort(
        key=lambda row: (
            pd.isna(row["roc_auc"]),
            0.0 if pd.isna(row["roc_auc"]) else -float(row["roc_auc"]),
            str(row["approach"]),
        )
    )
    return tuple(str(row["approach"]) for row in rows[:count])


def component_text(approaches: tuple[str, ...]) -> str:
    return "|".join(approaches)


def positive_proba(model: Any, frame: pd.DataFrame | np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    positive_index = classes.index(1)
    return np.asarray(model.predict_proba(frame)[:, positive_index], dtype=float)


def coefficient_rows(
    *,
    search: str,
    method: str,
    scope: str,
    feature_names: list[str],
    coefficients: np.ndarray,
    intercept: float,
) -> list[dict[str, Any]]:
    rows = [
        {
            "search_name": search,
            "method": method,
            "scope": scope,
            "feature": "__intercept__",
            "coefficient": float(intercept),
        }
    ]
    rows.extend(
        {
            "search_name": search,
            "method": method,
            "scope": scope,
            "feature": str(feature),
            "coefficient": float(coefficient),
        }
        for feature, coefficient in zip(feature_names, coefficients, strict=True)
    )
    return rows


def fit_per_search_stacker(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    approaches: tuple[str, ...],
    *,
    method: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    from sklearn.linear_model import LogisticRegression

    columns = [approach_score_col(approach) for approach in approaches]
    model = LogisticRegression(max_iter=2000, solver="liblinear", random_state=base_sweep.DEFAULT_SEED)
    model.fit(validation[columns], validation[TARGET_COL].astype(int))
    rows = coefficient_rows(
        search=str(validation["search_name"].iloc[0]),
        method=method,
        scope="per_search",
        feature_names=list(approaches),
        coefficients=np.asarray(model.coef_[0], dtype=float),
        intercept=float(model.intercept_[0]),
    )
    return positive_proba(model, validation[columns]), positive_proba(model, test[columns]), rows


def make_global_stacker(approaches: tuple[str, ...]):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    score_columns = [approach_score_col(approach) for approach in approaches]
    features = ColumnTransformer(
        [
            ("scores", "passthrough", score_columns),
            ("search", OneHotEncoder(handle_unknown="ignore"), ["search_name"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("features", features),
            (
                "model",
                LogisticRegression(max_iter=2000, solver="liblinear", random_state=base_sweep.DEFAULT_SEED),
            ),
        ]
    )


def fit_global_stacker(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    approaches: tuple[str, ...],
    *,
    method: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    model = make_global_stacker(approaches)
    model.fit(validation, validation[TARGET_COL].astype(int))
    fitted_features = model.named_steps["features"]
    classifier = model.named_steps["model"]
    try:
        feature_names = list(fitted_features.get_feature_names_out())
    except Exception:
        feature_names = [approach_score_col(approach) for approach in approaches]
    rows = coefficient_rows(
        search="__all__",
        method=method,
        scope="global_with_search",
        feature_names=feature_names,
        coefficients=np.asarray(classifier.coef_[0], dtype=float),
        intercept=float(classifier.intercept_[0]),
    )
    return positive_proba(model, validation), positive_proba(model, test), rows


def method_metrics(
    *,
    search: str,
    method: str,
    method_kind: str,
    approaches: tuple[str, ...],
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, Any]:
    validation_roc_auc, validation_pr_auc = voting.safe_auc(validation_labels, validation_scores)
    roc_auc, pr_auc = voting.safe_auc(test_labels, test_scores)
    alert = choose_threshold(
        validation_labels,
        validation_scores,
        min_precision=base_sweep.PROMOTION_PRECISION,
        min_count=base_sweep.VALIDATION_MIN_COUNT,
    )
    test_alert = threshold_metrics(test_labels, test_scores, float(alert["threshold"]))
    test_positive_rows = int(test_labels.sum())
    top = {k: precision_at_k(test_labels, test_scores, k) for k in (10, 25, 50)}
    validation_alert_qualified = (
        int(alert.get("count", 0)) >= base_sweep.VALIDATION_MIN_COUNT
        and pd.notna(alert.get("precision"))
        and float(alert["precision"]) >= base_sweep.PROMOTION_PRECISION
    )
    return {
        "search_name": search,
        "method": method,
        "method_kind": method_kind,
        "component_count": int(len(approaches)),
        "component_approaches": component_text(approaches),
        "validation_rows": int(len(validation_labels)),
        "test_rows": int(len(test_labels)),
        "test_positive_rows": test_positive_rows,
        "sold_base_rate": float(test_labels.mean()) if len(test_labels) else np.nan,
        "validation_roc_auc": validation_roc_auc,
        "validation_pr_auc": validation_pr_auc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_at_10": top[10]["precision"],
        "positives_at_10": top[10]["positive_count"],
        "precision_at_25": top[25]["precision"],
        "positives_at_25": top[25]["positive_count"],
        "precision_at_50": top[50]["precision"],
        "positives_at_50": top[50]["positive_count"],
        "alert_threshold": float(alert["threshold"]),
        "validation_alert_qualified_80p_20n": bool(validation_alert_qualified),
        "validation_alert_count": int(alert.get("count", 0)),
        "validation_alert_precision": float(alert["precision"]) if pd.notna(alert.get("precision")) else np.nan,
        "test_alert_count": int(test_alert["count"]),
        "test_alert_precision": (
            float(test_alert["precision"]) if pd.notna(test_alert.get("precision")) else np.nan
        ),
        "test_alert_positive_count": int(test_alert["positive_count"]),
        "test_alert_recall": voting.safe_divide(int(test_alert["positive_count"]), test_positive_rows),
    }


def append_method(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    search: str,
    method: str,
    method_kind: str,
    approaches: tuple[str, ...],
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    metrics: list[dict[str, Any]],
) -> None:
    column = ensemble_score_col(method)
    validation[column] = np.asarray(validation_scores, dtype=float)
    test[column] = np.asarray(test_scores, dtype=float)
    metrics.append(
        method_metrics(
            search=search,
            method=method,
            method_kind=method_kind,
            approaches=approaches,
            validation_labels=validation[TARGET_COL].astype(int).to_numpy(),
            validation_scores=validation[column].to_numpy(dtype=float),
            test_labels=test[TARGET_COL].astype(int).to_numpy(),
            test_scores=test[column].to_numpy(dtype=float),
        )
    )


def selection_row(search: str, method: str, reason: str, approaches: tuple[str, ...]) -> dict[str, Any]:
    return {
        "search_name": search,
        "method": method,
        "selection_reason": reason,
        "component_count": int(len(approaches)),
        "component_approaches": component_text(approaches),
    }


def evaluate_search_methods(
    search: str,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    validation_labels = validation[TARGET_COL].astype(int).to_numpy()
    test_labels = test[TARGET_COL].astype(int).to_numpy()

    for approach in ALL_APPROACHES:
        metrics.append(
            method_metrics(
                search=search,
                method=approach,
                method_kind="base_model",
                approaches=(approach,),
                validation_labels=validation_labels,
                validation_scores=validation[approach_score_col(approach)].to_numpy(dtype=float),
                test_labels=test_labels,
                test_scores=test[approach_score_col(approach)].to_numpy(dtype=float),
            )
        )

    best_single = select_top_approaches(validation, ALL_APPROACHES, 1)
    append_method(
        validation,
        test,
        search=search,
        method="best_single_validation_auc",
        method_kind="validation_selected_single",
        approaches=best_single,
        validation_scores=component_score(validation, best_single, reducer="mean"),
        test_scores=component_score(test, best_single, reducer="mean"),
        metrics=metrics,
    )
    selections.append(selection_row(search, "best_single_validation_auc", "top validation ROC AUC", best_single))

    component_methods = (
        ("sum_9_scores", "score_sum", ALL_APPROACHES, "sum"),
        ("mean_9_scores", "score_mean", ALL_APPROACHES, "mean"),
        ("mean_nonduplicate_8_scores", "score_mean_reduced", NON_DUPLICATE_APPROACHES, "mean"),
        ("mean_lean_7_scores", "score_mean_reduced", LEAN_APPROACHES, "mean"),
    )
    for method, kind, approaches, reducer in component_methods:
        append_method(
            validation,
            test,
            search=search,
            method=method,
            method_kind=kind,
            approaches=approaches,
            validation_scores=component_score(validation, approaches, reducer=reducer),
            test_scores=component_score(test, approaches, reducer=reducer),
            metrics=metrics,
        )
        selections.append(selection_row(search, method, f"fixed {reducer} component set", approaches))

    for count in (3, 5):
        approaches = select_top_approaches(validation, ALL_APPROACHES, count)
        method = f"validation_top{count}_mean"
        append_method(
            validation,
            test,
            search=search,
            method=method,
            method_kind="validation_selected_score_mean",
            approaches=approaches,
            validation_scores=component_score(validation, approaches, reducer="mean"),
            test_scores=component_score(test, approaches, reducer="mean"),
            metrics=metrics,
        )
        selections.append(selection_row(search, method, "top validation ROC AUC components", approaches))

    for label, approaches in META_STACKER_SETS:
        method = f"per_search_logistic_stacker_{label}"
        validation_scores, test_scores, rows = fit_per_search_stacker(
            validation,
            test,
            approaches,
            method=method,
        )
        append_method(
            validation,
            test,
            search=search,
            method=method,
            method_kind="per_search_logistic_stacker",
            approaches=approaches,
            validation_scores=validation_scores,
            test_scores=test_scores,
            metrics=metrics,
        )
        selections.append(selection_row(search, method, "logistic stacker component set", approaches))
        coefficients.extend(rows)
    return metrics, selections, coefficients


def add_global_stackers(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    metrics: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    coefficients: list[dict[str, Any]],
) -> None:
    for label, approaches in META_STACKER_SETS:
        method = f"global_logistic_stacker_{label}_with_search"
        validation_scores, test_scores, rows = fit_global_stacker(
            validation,
            test,
            approaches,
            method=method,
        )
        column = ensemble_score_col(method)
        validation[column] = validation_scores
        test[column] = test_scores
        for search, validation_group in validation.groupby("search_name", sort=True):
            test_group = test[test["search_name"] == search]
            metrics.append(
                method_metrics(
                    search=str(search),
                    method=method,
                    method_kind="global_logistic_stacker_with_search",
                    approaches=approaches,
                    validation_labels=validation_group[TARGET_COL].astype(int).to_numpy(),
                    validation_scores=validation_group[column].to_numpy(dtype=float),
                    test_labels=test_group[TARGET_COL].astype(int).to_numpy(),
                    test_scores=test_group[column].to_numpy(dtype=float),
                )
            )
        selections.append(selection_row("__all__", method, "global logistic stacker component set", approaches))
        coefficients.extend(rows)


def sort_best(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        frame.sort_values(["search_name", metric, "pr_auc"], ascending=[True, False, False], na_position="last")
        .drop_duplicates("search_name", keep="first")
        .reset_index(drop=True)
    )


def write_report(out_dir: Path, metrics: pd.DataFrame) -> Path:
    path = assert_experiment_path(out_dir / "basic_5_stacking_report.md")
    comparison = metrics[metrics["method"].isin(MAIN_METHODS)].copy()
    comparison = comparison.sort_values(["search_name", "method"]).reset_index(drop=True)
    best_auc = sort_best(comparison, "roc_auc")
    best_p25 = sort_best(comparison, "precision_at_25")
    overview_cols = [
        ("search_name", "Search"),
        ("method", "Method"),
        ("roc_auc", "ROC AUC"),
        ("pr_auc", "PR AUC"),
        ("precision_at_25", "P@25"),
        ("precision_at_50", "P@50"),
        ("test_alert_precision", "Alert precision"),
        ("test_alert_count", "Alert count"),
    ]
    kinds = {
        "method": "text",
        "test_alert_count": "int",
        "precision_at_25": "pct",
        "precision_at_50": "pct",
        "test_alert_precision": "pct",
    }
    lines = [
        "# Basic 5 Score Stacking",
        "",
        "This experiment combines the nine saved `basic_5` sold-status model scores on offline rows.",
        "Base models were already fit on the train split. Score combinations and logistic stackers are fit or selected on validation scores and evaluated on held-out test scores.",
        "",
        "- `sum_9_scores` and `mean_9_scores` have the same ROC/PR ranking because one is a constant rescale of the other.",
        "- Reduced score sets drop the duplicate logistic snapshot first; the lean set also drops the hand-built price-rule score.",
        "- Validation top-3 and top-5 means choose their components by validation ROC AUC only.",
        "- Per-search logistic stackers see component scores for one search. Global stackers see component scores plus a one-hot search identity.",
        "- Alert thresholds are selected on validation scores with the existing >=80% precision and >=20 selected-item target when that gate is feasible.",
        "",
        "## Best Main ROC AUC",
        "",
        *voting.write_table(best_auc, overview_cols, kinds),
        "",
        "## Best Main P@25",
        "",
        *voting.write_table(best_p25, overview_cols, kinds),
        "",
        "## Main Comparisons",
        "",
        *voting.write_table(comparison, overview_cols, kinds),
        "",
        "## Files",
        "",
        "- `stacking_metrics.csv`: base-model and ensemble held-out metrics by search, including ROC AUC, PR AUC, P@10/P@25/P@50, and validation-selected alert-threshold precision.",
        "- `test_item_scores.csv`: held-out rows with the nine model scores and generated ensemble scores.",
        "- `validation_item_scores.csv`: validation rows used for component selection, stacker training, and alert thresholds.",
        "- `component_selection.csv`: which models enter the validation-selected and fixed component methods.",
        "- `stacker_coefficients.csv`: learned logistic-stacker coefficients and intercepts.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate basic_5 score sums and logistic stackers offline.")
    parser.add_argument("--original-run", default=voting.DEFAULT_ORIGINAL_RUN)
    parser.add_argument("--new-run", default=voting.DEFAULT_NEW_RUN)
    parser.add_argument("--search", action="append", default=[])
    parser.add_argument("--seed", type=int, default=base_sweep.DEFAULT_SEED)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_experiment_dirs()
    searches = list(dict.fromkeys(args.search or voting.DEFAULT_SEARCHES))
    out_dir = assert_experiment_path(
        Path(args.out_dir) if args.out_dir else EXPERIMENT_ROOT / run_id("basic_5_stacking")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    coefficient_parts: list[dict[str, Any]] = []
    for search in searches:
        validation, test = score_search_base_models(
            search,
            seed=args.seed,
            original_run=args.original_run,
            new_run=args.new_run,
        )
        search_metrics, search_selections, search_coefficients = evaluate_search_methods(search, validation, test)
        validation_parts.append(validation)
        test_parts.append(test)
        metric_rows.extend(search_metrics)
        selection_rows.extend(search_selections)
        coefficient_parts.extend(search_coefficients)
        print(
            f"[basic_5_stacking] search={search} validation_rows={len(validation)} test_rows={len(test)}",
            flush=True,
        )

    validation_scores = pd.concat(validation_parts, ignore_index=True)
    test_scores = pd.concat(test_parts, ignore_index=True)
    add_global_stackers(
        validation_scores,
        test_scores,
        metrics=metric_rows,
        selections=selection_rows,
        coefficients=coefficient_parts,
    )

    metrics = pd.DataFrame(metric_rows).sort_values(["search_name", "method"]).reset_index(drop=True)
    selections = pd.DataFrame(selection_rows).sort_values(["search_name", "method"]).reset_index(drop=True)
    coefficients = pd.DataFrame(coefficient_parts).sort_values(["method", "search_name", "feature"]).reset_index(drop=True)
    metrics.to_csv(assert_experiment_path(out_dir / "stacking_metrics.csv"), index=False)
    selections.to_csv(assert_experiment_path(out_dir / "component_selection.csv"), index=False)
    coefficients.to_csv(assert_experiment_path(out_dir / "stacker_coefficients.csv"), index=False)
    validation_scores.to_csv(assert_experiment_path(out_dir / "validation_item_scores.csv"), index=False)
    test_scores.to_csv(assert_experiment_path(out_dir / "test_item_scores.csv"), index=False)
    report_path = write_report(out_dir, metrics)
    write_manifest(
        out_dir / "manifest.json",
        command="basic_5_stacking.run",
        extra={
            "original_run": args.original_run,
            "new_run": args.new_run,
            "searches": searches,
            "seed": args.seed,
            "all_approaches": list(ALL_APPROACHES),
            "nonduplicate_approaches": list(NON_DUPLICATE_APPROACHES),
            "lean_approaches": list(LEAN_APPROACHES),
            "main_methods": list(MAIN_METHODS),
            "outputs": {
                "report": str(report_path),
                "metrics": str(out_dir / "stacking_metrics.csv"),
                "component_selection": str(out_dir / "component_selection.csv"),
                "stacker_coefficients": str(out_dir / "stacker_coefficients.csv"),
                "validation_item_scores": str(out_dir / "validation_item_scores.csv"),
                "test_item_scores": str(out_dir / "test_item_scores.csv"),
            },
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
