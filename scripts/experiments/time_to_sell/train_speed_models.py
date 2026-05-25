#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.deal_finder import model_sweep as base_sweep
from experiments.deal_finder.modeling import TARGET_COL, score_with_model
from experiments.full_scrape_model import compare_feature_modalities as feature_compare
from experiments.time_to_sell.build_speed_datasets import boolish
from experiments.time_to_sell.paths import (
    EXPERIMENT_ROOT,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    utc_now_iso,
    write_manifest,
)


ORIGINAL_BASIC_APPROACHES = (
    "logistic_v1_baseline",
    "logistic_snapshot_v2",
    "sgd_text_numeric_v1",
    "linear_svm_calibrated_v1",
    "numeric_tree_v1",
    "rules_price_v1",
)
NEW_BASIC_APPROACHES = (
    "random_forest_basic_v1",
    "hist_gradient_basic_numeric_v1",
    "xgboost_basic_v1",
)
FEATURE_MODES = ("basic5", "full_visual")
MODE_TO_COMPARE_MODE = {
    "basic5": "basic_5",
    "full_visual": "full_scrape_plus_visual",
}


def latest_speed_dataset_dir() -> Path:
    candidates = sorted(
        path for path in EXPERIMENT_ROOT.glob("speed_labels_*") if (path / "basic5_speed_dataset.csv").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No speed label datasets found under {EXPERIMENT_ROOT}")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train time-to-sell classifiers on speed-label datasets.")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--window", type=int, action="append", default=[])
    parser.add_argument("--feature-mode", choices=FEATURE_MODES, action="append", default=[])
    parser.add_argument("--search", action="append", default=[])
    parser.add_argument("--seed", type=int, default=base_sweep.DEFAULT_SEED)
    parser.add_argument("--min-rows", type=int, default=25)
    parser.add_argument("--max-dino-dims", type=int, default=None)
    parser.add_argument(
        "--include-upload-date",
        action="store_true",
        default=False,
        help="Allow Upload_date-derived full-scrape features. Default keeps them out to avoid freshness leakage.",
    )
    return parser.parse_args()


def approach_specs() -> list[base_sweep.ApproachSpec]:
    original = [spec for spec in base_sweep.APPROACHES if spec.name in ORIGINAL_BASIC_APPROACHES]
    experimental = [
        spec for spec in feature_compare.EXPERIMENTAL_BASIC_APPROACHES if spec.name in NEW_BASIC_APPROACHES
    ]
    by_name = {spec.name: spec for spec in [*original, *experimental]}
    return [by_name[name] for name in (*ORIGINAL_BASIC_APPROACHES, *NEW_BASIC_APPROACHES)]


def available_windows(frame: pd.DataFrame) -> list[int]:
    out: list[int] = []
    for column in frame.columns:
        if column.startswith("label_sold_within_") and column.endswith("h"):
            out.append(int(column.removeprefix("label_sold_within_").removesuffix("h")))
    return sorted(set(out))


def load_datasets(dataset_dir: Path, max_dino_dims: int | None) -> dict[str, pd.DataFrame]:
    basic = pd.read_csv(dataset_dir / "basic5_speed_dataset.csv", low_memory=False)
    full = pd.read_csv(dataset_dir / "full_visual_speed_dataset.csv", low_memory=False)
    if max_dino_dims is not None:
        dino_cols = sorted(column for column in full.columns if column.startswith("DinoEmbedding_"))
        drop_cols = dino_cols[int(max_dino_dims) :]
        if drop_cols:
            full = full.drop(columns=drop_cols)
    return {"basic5": basic, "full_visual": full}


def prepare_target_frame(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    label_col = f"label_sold_within_{window}h"
    eval_col = f"label_evaluated_{window}h"
    if label_col not in frame.columns or eval_col not in frame.columns:
        return frame.iloc[0:0].copy()
    evaluated = boolish(frame[eval_col]).fillna(False)
    labels = boolish(frame[label_col])
    out = frame.loc[evaluated & labels.notna()].copy()
    out[TARGET_COL] = boolish(out[label_col]).astype(int)
    out["offline_label_eligible"] = True
    out["speed_window_h"] = int(window)
    return out.reset_index(drop=True)


def numeric(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def auc_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(np.unique(y_true)) < 2:
            return {"roc_auc": np.nan, "pr_auc": np.nan}
        return {
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "pr_auc": float(average_precision_score(y_true, scores)),
        }
    except Exception:
        return {"roc_auc": np.nan, "pr_auc": np.nan}


def confusion_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = np.asarray(scores >= threshold, dtype=int)
    y = np.asarray(y_true, dtype=int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    accuracy = (tp + tn) / len(y) if len(y) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall and pd.notna(precision) else np.nan
    return {
        "threshold": float(threshold),
        "predicted_positive_rows": int(tp + fp),
        "true_positive_rows": tp,
        "false_positive_rows": fp,
        "true_negative_rows": tn,
        "false_negative_rows": fn,
        "precision": float(precision) if pd.notna(precision) else np.nan,
        "recall": float(recall) if pd.notna(recall) else np.nan,
        "accuracy": float(accuracy) if pd.notna(accuracy) else np.nan,
        "f1": float(f1) if pd.notna(f1) else np.nan,
    }


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, objective: str) -> float:
    if len(scores) == 0:
        return 1.0
    candidates = sorted(set(np.round(np.asarray(scores, dtype=float), 6).tolist() + [0.5]), reverse=True)
    best_threshold = candidates[0]
    best_key: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        metrics = confusion_metrics(y_true, scores, threshold)
        precision = numeric(metrics["precision"])
        recall = numeric(metrics["recall"])
        accuracy = numeric(metrics["accuracy"])
        f1 = numeric(metrics["f1"])
        predicted = numeric(metrics["predicted_positive_rows"])
        if objective == "accuracy":
            key = (accuracy, f1, precision if pd.notna(precision) else -1.0, -threshold)
        elif objective == "f1":
            key = (f1 if pd.notna(f1) else -1.0, precision if pd.notna(precision) else -1.0, recall, predicted)
        else:
            raise ValueError(f"Unknown threshold objective: {objective}")
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
    return float(best_threshold)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict[str, float | int]:
    if len(y_true) == 0:
        return {"k": int(k), "count": 0, "precision": np.nan, "positive_rows": 0}
    order = np.argsort(-np.asarray(scores, dtype=float))
    top = order[: min(int(k), len(order))]
    positives = int(np.asarray(y_true, dtype=int)[top].sum())
    count = int(len(top))
    return {
        "k": int(k),
        "count": count,
        "precision": float(positives / count) if count else np.nan,
        "positive_rows": positives,
    }


def evaluate_scores(frame: pd.DataFrame, scores: np.ndarray, val_thresholds: dict[str, float]) -> dict[str, Any]:
    y = frame[TARGET_COL].astype(int).to_numpy()
    top_10pct = max(1, int(np.ceil(len(y) * 0.10)))
    return {
        "rows": int(len(frame)),
        "positive_rows": int(y.sum()),
        "base_rate": float(y.mean()) if len(y) else np.nan,
        **auc_metrics(y, scores),
        "precision_at_5": precision_at_k(y, scores, 5),
        "precision_at_10": precision_at_k(y, scores, 10),
        "precision_at_25": precision_at_k(y, scores, 25),
        "precision_at_10pct": precision_at_k(y, scores, top_10pct),
        "threshold_f1": confusion_metrics(y, scores, val_thresholds["f1"]),
        "threshold_accuracy": confusion_metrics(y, scores, val_thresholds["accuracy"]),
    }


def feature_columns_for_mode(
    train: pd.DataFrame,
    *,
    mode: str,
    spec: base_sweep.ApproachSpec,
    embedding_cols: list[str],
    include_upload_date: bool,
) -> tuple[list[str], list[str]]:
    compare_mode = MODE_TO_COMPARE_MODE[mode]
    mode_spec = feature_compare.make_mode_spec(spec, compare_mode)
    numeric, text = feature_compare.select_mode_features(
        train,
        spec=mode_spec,
        mode=compare_mode,
        embedding_cols=embedding_cols,
        include_upload_date=include_upload_date,
    )
    return numeric, text


def train_one(
    frame: pd.DataFrame,
    *,
    mode: str,
    search_name: str,
    window: int,
    spec: base_sweep.ApproachSpec,
    seed: int,
    include_upload_date: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    started = time.perf_counter()
    work = feature_compare.add_full_engineered_features(frame)
    embedding_cols = sorted(column for column in work.columns if column.startswith("DinoEmbedding_"))
    splits = base_sweep.stratified_random_split(work, seed=seed)
    split_sizes = {
        "train": int(len(splits.train)),
        "validation": int(len(splits.validation)),
        "test": int(len(splits.test)),
    }
    common = {
        "feature_mode": mode,
        "search_name": search_name,
        "window_h": int(window),
        "approach": spec.name,
        "model_kind": spec.kind,
        "seed": int(seed),
        "split_rows": split_sizes,
    }
    if min(split_sizes.values()) == 0:
        return {**common, "status": "skipped", "reason": "empty split"}, pd.DataFrame()
    if splits.train[TARGET_COL].nunique() < 2:
        return {**common, "status": "skipped", "reason": "training split has only one label class"}, pd.DataFrame()

    numeric, text = feature_columns_for_mode(
        splits.train,
        mode=mode,
        spec=spec,
        embedding_cols=embedding_cols,
        include_upload_date=include_upload_date,
    )
    if not numeric and not text:
        return {**common, "status": "skipped", "reason": "no usable features"}, pd.DataFrame()
    if spec.kind == "linear_svm_calibrated" and splits.train[TARGET_COL].value_counts().min() < 3:
        return {**common, "status": "skipped", "reason": "calibrated SVM needs at least 3 train rows per class"}, pd.DataFrame()

    fit_frame = base_sweep.bounded_fit_frame(splits.train, spec, seed)
    model = base_sweep.make_model(spec, numeric, text)
    model.fit(fit_frame, fit_frame[TARGET_COL].astype(int))
    validation_scores = np.clip(np.asarray(score_with_model(model, splits.validation), dtype=float), 0.0, 1.0)
    test_scores = np.clip(np.asarray(score_with_model(model, splits.test), dtype=float), 0.0, 1.0)
    validation_y = splits.validation[TARGET_COL].astype(int).to_numpy()
    thresholds = {
        "f1": choose_threshold(validation_y, validation_scores, "f1"),
        "accuracy": choose_threshold(validation_y, validation_scores, "accuracy"),
    }
    validation = evaluate_scores(splits.validation, validation_scores, thresholds)
    test = evaluate_scores(splits.test, test_scores, thresholds)
    predictions = splits.test[
        [col for col in ("tracking_key", "item_id", "SearchName", "Title", "Brand", "Size", "Price", "Likes") if col in splits.test]
    ].copy()
    predictions["feature_mode"] = mode
    predictions["search_name"] = search_name
    predictions["window_h"] = int(window)
    predictions["approach"] = spec.name
    predictions["label"] = splits.test[TARGET_COL].astype(int).to_numpy()
    predictions["score"] = test_scores
    predictions["rank_in_split"] = pd.Series(test_scores).rank(ascending=False, method="first").astype(int).to_numpy()
    row = {
        **common,
        "status": "trained",
        "reason": "",
        "numeric_features": numeric,
        "text_features": text,
        "fit_train_rows": int(len(fit_frame)),
        "threshold_f1": float(thresholds["f1"]),
        "threshold_accuracy": float(thresholds["accuracy"]),
        "validation": validation,
        "test": test,
        "fit_seconds": float(time.perf_counter() - started),
    }
    return row, predictions


def flatten_result(row: dict[str, Any]) -> dict[str, Any]:
    validation = row.get("validation", {})
    test = row.get("test", {})
    test_f1 = test.get("threshold_f1", {})
    test_acc = test.get("threshold_accuracy", {})
    val_f1 = validation.get("threshold_f1", {})
    return {
        "feature_mode": row.get("feature_mode"),
        "search_name": row.get("search_name"),
        "window_h": row.get("window_h"),
        "approach": row.get("approach"),
        "model_kind": row.get("model_kind"),
        "status": row.get("status"),
        "reason": row.get("reason", ""),
        "seed": row.get("seed"),
        "train_rows": row.get("split_rows", {}).get("train"),
        "validation_rows": row.get("split_rows", {}).get("validation"),
        "test_rows": test.get("rows"),
        "test_positive_rows": test.get("positive_rows"),
        "test_base_rate": test.get("base_rate"),
        "test_roc_auc": test.get("roc_auc"),
        "test_pr_auc": test.get("pr_auc"),
        "test_precision_at_5": test.get("precision_at_5", {}).get("precision"),
        "test_precision_at_10": test.get("precision_at_10", {}).get("precision"),
        "test_precision_at_25": test.get("precision_at_25", {}).get("precision"),
        "test_precision_at_10pct": test.get("precision_at_10pct", {}).get("precision"),
        "threshold_f1": row.get("threshold_f1"),
        "validation_f1_threshold_precision": val_f1.get("precision"),
        "validation_f1_threshold_recall": val_f1.get("recall"),
        "validation_f1_threshold_f1": val_f1.get("f1"),
        "test_f1_threshold_precision": test_f1.get("precision"),
        "test_f1_threshold_recall": test_f1.get("recall"),
        "test_f1_threshold_accuracy": test_f1.get("accuracy"),
        "test_f1_threshold_f1": test_f1.get("f1"),
        "test_f1_threshold_count": test_f1.get("predicted_positive_rows"),
        "threshold_accuracy": row.get("threshold_accuracy"),
        "test_accuracy_threshold_precision": test_acc.get("precision"),
        "test_accuracy_threshold_recall": test_acc.get("recall"),
        "test_accuracy_threshold_accuracy": test_acc.get("accuracy"),
        "test_accuracy_threshold_f1": test_acc.get("f1"),
        "test_accuracy_threshold_count": test_acc.get("predicted_positive_rows"),
        "numeric_feature_count": len(row.get("numeric_features", [])),
        "text_feature_count": len(row.get("text_features", [])),
        "fit_seconds": row.get("fit_seconds"),
    }


def best_tables(metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    trained = metrics[metrics["status"] == "trained"].copy() if not metrics.empty else pd.DataFrame()
    if trained.empty:
        return {
            "best_by_search_window_mode": pd.DataFrame(),
            "best_by_window_mode": pd.DataFrame(),
            "best_by_approach_mode": pd.DataFrame(),
        }
    sort_cols = ["test_pr_auc", "test_roc_auc", "test_precision_at_10", "test_f1_threshold_precision"]
    ascending = [False, False, False, False]
    return {
        "best_by_search_window_mode": trained.sort_values(sort_cols, ascending=ascending)
        .drop_duplicates(["feature_mode", "search_name", "window_h"], keep="first")
        .reset_index(drop=True),
        "best_by_window_mode": trained.sort_values(sort_cols, ascending=ascending)
        .drop_duplicates(["feature_mode", "window_h"], keep="first")
        .reset_index(drop=True),
        "best_by_approach_mode": trained.sort_values(sort_cols, ascending=ascending)
        .drop_duplicates(["feature_mode", "approach"], keep="first")
        .reset_index(drop=True),
    }


def fmt(value: object, digits: int = 3) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "" if pd.isna(number) else f"{float(number):.{digits}f}"


def write_report(out_dir: Path, metrics: pd.DataFrame, tables: dict[str, pd.DataFrame], summary: dict[str, Any]) -> Path:
    path = assert_experiment_path(out_dir / "speed_model_report.md")
    lines = [
        "# Time-to-sell speed model sweep",
        "",
        f"Dataset folder: `{summary['dataset_dir']}`",
        f"Rows trained/evaluated: `{summary['trained_rows']}` trained, `{summary['skipped_rows']}` skipped.",
        "",
        "Models are trained per search and per sold-within-hour target. Ranking below is by test PR AUC, then ROC AUC.",
        "",
    ]
    best = tables["best_by_search_window_mode"]
    if not best.empty:
        key_windows = [window for window in (12, 24, 48, 72) if window in set(best["window_h"])]
        report_best = best[best["window_h"].isin(key_windows)].copy() if key_windows else best.copy()
        lines.extend(
            [
                "## Best Per Search / Window / Feature Set",
                "",
                "| mode | search | window | approach | test rows | base | ROC AUC | PR AUC | p@10 | precision@F1-thr | recall@F1-thr |",
                "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in report_best.sort_values(["window_h", "feature_mode", "search_name"]).iterrows():
            lines.append(
                f"| `{row['feature_mode']}` | `{row['search_name']}` | {int(row['window_h'])} | `{row['approach']}` | "
                f"{int(row['test_rows'])} | {fmt(row['test_base_rate'])} | {fmt(row['test_roc_auc'])} | "
                f"{fmt(row['test_pr_auc'])} | {fmt(row['test_precision_at_10'])} | "
                f"{fmt(row['test_f1_threshold_precision'])} | {fmt(row['test_f1_threshold_recall'])} |"
            )
    by_window = tables["best_by_window_mode"]
    if not by_window.empty:
        lines.extend(
            [
                "",
                "## Best Overall Rows By Window",
                "",
                "| mode | window | search | approach | ROC AUC | PR AUC | p@10 |",
                "|---|---:|---|---|---:|---:|---:|",
            ]
        )
        for _, row in by_window.sort_values(["window_h", "feature_mode"]).iterrows():
            lines.append(
                f"| `{row['feature_mode']}` | {int(row['window_h'])} | `{row['search_name']}` | `{row['approach']}` | "
                f"{fmt(row['test_roc_auc'])} | {fmt(row['test_pr_auc'])} | {fmt(row['test_precision_at_10'])} |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `full_visual` includes full scrape fields, visual quality features, and DINO embedding columns.",
            "- Upload-date text/numeric fields are excluded by default to reduce freshness leakage.",
            "- Very small searches/windows are noisy; use them as directional signals, not model-selection truth.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    ensure_experiment_dirs()
    dataset_dir = (args.dataset_dir or latest_speed_dataset_dir()).resolve()
    out_dir = args.out_dir or (EXPERIMENT_ROOT / "offline_runs" / run_id("speed_model_sweep"))
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets(dataset_dir, args.max_dino_dims)
    windows = sorted(set(args.window or available_windows(datasets["basic5"])))
    feature_modes = tuple(dict.fromkeys(args.feature_mode or FEATURE_MODES))
    searches = set(args.search or [])
    specs = approach_specs()

    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for mode in feature_modes:
        source = datasets[mode]
        for window in windows:
            target_frame = prepare_target_frame(source, window)
            if searches:
                target_frame = target_frame[target_frame["SearchName"].astype(str).isin(searches)].copy()
            for search_name, search_frame in target_frame.groupby("SearchName", sort=True):
                search_frame = search_frame.reset_index(drop=True)
                if len(search_frame) < args.min_rows or search_frame[TARGET_COL].nunique() < 2:
                    for spec in specs:
                        rows.append(
                            {
                                "feature_mode": mode,
                                "search_name": str(search_name),
                                "window_h": int(window),
                                "approach": spec.name,
                                "model_kind": spec.kind,
                                "seed": int(args.seed),
                                "status": "skipped",
                                "reason": "not enough rows or only one label class",
                                "split_rows": {"train": 0, "validation": 0, "test": 0},
                            }
                        )
                    continue
                for spec in specs:
                    print(
                        f"[speed_models] mode={mode} search={search_name} window={window}h approach={spec.name}",
                        flush=True,
                    )
                    try:
                        row, pred = train_one(
                            search_frame,
                            mode=mode,
                            search_name=str(search_name),
                            window=window,
                            spec=spec,
                            seed=args.seed,
                            include_upload_date=args.include_upload_date,
                        )
                    except Exception as exc:
                        row, pred = (
                            {
                                "feature_mode": mode,
                                "search_name": str(search_name),
                                "window_h": int(window),
                                "approach": spec.name,
                                "model_kind": spec.kind,
                                "seed": int(args.seed),
                                "status": "error",
                                "reason": f"{type(exc).__name__}: {exc}",
                                "split_rows": {"train": 0, "validation": 0, "test": 0},
                            },
                            pd.DataFrame(),
                        )
                    rows.append(row)
                    if not pred.empty:
                        predictions.append(pred)

    raw_path = assert_experiment_path(out_dir / "metrics.json")
    raw_path.write_text(json.dumps(base_sweep.to_builtin({"metrics": rows}), indent=2, sort_keys=True), encoding="utf-8")
    metrics = pd.DataFrame([flatten_result(row) for row in rows])
    metrics_path = assert_experiment_path(out_dir / "metrics_long.csv")
    metrics.to_csv(metrics_path, index=False)
    prediction_frame = pd.concat(predictions, ignore_index=True, sort=False) if predictions else pd.DataFrame()
    predictions_path = assert_experiment_path(out_dir / "test_predictions.csv")
    prediction_frame.to_csv(predictions_path, index=False)

    tables = best_tables(metrics)
    table_paths: dict[str, str] = {}
    for name, table in tables.items():
        table_path = assert_experiment_path(out_dir / f"{name}.csv")
        table.to_csv(table_path, index=False)
        table_paths[name] = str(table_path)

    trained_rows = int((metrics["status"] == "trained").sum()) if not metrics.empty else 0
    skipped_rows = int((metrics["status"] != "trained").sum()) if not metrics.empty else 0
    summary = {
        "created_at": utc_now_iso(),
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "feature_modes": list(feature_modes),
        "windows_h": windows,
        "searches": sorted(searches) if searches else "all",
        "approaches": [spec.name for spec in specs],
        "trained_rows": trained_rows,
        "skipped_rows": skipped_rows,
        "test_prediction_rows": int(len(prediction_frame)),
        "outputs": {
            "metrics_long": str(metrics_path),
            "metrics_json": str(raw_path),
            "test_predictions": str(predictions_path),
            **table_paths,
        },
    }
    summary_path = assert_experiment_path(out_dir / "summary.json")
    summary_path.write_text(json.dumps(base_sweep.to_builtin(summary), indent=2, sort_keys=True), encoding="utf-8")
    report_path = write_report(out_dir, metrics, tables, summary)
    write_manifest(
        out_dir / "manifest.json",
        command=" ".join(sys.argv),
        extra=base_sweep.to_builtin({**summary, "outputs": {**summary["outputs"], "report": str(report_path)}}),
    )
    print(json.dumps(base_sweep.to_builtin({**summary, "report": str(report_path)}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
