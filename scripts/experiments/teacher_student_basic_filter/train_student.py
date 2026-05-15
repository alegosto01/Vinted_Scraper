#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    from pathlib import Path as _Path

    _ROOT = _Path(__file__).resolve().parents[3]
    _SCRIPTS = _ROOT / "scripts"
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiments.teacher_student_basic_filter.paths import (
    MODELS_DIR,
    OFFLINE_RUNS_DIR,
    ROOT,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_json,
    write_manifest,
)

from experiments.deal_finder import model_sweep as base_sweep
from experiments.deal_finder.modeling import (
    TARGET_COL,
    load_pickle,
    score_with_model,
    text_values_to_str,
    to_numeric_frame,
)
from experiments.full_scrape_model.compare_feature_modalities import add_full_engineered_features


DEFAULT_TEACHER_RUN = "sold_status_feature_modalities_20260515_full_visual"
DEFAULT_SEED = 42
RECALL_TARGETS = (0.90, 0.95, 0.98, 0.99)
PREFERRED_RECALL_TARGET = 0.95
BASIC_NUMERIC = ["Price", "Likes"]
BASIC_TEXT = ["Title", "Brand", "Size"]
ID_COLUMNS = ["SearchName", "item_id", "Dataid", "Title", "Brand", "Size", "Price", "Likes", "Link"]


@dataclass(frozen=True)
class StudentSpec:
    name: str
    kind: str
    use_text: bool = True


STUDENT_SPECS = (
    StudentSpec("ridge_text_basic_student_v1", "ridge", use_text=True),
    StudentSpec("sgd_huber_basic_student_v1", "sgd_huber", use_text=True),
    StudentSpec("extra_trees_basic_student_v1", "extra_trees", use_text=True),
    StudentSpec("ridge_numeric_student_v1", "ridge", use_text=False),
)


def full_run_dir(run_name: str) -> Path:
    return ROOT / "data" / "experiments" / "full_scrape_model" / "offline_runs" / run_name


def full_models_dir() -> Path:
    return ROOT / "data" / "experiments" / "full_scrape_model" / "models"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_pickle(path: Path, obj: Any) -> None:
    path = assert_experiment_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(obj, handle)
    tmp.replace(path)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def model_metadata_path(run_name: str, search: str, mode: str, approach: str, seed: int) -> Path:
    return full_models_dir() / f"{run_name}_{search}_{mode}_{approach}_seed{seed}_metadata.json"


def load_best_mode_rows(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "best_by_search_mode.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing best-by-mode table: {path}")
    rows = pd.read_csv(path)
    required = {"search_name", "feature_mode", "approach", "threshold"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return rows


def metadata_for_best(best_rows: pd.DataFrame, *, search: str, mode: str, run_name: str, seed: int) -> dict[str, Any]:
    matches = best_rows[(best_rows["search_name"].astype(str) == str(search)) & (best_rows["feature_mode"].astype(str) == mode)]
    if matches.empty:
        raise ValueError(f"No best row found for search={search!r}, mode={mode!r}")
    row = matches.iloc[0]
    path = model_metadata_path(run_name, search, mode, str(row["approach"]), seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata for {search} {mode}: {path}")
    metadata = read_json(path)
    metadata["threshold"] = float(metadata.get("threshold", row.get("threshold")))
    metadata["approach"] = str(metadata.get("approach", row["approach"]))
    metadata["feature_mode"] = str(metadata.get("feature_mode", mode))
    metadata["metadata_path"] = str(path)
    return metadata


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def available_text_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    out: list[str] = []
    for col in columns:
        if col not in df.columns:
            continue
        if df[col].fillna("").astype(str).str.contains(base_sweep.TOKEN_RE, regex=True).any():
            out.append(col)
    return out


def make_student_model(spec: StudentSpec, numeric_features: list[str], text_features: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge, SGDRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    transformers = []
    if numeric_features:
        numeric_pipe = Pipeline(
            steps=[
                ("to_numeric", FunctionTransformer(to_numeric_frame, validate=False)),
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("numeric", numeric_pipe, numeric_features))
    if spec.use_text:
        for col in text_features:
            text_pipe = Pipeline(
                steps=[
                    ("to_text", FunctionTransformer(text_values_to_str, validate=False)),
                    ("tfidf", TfidfVectorizer(max_features=700, ngram_range=(1, 2), min_df=1)),
                ]
            )
            transformers.append((f"text_{col}", text_pipe, col))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)
    if spec.kind == "ridge":
        regressor = Ridge(alpha=1.0, random_state=DEFAULT_SEED)
    elif spec.kind == "sgd_huber":
        regressor = SGDRegressor(
            loss="huber",
            epsilon=0.08,
            alpha=0.0001,
            max_iter=3000,
            tol=1e-4,
            random_state=DEFAULT_SEED,
        )
    elif spec.kind == "extra_trees":
        regressor = ExtraTreesRegressor(
            n_estimators=180,
            min_samples_leaf=4,
            random_state=DEFAULT_SEED,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown student kind: {spec.kind}")
    return Pipeline([("features", preprocessor), ("model", regressor)])


def clipped_predict(model: Any, frame: pd.DataFrame) -> np.ndarray:
    return np.clip(np.asarray(model.predict(frame), dtype=float), 0.0, 1.0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)) if len(y_true) else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else np.nan,
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
    }
    try:
        out["pearson"] = float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson"))
        out["spearman"] = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
    except Exception:
        out["pearson"] = np.nan
        out["spearman"] = np.nan
    return out


def choose_threshold_for_teacher_recall(
    scores: np.ndarray,
    teacher_pass: np.ndarray,
    *,
    target_recall: float,
) -> dict[str, Any]:
    teacher_pass = np.asarray(teacher_pass, dtype=bool)
    teacher_total = int(teacher_pass.sum())
    if len(scores) == 0:
        return {"threshold": 1.0, "teacher_recall": np.nan, "selected_count": 0, "teacher_selected_count": 0}
    if teacher_total == 0:
        threshold = float(np.quantile(scores, 0.95))
        selected = scores >= threshold
        return {
            "threshold": threshold,
            "teacher_recall": np.nan,
            "selected_count": int(selected.sum()),
            "teacher_selected_count": 0,
        }
    candidates = sorted(set(np.round(np.asarray(scores, dtype=float), 6).tolist()), reverse=True)
    best: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    for threshold in candidates:
        selected = scores >= threshold
        teacher_selected = int((selected & teacher_pass).sum())
        recall = float(teacher_selected / teacher_total)
        row = {
            "threshold": float(threshold),
            "teacher_recall": recall,
            "selected_count": int(selected.sum()),
            "teacher_selected_count": teacher_selected,
        }
        if fallback is None or row["teacher_recall"] > fallback["teacher_recall"] or (
            row["teacher_recall"] == fallback["teacher_recall"] and row["selected_count"] < fallback["selected_count"]
        ):
            fallback = row
        if recall >= target_recall:
            if best is None or row["selected_count"] < best["selected_count"] or (
                row["selected_count"] == best["selected_count"] and row["threshold"] > best["threshold"]
            ):
                best = row
    return best or fallback or {"threshold": 1.0, "teacher_recall": 0.0, "selected_count": 0, "teacher_selected_count": 0}


def selection_metrics(
    frame: pd.DataFrame,
    *,
    scores: np.ndarray,
    threshold: float,
    teacher_pass: np.ndarray,
    teacher_scores: np.ndarray,
    label: str,
    search_name: str,
    model_name: str,
    split: str,
    target_recall: float | None,
) -> dict[str, Any]:
    y = frame[TARGET_COL].astype(int).to_numpy()
    selected = np.asarray(scores, dtype=float) >= float(threshold)
    teacher_pass = np.asarray(teacher_pass, dtype=bool)
    teacher_total = int(teacher_pass.sum())
    selected_count = int(selected.sum())
    teacher_selected_count = int((selected & teacher_pass).sum())
    sold_selected_count = int(y[selected].sum()) if selected_count else 0
    sold_total = int(y.sum())
    cascade_mask = selected & teacher_pass
    cascade_count = int(cascade_mask.sum())
    cascade_sold_count = int(y[cascade_mask].sum()) if cascade_count else 0
    return {
        "search_name": search_name,
        "model_name": model_name,
        "label": label,
        "split": split,
        "target_recall": target_recall,
        "threshold": float(threshold),
        "rows": int(len(frame)),
        "teacher_pass_count": teacher_total,
        "sold_count": sold_total,
        "selected_count": selected_count,
        "selected_rate": float(selected_count / len(frame)) if len(frame) else np.nan,
        "teacher_selected_count": teacher_selected_count,
        "teacher_recall": float(teacher_selected_count / teacher_total) if teacher_total else np.nan,
        "teacher_precision_among_selected": float(teacher_selected_count / selected_count) if selected_count else np.nan,
        "sold_selected_count": sold_selected_count,
        "sold_precision_among_selected": float(sold_selected_count / selected_count) if selected_count else np.nan,
        "sold_recall": float(sold_selected_count / sold_total) if sold_total else np.nan,
        "cascade_final_count": cascade_count,
        "cascade_final_sold_count": cascade_sold_count,
        "cascade_final_sold_precision": float(cascade_sold_count / cascade_count) if cascade_count else np.nan,
        "teacher_score_mean_selected": float(np.nanmean(teacher_scores[selected])) if selected_count else np.nan,
        "teacher_score_mean_missed_teacher": float(np.nanmean(teacher_scores[teacher_pass & ~selected])) if int((teacher_pass & ~selected).sum()) else np.nan,
    }


def train_one_search(
    *,
    search: str,
    teacher_run: str,
    best_rows: pd.DataFrame,
    seed: int,
    run_dir: Path,
    test_prediction_rows: list[pd.DataFrame],
) -> list[dict[str, Any]]:
    dataset_path = full_run_dir(teacher_run) / "datasets" / f"{search}.csv"
    if not dataset_path.exists():
        return [{"search_name": search, "status": "skipped", "reason": f"missing dataset: {dataset_path}"}]
    raw = pd.read_csv(dataset_path, low_memory=False)
    frame = base_sweep.prepare_sweep_frame(raw)
    if len(frame) < 50 or frame[TARGET_COL].nunique() < 2:
        return [{"search_name": search, "status": "skipped", "reason": "not enough eligible rows or one class"}]
    work = add_full_engineered_features(frame)
    splits = base_sweep.stratified_random_split(work, seed=seed)

    teacher_meta = metadata_for_best(best_rows, search=search, mode="full_scrape_plus_visual", run_name=teacher_run, seed=seed)
    baseline_meta = metadata_for_best(best_rows, search=search, mode="basic_5", run_name=teacher_run, seed=seed)
    teacher_model = load_pickle(Path(teacher_meta["artifact_path"]))
    baseline_model = load_pickle(Path(baseline_meta["artifact_path"]))
    teacher_threshold = float(teacher_meta["threshold"])
    baseline_threshold = float(baseline_meta["threshold"])

    split_scores: dict[str, dict[str, Any]] = {}
    for split_name, split_frame in (("train", splits.train), ("validation", splits.validation), ("test", splits.test)):
        teacher_scores = np.clip(score_with_model(teacher_model, split_frame), 0.0, 1.0)
        baseline_scores = np.clip(score_with_model(baseline_model, split_frame), 0.0, 1.0)
        teacher_pass = teacher_scores >= teacher_threshold
        split_scores[split_name] = {
            "frame": split_frame,
            "teacher_scores": teacher_scores,
            "teacher_pass": teacher_pass,
            "baseline_scores": baseline_scores,
        }

    numeric = available_columns(splits.train, BASIC_NUMERIC)
    text = available_text_columns(splits.train, BASIC_TEXT)
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for split_name in ("validation", "test"):
        data = split_scores[split_name]
        baseline_rows.append(
            selection_metrics(
                data["frame"],
                scores=data["baseline_scores"],
                threshold=baseline_threshold,
                teacher_pass=data["teacher_pass"],
                teacher_scores=data["teacher_scores"],
                label="baseline_current_threshold",
                search_name=search,
                model_name=str(baseline_meta["approach"]),
                split=split_name,
                target_recall=None,
            )
        )
    rows.extend(baseline_rows)

    student_predictions: dict[str, dict[str, np.ndarray]] = {}
    trained_models: list[dict[str, Any]] = []
    for spec in STUDENT_SPECS:
        started = time.perf_counter()
        model = make_student_model(spec, numeric, text)
        model.fit(splits.train, split_scores["train"]["teacher_scores"])
        artifact_path = MODELS_DIR / f"{run_dir.name}_{search}_{spec.name}_seed{seed}.pkl"
        save_pickle(artifact_path, model)
        metadata = {
            "experiment_family": "teacher_student_basic_filter",
            "teacher_run": teacher_run,
            "search_name": search,
            "student_name": spec.name,
            "student_kind": spec.kind,
            "artifact_path": str(artifact_path),
            "numeric_features": numeric,
            "text_features": text if spec.use_text else [],
            "target": "full_visual_teacher_score",
            "teacher": {
                "approach": teacher_meta["approach"],
                "feature_mode": teacher_meta["feature_mode"],
                "threshold": teacher_threshold,
                "metadata_path": teacher_meta["metadata_path"],
                "artifact_path": teacher_meta["artifact_path"],
            },
            "baseline": {
                "approach": baseline_meta["approach"],
                "threshold": baseline_threshold,
                "metadata_path": baseline_meta["metadata_path"],
                "artifact_path": baseline_meta["artifact_path"],
            },
            "seed": int(seed),
            "split": "stratified_random_60_20_20",
            "fit_seconds": float(time.perf_counter() - started),
        }
        write_json(artifact_path.with_name(artifact_path.stem + "_metadata.json"), to_builtin(metadata))
        trained_models.append(metadata)

        student_predictions[spec.name] = {}
        for split_name in ("train", "validation", "test"):
            data = split_scores[split_name]
            preds = clipped_predict(model, data["frame"])
            student_predictions[spec.name][split_name] = preds
        for split_name in ("validation", "test"):
            data = split_scores[split_name]
            preds = student_predictions[spec.name][split_name]
            reg = regression_metrics(data["teacher_scores"], preds)
            for target in RECALL_TARGETS:
                chosen = choose_threshold_for_teacher_recall(
                    student_predictions[spec.name]["validation"],
                    split_scores["validation"]["teacher_pass"],
                    target_recall=float(target),
                )
                row = selection_metrics(
                    data["frame"],
                    scores=preds,
                    threshold=float(chosen["threshold"]),
                    teacher_pass=data["teacher_pass"],
                    teacher_scores=data["teacher_scores"],
                    label="student_teacher_score",
                    search_name=search,
                    model_name=spec.name,
                    split=split_name,
                    target_recall=float(target),
                )
                row.update(reg)
                row["student_kind"] = spec.kind
                row["teacher_approach"] = teacher_meta["approach"]
                row["teacher_threshold"] = teacher_threshold
                row["baseline_approach"] = baseline_meta["approach"]
                row["baseline_threshold"] = baseline_threshold
                row["fit_seconds"] = metadata["fit_seconds"]
                row["artifact_path"] = str(artifact_path)
                rows.append(row)

    # Keep a row-level test sheet for the preferred student per search after best selection.
    metrics_df = pd.DataFrame(rows)
    preferred = choose_best_for_search(metrics_df, search=search, target_recall=PREFERRED_RECALL_TARGET)
    if preferred is not None:
        data = split_scores["test"]
        best_scores = student_predictions[preferred["model_name"]]["test"]
        selected = best_scores >= float(preferred["threshold"])
        teacher_selected = data["teacher_pass"]
        baseline_selected = data["baseline_scores"] >= baseline_threshold
        item_cols = [col for col in ID_COLUMNS if col in data["frame"].columns]
        test_rows = data["frame"][item_cols].copy()
        test_rows["offline_sold_label"] = data["frame"][TARGET_COL].astype(int).to_numpy()
        test_rows["TeacherModel"] = str(teacher_meta["approach"])
        test_rows["TeacherScore"] = data["teacher_scores"]
        test_rows["TeacherThreshold"] = teacher_threshold
        test_rows["TeacherPass"] = teacher_selected
        test_rows["StudentModel"] = preferred["model_name"]
        test_rows["StudentScore"] = best_scores
        test_rows["StudentThreshold"] = float(preferred["threshold"])
        test_rows["StudentPass"] = selected
        test_rows["BaselineModel"] = str(baseline_meta["approach"])
        test_rows["BaselineScore"] = data["baseline_scores"]
        test_rows["BaselineThreshold"] = baseline_threshold
        test_rows["BaselinePass"] = baseline_selected
        test_rows["SearchName"] = search
        test_prediction_rows.append(test_rows)

    for row in rows:
        row["status"] = "trained"
        row["seed"] = int(seed)
        row["teacher_run"] = teacher_run
    return rows


def choose_best_for_search(metrics_df: pd.DataFrame, *, search: str, target_recall: float) -> dict[str, Any] | None:
    if metrics_df.empty:
        return None
    subset = metrics_df[
        (metrics_df["search_name"].astype(str) == str(search))
        & (metrics_df["label"].astype(str) == "student_teacher_score")
        & (metrics_df["split"].astype(str) == "validation")
        & (metrics_df["target_recall"].astype(float).round(4) == round(float(target_recall), 4))
    ].copy()
    if subset.empty:
        return None
    subset["meets_target"] = subset["teacher_recall"].fillna(0.0) >= float(target_recall)
    subset = subset.sort_values(
        ["meets_target", "selected_rate", "teacher_recall", "teacher_precision_among_selected", "mae"],
        ascending=[False, True, False, False, True],
        kind="stable",
    )
    return subset.iloc[0].to_dict()


def best_student_table(metrics_df: pd.DataFrame, *, target_recall: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metrics_df.empty:
        return pd.DataFrame()
    for search in sorted(metrics_df["search_name"].dropna().astype(str).unique()):
        best_val = choose_best_for_search(metrics_df, search=search, target_recall=target_recall)
        if best_val is None:
            continue
        test_match = metrics_df[
            (metrics_df["search_name"].astype(str) == search)
            & (metrics_df["model_name"].astype(str) == str(best_val["model_name"]))
            & (metrics_df["label"].astype(str) == "student_teacher_score")
            & (metrics_df["split"].astype(str) == "test")
            & (metrics_df["target_recall"].astype(float).round(4) == round(float(target_recall), 4))
        ]
        test_row = test_match.iloc[0].to_dict() if not test_match.empty else {}
        baseline_val = metrics_df[
            (metrics_df["search_name"].astype(str) == search)
            & (metrics_df["label"].astype(str) == "baseline_current_threshold")
            & (metrics_df["split"].astype(str) == "validation")
        ]
        baseline_test = metrics_df[
            (metrics_df["search_name"].astype(str) == search)
            & (metrics_df["label"].astype(str) == "baseline_current_threshold")
            & (metrics_df["split"].astype(str) == "test")
        ]
        baseline_val_row = baseline_val.iloc[0].to_dict() if not baseline_val.empty else {}
        baseline_test_row = baseline_test.iloc[0].to_dict() if not baseline_test.empty else {}
        rows.append(
            {
                "search_name": search,
                "student_model": best_val.get("model_name"),
                "student_threshold": best_val.get("threshold"),
                "target_teacher_recall": target_recall,
                "validation_teacher_recall": best_val.get("teacher_recall"),
                "test_teacher_recall": test_row.get("teacher_recall"),
                "validation_selected_count": best_val.get("selected_count"),
                "test_selected_count": test_row.get("selected_count"),
                "validation_selected_rate": best_val.get("selected_rate"),
                "test_selected_rate": test_row.get("selected_rate"),
                "validation_sold_precision": best_val.get("sold_precision_among_selected"),
                "test_sold_precision": test_row.get("sold_precision_among_selected"),
                "validation_teacher_precision": best_val.get("teacher_precision_among_selected"),
                "test_teacher_precision": test_row.get("teacher_precision_among_selected"),
                "test_cascade_final_count": test_row.get("cascade_final_count"),
                "test_cascade_final_sold_precision": test_row.get("cascade_final_sold_precision"),
                "baseline_model": baseline_val_row.get("model_name"),
                "baseline_threshold": baseline_val_row.get("threshold"),
                "baseline_validation_teacher_recall": baseline_val_row.get("teacher_recall"),
                "baseline_test_teacher_recall": baseline_test_row.get("teacher_recall"),
                "baseline_validation_selected_count": baseline_val_row.get("selected_count"),
                "baseline_test_selected_count": baseline_test_row.get("selected_count"),
                "baseline_validation_sold_precision": baseline_val_row.get("sold_precision_among_selected"),
                "baseline_test_sold_precision": baseline_test_row.get("sold_precision_among_selected"),
                "teacher_approach": best_val.get("teacher_approach"),
                "teacher_threshold": best_val.get("teacher_threshold"),
                "artifact_path": best_val.get("artifact_path"),
            }
        )
    return pd.DataFrame(rows)


def write_report(run_dir: Path, metrics_df: pd.DataFrame, best_df: pd.DataFrame) -> Path:
    report_path = assert_experiment_path(run_dir / "teacher_student_report.md")
    lines = [
        "# Teacher-Student Basic Filter Experiment",
        "",
        f"Run folder: `{run_dir}`",
        "",
        "The teacher is the best `full_scrape_plus_visual` model from the full-scrape comparison run.",
        "The student sees only the cheap first-page fields: `Title`, `Brand`, `Size`, `Price`, and `Likes`.",
        "",
        "The student target is the teacher's continuous full+visual score, not the sold label directly.",
        "Thresholds are chosen on validation to keep high teacher-pass recall, because this stage should avoid throwing away items that the expensive second model likes.",
        "",
    ]
    if not best_df.empty:
        lines.extend(
            [
                "## Best Student At 95% Teacher Recall Target",
                "",
                "| search | student | threshold | val teacher recall | test teacher recall | test selected | baseline test teacher recall | test sold precision |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in best_df.sort_values("search_name").iterrows():
            lines.append(
                f"| {row['search_name']} | {row['student_model']} | {float(row['student_threshold']):.4f} | "
                f"{float(row['validation_teacher_recall']):.3f} | {float(row['test_teacher_recall']):.3f} | "
                f"{int(row['test_selected_count'])} | {float(row['baseline_test_teacher_recall']):.3f} | "
                f"{float(row['test_sold_precision']):.3f} |"
            )
        lines.extend(
            [
                "",
                "`target_tradeoff_by_search.csv` contains the same best-student selection repeated for 90%, 95%, 98%, and 99% teacher-recall targets.",
            ]
        )
    if not metrics_df.empty:
        rows = metrics_df[metrics_df["label"].astype(str) == "student_teacher_score"]
        lines.extend(["", "## Models Tried", ""])
        for model_name in sorted(rows["model_name"].dropna().astype(str).unique()):
            model_rows = rows[rows["model_name"].astype(str) == model_name]
            mae = model_rows[model_rows["split"] == "test"]["mae"].mean()
            recall = model_rows[
                (model_rows["split"] == "test")
                & (model_rows["target_recall"].astype(float).round(4) == round(PREFERRED_RECALL_TARGET, 4))
            ]["teacher_recall"].mean()
            lines.append(f"- `{model_name}`: mean test teacher recall at 95% target `{recall:.3f}`, mean test MAE `{mae:.3f}`.")
    lines.extend(
        [
            "",
            "## How To Read This",
            "",
            "- Higher `test_teacher_recall` means the first filter keeps more items that the full+visual teacher would approve.",
            "- Lower selected count means fewer items need expensive full scraping, but too low can miss good teacher candidates.",
            "- `test_sold_precision` is the offline sold-label precision among items kept by the student, before the second-stage teacher filter.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_experiment_dirs()
    run_dir = Path(args.out_dir) if args.out_dir else OFFLINE_RUNS_DIR / run_id("student_fullvisual_score")
    run_dir = assert_experiment_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    teacher_run_dir = full_run_dir(args.teacher_run)
    best_rows = load_best_mode_rows(teacher_run_dir)
    if args.all_searches:
        searches = sorted((teacher_run_dir / "datasets").glob("*.csv"))
        search_names = [path.stem for path in searches]
    else:
        search_names = args.search
    if not search_names:
        raise SystemExit("Use --all-searches or --search.")

    metrics: list[dict[str, Any]] = []
    test_prediction_rows: list[pd.DataFrame] = []
    started = time.perf_counter()
    for search in search_names:
        print(f"[teacher_student] train search={search}", flush=True)
        step = time.perf_counter()
        try:
            rows = train_one_search(
                search=search,
                teacher_run=args.teacher_run,
                best_rows=best_rows,
                seed=args.seed,
                run_dir=run_dir,
                test_prediction_rows=test_prediction_rows,
            )
            metrics.extend(rows)
            print(f"[teacher_student] done search={search} rows={len(rows)} seconds={time.perf_counter() - step:.1f}", flush=True)
        except Exception as exc:
            print(f"[teacher_student] error search={search}: {type(exc).__name__}: {exc}", flush=True)
            metrics.append({"search_name": search, "status": "error", "reason": f"{type(exc).__name__}: {exc}"})

    metrics_df = pd.DataFrame(metrics)
    metrics_path = assert_experiment_path(run_dir / "metrics_long.csv")
    metrics_df.to_csv(metrics_path, index=False)
    best_df = best_student_table(metrics_df, target_recall=PREFERRED_RECALL_TARGET)
    best_path = assert_experiment_path(run_dir / "best_student_by_search.csv")
    best_df.to_csv(best_path, index=False)
    tradeoff_frames = [best_student_table(metrics_df, target_recall=target) for target in RECALL_TARGETS]
    tradeoff_df = pd.concat([frame for frame in tradeoff_frames if not frame.empty], ignore_index=True) if tradeoff_frames else pd.DataFrame()
    tradeoff_path = assert_experiment_path(run_dir / "target_tradeoff_by_search.csv")
    tradeoff_df.to_csv(tradeoff_path, index=False)
    test_items_path = assert_experiment_path(run_dir / "test_scored_items.csv")
    if test_prediction_rows:
        pd.concat(test_prediction_rows, ignore_index=True).to_csv(test_items_path, index=False)
    else:
        pd.DataFrame().to_csv(test_items_path, index=False)
    report_path = write_report(run_dir, metrics_df, best_df)
    write_manifest(
        run_dir / "manifest.json",
        command=" ".join(sys.argv),
        extra=to_builtin(
            {
                "experiment": "teacher_student_basic_filter",
                "teacher_run": args.teacher_run,
                "searches": search_names,
                "seed": args.seed,
                "student_specs": [spec.__dict__ for spec in STUDENT_SPECS],
                "recall_targets": list(RECALL_TARGETS),
                "preferred_recall_target": PREFERRED_RECALL_TARGET,
                "outputs": {
                    "metrics_long": str(metrics_path),
                    "best_student_by_search": str(best_path),
                    "target_tradeoff_by_search": str(tradeoff_path),
                    "test_scored_items": str(test_items_path),
                    "report": str(report_path),
                },
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        ),
    )
    print(f"Output folder: {run_dir}")
    print(f"Report: {report_path}")
    return {
        "run_dir": str(run_dir),
        "metrics_long": str(metrics_path),
        "best_student_by_search": str(best_path),
        "target_tradeoff_by_search": str(tradeoff_path),
        "test_scored_items": str(test_items_path),
        "report": str(report_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a basic-feature student to imitate the full+visual teacher score.")
    parser.add_argument("--all-searches", action="store_true")
    parser.add_argument("--search", action="append", default=[])
    parser.add_argument("--teacher-run", default=DEFAULT_TEACHER_RUN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
