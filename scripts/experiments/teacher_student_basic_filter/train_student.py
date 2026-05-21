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
PRECISION_TARGETS = (0.85, 0.90, 0.95, 0.98)
PREFERRED_PRECISION_TARGET = 0.95
BINARY_TOP_K_TARGETS = (5, 10, 15, 20)
PREFERRED_BINARY_TOP_K = 10
OBJECTIVES = ("recall", "precision", "binary")
DEFAULT_OBJECTIVE = "recall"
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

STUDENT_SPECS_BINARY = (
    StudentSpec("logistic_text_basic_student_v1", "logistic", use_text=True),
    StudentSpec("sgd_logloss_text_basic_student_v1", "sgd_logloss", use_text=True),
    StudentSpec("extra_trees_binary_basic_student_v1", "extra_trees_binary", use_text=True),
    StudentSpec("logistic_numeric_student_v1", "logistic", use_text=False),
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


def make_student_model(spec: StudentSpec, numeric_features: list[str], text_features: list[str], objective: str = "recall"):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge, SGDRegressor, SGDClassifier, LogisticRegression
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
    is_binary = objective == "binary"
    if spec.kind == "ridge":
        estimator = Ridge(alpha=1.0, random_state=DEFAULT_SEED)
    elif spec.kind == "sgd_huber":
        estimator = SGDRegressor(
            loss="huber",
            epsilon=0.08,
            alpha=0.0001,
            max_iter=3000,
            tol=1e-4,
            random_state=DEFAULT_SEED,
        )
    elif spec.kind == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=180,
            min_samples_leaf=4,
            random_state=DEFAULT_SEED,
            n_jobs=-1,
        )
    elif spec.kind == "logistic":
        estimator = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=DEFAULT_SEED,
        )
    elif spec.kind == "sgd_logloss":
        estimator = SGDClassifier(
            loss="log_loss",
            alpha=0.0001,
            max_iter=3000,
            tol=1e-4,
            random_state=DEFAULT_SEED,
        )
    elif spec.kind == "extra_trees_binary":
        estimator = ExtraTreesClassifier(
            n_estimators=180,
            min_samples_leaf=4,
            random_state=DEFAULT_SEED,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown student kind: {spec.kind}")
    return Pipeline([("features", preprocessor), ("model", estimator)])


def clipped_predict(model: Any, frame: pd.DataFrame) -> np.ndarray:
    return np.clip(np.asarray(model.predict(frame), dtype=float), 0.0, 1.0)


def predict_proba_positive(model: Any, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.clip(np.asarray(proba[:, 1], dtype=float), 0.0, 1.0)
    return clipped_predict(model, frame)


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


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, accuracy_score

    out: dict[str, float] = {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "pearson": np.nan, "spearman": np.nan}
    if len(y_true) == 0:
        return out
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    try:
        if len(np.unique(y_true_arr)) > 1:
            out["roc_auc"] = float(roc_auc_score(y_true_arr, y_pred_arr))
            out["pr_auc"] = float(average_precision_score(y_true_arr, y_pred_arr))
        out["log_loss"] = float(log_loss(y_true_arr, np.clip(y_pred_arr, 1e-7, 1 - 1e-7)))
        out["accuracy"] = float(accuracy_score(y_true_arr, y_pred_arr >= 0.5))
    except Exception:
        pass
    return out


def precision_at_k(scores: np.ndarray, y_true: np.ndarray, k: int) -> float:
    if len(scores) == 0 or k <= 0:
        return float("nan")
    y_true = np.asarray(y_true, dtype=bool)
    top_k_idx = np.argsort(scores)[-k:]
    return float(y_true[top_k_idx].sum() / len(top_k_idx))


def recall_at_k(scores: np.ndarray, y_true: np.ndarray, k: int) -> float:
    if len(scores) == 0 or k <= 0:
        return float("nan")
    y_true = np.asarray(y_true, dtype=bool)
    total_pos = int(y_true.sum())
    if total_pos == 0:
        return float("nan")
    top_k_idx = np.argsort(scores)[-k:]
    return float(y_true[top_k_idx].sum() / total_pos)


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


def choose_threshold_for_teacher_precision(
    scores: np.ndarray,
    teacher_pass: np.ndarray,
    *,
    target_precision: float,
) -> dict[str, Any]:
    teacher_pass = np.asarray(teacher_pass, dtype=bool)
    teacher_total = int(teacher_pass.sum())
    if len(scores) == 0:
        return {"threshold": 1.0, "teacher_precision": np.nan, "selected_count": 0, "teacher_selected_count": 0}
    if teacher_total == 0:
        threshold = float(np.quantile(scores, 0.99))
        selected = scores >= threshold
        return {
            "threshold": threshold,
            "teacher_precision": np.nan,
            "selected_count": int(selected.sum()),
            "teacher_selected_count": 0,
        }
    candidates = sorted(set(np.round(np.asarray(scores, dtype=float), 6).tolist()), reverse=True)
    best: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    for threshold in candidates:
        selected = scores >= threshold
        selected_count = int(selected.sum())
        if selected_count == 0:
            continue
        teacher_selected = int((selected & teacher_pass).sum())
        precision = float(teacher_selected / selected_count)
        row = {
            "threshold": float(threshold),
            "teacher_precision": precision,
            "selected_count": selected_count,
            "teacher_selected_count": teacher_selected,
        }
        if fallback is None or row["teacher_precision"] > fallback["teacher_precision"] or (
            row["teacher_precision"] == fallback["teacher_precision"] and row["selected_count"] > fallback["selected_count"]
        ):
            fallback = row
        if precision >= target_precision:
            if best is None or row["selected_count"] > best["selected_count"] or (
                row["selected_count"] == best["selected_count"] and row["threshold"] < best["threshold"]
            ):
                best = row
    return best or fallback or {"threshold": 1.0, "teacher_precision": 0.0, "selected_count": 0, "teacher_selected_count": 0}


def choose_threshold_for_precision_at_k(
    scores: np.ndarray,
    teacher_pass: np.ndarray,
    *,
    k: int,
) -> dict[str, Any]:
    teacher_pass = np.asarray(teacher_pass, dtype=bool)
    if len(scores) == 0:
        return {"threshold": 1.0, "precision_at_k": np.nan, "selected_count": 0, "teacher_selected_count": 0, "k": k}
    top_k_idx = np.argsort(scores)[-k:]
    threshold = float(scores[top_k_idx[0]])
    selected = scores >= threshold
    selected_count = int(selected.sum())
    teacher_selected = int((selected & teacher_pass).sum())
    precision = float(teacher_pass[top_k_idx].sum() / len(top_k_idx))
    return {
        "threshold": threshold,
        "precision_at_k": precision,
        "selected_count": selected_count,
        "teacher_selected_count": teacher_selected,
        "k": k,
    }


def choose_threshold_for_pass_rate(
    scores: np.ndarray,
    teacher_pass: np.ndarray,
    *,
    target_pass_rate: float = 0.15,
    min_precision: float = 0.5,
) -> dict[str, Any]:
    """Choose a threshold that passes roughly target_pass_rate fraction of items,
    while keeping precision above min_precision if possible."""
    teacher_pass = np.asarray(teacher_pass, dtype=bool)
    if len(scores) == 0:
        return {"threshold": 1.0, "precision_at_k": np.nan, "selected_count": 0, "teacher_selected_count": 0, "pass_rate": target_pass_rate}
    sorted_scores = np.sort(scores)[::-1]
    n = len(scores)
    # Try thresholds at various percentiles, from high to low
    best = None
    for percentile in np.linspace(0.90, 0.50, 41):
        threshold = float(np.quantile(scores, percentile))
        selected = scores >= threshold
        selected_count = int(selected.sum())
        if selected_count == 0:
            continue
        teacher_selected = int((selected & teacher_pass).sum())
        precision = float(teacher_selected / selected_count)
        pass_rate = float(selected_count / n)
        row = {
            "threshold": threshold,
            "precision_at_k": precision,
            "selected_count": selected_count,
            "teacher_selected_count": teacher_selected,
            "pass_rate": pass_rate,
        }
        if precision >= min_precision and pass_rate <= target_pass_rate:
            if best is None or pass_rate > best["pass_rate"] or (
                pass_rate == best["pass_rate"] and precision > best["precision_at_k"]
            ):
                best = row
    # Fallback: if no threshold meets min_precision, pick the one with highest precision
    if best is None:
        candidates = sorted(set(np.round(np.asarray(scores, dtype=float), 6).tolist()), reverse=True)
        for threshold in candidates:
            selected = scores >= threshold
            selected_count = int(selected.sum())
            if selected_count == 0:
                continue
            teacher_selected = int((selected & teacher_pass).sum())
            precision = float(teacher_selected / selected_count)
            pass_rate = float(selected_count / n)
            row = {
                "threshold": threshold,
                "precision_at_k": precision,
                "selected_count": selected_count,
                "teacher_selected_count": teacher_selected,
                "pass_rate": pass_rate,
            }
            if best is None or precision > best["precision_at_k"]:
                best = row
    return best or {"threshold": 1.0, "precision_at_k": np.nan, "selected_count": 0, "teacher_selected_count": 0, "pass_rate": 0.0}


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
    target_precision: float | None = None,
    objective: str | None = None,
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
        "objective": objective,
        "target_recall": target_recall,
        "target_precision": target_precision,
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
    objective: str,
    targets: tuple[float, ...],
    preferred_target: float,
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
                target_precision=None,
                objective=objective,
            )
        )
    rows.extend(baseline_rows)

    student_predictions: dict[str, dict[str, np.ndarray]] = {}
    trained_models: list[dict[str, Any]] = []
    specs = STUDENT_SPECS_BINARY if objective == "binary" else STUDENT_SPECS
    for spec in specs:
        started = time.perf_counter()
        model = make_student_model(spec, numeric, text, objective=objective)
        if objective == "binary":
            train_y = split_scores["train"]["teacher_pass"].astype(int)
            model.fit(splits.train, train_y)
        else:
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
            "target": "teacher_pass_binary" if objective == "binary" else "full_visual_teacher_score",
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
            if objective == "binary":
                preds = predict_proba_positive(model, data["frame"])
            else:
                preds = clipped_predict(model, data["frame"])
            student_predictions[spec.name][split_name] = preds
        for split_name in ("validation", "test"):
            data = split_scores[split_name]
            preds = student_predictions[spec.name][split_name]
            if objective == "binary":
                reg = classification_metrics(data["teacher_pass"].astype(int), preds)
            else:
                reg = regression_metrics(data["teacher_scores"], preds)
            for target in targets:
                if objective == "recall":
                    chosen = choose_threshold_for_teacher_recall(
                        student_predictions[spec.name]["validation"],
                        split_scores["validation"]["teacher_pass"],
                        target_recall=float(target),
                    )
                    target_recall_value: float | None = float(target)
                    target_precision_value: float | None = None
                    target_binary_k_value: int | None = None
                elif objective == "precision":
                    chosen = choose_threshold_for_teacher_precision(
                        student_predictions[spec.name]["validation"],
                        split_scores["validation"]["teacher_pass"],
                        target_precision=float(target),
                    )
                    target_recall_value = None
                    target_precision_value = float(target)
                    target_binary_k_value = None
                else:
                    # Use pass-rate-based threshold to ensure enough items get through stage 1
                    chosen = choose_threshold_for_pass_rate(
                        student_predictions[spec.name]["validation"],
                        split_scores["validation"]["teacher_pass"],
                        target_pass_rate=0.15,
                        min_precision=0.5,
                    )
                    target_recall_value = None
                    target_precision_value = None
                    target_binary_k_value = int(target)
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
                    target_recall=target_recall_value,
                    target_precision=target_precision_value,
                    objective=objective,
                )
                if objective == "binary":
                    row["precision_at_k"] = precision_at_k(preds, data["teacher_pass"], int(target))
                    row["recall_at_k"] = recall_at_k(preds, data["teacher_pass"], int(target))
                    row["target_binary_k"] = target_binary_k_value
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
    preferred = choose_best_for_search(metrics_df, search=search, objective=objective, target=preferred_target)
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


def choose_best_for_search(
    metrics_df: pd.DataFrame,
    *,
    search: str,
    objective: str,
    target: float,
) -> dict[str, Any] | None:
    if metrics_df.empty:
        return None
    if objective == "recall":
        target_col = "target_recall"
    elif objective == "precision":
        target_col = "target_precision"
    elif objective == "binary":
        target_col = "target_binary_k"
    else:
        raise ValueError(f"Unknown objective: {objective}")
    subset = metrics_df[
        (metrics_df["search_name"].astype(str) == str(search))
        & (metrics_df["label"].astype(str) == "student_teacher_score")
        & (metrics_df["split"].astype(str) == "validation")
        & (pd.to_numeric(metrics_df[target_col], errors="coerce").round(4) == round(float(target), 4))
    ].copy()
    if subset.empty:
        return None
    if objective == "recall":
        subset["meets_target"] = subset["teacher_recall"].fillna(0.0) >= float(target)
        subset = subset.sort_values(
            ["meets_target", "selected_rate", "teacher_recall", "teacher_precision_among_selected", "mae"],
            ascending=[False, True, False, False, True],
            kind="stable",
        )
    elif objective == "precision":
        subset["meets_target"] = subset["teacher_precision_among_selected"].fillna(0.0) >= float(target)
        subset = subset.sort_values(
            ["meets_target", "selected_rate", "teacher_precision_among_selected", "teacher_recall", "mae"],
            ascending=[False, False, False, False, True],
            kind="stable",
        )
    else:
        # For binary/cascade use, prefer models that pass enough items while maintaining decent precision
        subset["meets_min_rate"] = subset["selected_rate"].fillna(0.0) >= 0.03
        subset = subset.sort_values(
            ["meets_min_rate", "precision_at_k", "selected_rate", "recall_at_k", "roc_auc"],
            ascending=[False, False, False, False, False],
            kind="stable",
        )
    return subset.iloc[0].to_dict()


def best_student_table(metrics_df: pd.DataFrame, *, objective: str, target: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metrics_df.empty:
        return pd.DataFrame()
    if objective == "recall":
        target_col = "target_recall"
    elif objective == "precision":
        target_col = "target_precision"
    else:
        target_col = "target_binary_k"
    for search in sorted(metrics_df["search_name"].dropna().astype(str).unique()):
        best_val = choose_best_for_search(metrics_df, search=search, objective=objective, target=target)
        if best_val is None:
            continue
        test_match = metrics_df[
            (metrics_df["search_name"].astype(str) == search)
            & (metrics_df["model_name"].astype(str) == str(best_val["model_name"]))
            & (metrics_df["label"].astype(str) == "student_teacher_score")
            & (metrics_df["split"].astype(str) == "test")
            & (pd.to_numeric(metrics_df[target_col], errors="coerce").round(4) == round(float(target), 4))
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
        row_out = {
            "search_name": search,
            "student_model": best_val.get("model_name"),
            "student_threshold": best_val.get("threshold"),
            "objective": objective,
            "target_teacher_recall": float(target) if objective == "recall" else np.nan,
            "target_teacher_precision": float(target) if objective == "precision" else np.nan,
            "target_binary_k": int(target) if objective == "binary" else np.nan,
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
            "baseline_validation_teacher_precision": baseline_val_row.get("teacher_precision_among_selected"),
            "baseline_test_teacher_precision": baseline_test_row.get("teacher_precision_among_selected"),
            "baseline_validation_selected_count": baseline_val_row.get("selected_count"),
            "baseline_test_selected_count": baseline_test_row.get("selected_count"),
            "baseline_validation_sold_precision": baseline_val_row.get("sold_precision_among_selected"),
            "baseline_test_sold_precision": baseline_test_row.get("sold_precision_among_selected"),
            "teacher_approach": best_val.get("teacher_approach"),
            "teacher_threshold": best_val.get("teacher_threshold"),
            "artifact_path": best_val.get("artifact_path"),
        }
        if objective == "binary":
            row_out["validation_precision_at_k"] = best_val.get("precision_at_k")
            row_out["test_precision_at_k"] = test_row.get("precision_at_k")
            row_out["validation_recall_at_k"] = best_val.get("recall_at_k")
            row_out["test_recall_at_k"] = test_row.get("recall_at_k")
            row_out["validation_roc_auc"] = best_val.get("roc_auc")
            row_out["test_roc_auc"] = test_row.get("roc_auc")
            row_out["validation_pr_auc"] = best_val.get("pr_auc")
            row_out["test_pr_auc"] = test_row.get("pr_auc")
        rows.append(row_out)
    return pd.DataFrame(rows)


def write_report(
    run_dir: Path,
    metrics_df: pd.DataFrame,
    best_df: pd.DataFrame,
    *,
    objective: str,
    preferred_target: float,
    targets: tuple[float, ...],
) -> Path:
    report_path = assert_experiment_path(run_dir / "teacher_student_report.md")
    target_pct = int(round(float(preferred_target) * 100))
    tradeoff_pcts = ", ".join(f"{int(round(float(t) * 100))}%" for t in targets)
    if objective == "recall":
        objective_blurb = (
            "Thresholds are chosen on validation to keep high teacher-pass recall, because this stage should "
            "avoid throwing away items that the expensive second model likes."
        )
        heading = f"## Best Student At {target_pct}% Teacher Recall Target"
        tradeoff_blurb = f"`target_tradeoff_by_search.csv` contains the same best-student selection repeated for {tradeoff_pcts} teacher-recall targets."
        table_header_metric = "val teacher recall | test teacher recall"
        baseline_metric_label = "baseline test teacher recall"
        target_col = "target_recall"
        metric_col = "teacher_recall"
        target_label = "teacher recall"
    elif objective == "precision":
        objective_blurb = (
            "Thresholds are chosen on validation to keep teacher-pass precision above the target, so the cheap "
            "first stage acts as a precision filter that only forwards items the expensive second model is very likely to approve."
        )
        heading = f"## Best Student At {target_pct}% Teacher Precision Target"
        tradeoff_blurb = f"`target_tradeoff_by_search.csv` contains the same best-student selection repeated for {tradeoff_pcts} teacher-precision targets."
        table_header_metric = "val teacher precision | test teacher precision"
        baseline_metric_label = "baseline test teacher precision"
        target_col = "target_precision"
        metric_col = "teacher_precision_among_selected"
        target_label = "teacher precision"
    else:
        objective_blurb = (
            "Students are binary classifiers trained to predict teacher-pass directly. "
            "The best model is chosen by precision@k on validation, because the cascade caps stage-1 at k items per search."
        )
        heading = f"## Best Binary Classifier At Top-{int(preferred_target)} Precision"
        tradeoff_blurb = f"`target_tradeoff_by_search.csv` contains the same best-student selection repeated for top-{', '.join(str(t) for t in targets)} precision targets."
        table_header_metric = "val precision@k | test precision@k"
        baseline_metric_label = "baseline test teacher precision"
        target_col = "target_binary_k"
        metric_col = "precision_at_k"
        target_label = "precision@k"
    lines = [
        "# Teacher-Student Basic Filter Experiment",
        "",
        f"Run folder: `{run_dir}`",
        f"Objective: `{objective}` (target `{preferred_target:.2f}`).",
        "",
        "The teacher is the best `full_scrape_plus_visual` model from the full-scrape comparison run.",
        "The student sees only the cheap first-page fields: `Title`, `Brand`, `Size`, `Price`, and `Likes`.",
        "",
    ]
    if objective == "binary":
        lines.append("The student target is the teacher's pass/fail label, trained as a binary classifier.")
    else:
        lines.append("The student target is the teacher's continuous full+visual score, not the sold label directly.")
    lines.extend([objective_blurb, ""])
    if not best_df.empty:
        if objective == "recall":
            val_metric_key = "validation_teacher_recall"
            test_metric_key = "test_teacher_recall"
            baseline_metric_key = "baseline_test_teacher_recall"
        elif objective == "precision":
            val_metric_key = "validation_teacher_precision"
            test_metric_key = "test_teacher_precision"
            baseline_metric_key = "baseline_test_teacher_precision"
        else:
            val_metric_key = "validation_precision_at_k"
            test_metric_key = "test_precision_at_k"
            baseline_metric_key = "baseline_test_teacher_precision"
        lines.extend(
            [
                heading,
                "",
                f"| search | student | threshold | {table_header_metric} | test selected | {baseline_metric_label} | test sold precision |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in best_df.sort_values("search_name").iterrows():
            val_metric = row.get(val_metric_key)
            test_metric = row.get(test_metric_key)
            baseline_metric = row.get(baseline_metric_key)
            lines.append(
                f"| {row['search_name']} | {row['student_model']} | {float(row['student_threshold']):.4f} | "
                f"{float(val_metric) if pd.notna(val_metric) else float('nan'):.3f} | "
                f"{float(test_metric) if pd.notna(test_metric) else float('nan'):.3f} | "
                f"{int(row['test_selected_count'])} | "
                f"{float(baseline_metric) if pd.notna(baseline_metric) else float('nan'):.3f} | "
                f"{float(row['test_sold_precision']) if pd.notna(row.get('test_sold_precision')) else float('nan'):.3f} |"
            )
        lines.extend(["", tradeoff_blurb])
    if not metrics_df.empty:
        rows = metrics_df[metrics_df["label"].astype(str) == "student_teacher_score"]
        lines.extend(["", "## Models Tried", ""])
        for model_name in sorted(rows["model_name"].dropna().astype(str).unique()):
            model_rows = rows[rows["model_name"].astype(str) == model_name]
            if objective == "binary":
                p_at_k = model_rows[model_rows["split"] == "test"]["precision_at_k"].mean()
                r_at_k = model_rows[model_rows["split"] == "test"]["recall_at_k"].mean()
                roc = model_rows[model_rows["split"] == "test"]["roc_auc"].mean()
                lines.append(
                    f"- `{model_name}`: mean test precision@k `{p_at_k:.3f}`, recall@k `{r_at_k:.3f}`, ROC-AUC `{roc:.3f}`."
                )
            else:
                mae = model_rows[model_rows["split"] == "test"]["mae"].mean()
                metric_value = model_rows[
                    (model_rows["split"] == "test")
                    & (pd.to_numeric(model_rows[target_col], errors="coerce").round(4) == round(float(preferred_target), 4))
                ][metric_col].mean()
                lines.append(
                    f"- `{model_name}`: mean test {target_label} at {target_pct}% target `{metric_value:.3f}`, mean test MAE `{mae:.3f}`."
                )
    if objective == "recall":
        howto = [
            "- Higher `test_teacher_recall` means the first filter keeps more items that the full+visual teacher would approve.",
            "- Lower selected count means fewer items need expensive full scraping, but too low can miss good teacher candidates.",
        ]
    elif objective == "precision":
        howto = [
            "- Higher `test_teacher_precision` means the first filter mostly forwards items the full+visual teacher would also approve.",
            "- Higher selected count is good as long as `test_teacher_precision` stays above the target.",
        ]
    else:
        howto = [
            "- Higher `precision_at_k` means more of the top-k items forwarded to stage 2 are actually teacher-approved.",
            "- Higher `recall_at_k` means the top-k items catch more of the total teacher-approved pool.",
        ]
    lines.extend(
        [
            "",
            "## How To Read This",
            "",
            *howto,
            "- `test_sold_precision` is the offline sold-label precision among items kept by the student, before the second-stage teacher filter.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_experiment_dirs()
    objective = str(args.objective).strip().lower()
    if objective not in OBJECTIVES:
        raise SystemExit(f"Unknown --objective {args.objective!r}, choose from {OBJECTIVES}")
    if objective == "recall":
        targets = RECALL_TARGETS
        preferred = float(args.target) if args.target is not None else PREFERRED_RECALL_TARGET
        default_prefix = "student_fullvisual_score"
    elif objective == "precision":
        targets = PRECISION_TARGETS
        preferred = float(args.target) if args.target is not None else PREFERRED_PRECISION_TARGET
        default_prefix = "student_fullvisual_score_precision"
    else:
        targets = BINARY_TOP_K_TARGETS
        preferred = int(args.target) if args.target is not None else PREFERRED_BINARY_TOP_K
        default_prefix = "student_binary_topk"
    run_dir = Path(args.out_dir) if args.out_dir else OFFLINE_RUNS_DIR / run_id(default_prefix)
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
        print(f"[teacher_student] train search={search} objective={objective} preferred_target={preferred}", flush=True)
        step = time.perf_counter()
        try:
            rows = train_one_search(
                search=search,
                teacher_run=args.teacher_run,
                best_rows=best_rows,
                seed=args.seed,
                run_dir=run_dir,
                test_prediction_rows=test_prediction_rows,
                objective=objective,
                targets=targets,
                preferred_target=preferred,
            )
            metrics.extend(rows)
            print(f"[teacher_student] done search={search} rows={len(rows)} seconds={time.perf_counter() - step:.1f}", flush=True)
        except Exception as exc:
            print(f"[teacher_student] error search={search}: {type(exc).__name__}: {exc}", flush=True)
            metrics.append({"search_name": search, "status": "error", "reason": f"{type(exc).__name__}: {exc}"})

    metrics_df = pd.DataFrame(metrics)
    metrics_path = assert_experiment_path(run_dir / "metrics_long.csv")
    metrics_df.to_csv(metrics_path, index=False)
    best_df = best_student_table(metrics_df, objective=objective, target=preferred)
    best_path = assert_experiment_path(run_dir / "best_student_by_search.csv")
    best_df.to_csv(best_path, index=False)
    tradeoff_frames = [best_student_table(metrics_df, objective=objective, target=target) for target in targets]
    tradeoff_df = pd.concat([frame for frame in tradeoff_frames if not frame.empty], ignore_index=True) if tradeoff_frames else pd.DataFrame()
    tradeoff_path = assert_experiment_path(run_dir / "target_tradeoff_by_search.csv")
    tradeoff_df.to_csv(tradeoff_path, index=False)
    test_items_path = assert_experiment_path(run_dir / "test_scored_items.csv")
    if test_prediction_rows:
        pd.concat(test_prediction_rows, ignore_index=True).to_csv(test_items_path, index=False)
    else:
        pd.DataFrame().to_csv(test_items_path, index=False)
    report_path = write_report(run_dir, metrics_df, best_df, objective=objective, preferred_target=preferred, targets=targets)
    manifest_extra: dict[str, Any] = {
        "experiment": "teacher_student_basic_filter",
        "teacher_run": args.teacher_run,
        "searches": search_names,
        "seed": args.seed,
        "student_specs": [spec.__dict__ for spec in (STUDENT_SPECS_BINARY if objective == "binary" else STUDENT_SPECS)],
        "objective": objective,
        "targets": list(targets),
        "preferred_target": preferred,
        "outputs": {
            "metrics_long": str(metrics_path),
            "best_student_by_search": str(best_path),
            "target_tradeoff_by_search": str(tradeoff_path),
            "test_scored_items": str(test_items_path),
            "report": str(report_path),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    if objective == "recall":
        manifest_extra["recall_targets"] = list(targets)
        manifest_extra["preferred_recall_target"] = preferred
    elif objective == "precision":
        manifest_extra["precision_targets"] = list(targets)
        manifest_extra["preferred_precision_target"] = preferred
    else:
        manifest_extra["binary_top_k_targets"] = list(targets)
        manifest_extra["preferred_binary_top_k"] = preferred
    write_manifest(run_dir / "manifest.json", command=" ".join(sys.argv), extra=to_builtin(manifest_extra))
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
    parser.add_argument("--objective", choices=list(OBJECTIVES), default=DEFAULT_OBJECTIVE,
                        help="Threshold-selection objective: 'recall' picks the smallest selected_count above a teacher-recall floor; 'precision' picks the largest selected_count above a teacher-precision floor; 'binary' trains classifiers that predict teacher-pass directly and selects by precision@k.")
    parser.add_argument("--target", type=float, default=None,
                        help="Preferred target value for the chosen objective (default 0.95 for both).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
