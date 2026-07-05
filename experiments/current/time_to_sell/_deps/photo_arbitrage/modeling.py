from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.current.time_to_sell._deps.photo_arbitrage.features import MODEL_FEATURES, add_photo_features, heuristic_bad_photo_probability
from experiments.current.time_to_sell._deps.photo_arbitrage.paths import MODELS_DIR, assert_photo_path, utc_now_iso, write_json
from experiments.current.time_to_sell._deps.photo_arbitrage.quality_methods import MethodConfig, add_quality_method_scores, model_feature_columns, normalize_methods


VALID_LABELS = {"photo_quality_bad", "photo_quality_good", "unclear", "not_item_photo"}
TRAINING_LABELS = {"photo_quality_bad": 1, "photo_quality_good": 0}
LABEL_SOURCE_MANUAL = "manual"
LABEL_SOURCE_FASHIONCLIP_PSEUDO = "fashionclip_pseudo"
LABEL_SOURCES = {LABEL_SOURCE_MANUAL, LABEL_SOURCE_FASHIONCLIP_PSEUDO}
MODEL_VERSION = "photo_quality_v1"


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
    except Exception:
        return False
    return True


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in VALID_LABELS else ""


def training_label_series(labels: pd.DataFrame, *, label_source: str = LABEL_SOURCE_MANUAL) -> pd.Series:
    if label_source not in LABEL_SOURCES:
        raise ValueError(f"Unknown label source: {label_source}")
    if label_source == LABEL_SOURCE_FASHIONCLIP_PSEUDO:
        raw = labels.get("FashionClipPseudoLabel", pd.Series(dtype=str))
    else:
        raw = labels.get("manual_label", pd.Series(dtype=str))
    return raw.reindex(labels.index, fill_value="").map(normalize_label)


def label_readiness_summary(labels: pd.DataFrame, *, label_source: str = LABEL_SOURCE_MANUAL) -> dict[str, Any]:
    selected = training_label_series(labels, label_source=label_source)
    counts = selected.value_counts(dropna=False).to_dict()
    bad_rows = int(counts.get("photo_quality_bad", 0))
    good_rows = int(counts.get("photo_quality_good", 0))
    trainable_rows = bad_rows + good_rows
    missing: list[str] = []
    if bad_rows < 2:
        missing.append(f"{2 - bad_rows} more photo_quality_bad")
    if good_rows < 2:
        missing.append(f"{2 - good_rows} more photo_quality_good")
    status = "ready" if not missing else "not_ready"
    return {
        "status": status,
        "label_source": label_source,
        "total_rows": int(len(labels)),
        "trainable_rows": trainable_rows,
        "bad_rows": bad_rows,
        "good_rows": good_rows,
        "unclear_rows": int(counts.get("unclear", 0)),
        "not_item_photo_rows": int(counts.get("not_item_photo", 0)),
        "blank_rows": int(counts.get("", 0)),
        "label_counts": {str(label): int(count) for label, count in counts.items()},
        "missing": missing,
        "message": "Ready to train." if status == "ready" else "Need " + " and ".join(missing) + " before training.",
    }


def resolve_feature_columns(method_config: MethodConfig | None = None) -> list[str]:
    columns = list(MODEL_FEATURES)
    if method_config is not None:
        for column in model_feature_columns(method_config.methods):
            if column not in columns:
                columns.append(column)
    return columns


def ensure_model_features(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    method_config: MethodConfig | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    if not set(MODEL_FEATURES).issubset(out.columns):
        out = add_photo_features(out)
    if method_config is not None:
        extra_columns = [column for column in feature_columns if column not in MODEL_FEATURES]
        missing_extra = [column for column in extra_columns if column not in out.columns]
        if missing_extra:
            out = add_quality_method_scores(out, config=method_config)
    for column in feature_columns:
        if column not in out.columns:
            out[column] = np.nan
    return out


def prepare_labeled_frame(
    labels: pd.DataFrame,
    *,
    method_config: MethodConfig | None = None,
    label_source: str = LABEL_SOURCE_MANUAL,
) -> pd.DataFrame:
    out = labels.copy()
    if "manual_label" not in out.columns:
        out["manual_label"] = ""
    out["manual_label"] = out["manual_label"].map(normalize_label)
    out["TrainingLabel"] = training_label_series(out, label_source=label_source)
    out["TrainingLabelSource"] = label_source
    out = out[out["TrainingLabel"].isin(TRAINING_LABELS)].copy()
    out["PhotoQualityTarget"] = out["TrainingLabel"].map(TRAINING_LABELS).astype(int)
    feature_columns = resolve_feature_columns(method_config)
    out = ensure_model_features(out, feature_columns=feature_columns, method_config=method_config)
    return out.reset_index(drop=True)


def make_photo_quality_pipeline():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
        ]
    )


def make_photo_quality_estimator(feature_columns: list[str]) -> tuple[Any, str]:
    if sklearn_available():
        return make_photo_quality_pipeline(), "logistic_regression"
    return CentroidPhotoQualityModel(feature_columns), "centroid_fallback"


class CentroidPhotoQualityModel:
    """Small fallback classifier used when scikit-learn is unavailable."""

    def __init__(self, features: list[str]):
        self.features = features
        self.fill_values: np.ndarray | None = None
        self.bad_center: np.ndarray | None = None
        self.good_center: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "CentroidPhotoQualityModel":
        values = x.apply(pd.to_numeric, errors="coerce")
        fill = values.median(numeric_only=True).fillna(0.0)
        arr = values.fillna(fill).to_numpy(dtype=float)
        target = y.to_numpy(dtype=int)
        self.fill_values = fill.to_numpy(dtype=float)
        self.bad_center = arr[target == 1].mean(axis=0)
        self.good_center = arr[target == 0].mean(axis=0)
        return self

    def _matrix(self, x: pd.DataFrame) -> np.ndarray:
        values = x[self.features].apply(pd.to_numeric, errors="coerce")
        fill = self.fill_values if self.fill_values is not None else np.zeros(len(self.features))
        return values.fillna(pd.Series(fill, index=self.features)).to_numpy(dtype=float)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        arr = self._matrix(x)
        bad = self.bad_center if self.bad_center is not None else np.zeros(arr.shape[1])
        good = self.good_center if self.good_center is not None else np.zeros(arr.shape[1])
        dist_bad = np.linalg.norm(arr - bad, axis=1)
        dist_good = np.linalg.norm(arr - good, axis=1)
        prob_bad = dist_good / np.maximum(dist_bad + dist_good, 1e-9)
        return np.column_stack([1.0 - prob_bad, prob_bad])


def safe_metric(value: Any, digits: int = 4) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not np.isfinite(number):
        return None
    return round(number, digits)


def safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return safe_metric(numerator / denominator)


def binary_auc(labels: pd.Series, scores: pd.Series) -> float | None:
    work = pd.DataFrame({"label": labels, "score": scores}).dropna()
    if work.empty:
        return None
    positives = int((work["label"] == 1).sum())
    negatives = int((work["label"] == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = work["score"].rank(method="average")
    positive_rank_sum = float(ranks[work["label"] == 1].sum())
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return safe_metric(auc)


def binary_classification_metrics(labels: pd.Series, scores: pd.Series, *, threshold: float = 0.5) -> dict[str, Any]:
    target = pd.Series(labels).astype(int).reset_index(drop=True)
    score_values = pd.to_numeric(pd.Series(scores), errors="coerce").reset_index(drop=True)
    valid = score_values.notna() & np.isfinite(score_values)
    target = target[valid]
    score_values = score_values[valid]
    if target.empty:
        return {"status": "skipped", "reason": "No valid scores."}
    predicted_bad = score_values >= float(threshold)
    actual_bad = target == 1
    tp = int((predicted_bad & actual_bad).sum())
    fp = int((predicted_bad & ~actual_bad).sum())
    tn = int((~predicted_bad & ~actual_bad).sum())
    fn = int((~predicted_bad & actual_bad).sum())
    total = int(len(target))
    return {
        "status": "evaluated",
        "rows": total,
        "positive_rows": int(actual_bad.sum()),
        "negative_rows": int((~actual_bad).sum()),
        "threshold": float(threshold),
        "auc_bad_vs_good": binary_auc(target, score_values),
        "accuracy": safe_divide(tp + tn, total),
        "bad_precision": safe_divide(tp, tp + fp),
        "bad_recall": safe_divide(tp, tp + fn),
        "good_precision": safe_divide(tn, tn + fn),
        "good_recall": safe_divide(tn, tn + fp),
        "mean_bad_probability": safe_metric(score_values.mean()),
        "true_bad": tp,
        "false_bad": fp,
        "true_good": tn,
        "false_good": fn,
    }


def stratified_fold_indices(labels: pd.Series, *, max_folds: int = 5) -> list[list[int]]:
    target = pd.Series(labels).astype(int).reset_index(drop=True)
    counts = target.value_counts()
    if len(counts) < 2 or int(counts.min()) < 2:
        return []
    fold_count = max(2, min(int(max_folds), int(counts.min())))
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    for class_value in sorted(target.unique()):
        indices = target[target == class_value].index.tolist()
        for offset, idx in enumerate(indices):
            folds[offset % fold_count].append(int(idx))
    return [sorted(fold) for fold in folds if fold]


def cross_validate_photo_quality_model(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    feature_columns: list[str],
    max_folds: int = 5,
) -> dict[str, Any]:
    target = pd.Series(y).astype(int).reset_index(drop=True)
    features = x.reset_index(drop=True)
    folds = stratified_fold_indices(target, max_folds=max_folds)
    class_counts = {str(label): int(count) for label, count in target.value_counts().sort_index().items()}
    if not folds:
        return {
            "status": "skipped",
            "reason": "Need at least two usable labels in each class for stratified evaluation.",
            "class_counts": class_counts,
        }
    scores = pd.Series(np.nan, index=features.index, dtype=float)
    all_indices = set(features.index.tolist())
    estimator_kind = ""
    for test_idx in folds:
        test_set = set(test_idx)
        train_idx = sorted(all_indices - test_set)
        if target.iloc[train_idx].nunique() < 2:
            return {
                "status": "skipped",
                "reason": "A cross-validation training fold had only one class.",
                "class_counts": class_counts,
            }
        estimator, estimator_kind = make_photo_quality_estimator(feature_columns)
        try:
            estimator.fit(features.iloc[train_idx], target.iloc[train_idx])
            scores.iloc[test_idx] = estimator.predict_proba(features.iloc[test_idx])[:, 1]
        except Exception as exc:
            return {
                "status": "skipped",
                "reason": f"Cross-validation failed: {type(exc).__name__}",
                "class_counts": class_counts,
            }
    metrics = binary_classification_metrics(target, scores)
    metrics.update(
        {
            "status": "cross_validated",
            "method": "deterministic_stratified_kfold",
            "folds": int(len(folds)),
            "estimator_kind": estimator_kind,
            "class_counts": class_counts,
        }
    )
    return metrics


def train_photo_quality_model(
    labels: pd.DataFrame,
    *,
    method_config: MethodConfig | None = None,
    label_source: str = LABEL_SOURCE_MANUAL,
) -> tuple[Any | None, dict[str, Any]]:
    normalized_method_config = (
        MethodConfig(
            methods=normalize_methods(method_config.methods),
            pyiqa_model=method_config.pyiqa_model,
            aesthetic_model=method_config.aesthetic_model,
            fashionclip_model=method_config.fashionclip_model,
            fashionclip_local_files_only=method_config.fashionclip_local_files_only,
            dino_model=method_config.dino_model,
            max_images_per_item=method_config.max_images_per_item,
            device=method_config.device,
        )
        if method_config is not None
        else None
    )
    feature_columns = resolve_feature_columns(normalized_method_config)
    selected_labels = training_label_series(labels, label_source=label_source)
    frame = prepare_labeled_frame(labels, method_config=normalized_method_config, label_source=label_source)
    readiness = label_readiness_summary(labels, label_source=label_source)
    metadata = {
        "model_version": MODEL_VERSION,
        "created_at": utc_now_iso(),
        "features": feature_columns,
        "target": "photo_quality_bad_vs_good",
        "label_source": label_source,
        "label_readiness": readiness,
        "label_counts": selected_labels.value_counts(dropna=False).to_dict(),
        "manual_label_counts": labels.get("manual_label", pd.Series(dtype=str)).reindex(labels.index, fill_value="").map(normalize_label).value_counts(dropna=False).to_dict(),
        "training_rows": int(len(frame)),
        "status": "not_trained",
    }
    target_counts = frame["PhotoQualityTarget"].value_counts().to_dict() if "PhotoQualityTarget" in frame.columns else {}
    if "FashionClipPseudoLabel" in labels.columns:
        metadata["fashionclip_pseudo_label_counts"] = labels["FashionClipPseudoLabel"].reindex(labels.index, fill_value="").map(normalize_label).value_counts(dropna=False).to_dict()
    if normalized_method_config is not None:
        metadata["quality_method_config"] = {
            "methods": list(normalized_method_config.methods),
            "pyiqa_model": normalized_method_config.pyiqa_model,
            "aesthetic_model": normalized_method_config.aesthetic_model,
            "fashionclip_model": normalized_method_config.fashionclip_model,
            "fashionclip_local_files_only": bool(normalized_method_config.fashionclip_local_files_only),
            "dino_model": normalized_method_config.dino_model,
            "max_images_per_item": int(normalized_method_config.max_images_per_item),
            "device": normalized_method_config.device,
        }
    if readiness["status"] != "ready" or len(frame) < 4 or frame["PhotoQualityTarget"].nunique() < 2 or min(target_counts.values(), default=0) < 2:
        metadata["reason"] = readiness["message"]
        return None, metadata
    x = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = frame["PhotoQualityTarget"].astype(int)
    model, model_kind = make_photo_quality_estimator(feature_columns)
    metadata["model_kind"] = model_kind
    model.fit(x, y)
    train_scores = model.predict_proba(x)[:, 1]
    training_metrics = binary_classification_metrics(y, pd.Series(train_scores, index=x.index))
    evaluation = cross_validate_photo_quality_model(x, y, feature_columns=feature_columns)
    metadata.update(
        {
            "status": "trained",
            "positive_rows": int(y.sum()),
            "negative_rows": int((1 - y).sum()),
            "training_mean_bad_probability": float(np.mean(train_scores)),
            "training_metrics": training_metrics,
            "evaluation": evaluation,
        }
    )
    return model, metadata


def save_model(model: Any, metadata: dict[str, Any], *, model_dir: Path = MODELS_DIR) -> tuple[Path | None, Path]:
    model_dir = assert_photo_path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now('UTC').strftime("%Y%m%d_%H%M%S")
    metadata_path = model_dir / f"{MODEL_VERSION}_{timestamp}_metadata.json"
    model_path = model_dir / f"{MODEL_VERSION}_{timestamp}.pkl"
    if model is not None:
        tmp = model_path.with_suffix(model_path.suffix + ".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(model, handle)
        tmp.replace(model_path)
        metadata["artifact_path"] = str(model_path)
        latest_model = model_dir / f"{MODEL_VERSION}_latest.pkl"
        latest_tmp = latest_model.with_suffix(latest_model.suffix + ".tmp")
        with latest_tmp.open("wb") as handle:
            pickle.dump(model, handle)
        latest_tmp.replace(latest_model)
    else:
        model_path = None
        metadata["artifact_path"] = ""
    write_json(metadata_path, metadata)
    write_json(model_dir / f"{MODEL_VERSION}_latest_metadata.json", metadata)
    return model_path, metadata_path


def load_latest_model(model_dir: Path = MODELS_DIR) -> tuple[Any | None, dict[str, Any]]:
    metadata_path = model_dir / f"{MODEL_VERSION}_latest_metadata.json"
    if not metadata_path.exists():
        return None, {"status": "missing", "model_version": "heuristic_v0"}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifact = metadata.get("artifact_path") or str(model_dir / f"{MODEL_VERSION}_latest.pkl")
    path = Path(artifact)
    if not path.exists():
        latest = model_dir / f"{MODEL_VERSION}_latest.pkl"
        path = latest if latest.exists() else path
    if not path.exists():
        return None, metadata
    with path.open("rb") as handle:
        model = pickle.load(handle)
    return model, metadata


def method_config_from_metadata(metadata: dict[str, Any] | None) -> MethodConfig | None:
    payload = dict(metadata.get("quality_method_config") or {}) if metadata else {}
    methods = payload.get("methods")
    if not methods:
        return None
    return MethodConfig(
        methods=normalize_methods(methods),
        pyiqa_model=str(payload.get("pyiqa_model", MethodConfig.pyiqa_model)),
        aesthetic_model=str(payload.get("aesthetic_model", MethodConfig.aesthetic_model)),
        fashionclip_model=str(payload.get("fashionclip_model", MethodConfig.fashionclip_model)),
        fashionclip_local_files_only=bool(payload.get("fashionclip_local_files_only", True)),
        dino_model=str(payload.get("dino_model", MethodConfig.dino_model)),
        max_images_per_item=int(payload.get("max_images_per_item", 1)),
        device=str(payload.get("device", "auto")),
    )


def score_bad_photo_probability(
    frame: pd.DataFrame,
    model: Any | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[pd.Series, str]:
    method_config = method_config_from_metadata(metadata)
    feature_columns = list(metadata.get("features") or MODEL_FEATURES) if metadata else list(MODEL_FEATURES)
    work = ensure_model_features(frame, feature_columns=feature_columns, method_config=method_config)
    if model is None:
        return heuristic_bad_photo_probability(work), "heuristic_v0"
    x = work[feature_columns].apply(pd.to_numeric, errors="coerce")
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)[:, 1]
    else:
        raw = model.decision_function(x)
        scores = 1.0 / (1.0 + np.exp(-raw))
    return pd.Series(np.clip(scores, 0.0, 1.0), index=frame.index), MODEL_VERSION
