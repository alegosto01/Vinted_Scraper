from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.deal_finder.dataset import build_datasets, parse_timestamp_series, safe_feature_columns
from experiments.deal_finder.paths import (
    MODELS_DIR,
    OFFLINE_RUNS_DIR,
    assert_experiment_path,
    ensure_experiment_dirs,
    read_json,
    run_id,
    write_json,
    write_manifest,
)


NUMERIC_CANDIDATES = [
    "Price",
    "Likes",
    "Page",
]
TEXT_CANDIDATES = ["Title", "Brand", "Size", "SearchName"]
FEATURE_POLICY = "snapshot_raw_v1"
TARGET_COL = "offline_sold_label"
ELIGIBLE_COL = "offline_label_eligible"
TOKEN_RE = re.compile(r"[A-Za-z0-9]{2,}")


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
    except Exception:
        return False
    return True


def to_numeric_frame(values):
    if isinstance(values, pd.DataFrame):
        return values.apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame(values).apply(pd.to_numeric, errors="coerce")


def text_values_to_str(values):
    if isinstance(values, pd.DataFrame):
        series = values.iloc[:, 0]
    else:
        series = pd.Series(values)
    return series.fillna("").astype(str)


def select_model_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    safe = set(safe_feature_columns(df))
    numeric = [c for c in NUMERIC_CANDIDATES if c in df.columns and c in safe]
    text = []
    for col in TEXT_CANDIDATES:
        if col not in df.columns or col not in safe:
            continue
        values = df[col].fillna("").astype(str)
        if values.map(lambda value: bool(TOKEN_RE.search(value))).any():
            text.append(col)
    return numeric, text


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "observation_ts" in out.columns:
        out["_split_ts"] = pd.to_datetime(out["observation_ts"], errors="coerce", utc=True)
    elif "SearchDate" in out.columns:
        out["_split_ts"] = parse_timestamp_series(out["SearchDate"])
    else:
        out["_split_ts"] = pd.NaT
    if ELIGIBLE_COL not in out.columns:
        return out.iloc[0:0].copy()
    eligible = out[ELIGIBLE_COL]
    if str(eligible.dtype).lower() not in {"bool", "boolean"}:
        eligible = eligible.astype(str).str.lower().isin(["true", "1", "yes"])
    out = out[eligible.fillna(False)].copy()
    out = out.dropna(subset=["_split_ts"])
    if "item_id" in out.columns:
        out = out.sort_values("_split_ts", kind="stable").drop_duplicates("item_id", keep="first")
    return out.reset_index(drop=True)


def time_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> SplitFrames:
    ordered = df.sort_values("_split_ts", kind="stable").reset_index(drop=True)
    n = len(ordered)
    train_end = int(np.floor(n * train_frac))
    val_end = int(np.floor(n * (train_frac + val_frac)))
    return SplitFrames(
        train=ordered.iloc[:train_end].copy(),
        validation=ordered.iloc[train_end:val_end].copy(),
        test=ordered.iloc[val_end:].copy(),
    )


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"k": k, "count": 0, "precision": np.nan, "positive_count": 0}
    order = np.argsort(-scores)
    top = order[: min(k, len(order))]
    positives = int(np.asarray(y_true)[top].sum())
    count = int(len(top))
    return {"k": k, "count": count, "precision": float(positives / count) if count else np.nan, "positive_count": positives}


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    mask = scores >= threshold
    count = int(mask.sum())
    positives = int(np.asarray(y_true)[mask].sum()) if count else 0
    return {
        "threshold": float(threshold),
        "count": count,
        "precision": float(positives / count) if count else np.nan,
        "positive_count": positives,
    }


def required_material_precision(base_rate: float) -> float:
    if not np.isfinite(base_rate):
        base_rate = 0.0
    base_rate = min(max(float(base_rate), 0.0), 1.0)
    return min(0.95, max(0.60, base_rate + 0.20, base_rate * 1.25))


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, *, min_precision: float = 0.60, min_count: int = 20) -> dict[str, Any]:
    if len(scores) == 0:
        return {"threshold": 1.0, "precision": np.nan, "count": 0, "positive_count": 0}
    candidates = sorted(set(np.round(np.asarray(scores, dtype=float), 4).tolist() + [0.5, 0.6, 0.7, 0.8, 0.9]), reverse=True)
    valid = []
    for threshold in candidates:
        row = threshold_metrics(y_true, scores, threshold)
        if row["count"] >= min_count and pd.notna(row["precision"]) and row["precision"] >= min_precision:
            valid.append(row)
    if valid:
        valid.sort(key=lambda r: (r["precision"], r["positive_count"], r["count"], r["threshold"]), reverse=True)
        return valid[0]
    rows = [threshold_metrics(y_true, scores, threshold) for threshold in candidates]
    rows.sort(key=lambda r: (0.0 if pd.isna(r["precision"]) else r["precision"], r["positive_count"], r["count"]), reverse=True)
    return rows[0] if rows else {"threshold": 1.0, "precision": np.nan, "count": 0, "positive_count": 0}


def make_pipeline(numeric_features: list[str], text_features: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
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
    for col in text_features:
        text_pipe = Pipeline(
            steps=[
                ("to_text", FunctionTransformer(text_values_to_str, validate=False)),
                ("tfidf", TfidfVectorizer(max_features=300, ngram_range=(1, 2), min_df=1)),
            ]
        )
        transformers.append((f"text_{col}", text_pipe, col))
    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)
    return Pipeline(
        steps=[
            ("features", pre),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
        ]
    )


def score_with_model(model, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(frame)[:, 1]
    return model.decision_function(frame)


def save_pickle(path: Path, obj: Any) -> None:
    path = assert_experiment_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(obj, handle)
    tmp.replace(path)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def evaluate_split(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y = frame[TARGET_COL].astype(int).to_numpy()
    out = {
        "rows": int(len(frame)),
        "positive_rows": int(y.sum()),
        "base_rate": float(y.mean()) if len(y) else np.nan,
        "threshold": threshold_metrics(y, scores, threshold),
        "precision_at": {f"p@{k}": precision_at_k(y, scores, k) for k in (10, 25, 50)},
    }
    return out


def promoted(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    val = metrics.get("validation", {})
    test = metrics.get("test", {})
    val_thr = val.get("threshold", {})
    test_thr = test.get("threshold", {})
    val_p10 = val.get("precision_at", {}).get("p@10", {})
    test_p10 = test.get("precision_at", {}).get("p@10", {})
    base = float(test.get("base_rate", 0.0) or 0.0)
    required_precision = required_material_precision(base)

    checks = [
        (val_thr.get("precision", 0.0) >= 0.60, "validation threshold precision >= 60%"),
        (test_thr.get("precision", 0.0) >= 0.60, "test threshold precision >= 60%"),
        (val_p10.get("precision", 0.0) >= 0.60, "validation precision@10 >= 60%"),
        (test_p10.get("precision", 0.0) >= 0.60, "test precision@10 >= 60%"),
        (val_thr.get("count", 0) >= 20, "at least 20 validation recommendations above threshold"),
        (test_thr.get("count", 0) >= 10, "at least 10 test recommendations above threshold"),
        (
            test_thr.get("precision", 0.0) >= required_precision,
            f"threshold precision materially beats base rate (required >= {required_precision:.2f})",
        ),
    ]
    for ok, reason in checks:
        if not ok:
            reasons.append(reason)
    return not reasons, reasons


def train_one_dataset(dataset_path: Path, run_dir: Path, *, model_prefix: str) -> dict[str, Any]:
    df = pd.read_csv(dataset_path)
    model_frame = prepare_model_frame(df)
    search_name = dataset_path.stem
    if len(model_frame) < 50 or model_frame[TARGET_COL].nunique() < 2:
        return {
            "search_name": search_name,
            "status": "skipped",
            "reason": "not enough eligible rows or only one label class",
            "eligible_rows": int(len(model_frame)),
            "positive_rows": int(model_frame[TARGET_COL].sum()) if TARGET_COL in model_frame else 0,
            "qualified_for_paper_trading": False,
        }
    if not sklearn_available():
        return {
            "search_name": search_name,
            "status": "skipped",
            "reason": "scikit-learn is not available",
            "eligible_rows": int(len(model_frame)),
            "qualified_for_paper_trading": False,
        }

    splits = time_split(model_frame)
    if min(len(splits.train), len(splits.validation), len(splits.test)) == 0:
        return {
            "search_name": search_name,
            "status": "skipped",
            "reason": "time split produced an empty split",
            "eligible_rows": int(len(model_frame)),
            "qualified_for_paper_trading": False,
        }
    if splits.train[TARGET_COL].nunique() < 2:
        return {
            "search_name": search_name,
            "status": "skipped",
            "reason": "training split has only one label class",
            "eligible_rows": int(len(model_frame)),
            "qualified_for_paper_trading": False,
        }

    numeric, text = select_model_features(splits.train)
    if not numeric and not text:
        return {
            "search_name": search_name,
            "status": "skipped",
            "reason": "no usable training-time features",
            "eligible_rows": int(len(model_frame)),
            "qualified_for_paper_trading": False,
        }
    model = make_pipeline(numeric, text)
    y_train = splits.train[TARGET_COL].astype(int)
    model.fit(splits.train, y_train)
    val_scores = score_with_model(model, splits.validation)
    test_scores = score_with_model(model, splits.test)
    threshold = choose_threshold(splits.validation[TARGET_COL].astype(int).to_numpy(), val_scores)["threshold"]

    artifact_path = MODELS_DIR / f"{model_prefix}_{search_name}_logistic.pkl"
    save_pickle(artifact_path, model)
    metadata = {
        "search_name": search_name,
        "model_kind": "logistic_v1",
        "artifact_path": str(artifact_path),
        "numeric_features": numeric,
        "text_features": text,
        "feature_policy": FEATURE_POLICY,
        "threshold": float(threshold),
        "label": TARGET_COL,
        "live_success_label": "sold_within_2d",
    }
    write_json(MODELS_DIR / f"{model_prefix}_{search_name}_metadata.json", metadata)

    metrics = {
        "search_name": search_name,
        "status": "trained",
        "model_metadata": metadata,
        "split_rows": {
            "train": int(len(splits.train)),
            "validation": int(len(splits.validation)),
            "test": int(len(splits.test)),
        },
        "validation": evaluate_split(splits.validation, val_scores, float(threshold)),
        "test": evaluate_split(splits.test, test_scores, float(threshold)),
    }
    ok, reasons = promoted(metrics)
    metrics["qualified_for_paper_trading"] = bool(ok)
    metrics["promotion_failures"] = reasons
    return metrics


def train_offline(
    *,
    searches: list[str] | None = None,
    limit_rows: int | None = None,
    out_dir: Path | None = None,
    auto_paper_trading: bool = True,
) -> dict[str, Any]:
    ensure_experiment_dirs()
    if out_dir is None:
        out_dir = OFFLINE_RUNS_DIR / run_id("offline")
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_results = build_datasets(searches=searches, out_dir=out_dir, limit_rows=limit_rows)
    model_prefix = out_dir.name
    metrics = []
    for dataset in dataset_results:
        metrics.append(train_one_dataset(dataset.path, out_dir, model_prefix=model_prefix))

    qualified = [
        {
            "search_name": row["search_name"],
            "model_metadata": row.get("model_metadata", {}),
        }
        for row in metrics
        if row.get("qualified_for_paper_trading")
    ]
    summary = {
        "run_dir": str(out_dir),
        "dataset_count": len(dataset_results),
        "model_count": len([m for m in metrics if m.get("status") == "trained"]),
        "qualified_count": len(qualified),
        "auto_paper_trading": bool(auto_paper_trading),
        "metrics": metrics,
        "qualified_searches": qualified,
    }
    write_json(out_dir / "metrics_summary.json", summary)
    pd.DataFrame(
        [
            {
                "search_name": row.get("search_name"),
                "status": row.get("status"),
                "eligible_rows": row.get("eligible_rows"),
                "qualified_for_paper_trading": row.get("qualified_for_paper_trading"),
                "threshold": row.get("model_metadata", {}).get("threshold"),
                "validation_precision": row.get("validation", {}).get("threshold", {}).get("precision"),
                "validation_count": row.get("validation", {}).get("threshold", {}).get("count"),
                "test_precision": row.get("test", {}).get("threshold", {}).get("precision"),
                "test_count": row.get("test", {}).get("threshold", {}).get("count"),
                "test_precision_at_10": row.get("test", {}).get("precision_at", {}).get("p@10", {}).get("precision"),
                "reason": row.get("reason") or "; ".join(row.get("promotion_failures", [])),
            }
            for row in metrics
        ]
    ).to_csv(out_dir / "metrics_summary.csv", index=False)
    write_json(out_dir / "qualified_searches.json", {"qualified_searches": qualified})
    if qualified and auto_paper_trading:
        from experiments.deal_finder.paper_trading import collect_snapshot

        summary["paper_trading_start"] = collect_snapshot(
            max_searches=3,
            offline_run=out_dir,
            dry_run=False,
        )
    write_manifest(out_dir / "manifest.json", command="train_offline", extra=summary)
    return summary


def latest_offline_run() -> Path | None:
    if not OFFLINE_RUNS_DIR.exists():
        return None
    runs = [p for p in OFFLINE_RUNS_DIR.iterdir() if p.is_dir() and (p / "metrics_summary.json").exists()]
    return max(runs, key=lambda p: (p / "metrics_summary.json").stat().st_mtime) if runs else None


def load_qualified_searches(run_dir: Path | None = None) -> list[dict[str, Any]]:
    run_dir = run_dir or latest_offline_run()
    if run_dir is None:
        return []
    data = read_json(run_dir / "qualified_searches.json", {"qualified_searches": []})
    return data.get("qualified_searches", [])


def load_model_artifact(metadata: dict[str, Any]):
    return load_pickle(Path(metadata["artifact_path"]))


def score_candidates(candidates: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    model = load_model_artifact(metadata)
    scored = candidates.copy()
    for col in metadata.get("numeric_features", []):
        if col not in scored.columns:
            scored[col] = np.nan
    for col in metadata.get("text_features", []):
        if col not in scored.columns:
            scored[col] = ""
    scored["model_probability"] = score_with_model(model, scored)
    threshold = float(metadata.get("threshold", 1.0))
    scored["model_threshold"] = threshold
    scored["above_threshold"] = scored["model_probability"] >= threshold
    scored["model_version"] = Path(metadata.get("artifact_path", "model")).stem
    scored = scored.sort_values("model_probability", ascending=False, kind="stable").reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)
    return scored


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train and evaluate deal-finder models offline.")
    ap.add_argument("--all-searches", action="store_true")
    ap.add_argument("--search", action="append", default=[])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit-rows", type=int, default=None)
    ap.add_argument("--no-auto-paper-trading", action="store_true", help="Do not auto-start paper-trading for qualified searches.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all_searches and not args.search:
        raise SystemExit("Use --all-searches or at least one --search.")
    summary = train_offline(
        searches=None if args.all_searches else args.search,
        limit_rows=args.limit_rows,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        auto_paper_trading=not args.no_auto_paper_trading,
    )
    print(json.dumps({k: summary[k] for k in ("run_dir", "model_count", "qualified_count")}, indent=2))


if __name__ == "__main__":
    main()
