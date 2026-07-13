#!/usr/bin/env python3
"""Explain the live-trained giant_basic_visual sender model with SHAP."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.current.giant_basic_visual import apply_live_trained_to_live_collector as live  # noqa: E402
from experiments.current.giant_basic_visual.paths import (  # noqa: E402
    EXPERIMENT_ROOT,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_json,
    write_manifest,
)

SCORE_COL = live.SCORE_COL
THRESHOLD_COL = live.THRESHOLD_COL
PASS_COL = live.PASS_COL
IDENTITY_COLUMNS = ["SearchName", "item_id", "Dataid", "Link", "Title", "Brand", "Size", "Price", "Likes"]


def latest_scored_path(live_scoring_root: Path | None = None) -> Path:
    root = live_scoring_root or (EXPERIMENT_ROOT / "live_scoring")
    matches = sorted(root.glob("live_scoring_*/live_scored_items.csv"), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No live_scored_items.csv files found under {root}")
    return matches[-1]


def choose_rows(frame: pd.DataFrame, *, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.reset_index(drop=True).copy()
    return frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def prepare_feature_frame(frame: pd.DataFrame, model: Any) -> pd.DataFrame:
    prepared = live.prepare_feature_frame(frame)
    raw = prepared[live.VISUAL_FEATURES].astype(float)
    return live.align_feature_frame_to_model(raw, model)


def transformed_feature_frame(model: Any, feature_frame: pd.DataFrame) -> pd.DataFrame:
    steps = getattr(model, "named_steps", {})
    imputer = steps.get("imputer") if isinstance(steps, dict) else None
    transformed = imputer.transform(feature_frame) if imputer is not None else feature_frame.to_numpy()
    return pd.DataFrame(transformed, columns=feature_frame.columns, index=feature_frame.index)


def shap_values(model: Any, background: pd.DataFrame, sample: pd.DataFrame) -> tuple[np.ndarray, float | None, str]:
    try:
        import shap
    except Exception as exc:
        raise RuntimeError("The shap package is required for live visual model explanations.") from exc

    estimator = getattr(model, "named_steps", {}).get("model", model)
    explainer = shap.TreeExplainer(estimator, data=background.to_numpy())
    raw_values = explainer.shap_values(sample.to_numpy(), check_additivity=False)
    if isinstance(raw_values, list):
        values = raw_values[1] if len(raw_values) > 1 else raw_values[0]
    elif isinstance(raw_values, np.ndarray) and raw_values.ndim == 3:
        values = raw_values[:, :, 1] if raw_values.shape[-1] > 1 else raw_values[:, :, 0]
    else:
        values = raw_values
    expected = explainer.expected_value
    if isinstance(expected, (list, tuple, np.ndarray)):
        expected_value = float(np.asarray(expected).reshape(-1)[-1])
    else:
        expected_value = float(expected) if expected is not None else None
    return np.asarray(values, dtype=float), expected_value, "tree_shap_check_additivity_false"


def feature_importance_frame(values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for idx, feature in enumerate(feature_names):
        col_values = values[:, idx]
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(np.nanmean(np.abs(col_values))),
                "mean_shap": float(np.nanmean(col_values)),
                "positive_rate": float(np.nanmean(col_values > 0)),
            }
        )
    out = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False, kind="stable").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def likes_summary(importance: pd.DataFrame) -> dict[str, Any]:
    match = importance[importance["feature"].astype(str).eq("Likes")]
    if match.empty:
        return {"feature": "Likes", "present": False}
    row = match.iloc[0]
    return {
        "feature": "Likes",
        "present": True,
        "rank": int(row["rank"]),
        "mean_abs_shap": float(row["mean_abs_shap"]),
        "mean_shap": float(row["mean_shap"]),
        "positive_rate": float(row["positive_rate"]),
    }


def item_explanations_frame(
    sample: pd.DataFrame,
    values: np.ndarray,
    feature_names: list[str],
    *,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    likes_idx = feature_index.get("Likes")
    for row_pos, item in sample.reset_index(drop=True).iterrows():
        row_values = values[row_pos]
        ordered = np.argsort(-np.abs(row_values))[:top_n]
        positive = []
        negative = []
        for feature_idx in ordered:
            payload = {"feature": feature_names[int(feature_idx)], "shap_value": float(row_values[int(feature_idx)])}
            if payload["shap_value"] >= 0:
                positive.append(payload)
            else:
                negative.append(payload)
        identity = {col: item.get(col) for col in IDENTITY_COLUMNS if col in sample.columns}
        rows.append(
            {
                **identity,
                "score": item.get(SCORE_COL, np.nan),
                "threshold": item.get(THRESHOLD_COL, np.nan),
                "passed": item.get(PASS_COL, np.nan),
                "likes_shap_value": float(row_values[likes_idx]) if likes_idx is not None else np.nan,
                "top_positive_drivers": json.dumps(positive[:top_n], ensure_ascii=True),
                "top_negative_drivers": json.dumps(negative[:top_n], ensure_ascii=True),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    *,
    scored_path: Path,
    model_path: Path,
    explainer_method: str,
    expected_value: float | None,
    rows_loaded: int,
    rows_explained: int,
    telegram_eligible_rows: int,
    likes: dict[str, Any],
    importance: pd.DataFrame,
) -> None:
    lines = [
        "# Giant Basic Visual SHAP / Likes Report",
        "",
        f"Scored data: `{scored_path}`",
        f"Model: `{model_path}`",
        f"Explainer: `{explainer_method}`",
        f"Expected value: `{expected_value}`",
        "",
        "## Scope",
        "",
        f"- Rows loaded: `{rows_loaded}`",
        f"- Rows explained: `{rows_explained}`",
        f"- Telegram-eligible rows in scored data: `{telegram_eligible_rows}`",
        "",
        "## Likes",
        "",
    ]
    if likes.get("present"):
        lines.extend(
            [
                f"- Rank by mean absolute SHAP: `{likes['rank']}`",
                f"- Mean absolute SHAP: `{likes['mean_abs_shap']:.6f}`",
                f"- Mean SHAP: `{likes['mean_shap']:.6f}`",
                f"- Positive-rate: `{likes['positive_rate']:.3f}`",
            ]
        )
    else:
        lines.append("- `Likes` was not present in the explained feature matrix.")
    lines.extend(
        [
            "",
            "## Top Features",
            "",
            "| rank | feature | mean abs SHAP | mean SHAP | positive-rate |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in importance.head(25).iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['feature']} | {float(row['mean_abs_shap']):.6f} | "
            f"{float(row['mean_shap']):.6f} | {float(row['positive_rate']):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_experiment_dirs()
    scored_path = Path(args.scored_path) if args.scored_path else latest_scored_path()
    frame = pd.read_csv(scored_path, low_memory=False)
    if frame.empty:
        raise RuntimeError(f"No rows to explain in {scored_path}")

    model_path = Path(args.model_path) if args.model_path else live.MODEL_PATH
    model = joblib.load(model_path)
    background = choose_rows(frame, max_rows=max(1, int(args.max_background_rows)), seed=int(args.seed))
    sample = choose_rows(frame, max_rows=max(1, int(args.max_explain_rows)), seed=int(args.seed) + 7)

    background_features = transformed_feature_frame(model, prepare_feature_frame(background, model))
    sample_features = transformed_feature_frame(model, prepare_feature_frame(sample, model))
    values, expected_value, method = shap_values(model, background_features, sample_features)
    feature_names = list(sample_features.columns)

    importance = feature_importance_frame(values, feature_names)
    likes = likes_summary(importance)
    item_explanations = item_explanations_frame(sample, values, feature_names, top_n=max(1, int(args.top_item_features)))

    out_dir = Path(args.out_dir) if args.out_dir else EXPERIMENT_ROOT / "shap_likes" / run_id("live_shap_likes")
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    importance_path = out_dir / "feature_importance.csv"
    item_path = out_dir / "item_explanations.csv"
    likes_path = out_dir / "likes_summary.json"
    report_path = out_dir / "shap_report.md"
    manifest_path = out_dir / "manifest.json"

    importance.to_csv(importance_path, index=False)
    item_explanations.to_csv(item_path, index=False)
    price = pd.to_numeric(frame.get("Price"), errors="coerce")
    passed = frame.get(PASS_COL, pd.Series(False, index=frame.index)).astype(str).str.lower().isin({"true", "1", "1.0"})
    telegram_eligible_rows = int((passed & price.gt(live.TELEGRAM_MIN_PRICE_EUR)).sum())
    write_json(likes_path, likes)
    write_report(
        report_path,
        scored_path=scored_path,
        model_path=model_path,
        explainer_method=method,
        expected_value=expected_value,
        rows_loaded=int(len(frame)),
        rows_explained=int(len(sample)),
        telegram_eligible_rows=telegram_eligible_rows,
        likes=likes,
        importance=importance,
    )
    manifest = {
        "scored_path": str(scored_path),
        "model_path": str(model_path),
        "explainer_method": method,
        "expected_value": expected_value,
        "rows_loaded": int(len(frame)),
        "rows_explained": int(len(sample)),
        "telegram_eligible_rows": telegram_eligible_rows,
        "outputs": {
            "feature_importance": str(importance_path),
            "item_explanations": str(item_path),
            "likes_summary": str(likes_path),
            "report": str(report_path),
        },
    }
    write_manifest(manifest_path, command="giant_basic_visual.live_shap_likes_report", extra=manifest)
    print(json.dumps({"out_dir": str(out_dir), **manifest}, indent=2), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP/Likes report for the live giant_basic_visual sender.")
    parser.add_argument("--scored-path", default=None, help="Optional live_scored_items.csv. Defaults to latest live scoring output.")
    parser.add_argument("--model-path", default=None, help="Optional model artifact. Defaults to production live-trained model.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-background-rows", type=int, default=1000)
    parser.add_argument("--max-explain-rows", type=int, default=1000)
    parser.add_argument("--top-item-features", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
