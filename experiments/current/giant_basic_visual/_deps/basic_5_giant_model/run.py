#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
for _path in (ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.current.giant_basic_visual._deps.basic_5_giant_model.paths import (  # noqa: E402
    EXPERIMENT_ROOT,
    MODELS_DIR,
    OFFLINE_RUNS_DIR,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_json,
    write_manifest,
)
from experiments.current.giant_basic_visual._deps.deal_finder import model_sweep as base_sweep  # noqa: E402
from experiments.current.giant_basic_visual._deps.deal_finder.modeling import (  # noqa: E402
    ELIGIBLE_COL,
    TARGET_COL,
    choose_threshold,
    score_with_model,
)
from experiments.current.giant_basic_visual._deps.full_scrape_model.compare_feature_modalities import (  # noqa: E402
    BASIC_NUMERIC,
    BASIC_TEXT,
    EXPERIMENTAL_BASIC_APPROACHES,
)
from experiments.current.giant_basic_visual._deps.full_scrape_model.dataset import TASK_SOLD_STATUS, build_search_dataset  # noqa: E402


DEFAULT_SEARCHES = (
    "griffati_donna_all",
    "griffati_uomo_all",
    "gucci",
    "nike",
    "prada",
    "ps4",
    "telefoni",
    "hobby_collezionismo",
    "donna_accessori_gioielli",
)

BASIC5_APPROACH_NAMES = (
    "logistic_v1_baseline",
    "logistic_snapshot_v2",
    "sgd_text_numeric_v1",
    "linear_svm_calibrated_v1",
    "numeric_tree_v1",
    "rules_price_v1",
    "random_forest_basic_v1",
    "hist_gradient_basic_numeric_v1",
    "xgboost_basic_v1",
)

ID_COLUMNS = ["SearchName", "item_id", "Dataid", "Title", "Brand", "Size", "Price", "Likes", "Link"]


def available_basic5_specs() -> tuple[base_sweep.ApproachSpec, ...]:
    by_name = {spec.name: spec for spec in (*base_sweep.APPROACHES, *EXPERIMENTAL_BASIC_APPROACHES)}
    return tuple(by_name[name] for name in BASIC5_APPROACH_NAMES)


def sanitize_search_name(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return text.strip("_") or "unknown"


def search_onehot_column(search: str) -> str:
    return f"search__{sanitize_search_name(search)}"


def add_search_onehot(df: pd.DataFrame, searches: list[str] | tuple[str, ...]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    cols: list[str] = []
    search_values = out["SearchName"].fillna("").astype(str)
    for search in searches:
        col = search_onehot_column(search)
        out[col] = search_values.eq(str(search)).astype(int)
        cols.append(col)
    return out, cols


def load_combined_dataset(searches: list[str] | tuple[str, ...] = DEFAULT_SEARCHES) -> pd.DataFrame:
    frames = []
    for search in searches:
        frame = build_search_dataset(TASK_SOLD_STATUS, search)
        if frame.empty:
            raise ValueError(f"No sold-status rows found for search: {search}")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["SearchName"] = combined["SearchName"].fillna("").astype(str)
    return combined


def prepare_global_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if ELIGIBLE_COL not in out.columns or TARGET_COL not in out.columns:
        return out.iloc[0:0].copy()
    eligible = base_sweep.parse_bool_series(out[ELIGIBLE_COL])
    out = out[eligible].copy()
    out["item_id"] = out.get("item_id", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
    out = out[out["item_id"].str.len() > 0].copy()
    out = out.drop_duplicates(subset=["SearchName", "item_id"], keep="last")
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce").fillna(0).astype(int)
    out = base_sweep.add_engineered_snapshot_features(out)
    return out.reset_index(drop=True)


def stratified_global_split(
    df: pd.DataFrame,
    *,
    seed: int = base_sweep.DEFAULT_SEED,
    train_frac: float = 0.60,
    validation_frac: float = 0.20,
) -> base_sweep.SplitFrames:
    from sklearn.model_selection import train_test_split

    if df.empty:
        return base_sweep.SplitFrames(df.copy(), df.copy(), df.copy())
    strata = df["SearchName"].astype(str) + "__" + df[TARGET_COL].astype(str)
    if strata.value_counts().min() < 2:
        strata = df[TARGET_COL].astype(str)
    stratify = strata if strata.value_counts().min() >= 2 else None
    train, rest = train_test_split(
        df,
        train_size=train_frac,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    rest_strata = rest["SearchName"].astype(str) + "__" + rest[TARGET_COL].astype(str)
    rest_stratify = rest_strata if rest_strata.value_counts().min() >= 2 else None
    val_size = validation_frac / max(1.0 - train_frac, 1e-6)
    validation, test = train_test_split(
        rest,
        train_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=rest_stratify,
    )
    return base_sweep.SplitFrames(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def _has_text_signal(frame: pd.DataFrame, column: str) -> bool:
    return (
        column in frame.columns
        and frame[column].fillna("").astype(str).str.contains(base_sweep.TOKEN_RE, regex=True).any()
    )


def select_basic5_features(
    train_df: pd.DataFrame,
    spec: base_sweep.ApproachSpec,
    search_onehot_cols: list[str],
) -> tuple[list[str], list[str]]:
    if spec.kind == "rules":
        numeric = [col for col in ("price_num", "likes_num", "page_num", "Price", "Likes", "Page") if col in train_df.columns]
        return numeric, []
    numeric = [col for col in BASIC_NUMERIC if col in train_df.columns]
    if spec.use_engineered:
        numeric.extend(col for col in base_sweep.ENGINEERED_NUMERIC if col in train_df.columns and col not in numeric)
    numeric.extend(col for col in search_onehot_cols if col in train_df.columns)
    text = [col for col in BASIC_TEXT if spec.use_text and _has_text_signal(train_df, col)]
    numeric = base_sweep.safe_selected_features(numeric, train_df)
    text = base_sweep.safe_selected_features(text, train_df)
    return numeric, text


def save_pickle(path: Path, obj: Any) -> None:
    path = assert_experiment_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(obj, handle)
    tmp.replace(path)


def calibration_metrics(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    try:
        from sklearn.metrics import brier_score_loss
    except Exception:
        return {"brier": np.nan, "ece": np.nan}
    if len(y_true) == 0:
        return {"brier": np.nan, "ece": np.nan}
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (scores >= lo) & (scores < hi)
        if hi == 1.0:
            mask = (scores >= lo) & (scores <= hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(scores[mask].mean()) - float(y_true[mask].mean()))
    return {"brier": float(brier_score_loss(y_true, scores)), "ece": float(ece)}


def per_search_metrics(frame: pd.DataFrame, scores: np.ndarray, *, threshold: float, approach: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for search, group in frame.groupby("SearchName", sort=True):
        positions = group.index.to_numpy(dtype=int)
        group_scores = np.asarray(scores, dtype=float)[positions]
        labels = group[TARGET_COL].astype(int).to_numpy()
        evaluated = base_sweep.evaluate_scores(group, group_scores, threshold)
        rows.append(
            {
                "search_name": str(search),
                "approach": approach,
                "rows": evaluated["rows"],
                "positive_rows": evaluated["positive_rows"],
                "base_rate": evaluated["base_rate"],
                "threshold": threshold,
                "threshold_count": evaluated["threshold"]["count"],
                "threshold_precision": evaluated["threshold"]["precision"],
                "threshold_recall": evaluated["threshold"]["recall"],
                "precision_at_10": evaluated["precision_at"]["p@10"]["precision"],
                "precision_at_25": evaluated["precision_at"]["p@25"]["precision"],
                "precision_at_50": evaluated["precision_at"]["p@50"]["precision"],
                "roc_auc": evaluated["roc_auc"],
                "pr_auc": evaluated["pr_auc"],
                **calibration_metrics(labels, group_scores),
            }
        )
    return rows


def score_frame(frame: pd.DataFrame, scores_by_approach: dict[str, np.ndarray]) -> pd.DataFrame:
    keep = [col for col in ID_COLUMNS if col in frame.columns]
    out = frame[keep + [TARGET_COL]].copy()
    for approach, scores in scores_by_approach.items():
        out[f"score__{approach}"] = np.asarray(scores, dtype=float)
    return out


def train_one_approach(
    splits: base_sweep.SplitFrames,
    *,
    spec: base_sweep.ApproachSpec,
    search_onehot_cols: list[str],
    seed: int,
    run_name: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    started = time.perf_counter()
    mode_spec = replace(spec, use_image=False, feature_policy=f"{spec.feature_policy}__global_basic_5_with_search")
    numeric, text = select_basic5_features(splits.train, mode_spec, search_onehot_cols)
    if not numeric and not text:
        raise ValueError("no usable basic_5 features")
    if mode_spec.kind == "linear_svm_calibrated" and splits.train[TARGET_COL].value_counts().min() < 3:
        raise ValueError("calibrated SVM needs at least 3 training rows per class")
    fit_frame = base_sweep.bounded_fit_frame(splits.train, mode_spec, seed)
    model = base_sweep.make_model(mode_spec, numeric, text)
    model.fit(fit_frame, fit_frame[TARGET_COL].astype(int))

    validation_scores = np.clip(np.asarray(score_with_model(model, splits.validation), dtype=float), 0.0, 1.0)
    test_scores = np.clip(np.asarray(score_with_model(model, splits.test), dtype=float), 0.0, 1.0)
    threshold_row = choose_threshold(
        splits.validation[TARGET_COL].astype(int).to_numpy(),
        validation_scores,
        min_precision=base_sweep.PROMOTION_PRECISION,
        min_count=base_sweep.VALIDATION_MIN_COUNT,
    )
    threshold = float(threshold_row["threshold"])

    artifact_path = MODELS_DIR / f"{run_name}_global_basic_5_{mode_spec.name}_seed{seed}.pkl"
    metadata_path = MODELS_DIR / f"{run_name}_global_basic_5_{mode_spec.name}_seed{seed}_metadata.json"
    save_pickle(artifact_path, model)
    metadata = {
        "experiment_family": "basic_5_giant_model",
        "feature_mode": "global_basic_5",
        "approach": mode_spec.name,
        "model_kind": mode_spec.kind,
        "artifact_path": str(artifact_path),
        "numeric_features": numeric,
        "text_features": text,
        "search_onehot_features": search_onehot_cols,
        "feature_policy": mode_spec.feature_policy,
        "threshold": threshold,
        "label": TARGET_COL,
        "seed": int(seed),
        "split": "stratified_search_label_random_60_20_20",
        "train_rows": int(len(splits.train)),
        "fit_train_rows": int(len(fit_frame)),
        "validation_rows": int(len(splits.validation)),
        "test_rows": int(len(splits.test)),
    }
    write_json(metadata_path, base_sweep.to_builtin(metadata))

    validation_eval = base_sweep.evaluate_scores(splits.validation, validation_scores, threshold)
    test_eval = base_sweep.evaluate_scores(splits.test, test_scores, threshold)
    promotion_ok, promotion_failures = base_sweep.promoted_80(
        {"validation": validation_eval, "test": test_eval},
        live_ready=bool(mode_spec.live_ready),
        feature_leakage=base_sweep.has_leakage_features(numeric + text),
    )
    labels = splits.test[TARGET_COL].astype(int).to_numpy()
    row = {
        "approach": mode_spec.name,
        "status": "trained",
        "seed": int(seed),
        "artifact_path": str(artifact_path),
        "metadata_path": str(metadata_path),
        "model_kind": mode_spec.kind,
        "feature_policy": mode_spec.feature_policy,
        "numeric_features": numeric,
        "text_features": text,
        "threshold": threshold,
        "train_rows": int(len(splits.train)),
        "fit_train_rows": int(len(fit_frame)),
        "validation_rows": int(len(splits.validation)),
        "test_rows": int(len(splits.test)),
        "validation_base_rate": validation_eval["base_rate"],
        "validation_threshold_precision": validation_eval["threshold"]["precision"],
        "validation_threshold_count": validation_eval["threshold"]["count"],
        "validation_precision_at_10": validation_eval["precision_at"]["p@10"]["precision"],
        "validation_roc_auc": validation_eval["roc_auc"],
        "validation_pr_auc": validation_eval["pr_auc"],
        "test_base_rate": test_eval["base_rate"],
        "test_threshold_precision": test_eval["threshold"]["precision"],
        "test_threshold_count": test_eval["threshold"]["count"],
        "test_threshold_recall": test_eval["threshold"]["recall"],
        "test_precision_at_10": test_eval["precision_at"]["p@10"]["precision"],
        "test_precision_at_25": test_eval["precision_at"]["p@25"]["precision"],
        "test_precision_at_50": test_eval["precision_at"]["p@50"]["precision"],
        "test_roc_auc": test_eval["roc_auc"],
        "test_pr_auc": test_eval["pr_auc"],
        **calibration_metrics(labels, test_scores),
        "qualified_for_paper_trading": bool(promotion_ok),
        "promotion_failures": "; ".join(promotion_failures),
        "fit_seconds": float(time.perf_counter() - started),
    }
    return row, validation_scores, test_scores


def write_report(
    run_dir: Path,
    *,
    metrics: pd.DataFrame,
    per_search: pd.DataFrame,
    searches: list[str],
) -> Path:
    path = assert_experiment_path(run_dir / "basic_5_giant_model_report.md")

    def fmt(value: object, decimals: int = 3) -> str:
        try:
            if value is None or pd.isna(value):
                return "-"
            return f"{float(value):.{decimals}f}"
        except Exception:
            return str(value)

    trained = metrics[metrics["status"].eq("trained")].copy()
    best_auc = trained.sort_values("test_roc_auc", ascending=False, na_position="last").head(1)
    best_p25 = trained.sort_values("test_precision_at_25", ascending=False, na_position="last").head(1)
    lines = [
        "# Basic 5 Giant Model",
        "",
        f"Run folder: `{run_dir}`",
        "",
        "This experiment trains one global sold/not-sold model across the active searches.",
        "Each approach uses the Basic5 fields (`Price`, `Likes`, `Title`, `Brand`, `Size`) plus one-hot `SearchName` features.",
        "",
        f"Searches: `{', '.join(searches)}`.",
        "",
        "## Best Global Rows",
        "",
        "| selector | approach | AUC | PR AUC | P@25 | threshold precision | threshold count | threshold |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, frame in (("best_auc", best_auc), ("best_p25", best_p25)):
        if frame.empty:
            continue
        row = frame.iloc[0]
        lines.append(
            f"| {label} | {row['approach']} | {fmt(row['test_roc_auc'])} | {fmt(row['test_pr_auc'])} | "
            f"{fmt(row['test_precision_at_25'])} | {fmt(row['test_threshold_precision'])} | "
            f"{int(row['test_threshold_count']) if pd.notna(row['test_threshold_count']) else 0} | {fmt(row['threshold'], 4)} |"
        )
    lines.extend(["", "## All Nine Approaches", ""])
    lines.extend(
        [
            "| approach | AUC | PR AUC | P@10 | P@25 | P@50 | threshold precision | threshold count | Brier | ECE | fit_s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in trained.sort_values("test_roc_auc", ascending=False, na_position="last").iterrows():
        lines.append(
            f"| {row['approach']} | {fmt(row['test_roc_auc'])} | {fmt(row['test_pr_auc'])} | "
            f"{fmt(row['test_precision_at_10'])} | {fmt(row['test_precision_at_25'])} | {fmt(row['test_precision_at_50'])} | "
            f"{fmt(row['test_threshold_precision'])} | {int(row['test_threshold_count']) if pd.notna(row['test_threshold_count']) else 0} | "
            f"{fmt(row['brier'])} | {fmt(row['ece'])} | {fmt(row['fit_seconds'], 1)} |"
        )
    lines.extend(["", "## Per-Search AUC For Top Global AUC Model", ""])
    if not best_auc.empty:
        approach = str(best_auc.iloc[0]["approach"])
        top_search = per_search[per_search["approach"].eq(approach)].sort_values("search_name")
        lines.extend(["| search | rows | base rate | AUC | PR AUC | P@25 | threshold precision | count |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for _, row in top_search.iterrows():
            lines.append(
                f"| {row['search_name']} | {int(row['rows'])} | {fmt(row['base_rate'])} | {fmt(row['roc_auc'])} | "
                f"{fmt(row['pr_auc'])} | {fmt(row['precision_at_25'])} | {fmt(row['threshold_precision'])} | "
                f"{int(row['threshold_count']) if pd.notna(row['threshold_count']) else 0} |"
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `metrics_long.csv`: global held-out metrics for the nine approaches.",
            "- `per_search_metrics.csv`: held-out metrics by approach and search.",
            "- `validation_scores.csv` / `test_scores.csv`: item-level scores for all trained approaches.",
            "- `dataset_summary.csv`: row and label counts by search before splitting.",
            "- `manifest.json`: run settings and git snapshot.",
            "",
            "## Notes",
            "",
            "- This is offline-only; it does not start paper trading.",
            "- Thresholds are selected globally on the validation split using the existing >=80% precision and >=20 selected-item rule.",
            "- Because the threshold is global, per-search threshold precision can vary substantially.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(
    *,
    searches: list[str] | None = None,
    approaches: list[str] | None = None,
    seed: int = base_sweep.DEFAULT_SEED,
    out_dir: Path | None = None,
    limit_rows: int | None = None,
) -> dict[str, Any]:
    ensure_experiment_dirs()
    selected_searches = list(dict.fromkeys(searches or DEFAULT_SEARCHES))
    specs = [spec for spec in available_basic5_specs() if approaches is None or spec.name in set(approaches)]
    if not specs:
        raise ValueError("No approaches selected")
    out_dir = assert_experiment_path(out_dir if out_dir is not None else OFFLINE_RUNS_DIR / run_id("basic_5_giant"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[basic_5_giant] loading searches={selected_searches}", flush=True)
    raw = load_combined_dataset(selected_searches)
    frame = prepare_global_frame(raw)
    if limit_rows is not None and limit_rows > 0:
        parts = []
        for _, group in frame.groupby(["SearchName", TARGET_COL], sort=False):
            parts.append(group.sample(min(limit_rows, len(group)), random_state=seed))
        frame = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    frame, search_onehot_cols = add_search_onehot(frame, selected_searches)

    if frame[TARGET_COL].nunique() < 2:
        raise ValueError("Combined dataset has only one label class")
    dataset_summary = (
        frame.groupby("SearchName", sort=True)[TARGET_COL]
        .agg(rows="size", positive_rows="sum", base_rate="mean")
        .reset_index()
    )
    dataset_summary.to_csv(assert_experiment_path(out_dir / "dataset_summary.csv"), index=False)
    frame.to_csv(assert_experiment_path(out_dir / "combined_dataset.csv"), index=False)

    splits = stratified_global_split(frame, seed=seed)
    split_summary = pd.DataFrame(
        [
            {
                "split": name,
                "rows": len(part),
                "positive_rows": int(part[TARGET_COL].sum()),
                "base_rate": float(part[TARGET_COL].mean()) if len(part) else np.nan,
            }
            for name, part in (("train", splits.train), ("validation", splits.validation), ("test", splits.test))
        ]
    )
    split_summary.to_csv(assert_experiment_path(out_dir / "split_summary.csv"), index=False)
    print(
        f"[basic_5_giant] rows={len(frame)} train={len(splits.train)} validation={len(splits.validation)} test={len(splits.test)}",
        flush=True,
    )

    metric_rows: list[dict[str, Any]] = []
    per_search_rows: list[dict[str, Any]] = []
    validation_scores: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}
    for spec in specs:
        step = time.perf_counter()
        print(f"[basic_5_giant] training {spec.name}", flush=True)
        try:
            row, val_score, test_score = train_one_approach(
                splits,
                spec=spec,
                search_onehot_cols=search_onehot_cols,
                seed=seed,
                run_name=out_dir.name,
            )
            metric_rows.append(row)
            validation_scores[spec.name] = val_score
            test_scores[spec.name] = test_score
            per_search_rows.extend(per_search_metrics(splits.test, test_score, threshold=float(row["threshold"]), approach=spec.name))
            print(
                f"[basic_5_giant] done {spec.name}: AUC={row['test_roc_auc']:.3f} "
                f"P@25={row['test_precision_at_25']:.3f} seconds={time.perf_counter() - step:.1f}",
                flush=True,
            )
        except Exception as exc:
            print(f"[basic_5_giant] error {spec.name}: {type(exc).__name__}: {exc}", flush=True)
            metric_rows.append({"approach": spec.name, "status": "error", "reason": f"{type(exc).__name__}: {exc}", "seed": seed})

    metrics = pd.DataFrame(metric_rows)
    per_search = pd.DataFrame(per_search_rows)
    metrics.to_csv(assert_experiment_path(out_dir / "metrics_long.csv"), index=False)
    per_search.to_csv(assert_experiment_path(out_dir / "per_search_metrics.csv"), index=False)
    score_frame(splits.validation, validation_scores).to_csv(assert_experiment_path(out_dir / "validation_scores.csv"), index=False)
    score_frame(splits.test, test_scores).to_csv(assert_experiment_path(out_dir / "test_scores.csv"), index=False)
    write_json(out_dir / "results.json", base_sweep.to_builtin({"metrics": metric_rows, "per_search_metrics": per_search_rows}))
    report_path = write_report(out_dir, metrics=metrics, per_search=per_search, searches=selected_searches)

    trained_count = int(metrics["status"].eq("trained").sum()) if "status" in metrics else 0
    summary = {
        "run_dir": str(out_dir),
        "seed": seed,
        "searches": selected_searches,
        "approaches": [spec.name for spec in specs],
        "trained_rows": trained_count,
        "outputs": {
            "report": str(report_path),
            "metrics_long": str(out_dir / "metrics_long.csv"),
            "per_search_metrics": str(out_dir / "per_search_metrics.csv"),
            "validation_scores": str(out_dir / "validation_scores.csv"),
            "test_scores": str(out_dir / "test_scores.csv"),
            "dataset_summary": str(out_dir / "dataset_summary.csv"),
        },
    }
    write_manifest(out_dir / "manifest.json", command="basic_5_giant_model.run", extra=base_sweep.to_builtin(summary))
    print(json.dumps({"run_dir": str(out_dir), "trained_rows": trained_count, "report": str(report_path)}, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one global basic_5 model family across six searches.")
    parser.add_argument("--search", action="append", default=[], help="Search to include. Defaults to the active searches.")
    parser.add_argument("--approach", action="append", default=[], choices=BASIC5_APPROACH_NAMES)
    parser.add_argument("--seed", type=int, default=base_sweep.DEFAULT_SEED)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit-rows", type=int, default=None, help="Per search/label sample cap for smoke runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        searches=args.search or None,
        approaches=args.approach or None,
        seed=args.seed,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        limit_rows=args.limit_rows,
    )


if __name__ == "__main__":
    main()
