#!/usr/bin/env python3
"""Train the full-scrape giant model family across the main searches.

Mirrors ``basic_5_giant_model.run`` (one global sold/not-sold model pooled
across searches, stratified 60/20/20 split, global validation threshold) but
uses the *full-scrape* feature set and the extended model zoo in
``model_zoo.ZOO``. Offline-only; it does not start paper trading and never
touches Telegram.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.current.full_scrape_giant_model._deps.deal_finder import model_sweep as base_sweep  # noqa: E402
from experiments.current.full_scrape_giant_model._deps.deal_finder.modeling import TARGET_COL, choose_threshold, score_with_model  # noqa: E402
from experiments.current.full_scrape_giant_model import metrics as fm  # noqa: E402
from experiments.current.full_scrape_giant_model import model_zoo as mz  # noqa: E402
from experiments.current.full_scrape_giant_model.dataset import (  # noqa: E402
    MAIN_SEARCHES,
    dataset_summary,
    load_giant_frame,
    prepare_full_frame,
)
from experiments.current.full_scrape_giant_model.image_filter import offline_image_filter  # noqa: E402
from experiments.current.full_scrape_giant_model.live_objective import LIVE_SPEC_NAMES  # noqa: E402
from experiments.current.full_scrape_giant_model.visual_features import merge_offline_visual_features  # noqa: E402
from experiments.current.full_scrape_giant_model.paths import (  # noqa: E402
    MODELS_DIR,
    OFFLINE_RUNS_DIR,
    REPORTS_DIR,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_json,
    write_manifest,
)

ID_COLUMNS = ["SearchName", "item_id", "Dataid", "Title", "Brand", "Size", "Price", "Likes", "Link"]


def stratified_global_split(
    df: pd.DataFrame,
    *,
    seed: int = base_sweep.DEFAULT_SEED,
    train_frac: float = 0.60,
    validation_frac: float = 0.20,
) -> base_sweep.SplitFrames:
    """60/20/20 split stratified by SearchName x label (mirrors basic5)."""
    from sklearn.model_selection import train_test_split

    if df.empty:
        return base_sweep.SplitFrames(df.copy(), df.copy(), df.copy())
    strata = df["SearchName"].astype(str) + "__" + df[TARGET_COL].astype(str)
    stratify = strata if strata.value_counts().min() >= 2 else (
        df[TARGET_COL].astype(str) if df[TARGET_COL].value_counts().min() >= 2 else None
    )
    train, rest = train_test_split(df, train_size=train_frac, random_state=seed, shuffle=True, stratify=stratify)
    rest_strata = rest["SearchName"].astype(str) + "__" + rest[TARGET_COL].astype(str)
    rest_stratify = rest_strata if rest_strata.value_counts().min() >= 2 else None
    val_size = validation_frac / max(1.0 - train_frac, 1e-6)
    validation, test = train_test_split(rest, train_size=val_size, random_state=seed, shuffle=True, stratify=rest_stratify)
    return base_sweep.SplitFrames(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def calibration_metrics(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    try:
        from sklearn.metrics import brier_score_loss
    except Exception:
        return {"brier": np.nan, "ece": np.nan}
    if len(y_true) == 0:
        return {"brier": np.nan, "ece": np.nan}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scores >= lo) & (scores <= hi) if hi == 1.0 else (scores >= lo) & (scores < hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(scores[mask].mean()) - float(y_true[mask].mean()))
    return {"brier": float(brier_score_loss(y_true, scores)), "ece": float(ece)}


def save_pickle(path: Path, obj: Any) -> None:
    path = assert_experiment_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(obj, handle)
    tmp.replace(path)


def _prefixed(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    keep = ("base_rate", "selected_count", "coverage", "precision", "recall", "lift",
            "precision_at_10", "precision_at_25", "precision_at_50", "roc_auc", "pr_auc")
    return {f"{prefix}_{k}": metrics.get(k) for k in keep}


def train_one(
    splits: base_sweep.SplitFrames,
    *,
    spec: mz.GiantSpec,
    seed: int,
    run_name: str,
    feature_mode: str = "full_scrape",
    visual_cols: list[str] | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    started = time.perf_counter()
    numeric, categorical, text = mz.select_full_features(splits.train, spec)
    if not numeric and not categorical and not text:
        raise ValueError("no usable full-scrape features")
    visual_used: list[str] = []
    if feature_mode == "full_scrape_visual" and visual_cols:
        extra = [c for c in visual_cols if mz._has_numeric_signal(splits.train, c) and c not in numeric]
        visual_used = base_sweep.safe_selected_features(extra, splits.train)
        numeric = numeric + visual_used
    model = mz.make_model(spec, numeric, categorical, text)
    model.fit(splits.train, splits.train[TARGET_COL].astype(int))

    val_scores = np.clip(np.asarray(score_with_model(model, splits.validation), dtype=float), 0.0, 1.0)
    test_scores = np.clip(np.asarray(score_with_model(model, splits.test), dtype=float), 0.0, 1.0)
    threshold = float(
        choose_threshold(
            splits.validation[TARGET_COL].astype(int).to_numpy(),
            val_scores,
            min_precision=base_sweep.PROMOTION_PRECISION,
            min_count=base_sweep.VALIDATION_MIN_COUNT,
        )["threshold"]
    )

    family = "full_scrape_visual" if feature_mode == "full_scrape_visual" else "full_scrape"
    artifact_path = MODELS_DIR / f"{run_name}_{family}_{spec.name}_seed{seed}.pkl"
    save_pickle(artifact_path, model)
    metadata = {
        "experiment_family": "full_scrape_giant_model",
        "feature_mode": feature_mode,
        "approach": spec.name,
        "model_kind": spec.kind,
        "origin": spec.origin,
        "artifact_path": str(artifact_path),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "text_features": text,
        "visual_features": visual_used,
        "threshold": threshold,
        "label": TARGET_COL,
        "seed": int(seed),
        "split": "stratified_search_label_random_60_20_20",
    }
    write_json(MODELS_DIR / f"{run_name}_{family}_{spec.name}_seed{seed}_metadata.json", base_sweep.to_builtin(metadata))

    y_val = splits.validation[TARGET_COL].astype(int).to_numpy()
    y_test = splits.test[TARGET_COL].astype(int).to_numpy()
    val_m = fm.summarize(y_val, val_scores, threshold)
    test_m = fm.summarize(y_test, test_scores, threshold)
    row = {
        "approach": spec.name,
        "status": "trained",
        "kind": spec.kind,
        "origin": spec.origin,
        "notes": spec.notes,
        "seed": int(seed),
        "n_numeric": len(numeric),
        "n_categorical": len(categorical),
        "n_text": len(text),
        "threshold": threshold,
        "train_rows": int(len(splits.train)),
        "validation_rows": int(len(splits.validation)),
        "test_rows": int(len(splits.test)),
        **_prefixed(val_m, "val"),
        **_prefixed(test_m, "test"),
        **calibration_metrics(y_test, test_scores),
        "fit_seconds": float(time.perf_counter() - started),
    }
    return row, val_scores, test_scores


def per_search_rows(frame: pd.DataFrame, scores: np.ndarray, *, threshold: float, approach: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scores = np.asarray(scores, dtype=float)
    for search, group in frame.groupby("SearchName", sort=True):
        idx = group.index.to_numpy(dtype=int)
        y = group[TARGET_COL].astype(int).to_numpy()
        m = fm.summarize(y, scores[idx], threshold)
        rows.append({"search_name": str(search), "approach": approach, **m})
    return rows


def score_frame(frame: pd.DataFrame, scores_by_approach: dict[str, np.ndarray]) -> pd.DataFrame:
    keep = [c for c in ID_COLUMNS if c in frame.columns]
    out = frame[keep + [TARGET_COL]].copy()
    for approach, scores in scores_by_approach.items():
        out[f"score__{approach}"] = np.asarray(scores, dtype=float)
    return out


def run(
    *,
    searches: list[str] | None = None,
    approaches: list[str] | None = None,
    feature_mode: str = "full_scrape",
    seed: int = base_sweep.DEFAULT_SEED,
    out_dir: Path | None = None,
    limit_rows: int | None = None,
    skip_image_filter: bool = False,
) -> dict[str, Any]:
    ensure_experiment_dirs()
    selected_searches = list(dict.fromkeys(searches or MAIN_SEARCHES))
    if approaches is None and feature_mode == "full_scrape_visual":
        approaches = list(LIVE_SPEC_NAMES)
    specs = [s for s in mz.ZOO if approaches is None or s.name in set(approaches)]
    if not specs:
        raise ValueError("No approaches selected")
    default_prefix = "full_scrape_giant_visual" if feature_mode == "full_scrape_visual" else "full_scrape_giant"
    out_dir = assert_experiment_path(out_dir if out_dir is not None else OFFLINE_RUNS_DIR / run_id(default_prefix))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[full_scrape_giant] loading searches={selected_searches} feature_mode={feature_mode}", flush=True)
    frame = prepare_full_frame(load_giant_frame(selected_searches))
    total_before = int(len(frame))

    # --- main-image filtering accounting ---
    if skip_image_filter:
        image_acc = pd.DataFrame([{"search_name": "__TOTAL__", "total_rows": total_before, "kept_rows": total_before, "skipped_rows": 0}])
    else:
        frame, image_acc = offline_image_filter(frame)
    image_acc.to_csv(assert_experiment_path(out_dir / "image_filter_summary.csv"), index=False)
    total_after = int(len(frame))
    print(f"[full_scrape_giant] image filter: rows {total_before} -> {total_after}", flush=True)

    if limit_rows is not None and limit_rows > 0:
        parts = []
        for _, group in frame.groupby(["SearchName", TARGET_COL], sort=False):
            parts.append(group.sample(min(limit_rows, len(group)), random_state=seed))
        frame = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    visual_cols: list[str] = []
    if feature_mode == "full_scrape_visual":
        frame, visual_cols = merge_offline_visual_features(frame)
        print(f"[full_scrape_giant] visual features merged: {len(visual_cols)} ({', '.join(visual_cols)})", flush=True)

    if frame[TARGET_COL].nunique() < 2:
        raise ValueError("Combined dataset has only one label class after filtering")

    summary = dataset_summary(frame)
    summary.to_csv(assert_experiment_path(out_dir / "dataset_summary.csv"), index=False)

    splits = stratified_global_split(frame, seed=seed)
    pd.DataFrame(
        [
            {"split": name, "rows": len(part), "positive_rows": int(part[TARGET_COL].sum()),
             "base_rate": float(part[TARGET_COL].mean()) if len(part) else np.nan}
            for name, part in (("train", splits.train), ("validation", splits.validation), ("test", splits.test))
        ]
    ).to_csv(assert_experiment_path(out_dir / "split_summary.csv"), index=False)
    print(f"[full_scrape_giant] rows={len(frame)} train={len(splits.train)} val={len(splits.validation)} test={len(splits.test)}", flush=True)

    metric_rows: list[dict[str, Any]] = []
    ps_rows: list[dict[str, Any]] = []
    val_scores: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}
    for spec in specs:
        step = time.perf_counter()
        print(f"[full_scrape_giant] training {spec.name}", flush=True)
        try:
            row, vs, ts = train_one(splits, spec=spec, seed=seed, run_name=out_dir.name, feature_mode=feature_mode, visual_cols=visual_cols)
            metric_rows.append(row)
            val_scores[spec.name] = vs
            test_scores[spec.name] = ts
            ps_rows.extend(per_search_rows(splits.test, ts, threshold=float(row["threshold"]), approach=spec.name))
            print(f"[full_scrape_giant] done {spec.name}: AUC={row['test_roc_auc']:.3f} "
                  f"P@25={row['test_precision_at_25']:.3f} cov={row['test_coverage']:.3f} {time.perf_counter()-step:.1f}s", flush=True)
        except Exception as exc:
            print(f"[full_scrape_giant] error {spec.name}: {type(exc).__name__}: {exc}", flush=True)
            metric_rows.append({"approach": spec.name, "status": "error", "kind": spec.kind,
                                "origin": spec.origin, "reason": f"{type(exc).__name__}: {exc}", "seed": seed})

    metrics_df = pd.DataFrame(metric_rows)
    per_search_df = pd.DataFrame(ps_rows)
    metrics_df.to_csv(assert_experiment_path(out_dir / "metrics_long.csv"), index=False)
    per_search_df.to_csv(assert_experiment_path(out_dir / "per_search_metrics.csv"), index=False)
    score_frame(splits.validation, val_scores).to_csv(assert_experiment_path(out_dir / "validation_scores.csv"), index=False)
    score_frame(splits.test, test_scores).to_csv(assert_experiment_path(out_dir / "test_scores.csv"), index=False)
    write_json(out_dir / "results.json", base_sweep.to_builtin({"metrics": metric_rows, "per_search_metrics": ps_rows}))

    trained = int(metrics_df.get("status", pd.Series(dtype=object)).eq("trained").sum())
    summary_payload = {
        "run_dir": str(out_dir),
        "seed": seed,
        "feature_mode": feature_mode,
        "visual_features": visual_cols,
        "searches": selected_searches,
        "approaches": [s.name for s in specs],
        "trained_rows": trained,
        "rows_before_image_filter": total_before,
        "rows_after_image_filter": total_after,
        "telegram_policy_changed": False,
    }
    write_manifest(out_dir / "manifest.json", command="full_scrape_giant_model.run", extra=base_sweep.to_builtin(summary_payload))

    # Best-effort markdown report (separate module; tolerate absence).
    try:
        from experiments.current.full_scrape_giant_model.report_offline import write_offline_report
        report_path = write_offline_report(out_dir)
        summary_payload["report"] = str(report_path)
    except Exception as exc:  # pragma: no cover
        print(f"[full_scrape_giant] report skipped: {type(exc).__name__}: {exc}", flush=True)

    print(json.dumps({"run_dir": str(out_dir), "trained_rows": trained,
                      "rows_before_image_filter": total_before, "rows_after_image_filter": total_after}, indent=2), flush=True)
    return summary_payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the full-scrape giant model family across the main searches.")
    p.add_argument("--search", action="append", default=[], help="Search to include. Defaults to the main searches.")
    p.add_argument("--approach", action="append", default=[], choices=[s.name for s in mz.ZOO])
    p.add_argument("--feature-mode", default="full_scrape", choices=["full_scrape", "full_scrape_visual"])
    p.add_argument("--seed", type=int, default=base_sweep.DEFAULT_SEED)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--limit-rows", type=int, default=None, help="Per search/label sample cap for smoke runs.")
    p.add_argument("--skip-image-filter", action="store_true", help="Train on all rows (still reports counts).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        searches=args.search or None,
        approaches=args.approach or None,
        feature_mode=args.feature_mode,
        seed=args.seed,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        limit_rows=args.limit_rows,
        skip_image_filter=args.skip_image_filter,
    )


if __name__ == "__main__":
    main()
