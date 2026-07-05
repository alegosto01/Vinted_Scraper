#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
for _path in (ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.current.giant_basic_visual._deps.basic_5_giant_model import run as basic5  # noqa: E402
from experiments.current.giant_basic_visual._deps.deal_finder import model_sweep as base_sweep  # noqa: E402
from experiments.current.giant_basic_visual._deps.deal_finder.modeling import TARGET_COL, choose_threshold, score_with_model  # noqa: E402
from experiments.current.giant_basic_visual._deps.full_scrape_model.compare_feature_modalities import BASIC_NUMERIC, BASIC_TEXT  # noqa: E402
from experiments.current.giant_basic_visual.features import (  # noqa: E402
    FEATURE_MODES,
    filter_rows_with_main_image,
    is_diagnostic_mode,
    merge_quality_scores,
    mode_live_ready,
    feature_columns_for_mode,
    assert_no_raw_dino_features,
)
from experiments.current.giant_basic_visual.paths import (  # noqa: E402
    MODELS_DIR,
    OFFLINE_RUNS_DIR,
    ROOT,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_json,
    write_manifest,
)


DEFAULT_SEARCHES = (
    "griffati_donna_all",
    "griffati_uomo_all",
    "gucci",
    "nike",
    "prada",
    "ps4",
    "telefoni",
)
DEFAULT_VISUAL_SOURCE = (
    ROOT
    / "data"
    / "experiments"
    / "photo_arbitrage"
    / "features"
    / "sold_unsold_visuals_20260514_full"
    / "combined_scored.csv"
)
ID_COLUMNS = [
    "SearchName",
    "item_id",
    "Dataid",
    "Title",
    "Brand",
    "Size",
    "Price",
    "Likes",
    "Link",
    "LocalPrimaryImagePath",
]


def _has_text_signal(frame: pd.DataFrame, column: str) -> bool:
    return (
        column in frame.columns
        and frame[column].fillna("").astype(str).str.contains(base_sweep.TOKEN_RE, regex=True).any()
    )


def selected_modes(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return FEATURE_MODES
    unknown = sorted(set(values) - set(FEATURE_MODES))
    if unknown:
        raise ValueError(f"Unknown feature modes: {', '.join(unknown)}")
    return tuple(dict.fromkeys(values))


def selected_specs(values: list[str] | None) -> list[base_sweep.ApproachSpec]:
    specs = [spec for spec in basic5.available_basic5_specs() if not values or spec.name in set(values)]
    if not specs:
        raise ValueError("No approaches selected")
    return specs


def select_features(
    train_df: pd.DataFrame,
    spec: base_sweep.ApproachSpec,
    search_onehot_cols: list[str],
    mode: str,
) -> tuple[list[str], list[str]]:
    if spec.kind == "rules":
        numeric = [col for col in ("price_num", "likes_num", "page_num", "Price", "Likes", "Page") if col in train_df.columns]
        return numeric, []
    numeric = [col for col in BASIC_NUMERIC if col in train_df.columns]
    numeric.extend(col for col in search_onehot_cols if col in train_df.columns)
    numeric.extend(col for col in feature_columns_for_mode(mode) if col in train_df.columns and train_df[col].notna().any())
    text = [col for col in BASIC_TEXT if spec.use_text and _has_text_signal(train_df, col)]
    numeric = base_sweep.safe_selected_features(numeric, train_df)
    text = base_sweep.safe_selected_features(text, train_df)
    assert_no_raw_dino_features(numeric + text)
    return numeric, text


def save_pickle(path: Path, obj: Any) -> None:
    path = assert_experiment_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(obj, handle)
    tmp.replace(path)


def train_one_approach(
    splits: base_sweep.SplitFrames,
    *,
    spec: base_sweep.ApproachSpec,
    search_onehot_cols: list[str],
    seed: int,
    run_name: str,
    mode: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    started = time.perf_counter()
    mode_spec = replace(
        spec,
        use_image=False,
        live_ready=bool(spec.live_ready and mode_live_ready(mode)),
        feature_policy=f"{spec.feature_policy}__global_basic_5_with_search__{mode}",
    )
    numeric, text = select_features(splits.train, mode_spec, search_onehot_cols, mode)
    if not numeric and not text:
        raise ValueError("no usable basic_5 visual features")
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

    artifact_path = MODELS_DIR / f"{run_name}_{mode}_{mode_spec.name}_seed{seed}.pkl"
    metadata_path = MODELS_DIR / f"{run_name}_{mode}_{mode_spec.name}_seed{seed}_metadata.json"
    save_pickle(artifact_path, model)
    metadata = {
        "experiment_family": "giant_basic_visual",
        "feature_mode": mode,
        "diagnostic": bool(is_diagnostic_mode(mode)),
        "live_ready": bool(mode_spec.live_ready),
        "telegram_policy_changed": False,
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
        "split": "stratified_search_label_random_60_20_20_fixed_after_main_image_filter",
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
        "feature_mode": mode,
        "approach": mode_spec.name,
        "status": "trained",
        "seed": int(seed),
        "artifact_path": str(artifact_path),
        "metadata_path": str(metadata_path),
        "model_kind": mode_spec.kind,
        "feature_policy": mode_spec.feature_policy,
        "diagnostic": bool(is_diagnostic_mode(mode)),
        "live_ready": bool(mode_spec.live_ready),
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
        "validation_threshold_recall": validation_eval["threshold"]["recall"],
        "validation_precision_at_10": validation_eval["precision_at"]["p@10"]["precision"],
        "validation_precision_at_25": validation_eval["precision_at"]["p@25"]["precision"],
        "validation_precision_at_50": validation_eval["precision_at"]["p@50"]["precision"],
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
        **basic5.calibration_metrics(labels, test_scores),
        "qualified_for_paper_trading": bool(promotion_ok),
        "promotion_failures": "; ".join(promotion_failures),
        "fit_seconds": float(time.perf_counter() - started),
    }
    return row, validation_scores, test_scores


def score_frame(frame: pd.DataFrame, scores_by_key: dict[str, np.ndarray]) -> pd.DataFrame:
    keep = [col for col in ID_COLUMNS if col in frame.columns]
    out = frame[keep + [TARGET_COL]].copy()
    for key, scores in scores_by_key.items():
        out[f"score__{key}"] = np.asarray(scores, dtype=float)
    return out


def split_summary_rows(splits: base_sweep.SplitFrames) -> list[dict[str, Any]]:
    return [
        {
            "split": name,
            "rows": len(part),
            "positive_rows": int(part[TARGET_COL].sum()) if TARGET_COL in part else 0,
            "base_rate": float(part[TARGET_COL].mean()) if len(part) else np.nan,
        }
        for name, part in (("train", splits.train), ("validation", splits.validation), ("test", splits.test))
    ]


def make_lift_tables(metrics: pd.DataFrame, per_search: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trained = metrics[metrics["status"].eq("trained")].copy()
    control = trained[trained["feature_mode"].eq("basic_5_control")][
        ["approach", "test_roc_auc", "test_pr_auc", "test_precision_at_25", "test_threshold_precision", "test_threshold_count"]
    ].rename(
        columns={
            "test_roc_auc": "control_test_roc_auc",
            "test_pr_auc": "control_test_pr_auc",
            "test_precision_at_25": "control_test_precision_at_25",
            "test_threshold_precision": "control_test_threshold_precision",
            "test_threshold_count": "control_test_threshold_count",
        }
    )
    lift = trained.merge(control, on="approach", how="left")
    for col in ("test_roc_auc", "test_pr_auc", "test_precision_at_25", "test_threshold_precision", "test_threshold_count"):
        base_col = f"control_{col}"
        if base_col in lift.columns:
            lift[f"delta_{col}"] = lift[col] - lift[base_col]

    if per_search.empty:
        return lift, pd.DataFrame()
    search_control = per_search[per_search["feature_mode"].eq("basic_5_control")][
        ["search_name", "approach", "roc_auc", "pr_auc", "precision_at_25", "threshold_precision", "threshold_count"]
    ].rename(
        columns={
            "roc_auc": "control_roc_auc",
            "pr_auc": "control_pr_auc",
            "precision_at_25": "control_precision_at_25",
            "threshold_precision": "control_threshold_precision",
            "threshold_count": "control_threshold_count",
        }
    )
    search_lift = per_search.merge(search_control, on=["search_name", "approach"], how="left")
    for col in ("roc_auc", "pr_auc", "precision_at_25", "threshold_precision", "threshold_count"):
        base_col = f"control_{col}"
        if base_col in search_lift.columns:
            search_lift[f"delta_{col}"] = search_lift[col] - search_lift[base_col]
    return lift, search_lift


def best_tables(metrics: pd.DataFrame, per_search: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trained = metrics[metrics["status"].eq("trained")].copy()
    best_by_mode = (
        trained.sort_values(["feature_mode", "test_roc_auc"], ascending=[True, False], na_position="last")
        .groupby("feature_mode", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    if per_search.empty:
        return best_by_mode, pd.DataFrame(), pd.DataFrame()
    best_by_search_mode = (
        per_search.sort_values(["search_name", "feature_mode", "roc_auc"], ascending=[True, True, False], na_position="last")
        .groupby(["search_name", "feature_mode"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_by_search = (
        per_search.sort_values(["search_name", "roc_auc"], ascending=[True, False], na_position="last")
        .groupby("search_name", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return best_by_mode, best_by_search_mode, best_by_search


def write_report(
    run_dir: Path,
    *,
    metrics: pd.DataFrame,
    per_search: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    skipped_summary: pd.DataFrame,
    lift: pd.DataFrame,
    search_lift: pd.DataFrame,
    searches: list[str],
    before_rows: int,
    after_rows: int,
    visual_merge: dict[str, Any],
) -> Path:
    path = assert_experiment_path(run_dir / "giant_basic_visual_report.md")

    def fmt(value: object, decimals: int = 3) -> str:
        try:
            if value is None or pd.isna(value):
                return "-"
            return f"{float(value):.{decimals}f}"
        except Exception:
            return str(value)

    trained = metrics[metrics["status"].eq("trained")].copy()
    best_overall = trained.sort_values("test_roc_auc", ascending=False, na_position="last").head(1)
    lines = [
        "# Giant Basic Visual",
        "",
        f"Run folder: `{run_dir}`",
        "",
        "This is a shadow experiment. Telegram policy was not changed.",
        "Only `LocalPrimaryImagePath` is used. Rows without a readable main image are skipped.",
        "",
        "## Dataset",
        "",
        f"- Rows before main-image filter: `{before_rows}`",
        f"- Rows after main-image filter: `{after_rows}`",
        f"- Rows skipped: `{before_rows - after_rows}`",
        f"- Visual score rows matched: `{visual_merge.get('matched_rows', 0)}` from `{visual_merge.get('score_rows', 0)}` cached score rows",
        f"- Searches: `{', '.join(searches)}`",
        "",
        "| search | rows | positive | base rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in dataset_summary.iterrows():
        lines.append(f"| {row['SearchName']} | {int(row['rows'])} | {int(row['positive_rows'])} | {fmt(row['base_rate'])} |")

    if not skipped_summary.empty:
        lines.extend(["", "## Skipped Rows", "", "| search | reason | rows |", "| --- | --- | ---: |"])
        for _, row in skipped_summary.iterrows():
            lines.append(f"| {row['SearchName']} | {row['skip_reason']} | {int(row['rows'])} |")

    lines.extend(["", "## Best Overall", ""])
    if not best_overall.empty:
        row = best_overall.iloc[0]
        lines.append(
            f"`{row['feature_mode']} / {row['approach']}`: AUC `{fmt(row['test_roc_auc'])}`, "
            f"PR-AUC `{fmt(row['test_pr_auc'])}`, P@25 `{fmt(row['test_precision_at_25'])}`, "
            f"threshold precision `{fmt(row['test_threshold_precision'])}`, selected `{int(row['test_threshold_count'])}`."
        )

    lines.extend(
        [
            "",
            "## All Models",
            "",
            "| mode | approach | AUC | PR AUC | P@10 | P@25 | P@50 | threshold precision | recall | selected | threshold | live ready |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, row in trained.sort_values(["feature_mode", "test_roc_auc"], ascending=[True, False], na_position="last").iterrows():
        lines.append(
            f"| {row['feature_mode']} | {row['approach']} | {fmt(row['test_roc_auc'])} | {fmt(row['test_pr_auc'])} | "
            f"{fmt(row['test_precision_at_10'])} | {fmt(row['test_precision_at_25'])} | {fmt(row['test_precision_at_50'])} | "
            f"{fmt(row['test_threshold_precision'])} | {fmt(row['test_threshold_recall'])} | "
            f"{int(row['test_threshold_count']) if pd.notna(row['test_threshold_count']) else 0} | {fmt(row['threshold'], 4)} | "
            f"{bool(row['live_ready'])} |"
        )

    lines.extend(["", "## Per-Search Lift Vs Control", ""])
    if not search_lift.empty:
        rows = search_lift[~search_lift["feature_mode"].eq("basic_5_control")].copy()
        rows = rows.sort_values(["search_name", "delta_roc_auc"], ascending=[True, False], na_position="last")
        lines.extend(
            [
                "| search | mode | approach | AUC | delta AUC | PR AUC | delta PR AUC | selected | threshold precision |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in rows.iterrows():
            lines.append(
                f"| {row['search_name']} | {row['feature_mode']} | {row['approach']} | {fmt(row['roc_auc'])} | "
                f"{fmt(row['delta_roc_auc'])} | {fmt(row['pr_auc'])} | {fmt(row['delta_pr_auc'])} | "
                f"{int(row['threshold_count']) if pd.notna(row['threshold_count']) else 0} | {fmt(row['threshold_precision'])} |"
            )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `filtered_dataset.csv`: training rows after main-image filter.",
            "- `skipped_main_image_rows.csv`: skipped rows and reasons.",
            "- `metrics_long.csv`: global metrics for every mode/model.",
            "- `per_search_metrics.csv`: held-out per-search metrics.",
            "- `validation_scores.csv` / `test_scores.csv`: item-level scores.",
            "- `mode_lift.csv` / `per_search_lift.csv`: lift against Basic5 control on the same rows/split.",
            "",
            "## Safety",
            "",
            "- No Telegram messages were sent.",
            "- `telegram_sent_items.csv` was not read or edited.",
            "- `main_image_dino_outlier_diagnostic` is diagnostic only and not live-ready.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(
    *,
    searches: list[str] | None = None,
    approaches: list[str] | None = None,
    feature_modes: list[str] | None = None,
    seed: int = base_sweep.DEFAULT_SEED,
    visual_source: Path | None = DEFAULT_VISUAL_SOURCE,
    out_dir: Path | None = None,
    limit_rows: int | None = None,
) -> dict[str, Any]:
    ensure_experiment_dirs()
    selected_searches = list(dict.fromkeys(searches or DEFAULT_SEARCHES))
    modes = selected_modes(feature_modes)
    specs = selected_specs(approaches)
    out_dir = assert_experiment_path(out_dir if out_dir is not None else OFFLINE_RUNS_DIR / run_id("giant_basic_visual"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[giant_basic_visual] loading searches={selected_searches}", flush=True)
    raw = basic5.load_combined_dataset(selected_searches)
    frame = basic5.prepare_global_frame(raw)
    before_rows = int(len(frame))
    if limit_rows is not None and limit_rows > 0:
        parts = []
        for _, group in frame.groupby(["SearchName", TARGET_COL], sort=False):
            parts.append(group.sample(min(limit_rows, len(group)), random_state=seed))
        frame = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        before_rows = int(len(frame))

    print("[giant_basic_visual] filtering rows to readable LocalPrimaryImagePath only", flush=True)
    filtered, skipped = filter_rows_with_main_image(frame)
    filtered, visual_merge = merge_quality_scores(filtered, visual_source)
    filtered, search_onehot_cols = basic5.add_search_onehot(filtered, selected_searches)
    after_rows = int(len(filtered))
    if after_rows == 0:
        raise ValueError("No rows survived the main-image filter")
    if filtered[TARGET_COL].nunique() < 2:
        raise ValueError("Filtered dataset has only one label class")

    dataset_summary = (
        filtered.groupby("SearchName", sort=True)[TARGET_COL]
        .agg(rows="size", positive_rows="sum", base_rate="mean")
        .reset_index()
    )
    skipped_summary = (
        skipped.groupby(["SearchName", "skip_reason"], dropna=False)
        .size()
        .reset_index(name="rows")
        if not skipped.empty
        else pd.DataFrame(columns=["SearchName", "skip_reason", "rows"])
    )
    filtered.to_csv(assert_experiment_path(out_dir / "filtered_dataset.csv"), index=False)
    skipped.to_csv(assert_experiment_path(out_dir / "skipped_main_image_rows.csv"), index=False)
    dataset_summary.to_csv(assert_experiment_path(out_dir / "dataset_summary.csv"), index=False)
    skipped_summary.to_csv(assert_experiment_path(out_dir / "skipped_summary.csv"), index=False)

    splits = basic5.stratified_global_split(filtered, seed=seed)
    pd.DataFrame(split_summary_rows(splits)).to_csv(assert_experiment_path(out_dir / "split_summary.csv"), index=False)
    print(
        f"[giant_basic_visual] rows={after_rows}/{before_rows} train={len(splits.train)} "
        f"validation={len(splits.validation)} test={len(splits.test)}",
        flush=True,
    )

    metric_rows: list[dict[str, Any]] = []
    per_search_rows: list[dict[str, Any]] = []
    validation_scores: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}
    for mode in modes:
        for spec in specs:
            key = f"{mode}__{spec.name}"
            step = time.perf_counter()
            print(f"[giant_basic_visual] training {mode} / {spec.name}", flush=True)
            try:
                row, val_score, test_score = train_one_approach(
                    splits,
                    spec=spec,
                    search_onehot_cols=search_onehot_cols,
                    seed=seed,
                    run_name=out_dir.name,
                    mode=mode,
                )
                metric_rows.append(row)
                validation_scores[key] = val_score
                test_scores[key] = test_score
                search_rows = basic5.per_search_metrics(splits.test, test_score, threshold=float(row["threshold"]), approach=spec.name)
                for search_row in search_rows:
                    search_row["feature_mode"] = mode
                    search_row["diagnostic"] = bool(is_diagnostic_mode(mode))
                    search_row["live_ready"] = bool(row["live_ready"])
                per_search_rows.extend(search_rows)
                print(
                    f"[giant_basic_visual] done {mode}/{spec.name}: AUC={row['test_roc_auc']:.3f} "
                    f"P@25={row['test_precision_at_25']:.3f} seconds={time.perf_counter() - step:.1f}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[giant_basic_visual] error {mode}/{spec.name}: {type(exc).__name__}: {exc}", flush=True)
                metric_rows.append(
                    {
                        "feature_mode": mode,
                        "approach": spec.name,
                        "status": "error",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "seed": seed,
                        "diagnostic": bool(is_diagnostic_mode(mode)),
                        "live_ready": False,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    per_search = pd.DataFrame(per_search_rows)
    lift, search_lift = make_lift_tables(metrics, per_search)
    best_by_mode, best_by_search_mode, best_by_search = best_tables(metrics, per_search)

    metrics.to_csv(assert_experiment_path(out_dir / "metrics_long.csv"), index=False)
    per_search.to_csv(assert_experiment_path(out_dir / "per_search_metrics.csv"), index=False)
    lift.to_csv(assert_experiment_path(out_dir / "mode_lift.csv"), index=False)
    search_lift.to_csv(assert_experiment_path(out_dir / "per_search_lift.csv"), index=False)
    best_by_mode.to_csv(assert_experiment_path(out_dir / "best_by_mode.csv"), index=False)
    best_by_search_mode.to_csv(assert_experiment_path(out_dir / "best_by_search_mode.csv"), index=False)
    best_by_search.to_csv(assert_experiment_path(out_dir / "best_by_search.csv"), index=False)
    score_frame(splits.validation, validation_scores).to_csv(assert_experiment_path(out_dir / "validation_scores.csv"), index=False)
    score_frame(splits.test, test_scores).to_csv(assert_experiment_path(out_dir / "test_scores.csv"), index=False)
    write_json(
        out_dir / "results.json",
        base_sweep.to_builtin(
            {
                "metrics": metric_rows,
                "per_search_metrics": per_search_rows,
                "before_rows": before_rows,
                "after_rows": after_rows,
                "visual_merge": visual_merge,
            }
        ),
    )
    report_path = write_report(
        out_dir,
        metrics=metrics,
        per_search=per_search,
        dataset_summary=dataset_summary,
        skipped_summary=skipped_summary,
        lift=lift,
        search_lift=search_lift,
        searches=selected_searches,
        before_rows=before_rows,
        after_rows=after_rows,
        visual_merge=visual_merge,
    )

    trained_count = int(metrics["status"].eq("trained").sum()) if "status" in metrics else 0
    summary = {
        "run_dir": str(out_dir),
        "seed": seed,
        "searches": selected_searches,
        "approaches": [spec.name for spec in specs],
        "feature_modes": list(modes),
        "rows_before_main_image_filter": before_rows,
        "rows_after_main_image_filter": after_rows,
        "trained_rows": trained_count,
        "telegram_policy_changed": False,
        "outputs": {
            "report": str(report_path),
            "metrics_long": str(out_dir / "metrics_long.csv"),
            "per_search_metrics": str(out_dir / "per_search_metrics.csv"),
            "mode_lift": str(out_dir / "mode_lift.csv"),
            "per_search_lift": str(out_dir / "per_search_lift.csv"),
        },
    }
    write_manifest(out_dir / "manifest.json", command="giant_basic_visual.run", extra=base_sweep.to_builtin(summary))
    print(json.dumps({"run_dir": str(out_dir), "trained_rows": trained_count, "report": str(report_path)}, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Basic5 giant models with main-image-only visual features.")
    parser.add_argument("--search", action="append", default=[], help="Search to include.")
    parser.add_argument("--approach", action="append", default=[], choices=basic5.BASIC5_APPROACH_NAMES)
    parser.add_argument("--feature-mode", action="append", default=[], choices=FEATURE_MODES)
    parser.add_argument("--seed", type=int, default=base_sweep.DEFAULT_SEED)
    parser.add_argument("--visual-source", default=str(DEFAULT_VISUAL_SOURCE))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit-rows", type=int, default=None, help="Per search/label sample cap for smoke runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visual_source = Path(args.visual_source) if args.visual_source else None
    run(
        searches=args.search or None,
        approaches=args.approach or None,
        feature_modes=args.feature_mode or None,
        seed=args.seed,
        visual_source=visual_source,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        limit_rows=args.limit_rows,
    )


if __name__ == "__main__":
    main()

