#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.deal_finder import model_sweep as base_sweep
from experiments.deal_finder.modeling import TARGET_COL, score_with_model
from experiments.full_scrape_model.dataset import (
    TASK_SOLD_STATUS,
    available_searches,
    build_search_dataset,
    normalize_id_value,
)
from experiments.full_scrape_model.paths import (
    EXPERIMENT_ROOT,
    MODELS_DIR,
    OFFLINE_RUNS_DIR,
    ROOT,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_json,
    write_manifest,
)


DEFAULT_VISUAL_RUN = "sold_unsold_visuals_20260514_full"
DEFAULT_EXCLUDED_SEARCHES = {"Borse_Griffate", "Scarpe_Griffate"}
FEATURE_MODES = ("basic_5", "full_scrape", "full_scrape_plus_visual")

BASIC_NUMERIC = ["Price", "Likes"]
BASIC_TEXT = ["Title", "Brand", "Size"]

FULL_NUMERIC = [
    "Price",
    "Likes",
    "Page",
    "SearchCount",
    "Interested_count",
    "View_count",
    "ReviewsCount",
    "Stars",
    "PictureCount",
    "VisiblePictureCount",
    "HiddenPictureCount",
    "Upload_date_days",
    "description_char_len",
    "description_token_count",
    "condition_present",
    "seller_present",
    "location_present",
    "has_reviews",
    "has_stars",
]

FULL_TEXT = [
    "Title",
    "Brand",
    "Size",
    "Description",
    "Condition",
    "Upload_date",
    "SellerName",
    "Location",
]

VISUAL_NUMERIC = [
    "SimpleBadPhotoScore",
    "PyiqaQualityScore",
    "PyiqaBadPhotoScore",
    "AestheticGoodScore",
    "AestheticBadPhotoScore",
    "DinoEmbeddingDim",
    "DinoEmbeddingNorm",
    "DinoOutlierScore",
    "CombinedBadPhotoScore",
    "PhotoImageCount",
    "PhotoUsableImageCount",
    "PhotoPrimaryWidth",
    "PhotoPrimaryHeight",
    "PhotoPrimaryAspectRatio",
    "PhotoPrimaryBrightness",
    "PhotoPrimaryContrast",
    "PhotoPrimarySaturation",
    "PhotoPrimarySharpness",
    "PhotoPrimaryEdgeDensity",
    "PhotoAvgBrightness",
    "PhotoAvgContrast",
    "PhotoAvgSaturation",
    "PhotoAvgSharpness",
    "PhotoAvgEdgeDensity",
    "PhotoDuplicateImageCount",
    "PhotoLowQualityFraction",
    "PhotoScreenshotRiskFraction",
    "PhotoOnlyOneImage",
    "PhotoMissingImages",
    "PictureCountNum",
]

VISUAL_MERGE_COLUMNS = ["SearchName", "item_id", "Dataid", "Link", *VISUAL_NUMERIC, "DinoEmbedding"]


def patch_base_paths() -> None:
    base_sweep.MODELS_DIR = MODELS_DIR
    base_sweep.assert_experiment_path = assert_experiment_path
    base_sweep.write_json = write_json
    base_sweep.write_manifest = write_manifest


def visual_source_path(run_name: str) -> Path:
    return ROOT / "data" / "experiments" / "photo_arbitrage" / "features" / run_name / "combined_scored.csv"


def add_identity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "item_id" in out.columns:
        out["item_id"] = out["item_id"].map(normalize_id_value)
    else:
        out["item_id"] = ""
    if "Dataid" in out.columns:
        data_ids = out["Dataid"].map(normalize_id_value)
        out["item_id"] = out["item_id"].where(out["item_id"].astype(str).str.len() > 0, data_ids)
    if "Link" in out.columns:
        links = out["Link"].fillna("").astype(str).str.strip()
        out["item_id"] = out["item_id"].where(out["item_id"].astype(str).str.len() > 0, links)
    return out


def read_visual_features(path: Path, *, include_dino_embedding: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Visual feature file not found: {path}")
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in VISUAL_MERGE_COLUMNS if col in header.columns]
    if not include_dino_embedding and "DinoEmbedding" in usecols:
        usecols.remove("DinoEmbedding")
    visual = pd.read_csv(path, usecols=usecols, low_memory=False)
    visual = add_identity(visual)
    visual = visual[visual["item_id"].astype(str).str.len() > 0].copy()
    keep_cols = ["SearchName", "item_id"] + [
        col for col in VISUAL_NUMERIC if col in visual.columns
    ]
    if include_dino_embedding and "DinoEmbedding" in visual.columns:
        keep_cols.append("DinoEmbedding")
    visual = visual[keep_cols].drop_duplicates(subset=["SearchName", "item_id"], keep="last")
    return visual.reset_index(drop=True)


def parse_embedding(value: object) -> list[float] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except Exception:
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    try:
        return [float(item) for item in parsed]
    except Exception:
        return None


def add_dino_embedding_columns(df: pd.DataFrame, *, max_dims: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    if "DinoEmbedding" not in df.columns:
        return df, []
    parsed: dict[int, list[float]] = {}
    dim = 0
    for idx, value in df["DinoEmbedding"].items():
        vector = parse_embedding(value)
        if vector:
            parsed[idx] = vector
            dim = max(dim, len(vector))
    if not parsed or dim <= 0:
        return df.drop(columns=["DinoEmbedding"], errors="ignore"), []
    if max_dims is not None and max_dims > 0:
        dim = min(dim, int(max_dims))
    matrix = np.full((len(df), dim), np.nan, dtype=np.float32)
    index_pos = {idx: pos for pos, idx in enumerate(df.index)}
    for idx, vector in parsed.items():
        pos = index_pos.get(idx)
        if pos is None:
            continue
        take = min(dim, len(vector))
        matrix[pos, :take] = np.asarray(vector[:take], dtype=np.float32)
    cols = [f"DinoEmbedding_{i:04d}" for i in range(dim)]
    emb_df = pd.DataFrame(matrix, index=df.index, columns=cols)
    out = pd.concat([df.drop(columns=["DinoEmbedding"], errors="ignore"), emb_df], axis=1)
    return out, cols


def text_len(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.len()


def token_count(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.findall(r"[A-Za-z0-9]+").map(len)


def present_flag(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("").astype(int)


def add_full_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = base_sweep.add_engineered_snapshot_features(df)
    description = out["Description"] if "Description" in out.columns else pd.Series([""] * len(out), index=out.index)
    condition = out["Condition"] if "Condition" in out.columns else pd.Series([""] * len(out), index=out.index)
    seller = out["SellerName"] if "SellerName" in out.columns else pd.Series([""] * len(out), index=out.index)
    location = out["Location"] if "Location" in out.columns else pd.Series([""] * len(out), index=out.index)
    out["description_char_len"] = text_len(description)
    out["description_token_count"] = token_count(description)
    out["condition_present"] = present_flag(condition)
    out["seller_present"] = present_flag(seller)
    out["location_present"] = present_flag(location)
    out["has_reviews"] = pd.to_numeric(out.get("ReviewsCount", pd.Series(index=out.index)), errors="coerce").fillna(0).gt(0).astype(int)
    out["has_stars"] = pd.to_numeric(out.get("Stars", pd.Series(index=out.index)), errors="coerce").notna().astype(int)
    return out


def available_text_features(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    out: list[str] = []
    for col in candidates:
        if col not in df.columns:
            continue
        values = df[col].fillna("").astype(str)
        if values.str.contains(base_sweep.TOKEN_RE, regex=True).any():
            out.append(col)
    return out


def available_numeric_features(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [col for col in candidates if col in df.columns]


def mode_feature_pools(df: pd.DataFrame, mode: str, embedding_cols: list[str]) -> tuple[list[str], list[str]]:
    if mode == "basic_5":
        return available_numeric_features(df, BASIC_NUMERIC), available_text_features(df, BASIC_TEXT)
    if mode == "full_scrape":
        return available_numeric_features(df, FULL_NUMERIC), available_text_features(df, FULL_TEXT)
    if mode == "full_scrape_plus_visual":
        numeric = available_numeric_features(df, FULL_NUMERIC + VISUAL_NUMERIC + embedding_cols)
        return numeric, available_text_features(df, FULL_TEXT)
    raise ValueError(f"Unknown feature mode: {mode}")


def select_mode_features(
    df: pd.DataFrame,
    *,
    spec: base_sweep.ApproachSpec,
    mode: str,
    embedding_cols: list[str],
) -> tuple[list[str], list[str]]:
    numeric_pool, text_pool = mode_feature_pools(df, mode, embedding_cols)
    if spec.kind == "rules":
        return [col for col in ("price_num", "likes_num", "page_num", "Price", "Likes", "Page") if col in df.columns], []
    if spec.name == "logistic_v1_baseline":
        numeric = available_numeric_features(df, BASIC_NUMERIC if mode == "basic_5" else FULL_NUMERIC)
        if mode == "full_scrape_plus_visual":
            numeric = available_numeric_features(df, FULL_NUMERIC + VISUAL_NUMERIC + embedding_cols)
        text = text_pool
    elif spec.use_text:
        numeric = numeric_pool
        text = text_pool
    else:
        numeric = numeric_pool
        text = []
    numeric = base_sweep.safe_selected_features(numeric, df)
    text = base_sweep.safe_selected_features(text, df)
    return numeric, text


def make_mode_spec(spec: base_sweep.ApproachSpec, mode: str) -> base_sweep.ApproachSpec:
    # The comparison controls image/visual inputs via feature_mode, so do not let
    # the legacy visual_basic_v1 path recompute a different image feature set.
    return replace(spec, use_image=False, feature_policy=f"{spec.feature_policy}__{mode}")


def evaluate_model(
    frame: pd.DataFrame,
    *,
    search_name: str,
    mode: str,
    spec: base_sweep.ApproachSpec,
    seed: int,
    run_prefix: str,
    embedding_cols: list[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    work = add_full_engineered_features(frame)
    splits = base_sweep.stratified_random_split(work, seed=seed)
    if min(len(splits.train), len(splits.validation), len(splits.test)) == 0:
        return {"search_name": search_name, "feature_mode": mode, "approach": spec.name, "seed": seed, "status": "skipped", "reason": "empty split"}
    if splits.train[TARGET_COL].nunique() < 2:
        return {"search_name": search_name, "feature_mode": mode, "approach": spec.name, "seed": seed, "status": "skipped", "reason": "training split has only one label class"}

    mode_spec = make_mode_spec(spec, mode)
    numeric, text = select_mode_features(splits.train, spec=mode_spec, mode=mode, embedding_cols=embedding_cols)
    if not numeric and not text:
        return {"search_name": search_name, "feature_mode": mode, "approach": spec.name, "seed": seed, "status": "skipped", "reason": "no usable features"}
    if mode_spec.kind == "linear_svm_calibrated" and splits.train[TARGET_COL].value_counts().min() < 3:
        return {
            "search_name": search_name,
            "feature_mode": mode,
            "approach": spec.name,
            "seed": seed,
            "status": "skipped",
            "reason": "calibrated SVM needs at least 3 training rows per class",
        }

    fit_frame = base_sweep.bounded_fit_frame(splits.train, mode_spec, seed)
    model = base_sweep.make_model(mode_spec, numeric, text)
    model.fit(fit_frame, fit_frame[TARGET_COL].astype(int))
    val_scores = score_with_model(model, splits.validation)
    test_scores = score_with_model(model, splits.test)
    threshold = base_sweep.choose_threshold(
        splits.validation[TARGET_COL].astype(int).to_numpy(),
        val_scores,
        min_precision=base_sweep.PROMOTION_PRECISION,
        min_count=base_sweep.VALIDATION_MIN_COUNT,
    )["threshold"]
    feature_leakage = base_sweep.has_leakage_features(numeric + text)

    artifact_path = MODELS_DIR / f"{run_prefix}_{search_name}_{mode}_{spec.name}_seed{seed}.pkl"
    base_sweep.save_pickle(artifact_path, model)
    metadata = {
        "experiment_family": "full_scrape_model",
        "comparison": "feature_modalities",
        "task": "sold_status",
        "search_name": search_name,
        "feature_mode": mode,
        "approach": spec.name,
        "model_kind": spec.kind,
        "artifact_path": str(artifact_path),
        "numeric_features": numeric,
        "text_features": text,
        "threshold": float(threshold),
        "label": TARGET_COL,
        "seed": int(seed),
        "split": "stratified_random_60_20_20",
        "input_rows": int(len(frame)),
        "train_rows": int(len(splits.train)),
        "fit_train_rows": int(len(fit_frame)),
        "visual_embedding_dims_used": int(len([c for c in numeric if c.startswith("DinoEmbedding_")])),
    }
    write_json(artifact_path.with_name(artifact_path.stem + "_metadata.json"), base_sweep.to_builtin(metadata))

    metrics = {
        "search_name": search_name,
        "feature_mode": mode,
        "approach": spec.name,
        "status": "trained",
        "seed": int(seed),
        "model_metadata": metadata,
        "feature_policy": mode_spec.feature_policy,
        "live_ready": False,
        "requires_images": mode == "full_scrape_plus_visual",
        "numeric_features": numeric,
        "text_features": text,
        "feature_leakage": bool(feature_leakage),
        "threshold": float(threshold),
        "split_rows": {
            "train": int(len(splits.train)),
            "fit_train": int(len(fit_frame)),
            "validation": int(len(splits.validation)),
            "test": int(len(splits.test)),
        },
        "validation": base_sweep.evaluate_scores(splits.validation, val_scores, float(threshold)),
        "test": base_sweep.evaluate_scores(splits.test, test_scores, float(threshold)),
    }
    ok, reasons = base_sweep.promoted_80(metrics, live_ready=True, feature_leakage=feature_leakage)
    metrics["qualified_for_paper_trading"] = bool(ok)
    metrics["promotion_failures"] = reasons
    metrics["fit_seconds"] = float(time.perf_counter() - started)
    return metrics


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    flat = base_sweep.flatten_metrics(row)
    flat["feature_mode"] = row.get("feature_mode")
    flat["requires_images"] = row.get("requires_images")
    metadata = row.get("model_metadata", {})
    flat["visual_embedding_dims_used"] = metadata.get("visual_embedding_dims_used", 0) if isinstance(metadata, dict) else 0
    return flat


def best_tables(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trained = metrics_df[metrics_df["status"] == "trained"].copy() if not metrics_df.empty else pd.DataFrame()
    if trained.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    sort_cols = ["qualified_for_paper_trading", "test_precision", "test_precision_at_10", "test_count", "test_roc_auc"]
    ascending = [False, False, False, False, False]
    best_by_search_mode = (
        trained.sort_values(sort_cols, ascending=ascending)
        .drop_duplicates(["search_name", "feature_mode"], keep="first")
        .reset_index(drop=True)
    )
    best_by_search = (
        trained.sort_values(sort_cols, ascending=ascending)
        .drop_duplicates(["search_name"], keep="first")
        .reset_index(drop=True)
    )
    best_by_approach_mode = (
        trained.sort_values(sort_cols, ascending=ascending)
        .drop_duplicates(["approach", "feature_mode"], keep="first")
        .reset_index(drop=True)
    )
    return best_by_search_mode, best_by_search, best_by_approach_mode


def modality_comparison(best_by_search_mode: pd.DataFrame) -> pd.DataFrame:
    if best_by_search_mode.empty:
        return pd.DataFrame()
    keep = [
        "search_name",
        "feature_mode",
        "approach",
        "threshold",
        "validation_precision",
        "validation_count",
        "test_precision",
        "test_count",
        "test_precision_at_10",
        "test_roc_auc",
        "test_pr_auc",
    ]
    return best_by_search_mode[[col for col in keep if col in best_by_search_mode.columns]].sort_values(
        ["search_name", "feature_mode"], kind="stable"
    )


def write_report(run_dir: Path, metrics_df: pd.DataFrame, best_by_search_mode: pd.DataFrame, comparison: pd.DataFrame) -> Path:
    path = assert_experiment_path(run_dir / "feature_modality_report.md")
    lines = [
        "# Full Scrape Feature Modality Comparison",
        "",
        f"Run folder: `{run_dir}`",
        "",
        "This compares the same sold-status model families across three input modes:",
        "",
        "- `basic_5`: only `Title`, `Brand`, `Size`, `Price`, and `Likes` as source inputs.",
        "- `full_scrape`: basic fields plus full item/seller metadata, without photo-arbitrage visual features.",
        "- `full_scrape_plus_visual`: full-scrape fields plus photo-arbitrage numeric visual features and DINO embedding dimensions when available.",
        "",
        f"Trained rows: `{int((metrics_df['status'] == 'trained').sum()) if not metrics_df.empty else 0}`",
        "",
    ]
    if not comparison.empty:
        lines.extend(
            [
                "## Best Model Per Search And Mode",
                "",
                "| search | mode | approach | threshold | val precision | test precision | test p@10 | test AUC | test PR AUC | test count |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in comparison.iterrows():
            lines.append(
                f"| {row.get('search_name')} | {row.get('feature_mode')} | {row.get('approach')} | "
                f"{float(row.get('threshold', np.nan)):.4f} | "
                f"{float(row.get('validation_precision', np.nan)):.3f} | "
                f"{float(row.get('test_precision', np.nan)):.3f} | "
                f"{float(row.get('test_precision_at_10', np.nan)):.3f} | "
                f"{float(row.get('test_roc_auc', np.nan)):.3f} | "
                f"{float(row.get('test_pr_auc', np.nan)):.3f} | "
                f"{int(row.get('test_count', 0)) if pd.notna(row.get('test_count', np.nan)) else 0} |"
            )
    if not best_by_search_mode.empty:
        lines.extend(["", "## Visual Lift Summary", ""])
        for search, group in best_by_search_mode.groupby("search_name", sort=True):
            by_mode = group.set_index("feature_mode")
            base_auc = by_mode.get("test_roc_auc", pd.Series(dtype=float)).get("full_scrape", np.nan)
            visual_auc = by_mode.get("test_roc_auc", pd.Series(dtype=float)).get("full_scrape_plus_visual", np.nan)
            base_precision = by_mode.get("test_precision", pd.Series(dtype=float)).get("full_scrape", np.nan)
            visual_precision = by_mode.get("test_precision", pd.Series(dtype=float)).get("full_scrape_plus_visual", np.nan)
            if pd.notna(base_auc) and pd.notna(visual_auc):
                lines.append(
                    f"- `{search}`: visual vs full nonvisual AUC lift `{visual_auc - base_auc:+.3f}`, "
                    f"test precision change `{visual_precision - base_precision:+.3f}`."
                )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare basic, full-scrape, and full-scrape-plus-visual feature modes.")
    parser.add_argument("--all-searches", action="store_true")
    parser.add_argument("--search", action="append", default=[])
    parser.add_argument("--include-excluded-searches", action="store_true")
    parser.add_argument("--visual-run", default=DEFAULT_VISUAL_RUN)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--approach", action="append", default=[])
    parser.add_argument("--seed", type=int, default=base_sweep.DEFAULT_SEED)
    parser.add_argument("--include-dino-embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-dino-dims", type=int, default=None, help="Optional cap for DINO embedding dimensions; default uses all.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.all_searches and not args.search:
        raise SystemExit("Use --all-searches or at least one --search.")
    ensure_experiment_dirs()
    patch_base_paths()
    out_dir = Path(args.out_dir) if args.out_dir else OFFLINE_RUNS_DIR / run_id("sold_status_feature_modalities")
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = out_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    visual_path = visual_source_path(args.visual_run)
    visual = read_visual_features(visual_path, include_dino_embedding=args.include_dino_embedding)
    selected_names = set(args.approach or [spec.name for spec in base_sweep.APPROACHES])
    approaches = [spec for spec in base_sweep.APPROACHES if spec.name in selected_names]

    searches = args.search or available_searches(TASK_SOLD_STATUS)
    if not args.include_excluded_searches:
        searches = [search for search in searches if search not in DEFAULT_EXCLUDED_SEARCHES]

    metrics: list[dict[str, Any]] = []
    dataset_summary: list[dict[str, Any]] = []
    for search in searches:
        frame = build_search_dataset(TASK_SOLD_STATUS, search)
        frame = base_sweep.prepare_sweep_frame(frame)
        if args.limit_rows is not None and args.limit_rows > 0:
            frame = frame.sample(n=min(args.limit_rows, len(frame)), random_state=args.seed).reset_index(drop=True)
        if frame.empty:
            print(f"[feature_modalities] {search}: empty eligible frame")
            continue
        frame = add_identity(frame)
        search_visual = visual[visual["SearchName"].astype(str) == str(search)].copy() if "SearchName" in visual.columns else visual.copy()
        merged = frame.merge(search_visual.drop(columns=["SearchName"], errors="ignore"), on="item_id", how="left")
        merged, embedding_cols = add_dino_embedding_columns(merged, max_dims=args.max_dino_dims)
        dataset_path = assert_experiment_path(datasets_dir / f"{search}.csv")
        merged.to_csv(dataset_path, index=False)
        dataset_summary.append(
            {
                "search_name": search,
                "rows": int(len(merged)),
                "positive_rows": int(merged[TARGET_COL].sum()),
                "visual_rows": int(merged[[c for c in VISUAL_NUMERIC if c in merged.columns]].notna().any(axis=1).sum()),
                "dino_embedding_dims": int(len(embedding_cols)),
                "dataset_path": str(dataset_path),
            }
        )
        print(
            f"[feature_modalities] {search}: rows={len(merged)} positives={int(merged[TARGET_COL].sum())} "
            f"visual_rows={dataset_summary[-1]['visual_rows']} dino_dims={len(embedding_cols)}",
            flush=True,
        )
        if len(merged) < 50 or merged[TARGET_COL].nunique() < 2:
            for mode in FEATURE_MODES:
                for spec in approaches:
                    metrics.append(
                        {
                            "search_name": search,
                            "feature_mode": mode,
                            "approach": spec.name,
                            "status": "skipped",
                            "seed": args.seed,
                            "reason": "not enough eligible rows or only one label class",
                        }
                    )
            continue
        for mode in FEATURE_MODES:
            for spec in approaches:
                started = time.perf_counter()
                print(f"[feature_modalities] train search={search} mode={mode} approach={spec.name}", flush=True)
                try:
                    row = evaluate_model(
                        merged,
                        search_name=search,
                        mode=mode,
                        spec=spec,
                        seed=args.seed,
                        run_prefix=out_dir.name,
                        embedding_cols=embedding_cols,
                    )
                    metrics.append(row)
                    print(
                        f"[feature_modalities] done search={search} mode={mode} approach={spec.name} "
                        f"status={row.get('status')} seconds={time.perf_counter() - started:.1f}",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[feature_modalities] error search={search} mode={mode} approach={spec.name}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    metrics.append(
                        {
                            "search_name": search,
                            "feature_mode": mode,
                            "approach": spec.name,
                            "status": "error",
                            "seed": args.seed,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )

    write_json(out_dir / "metrics.json", base_sweep.to_builtin({"metrics": metrics}))
    metrics_df = pd.DataFrame([flatten(row) for row in metrics])
    metrics_path = assert_experiment_path(out_dir / "metrics_long.csv")
    metrics_df.to_csv(metrics_path, index=False)
    best_by_search_mode, best_by_search, best_by_approach_mode = best_tables(metrics_df)
    best_by_search_mode_path = assert_experiment_path(out_dir / "best_by_search_mode.csv")
    best_by_search_path = assert_experiment_path(out_dir / "best_by_search.csv")
    best_by_approach_mode_path = assert_experiment_path(out_dir / "best_by_approach_mode.csv")
    best_by_search_mode.to_csv(best_by_search_mode_path, index=False)
    best_by_search.to_csv(best_by_search_path, index=False)
    best_by_approach_mode.to_csv(best_by_approach_mode_path, index=False)
    comparison = modality_comparison(best_by_search_mode)
    comparison_path = assert_experiment_path(out_dir / "modality_comparison_by_search.csv")
    comparison.to_csv(comparison_path, index=False)
    report_path = write_report(out_dir, metrics_df, best_by_search_mode, comparison)
    write_manifest(
        out_dir / "manifest.json",
        command=" ".join(sys.argv),
        extra=base_sweep.to_builtin(
            {
                "task": TASK_SOLD_STATUS,
                "feature_modes": FEATURE_MODES,
                "searches": searches,
                "excluded_searches": sorted(DEFAULT_EXCLUDED_SEARCHES if not args.include_excluded_searches else []),
                "approaches": [spec.name for spec in approaches],
                "seed": args.seed,
                "visual_source": str(visual_path),
                "include_dino_embedding": bool(args.include_dino_embedding),
                "max_dino_dims": args.max_dino_dims,
                "dataset_summary": dataset_summary,
                "outputs": {
                    "metrics_long": str(metrics_path),
                    "best_by_search_mode": str(best_by_search_mode_path),
                    "best_by_search": str(best_by_search_path),
                    "best_by_approach_mode": str(best_by_approach_mode_path),
                    "modality_comparison": str(comparison_path),
                    "report": str(report_path),
                },
            }
        ),
    )
    print(f"Output folder: {out_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
