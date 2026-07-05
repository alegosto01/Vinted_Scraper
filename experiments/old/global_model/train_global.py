#!/usr/bin/env python3
"""Train global sold/not-sold models on all searches combined.

Two feature modes are trained:
  global_basic_5     – Price, Likes, Title, Brand, Size + SearchName (one-hot)
  global_full_visual – Full scrape + visual features + SearchName (one-hot)

Results are compared to the best per-search models from the most recent
sold_status_feature_modalities run and evaluated against live-tested items.
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

from experiments.old.global_model._deps.deal_finder import model_sweep as base_sweep
from experiments.old.global_model._deps.deal_finder.modeling import TARGET_COL, ELIGIBLE_COL, score_with_model
from experiments.old.global_model._deps.full_scrape_model.compare_feature_modalities import (
    BASIC_NUMERIC,
    BASIC_TEXT,
    VISUAL_NUMERIC,
    add_dino_embedding_columns,
    add_full_engineered_features,
    available_numeric_features,
    available_text_features,
    read_visual_features,
    visual_source_path,
    DEFAULT_VISUAL_RUN,
    DEFAULT_EXCLUDED_SEARCHES,
    full_numeric,
    full_text,
)
from experiments.old.global_model._deps.full_scrape_model.dataset import TASK_SOLD_STATUS
from experiments.old.global_model._deps.full_scrape_model.paths import write_json as fsm_write_json
from experiments.old.global_model.paths import (
    EXPERIMENT_ROOT,
    MODELS_DIR,
    OFFLINE_RUNS_DIR,
    PER_SEARCH_RUN_DIR,
    SOLD_STATUS_SOURCE,
    LIVE_BENCHMARK_CSV,
    CASCADE_LIVE_CSV,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_json,
    write_manifest,
)

DEFAULT_SEED = 42
FEATURE_MODES = ("global_basic_5", "global_full_visual")
APPROACHES = ["logistic_snapshot_v2", "sgd_text_numeric_v1", "numeric_tree_v1", "linear_svm_calibrated_v1"]
CALIBRATION_BINS = 10

# Searches excluded by default (too few samples or noisy)
EXCLUDED_SEARCHES = DEFAULT_EXCLUDED_SEARCHES


# ── Data loading ────────────────────────────────────────────────────────────

def _normalize_id(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def load_global_dataset(*, exclude_searches: set[str] | None = None) -> pd.DataFrame:
    if not SOLD_STATUS_SOURCE.exists():
        raise FileNotFoundError(f"Training source not found: {SOLD_STATUS_SOURCE}")
    df = pd.read_csv(SOLD_STATUS_SOURCE, low_memory=False)

    if exclude_searches:
        mask = df["SourceSearch"].astype(str).isin(exclude_searches) if "SourceSearch" in df.columns else pd.Series(False, index=df.index)
        df = df[~mask].copy()

    # Resolve item_id
    df["item_id"] = ""
    if "Dataid" in df.columns:
        df["item_id"] = df["Dataid"].map(_normalize_id)
    if "Link" in df.columns:
        links = df["Link"].fillna("").astype(str).str.strip()
        df["item_id"] = df["item_id"].where(df["item_id"].str.len() > 0, links)

    # Dedupe within each search to avoid item appearing twice
    df = df[df["item_id"].str.len() > 0].copy()
    source_col = "SourceSearch" if "SourceSearch" in df.columns else "SearchName"
    df = df.drop_duplicates(subset=[source_col, "item_id"], keep="last").reset_index(drop=True)

    # Labels
    label = df.get("MergedStatusBinary", pd.Series(index=df.index, dtype=object)).astype(str).str.strip().str.lower()
    df[TARGET_COL] = label.eq("sold").astype(int)
    df[ELIGIBLE_COL] = True
    df["training_task"] = TASK_SOLD_STATUS

    # Normalise search name column
    if "SourceSearch" in df.columns and "SearchName" not in df.columns:
        df["SearchName"] = df["SourceSearch"]
    elif "SourceSearch" in df.columns:
        df["SearchName"] = df["SourceSearch"].fillna(df["SearchName"])

    return df.reset_index(drop=True)


def add_search_onehot(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add one-hot columns for SearchName; return (df_with_cols, col_names)."""
    out = df.copy()
    searches = sorted(out["SearchName"].dropna().astype(str).unique())
    cols = []
    for s in searches:
        col = f"search__{s.lower().replace(' ', '_')}"
        out[col] = (out["SearchName"].astype(str) == s).astype(int)
        cols.append(col)
    return out, cols


def safe_select(candidates: list[str], df: pd.DataFrame, blocked: set[str] | None = None) -> list[str]:
    blocked = blocked or set()
    safe = {c for c in df.columns if not base_sweep.is_leakage_feature(c)}
    return [c for c in candidates if c in safe and c not in blocked]


# ── Stratified split ─────────────────────────────────────────────────────────

def stratified_global_split(
    df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratify on search × label so each split has all searches represented."""
    from sklearn.model_selection import train_test_split

    # Create strata: search_label combo
    strata = df["SearchName"].astype(str) + "__" + df[TARGET_COL].astype(str)
    min_class_size = strata.value_counts().min()

    if min_class_size < 5:
        # Fall back to label-only stratification
        strata = df[TARGET_COL].astype(str)

    train, rest = train_test_split(df, train_size=train_frac, random_state=seed, shuffle=True, stratify=strata)
    rest_strata = rest["SearchName"].astype(str) + "__" + rest[TARGET_COL].astype(str)
    rest_min = rest_strata.value_counts().min()
    val_size = val_frac / (1.0 - train_frac)
    val, test = train_test_split(
        rest,
        train_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=rest_strata if rest_min >= 2 else None,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ── Feature selection by mode ─────────────────────────────────────────────────

def mode_features(
    train_df: pd.DataFrame,
    *,
    mode: str,
    search_onehot_cols: list[str],
    embedding_cols: list[str],
) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, text_cols) for the given mode."""
    if mode == "global_basic_5":
        numeric = available_numeric_features(train_df, BASIC_NUMERIC + search_onehot_cols)
        text = available_text_features(train_df, BASIC_TEXT)
    elif mode == "global_full_visual":
        fn = full_numeric(include_upload_date=False)
        ft = full_text(include_upload_date=False)
        numeric = available_numeric_features(train_df, fn + VISUAL_NUMERIC + embedding_cols + search_onehot_cols)
        text = available_text_features(train_df, ft)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    # Strip leakage
    blocked = set(search_onehot_cols)  # keep onehot; leakage guard handles others
    numeric = safe_select(numeric, train_df)
    text = safe_select(text, train_df)
    return numeric, text


# ── Model training ─────────────────────────────────────────────────────────

def train_global_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    mode: str,
    approach_name: str,
    seed: int,
    search_onehot_cols: list[str],
    embedding_cols: list[str],
    run_prefix: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    spec = next((s for s in base_sweep.APPROACHES if s.name == approach_name), None)
    if spec is None:
        return {"mode": mode, "approach": approach_name, "status": "error", "reason": f"Unknown approach: {approach_name}"}

    # Always disable the legacy use_image path; visual features handled explicitly
    from dataclasses import replace as _replace
    spec = _replace(spec, use_image=False)

    numeric, text = mode_features(train_df, mode=mode, search_onehot_cols=search_onehot_cols, embedding_cols=embedding_cols)
    if not numeric and not text:
        return {"mode": mode, "approach": approach_name, "status": "skipped", "reason": "no usable features"}
    if train_df[TARGET_COL].nunique() < 2:
        return {"mode": mode, "approach": approach_name, "status": "skipped", "reason": "single label class in train"}
    if spec.kind == "linear_svm_calibrated" and train_df[TARGET_COL].value_counts().min() < 3:
        return {"mode": mode, "approach": approach_name, "status": "skipped", "reason": "calibrated SVM needs >= 3 rows per class"}

    fit_frame = base_sweep.bounded_fit_frame(train_df, spec, seed)
    model = base_sweep.make_model(spec, numeric, text)
    model.fit(fit_frame, fit_frame[TARGET_COL].astype(int))

    val_scores = score_with_model(model, val_df)
    test_scores = score_with_model(model, test_df)

    threshold = base_sweep.choose_threshold(
        val_df[TARGET_COL].astype(int).to_numpy(),
        val_scores,
        min_precision=base_sweep.PROMOTION_PRECISION,
        min_count=base_sweep.VALIDATION_MIN_COUNT,
    )["threshold"]

    artifact_path = MODELS_DIR / f"{run_prefix}_{mode}_{approach_name}_seed{seed}.pkl"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = artifact_path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(model, fh)
    tmp.replace(artifact_path)

    global_val = base_sweep.evaluate_scores(val_df, val_scores, float(threshold))
    global_test = base_sweep.evaluate_scores(test_df, test_scores, float(threshold))
    per_search_test = _per_search_eval(test_df, test_scores, threshold)
    calib = calibration_metrics(test_df[TARGET_COL].astype(int).to_numpy(), test_scores)

    return {
        "mode": mode,
        "approach": approach_name,
        "status": "trained",
        "seed": int(seed),
        "artifact_path": str(artifact_path),
        "numeric_features": numeric,
        "text_features": text,
        "threshold": float(threshold),
        "train_rows": int(len(train_df)),
        "fit_train_rows": int(len(fit_frame)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "global_val": global_val,
        "global_test": global_test,
        "per_search_test": per_search_test,
        "calibration": calib,
        "fit_seconds": float(time.perf_counter() - started),
    }


def _per_search_eval(df: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for search, idx in df.groupby("SearchName").groups.items():
        sub = df.loc[idx]
        sub_scores = scores[sub.index.map({v: i for i, v in enumerate(df.index)}.get)]
        y = sub[TARGET_COL].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            out[str(search)] = {"skipped": "single class", "rows": int(len(sub))}
            continue
        auc_row = base_sweep.auc_metrics(y, sub_scores)
        thr_row = base_sweep.threshold_metrics(y, sub_scores, threshold)
        out[str(search)] = {
            "rows": int(len(sub)),
            "positive_rows": int(y.sum()),
            "base_rate": float(y.mean()),
            "roc_auc": auc_row.get("roc_auc", np.nan),
            "pr_auc": auc_row.get("pr_auc", np.nan),
            "threshold_precision": thr_row.get("precision", np.nan),
            "threshold_count": thr_row.get("count", 0),
            "threshold_recall": float(thr_row.get("positive_count", 0) / y.sum()) if y.sum() > 0 else np.nan,
        }
    return out


# ── Calibration ───────────────────────────────────────────────────────────────

def calibration_metrics(y_true: np.ndarray, scores: np.ndarray, n_bins: int = CALIBRATION_BINS) -> dict[str, Any]:
    """Compute Brier score, ECE, and calibration curve data."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    if len(np.unique(y_true)) < 2:
        return {"brier": np.nan, "ece": np.nan, "curve": []}

    brier = float(brier_score_loss(y_true, scores))

    # ECE: expected calibration error
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (scores >= lo) & (scores < hi)
        if hi == 1.0:
            mask = (scores >= lo) & (scores <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = float(scores[mask].mean())
        bin_acc = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

    # Calibration curve (fraction of positives vs mean predicted probability)
    try:
        frac_pos, mean_pred = calibration_curve(y_true, scores, n_bins=n_bins, strategy="uniform")
        curve = [{"mean_pred": float(mp), "frac_pos": float(fp)} for mp, fp in zip(mean_pred, frac_pos)]
    except Exception:
        curve = []

    return {"brier": brier, "ece": float(ece), "curve": curve}


# ── Per-search comparison ─────────────────────────────────────────────────────

def load_per_search_results(run_dir: Path | None = None) -> pd.DataFrame:
    """Load the best per-search results from the most recent feature modalities run."""
    base = PER_SEARCH_RUN_DIR
    if run_dir is None:
        if not base.exists():
            return pd.DataFrame()
        runs = sorted(base.iterdir(), reverse=True)
        candidates = [r for r in runs if r.is_dir() and (r / "metrics_long.csv").exists() and "modalities" in r.name]
        if not candidates:
            return pd.DataFrame()
        run_dir = candidates[0]
    csv_path = run_dir / "metrics_long.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path, low_memory=False)
    return df[df["status"] == "trained"].copy()


def build_comparison_table(global_results: list[dict], per_search_df: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side global vs per-search best metrics for each search × mode."""
    rows = []
    mode_to_per_search_mode = {
        "global_basic_5": "basic_5",
        "global_full_visual": "full_scrape_plus_visual",
    }
    for result in global_results:
        if result.get("status") != "trained":
            continue
        mode = result["mode"]
        approach = result["approach"]
        per_mode = mode_to_per_search_mode.get(mode, mode)
        global_auc = result.get("global_test", {}).get("roc_auc", np.nan)
        global_prec = result.get("global_test", {}).get("threshold", {}).get("precision", np.nan)
        brier = result.get("calibration", {}).get("brier", np.nan)
        ece = result.get("calibration", {}).get("ece", np.nan)

        for search, ps_metrics in result.get("per_search_test", {}).items():
            if "skipped" in ps_metrics:
                continue
            # Per-search model best AUC for same mode
            ps_auc = np.nan
            ps_prec = np.nan
            ps_approach = ""
            if not per_search_df.empty:
                mask = (per_search_df["search_name"] == search) & (per_search_df["feature_mode"] == per_mode)
                subset = per_search_df[mask]
                if not subset.empty:
                    best_row = subset.sort_values("test_roc_auc", ascending=False).iloc[0]
                    ps_auc = float(best_row.get("test_roc_auc", np.nan))
                    ps_prec = float(best_row.get("test_precision", np.nan))
                    ps_approach = str(best_row.get("approach", ""))
            rows.append({
                "search": search,
                "mode": mode,
                "global_approach": approach,
                "per_search_approach": ps_approach,
                "global_search_roc_auc": float(ps_metrics.get("roc_auc", np.nan)),
                "per_search_roc_auc": ps_auc,
                "auc_delta": float(ps_metrics.get("roc_auc", np.nan)) - ps_auc,
                "global_search_precision": float(ps_metrics.get("threshold_precision", np.nan)),
                "per_search_precision": ps_prec,
                "precision_delta": float(ps_metrics.get("threshold_precision", np.nan)) - ps_prec,
                "global_brier": brier,
                "global_ece": ece,
                "test_rows": int(ps_metrics.get("rows", 0)),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Live benchmark testing ────────────────────────────────────────────────────

def score_live_items(
    model,
    live_df: pd.DataFrame,
    *,
    mode: str,
    numeric: list[str],
    text: list[str],
    search_onehot_cols: list[str],
) -> pd.DataFrame:
    """Apply a trained global model to live benchmark items."""
    scored = live_df.copy()
    # Add search one-hot for inference
    all_searches = [c.replace("search__", "") for c in search_onehot_cols]
    for s in all_searches:
        col = f"search__{s}"
        if col not in scored.columns:
            scored[col] = (scored.get("SearchName", pd.Series("", index=scored.index)).astype(str).str.lower() == s).astype(int)

    # Ensure required feature columns exist
    for col in numeric:
        if col not in scored.columns:
            scored[col] = np.nan
    for col in text:
        if col not in scored.columns:
            scored[col] = ""

    scored["global_score"] = score_with_model(model, scored)
    return scored


def evaluate_live_items(scored_df: pd.DataFrame, *, outcome_col: str = "sold_within_7d") -> dict[str, Any]:
    """Compare global model scores to original per-search model scores on live items.

    Uses last_recheck_status to include OnSale (not-sold) items as negatives.
    """
    df = scored_df.copy()

    # Use last_recheck_status when available to get full binary labels
    if "last_recheck_status" in df.columns:
        status = df["last_recheck_status"].astype(str).str.strip()
        known = status.isin(["Sold", "OnSale"])
        df = df[known].copy()
        df["_outcome"] = (status[known] == "Sold").astype(int)
        resolved = df
    else:
        has_outcome = df[outcome_col].notna() if outcome_col in df.columns else pd.Series(False, index=df.index)
        resolved = df[has_outcome].copy()
        resolved["_outcome"] = resolved[outcome_col].astype(int)

    if resolved.empty:
        return {"resolved_items": 0, "note": f"No items with known {outcome_col} outcome"}

    y_true = resolved["_outcome"].to_numpy()
    global_scores = resolved["global_score"].to_numpy()

    from sklearn.metrics import roc_auc_score, average_precision_score
    result: dict[str, Any] = {
        "resolved_items": int(len(resolved)),
        "positives": int(y_true.sum()),
        "base_rate": float(y_true.mean()),
        "outcome_source": "last_recheck_status" if "last_recheck_status" in resolved.columns else outcome_col,
    }

    if len(np.unique(y_true)) < 2:
        result["note"] = "Only one class in resolved items"
        return result

    result["global_roc_auc"] = float(roc_auc_score(y_true, global_scores))
    result["global_pr_auc"] = float(average_precision_score(y_true, global_scores))
    result["global_brier"] = float(calibration_metrics(y_true, global_scores)["brier"])
    result["global_ece"] = float(calibration_metrics(y_true, global_scores)["ece"])

    # Compare to original model scores if available
    if "model_probability" in resolved.columns:
        orig_scores = pd.to_numeric(resolved["model_probability"], errors="coerce").fillna(0.5).to_numpy()
        result["orig_roc_auc"] = float(roc_auc_score(y_true, orig_scores))
        result["orig_pr_auc"] = float(average_precision_score(y_true, orig_scores))
        result["orig_brier"] = float(calibration_metrics(y_true, orig_scores)["brier"])
        result["auc_delta_vs_orig"] = result["global_roc_auc"] - result["orig_roc_auc"]

    return result


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(
    run_dir: Path,
    global_results: list[dict],
    comparison_df: pd.DataFrame,
    live_eval: dict[str, Any],
    per_search_run_name: str,
) -> Path:
    path = assert_experiment_path(run_dir / "global_model_report.md")

    def _fmt(v, decimals=3) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:.{decimals}f}"

    lines = [
        "# Global Sold/Not-Sold Model — Training Report",
        "",
        f"Run folder: `{run_dir}`",
        "",
        "## Overview",
        "",
        "Trains a **single model across all searches** (sold/not-sold target) for two feature modes:",
        "",
        "- `global_basic_5` — Price, Likes, Title, Brand, Size + SearchName one-hot",
        "- `global_full_visual` — Full scrape metadata + photo-arbitrage visual features + SearchName one-hot",
        "",
        "The same approaches used in per-search experiments are tested.",
        f"Comparison baseline: `{per_search_run_name}`",
        "",
    ]

    # Summary table
    trained = [r for r in global_results if r.get("status") == "trained"]
    lines += [
        "## Global Test Metrics (best approach per mode)",
        "",
        "| mode | approach | test AUC | test PR AUC | test precision@thr | threshold | brier | ece | fit_s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    # Best per mode by AUC
    best_by_mode: dict[str, dict] = {}
    for r in trained:
        m = r["mode"]
        auc = r.get("global_test", {}).get("roc_auc", np.nan)
        if m not in best_by_mode or (pd.notna(auc) and auc > best_by_mode[m].get("global_test", {}).get("roc_auc", -1)):
            best_by_mode[m] = r
    for m in FEATURE_MODES:
        r = best_by_mode.get(m)
        if r is None:
            lines.append(f"| {m} | — | — | — | — | — | — | — | — |")
            continue
        gt = r.get("global_test", {})
        calib = r.get("calibration", {})
        lines.append(
            f"| {m} | {r['approach']} | {_fmt(gt.get('roc_auc'))} | {_fmt(gt.get('pr_auc'))} | "
            f"{_fmt(gt.get('threshold', {}).get('precision'))} | {_fmt(r.get('threshold'), 4)} | "
            f"{_fmt(calib.get('brier'))} | {_fmt(calib.get('ece'))} | {_fmt(r.get('fit_seconds'), 1)} |"
        )
    lines.append("")

    # All approaches table
    lines += [
        "## All Approaches",
        "",
        "| mode | approach | test AUC | test PR AUC | test precision@thr | brier | ece |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(trained, key=lambda x: (x["mode"], -x.get("global_test", {}).get("roc_auc", 0))):
        gt = r.get("global_test", {})
        calib = r.get("calibration", {})
        lines.append(
            f"| {r['mode']} | {r['approach']} | {_fmt(gt.get('roc_auc'))} | {_fmt(gt.get('pr_auc'))} | "
            f"{_fmt(gt.get('threshold', {}).get('precision'))} | {_fmt(calib.get('brier'))} | {_fmt(calib.get('ece'))} |"
        )
    lines.append("")

    # Per-search breakdown (best global model per mode)
    lines += [
        "## Per-Search Breakdown (best global model per mode)",
        "",
        "| search | mode | global AUC | per-search AUC | Δ AUC | global prec | per-search prec |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not comparison_df.empty:
        for _, row in comparison_df.sort_values(["search", "mode"]).iterrows():
            lines.append(
                f"| {row['search']} | {row['mode']} | {_fmt(row['global_search_roc_auc'])} | "
                f"{_fmt(row['per_search_roc_auc'])} | {_fmt(row['auc_delta'])} | "
                f"{_fmt(row['global_search_precision'])} | {_fmt(row['per_search_precision'])} |"
            )
    else:
        lines.append("| (no comparison data) |")
    lines.append("")

    # Calibration curves (text representation)
    lines += ["## Calibration Summary", ""]
    lines.append("Lower Brier score = better probability calibration. Lower ECE = better average calibration.")
    lines.append("")
    for m in FEATURE_MODES:
        r = best_by_mode.get(m)
        if r is None:
            continue
        calib = r.get("calibration", {})
        curve = calib.get("curve", [])
        lines.append(f"### {m} ({r['approach']})")
        lines.append(f"Brier: {_fmt(calib.get('brier'))} | ECE: {_fmt(calib.get('ece'))}")
        if curve:
            lines.append("")
            lines.append("| bin mean_pred | fraction_positive | gap |")
            lines.append("| ---: | ---: | ---: |")
            for pt in curve:
                gap = pt["mean_pred"] - pt["frac_pos"]
                lines.append(f"| {pt['mean_pred']:.2f} | {pt['frac_pos']:.2f} | {gap:+.2f} |")
        lines.append("")

    # Live testing
    lines += ["## Live Benchmark Items", ""]
    if not live_eval:
        lines.append("No live benchmark data available.")
    else:
        lines.append(f"Items with resolved outcome: **{live_eval.get('resolved_items', 0)}**")
        lines.append(f"Sold (positive): {live_eval.get('positives', 0)} / base rate: {_fmt(live_eval.get('base_rate'))}")
        lines.append("")
        if "global_roc_auc" in live_eval:
            lines.append(f"- Global model AUC: **{_fmt(live_eval.get('global_roc_auc'))}**")
            lines.append(f"- Global model PR-AUC: **{_fmt(live_eval.get('global_pr_auc'))}**")
            lines.append(f"- Global model Brier: **{_fmt(live_eval.get('global_brier'))}**")
            lines.append(f"- Global model ECE: **{_fmt(live_eval.get('global_ece'))}**")
        if "orig_roc_auc" in live_eval:
            lines.append(f"- Original per-search AUC: **{_fmt(live_eval.get('orig_roc_auc'))}**")
            lines.append(f"- AUC delta (global − per-search): **{_fmt(live_eval.get('auc_delta_vs_orig'))}**")
        if live_eval.get("note"):
            lines.append(f"  *Note: {live_eval['note']}*")
    lines.append("")

    lines += [
        "## Notes",
        "",
        "- Global model is trained on ALL searches simultaneously; per-search models train separately on each search.",
        "- SearchName is one-hot encoded as a numeric feature (not a text feature) so tree models can use it directly.",
        "- Stratified split ensures each search × label combination appears in all three splits.",
        "- Calibration: a perfectly calibrated model would have all calibration curve points on the diagonal.",
        "- ECE < 0.05 is generally considered well-calibrated. Brier baseline is the base rate squared.",
        "- Live benchmark: only items with resolved sold_within_7d outcomes are used for AUC calculation.",
        "  The live cohort is small (25 resolved items as of 2026-05-20) so results are indicative, not conclusive.",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def run(
    *,
    modes: list[str] | None = None,
    approaches: list[str] | None = None,
    seed: int = DEFAULT_SEED,
    visual_run: str = DEFAULT_VISUAL_RUN,
    include_dino: bool = True,
    max_dino_dims: int | None = None,
    limit_rows: int | None = None,
    out_dir: Path | None = None,
    exclude_searches: set[str] | None = None,
) -> dict[str, Any]:
    ensure_experiment_dirs()
    if out_dir is None:
        out_dir = OFFLINE_RUNS_DIR / run_id("global_sold_status")
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = modes or list(FEATURE_MODES)
    approach_names = approaches or APPROACHES
    exclude = exclude_searches if exclude_searches is not None else EXCLUDED_SEARCHES

    print(f"[global_model] Loading training data from {SOLD_STATUS_SOURCE.name}…", flush=True)
    df = load_global_dataset(exclude_searches=exclude)
    if limit_rows:
        parts = []
        for search, grp in df.groupby("SearchName"):
            parts.append(grp.sample(min(limit_rows, len(grp)), random_state=seed))
        df = pd.concat(parts, ignore_index=True)
    print(f"[global_model] Total rows: {len(df)} | searches: {sorted(df['SearchName'].unique())}", flush=True)

    # Visual features (only needed for full_visual mode)
    visual_df = pd.DataFrame()
    embedding_cols: list[str] = []
    if any("visual" in m for m in modes):
        vpath = visual_source_path(visual_run)
        try:
            print(f"[global_model] Loading visual features from {vpath.name}…", flush=True)
            visual_df = read_visual_features(vpath, include_dino_embedding=include_dino)
            print(f"[global_model] Visual rows: {len(visual_df)}", flush=True)
        except FileNotFoundError as exc:
            print(f"[global_model] WARNING: {exc} — visual mode will use no visual features", flush=True)

    all_results: list[dict] = []
    for mode in modes:
        print(f"\n[global_model] ── Mode: {mode} ──", flush=True)

        # Build dataset for this mode
        work = add_full_engineered_features(df.copy())
        if mode == "global_full_visual" and not visual_df.empty:
            # Merge visual features: match on SearchName + item_id
            visual_for_merge = visual_df.drop(columns=["SearchName"], errors="ignore")
            work = work.merge(visual_for_merge, on="item_id", how="left")
            work, embedding_cols = add_dino_embedding_columns(work, max_dims=max_dino_dims)

        # Add SearchName one-hot
        work, search_onehot_cols = add_search_onehot(work)

        # Filter eligible
        eligible_mask = base_sweep.parse_bool_series(work[ELIGIBLE_COL]) if ELIGIBLE_COL in work.columns else pd.Series(True, index=work.index)
        work = work[eligible_mask].copy()
        work[TARGET_COL] = pd.to_numeric(work[TARGET_COL], errors="coerce").fillna(0).astype(int)
        print(f"[global_model] Eligible rows: {len(work)} | positives: {int(work[TARGET_COL].sum())}", flush=True)

        train_df, val_df, test_df = stratified_global_split(work, seed=seed)
        print(f"[global_model] Split: train={len(train_df)} val={len(val_df)} test={len(test_df)}", flush=True)

        for approach_name in approach_names:
            print(f"[global_model] Training {mode} × {approach_name}…", flush=True)
            try:
                result = train_global_model(
                    train_df, val_df, test_df,
                    mode=mode,
                    approach_name=approach_name,
                    seed=seed,
                    search_onehot_cols=search_onehot_cols,
                    embedding_cols=embedding_cols,
                    run_prefix=out_dir.name,
                )
                all_results.append(result)
                gt = result.get("global_test", {})
                calib = result.get("calibration", {})
                print(
                    f"[global_model] done {mode}×{approach_name}: AUC={gt.get('roc_auc', float('nan')):.3f} "
                    f"brier={calib.get('brier', float('nan')):.3f} ece={calib.get('ece', float('nan')):.3f} "
                    f"s={result.get('fit_seconds', 0):.1f}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[global_model] ERROR {mode}×{approach_name}: {type(exc).__name__}: {exc}", flush=True)
                all_results.append({"mode": mode, "approach": approach_name, "status": "error", "reason": str(exc)})

    # Save metrics
    write_json(out_dir / "results.json", {"seed": seed, "results": base_sweep.to_builtin(all_results)})
    results_df_rows = []
    for r in all_results:
        gt = r.get("global_test", {})
        calib = r.get("calibration", {})
        results_df_rows.append({
            "mode": r.get("mode"),
            "approach": r.get("approach"),
            "status": r.get("status"),
            "threshold": r.get("threshold"),
            "train_rows": r.get("train_rows"),
            "test_rows": r.get("test_rows"),
            "test_roc_auc": gt.get("roc_auc"),
            "test_pr_auc": gt.get("pr_auc"),
            "test_precision": gt.get("threshold", {}).get("precision"),
            "test_count": gt.get("threshold", {}).get("count"),
            "test_recall": gt.get("threshold", {}).get("recall"),
            "brier": calib.get("brier"),
            "ece": calib.get("ece"),
            "fit_seconds": r.get("fit_seconds"),
        })
    results_df = pd.DataFrame(results_df_rows)
    results_df.to_csv(assert_experiment_path(out_dir / "results_summary.csv"), index=False)

    # Load per-search comparison
    per_search_df = load_per_search_results()
    per_search_run_name = "(not found)"
    if not per_search_df.empty:
        runs = sorted(PER_SEARCH_RUN_DIR.iterdir(), reverse=True) if PER_SEARCH_RUN_DIR.exists() else []
        cands = [r for r in runs if r.is_dir() and "modalities" in r.name and (r / "metrics_long.csv").exists()]
        if cands:
            per_search_run_name = cands[0].name

    # Best result per mode (by global AUC)
    best_by_mode: dict[str, dict] = {}
    for r in all_results:
        if r.get("status") != "trained":
            continue
        m = r["mode"]
        auc = r.get("global_test", {}).get("roc_auc", -1)
        if m not in best_by_mode or auc > best_by_mode[m].get("global_test", {}).get("roc_auc", -2):
            best_by_mode[m] = r

    comparison_df = build_comparison_table(list(best_by_mode.values()), per_search_df)
    if not comparison_df.empty:
        comparison_df.to_csv(assert_experiment_path(out_dir / "comparison_vs_per_search.csv"), index=False)

    # Live items evaluation (global_basic_5 only — live items have basic features)
    live_eval: dict[str, Any] = {}
    best_basic = best_by_mode.get("global_basic_5")
    if best_basic and best_basic.get("status") == "trained" and LIVE_BENCHMARK_CSV.exists():
        try:
            live_df_raw = pd.read_csv(LIVE_BENCHMARK_CSV, low_memory=False)
            live_work = add_full_engineered_features(live_df_raw.copy())
            _, onehot_cols = add_search_onehot(live_work)  # only for column list
            # Re-add onehot based on trained model's search columns
            model_pkl = pickle.loads(Path(best_basic["artifact_path"]).read_bytes())
            scored_live = score_live_items(
                model_pkl,
                live_work,
                mode="global_basic_5",
                numeric=best_basic["numeric_features"],
                text=best_basic["text_features"],
                search_onehot_cols=[c for c in best_basic["numeric_features"] if c.startswith("search__")],
            )
            scored_live.to_csv(assert_experiment_path(out_dir / "live_scored.csv"), index=False)
            live_eval = evaluate_live_items(scored_live)
            print(f"[global_model] Live eval: {live_eval}", flush=True)
        except Exception as exc:
            print(f"[global_model] WARNING: live evaluation failed: {exc}", flush=True)
            live_eval = {"error": str(exc)}

    # Write report
    report_path = write_report(out_dir, all_results, comparison_df, live_eval, per_search_run_name)
    print(f"\n[global_model] Report: {report_path}", flush=True)

    summary = {
        "run_dir": str(out_dir),
        "modes": modes,
        "approaches": approach_names,
        "seed": seed,
        "trained": len([r for r in all_results if r.get("status") == "trained"]),
        "per_search_comparison": per_search_run_name,
        "live_eval": base_sweep.to_builtin(live_eval),
        "report": str(report_path),
    }
    write_manifest(out_dir / "manifest.json", command="global_model.train_global", extra=base_sweep.to_builtin(summary))
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train global sold/not-sold models across all searches.")
    ap.add_argument("--mode", action="append", default=[], choices=list(FEATURE_MODES))
    ap.add_argument("--approach", action="append", default=[], choices=APPROACHES)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--visual-run", default=DEFAULT_VISUAL_RUN)
    ap.add_argument("--no-dino", action="store_true", help="Exclude DINO embedding from visual mode.")
    ap.add_argument("--max-dino-dims", type=int, default=None)
    ap.add_argument("--limit-rows", type=int, default=None, help="Max rows per search (for smoke testing).")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--include-excluded-searches", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        modes=args.mode or None,
        approaches=args.approach or None,
        seed=args.seed,
        visual_run=args.visual_run,
        include_dino=not args.no_dino,
        max_dino_dims=args.max_dino_dims,
        limit_rows=args.limit_rows,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        exclude_searches=set() if args.include_excluded_searches else None,
    )
    print(json.dumps({k: summary[k] for k in ("run_dir", "trained", "per_search_comparison")}, indent=2))


if __name__ == "__main__":
    main()
