from __future__ import annotations
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.current.full_scrape_giant_model._deps.deal_finder.modeling import threshold_metrics, precision_at_k
from experiments.current.full_scrape_giant_model._deps.deal_finder.model_sweep import auc_metrics


def summarize(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """
    Compute a flat dict of metrics for a single model evaluation.

    Args:
        y_true: Binary labels as array-like.
        scores: Predicted scores as array-like.
        threshold: Decision threshold.

    Returns:
        Flat dict with keys: rows, positive_rows, base_rate, threshold, selected_count,
        coverage, precision, recall, lift, precision_at_10, precision_at_25,
        precision_at_50, roc_auc, pr_auc.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)

    thr = threshold_metrics(y, s, threshold)
    recall = float(thr["positive_count"] / y.sum()) if y.sum() else np.nan
    base_rate = float(y.mean()) if len(y) else np.nan
    selected_count = int(thr["count"])
    coverage = float(thr["count"] / len(y)) if len(y) else np.nan
    lift = (
        float(thr["precision"] / base_rate)
        if (base_rate and pd.notna(thr["precision"]))
        else np.nan
    )

    p = {k: precision_at_k(y, s, k) for k in (10, 25, 50)}
    a = auc_metrics(y, s)

    return {
        "rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "base_rate": base_rate,
        "threshold": float(threshold),
        "selected_count": selected_count,
        "coverage": coverage,
        "precision": float(thr["precision"]) if pd.notna(thr["precision"]) else np.nan,
        "recall": recall,
        "lift": lift,
        "precision_at_10": p[10]["precision"],
        "precision_at_25": p[25]["precision"],
        "precision_at_50": p[50]["precision"],
        "roc_auc": a["roc_auc"],
        "pr_auc": a["pr_auc"],
    }


def best_model_overall(metrics_df: pd.DataFrame, metric: str = "roc_auc") -> pd.Series | None:
    """
    Return the single best model row by metric.

    If a 'status' column exists, filter to 'trained' rows first.
    Rows with NaN in the metric are placed last.

    Args:
        metrics_df: DataFrame with model metrics.
        metric: Column name to sort by (descending).

    Returns:
        Top row as Series, or None if DataFrame is empty.
    """
    df = metrics_df.copy()
    if "status" in df.columns:
        df = df[df["status"] == "trained"]
    if df.empty:
        return None
    df = df.sort_values(by=metric, ascending=False, na_position="last")
    return df.iloc[0]


def best_model_per_search(per_search_df: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """
    Return the best model per search_name.

    Sorts by metric descending, keeps first occurrence per search_name,
    and resets the index.

    Args:
        per_search_df: DataFrame with per-search model metrics.
        metric: Column name to sort by (descending).

    Returns:
        DataFrame with one row per search_name (the best by metric).
    """
    df = per_search_df.sort_values(by=metric, ascending=False, na_position="last")
    df = df.drop_duplicates(subset=["search_name"], keep="first")
    df = df.reset_index(drop=True)
    return df
