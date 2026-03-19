#!/usr/bin/env python3
"""
Evaluate how well DealScore separates sold vs unsold items.

Features:
- Adds SoldLabel on the fly by matching Dataid from sold_df into deals_ranked.csv
- Writes a labeled version of deals_ranked
- Computes summary stats for sold vs unsold
- Computes sell rate by DealScore quantile bins
- Computes Precision@TopK
- Exports false positives / false negatives for inspection
- Saves a few simple matplotlib charts

Usage:
  python evaluate_deal_score.py \
      --deals deals_ranked.csv \
      --sold sold_df.csv \
      --out_dir ./deal_score_eval
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_TOPK = [10, 20, 50, 100, 200, 500, 1000]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", required=True, help="Path to deals_ranked.csv")
    ap.add_argument("--sold", required=True, help="Path to sold_df.csv")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--id_col", default="Dataid", help="Identifier column shared by both CSVs")
    ap.add_argument("--score_col", default="DealScore", help="Deal score column in deals CSV")
    ap.add_argument("--search_col", default="SearchName", help="Optional search/grouping column")
    ap.add_argument("--product_col", default="ProductId", help="Optional product column")
    ap.add_argument("--variant_col", default="VariantId", help="Optional variant column")
    ap.add_argument("--bins", type=int, default=10, help="Number of quantile bins")
    ap.add_argument(
        "--topk",
        default=",".join(map(str, DEFAULT_TOPK)),
        help="Comma-separated list like 10,20,50,100",
    )
    ap.add_argument("--high_score_threshold", type=float, default=2.0)
    ap.add_argument("--low_score_threshold", type=float, default=-1.0)
    return ap.parse_args()


def safe_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def normalize_id_series(s: pd.Series) -> pd.Series:
    # robust string normalization so numeric/string ids still match
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def quantile_binning(values: pd.Series, bins: int) -> pd.Series:
    valid = values.notna()
    out = pd.Series([pd.NA] * len(values), index=values.index, dtype="object")
    n_unique = values[valid].nunique()
    if n_unique == 0:
        return out
    q = min(bins, n_unique)
    try:
        out.loc[valid] = pd.qcut(values.loc[valid], q=q, duplicates="drop")
    except ValueError:
        ranks = values.loc[valid].rank(method="first")
        out.loc[valid] = pd.qcut(ranks, q=min(q, ranks.nunique()), duplicates="drop")
    return out


def precision_at_k(df: pd.DataFrame, k_values: Iterable[int], score_col: str) -> pd.DataFrame:
    rows = []
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    n_total = len(ranked)
    for k in k_values:
        kk = min(k, n_total)
        if kk <= 0:
            continue
        top = ranked.head(kk)
        sold_rate = float(top["SoldLabel"].mean()) if len(top) else np.nan
        rows.append({"TopK": kk, "SoldRate": sold_rate, "SoldCount": int(top["SoldLabel"].sum()), "Count": int(len(top))})
    return pd.DataFrame(rows)


def correlation_summary(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    tmp = df[[score_col, "SoldLabel"]].copy()
    tmp[score_col] = safe_numeric_series(tmp[score_col])
    tmp = tmp.dropna(subset=[score_col])
    pearson = tmp[score_col].corr(tmp["SoldLabel"], method="pearson") if len(tmp) >= 2 else np.nan
    spearman = tmp[score_col].corr(tmp["SoldLabel"], method="spearman") if len(tmp) >= 2 else np.nan
    return pd.DataFrame([
        {"Metric": "pearson_corr", "Value": pearson},
        {"Metric": "spearman_corr", "Value": spearman},
        {"Metric": "n_scored_items", "Value": int(len(tmp))},
    ])


def sold_vs_unsold_summary(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    g = df.groupby("SoldLabel", dropna=False)[score_col]
    summary = g.agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    summary["SoldLabelName"] = summary["SoldLabel"].map({0: "unsold", 1: "sold"})
    return summary[["SoldLabel", "SoldLabelName", "count", "mean", "median", "std", "min", "max"]]


def grouped_summary(df: pd.DataFrame, by_col: str, score_col: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by_col):
        if pd.isna(key):
            continue
        rows.append(
            {
                by_col: key,
                "Count": int(len(g)),
                "SoldRate": float(g["SoldLabel"].mean()),
                "MeanDealScore": float(g[score_col].mean()) if g[score_col].notna().any() else np.nan,
                "MedianDealScore": float(g[score_col].median()) if g[score_col].notna().any() else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Count", "SoldRate"], ascending=[False, False])
    return out


def save_bar_plot(data: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path, rotate: bool = True) -> None:
    if data.empty:
        return
    fig = plt.figure(figsize=(10, 5))
    x = np.arange(len(data))
    plt.bar(x, data[y_col].to_numpy())
    plt.xticks(x, data[x_col].astype(str).tolist(), rotation=45 if rotate else 0, ha="right" if rotate else "center")
    plt.title(title)
    plt.ylabel(y_col)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_hist_by_class(df: pd.DataFrame, score_col: str, out_path: Path) -> None:
    scored = df.dropna(subset=[score_col]).copy()
    if scored.empty:
        return
    fig = plt.figure(figsize=(9, 5))
    sold = scored.loc[scored["SoldLabel"] == 1, score_col].to_numpy()
    unsold = scored.loc[scored["SoldLabel"] == 0, score_col].to_numpy()
    bins = 30
    if len(unsold):
        plt.hist(unsold, bins=bins, alpha=0.6, label="Unsold")
    if len(sold):
        plt.hist(sold, bins=bins, alpha=0.6, label="Sold")
    plt.xlabel(score_col)
    plt.ylabel("Count")
    plt.title("DealScore distribution by sold label")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_precision_plot(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["TopK"], df["SoldRate"], marker="o")
    plt.xlabel("Top K highest DealScore items")
    plt.ylabel("Sold rate")
    plt.title("Precision@K")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    deals = pd.read_csv(args.deals)
    sold = pd.read_csv(args.sold)

    if args.id_col not in deals.columns:
        raise ValueError(f"{args.id_col!r} not found in deals file")
    if args.id_col not in sold.columns:
        raise ValueError(f"{args.id_col!r} not found in sold file")
    if args.score_col not in deals.columns:
        raise ValueError(f"{args.score_col!r} not found in deals file")

    deals[args.id_col] = normalize_id_series(deals[args.id_col])
    sold[args.id_col] = normalize_id_series(sold[args.id_col])
    sold_ids = set(sold[args.id_col].dropna().unique().tolist())

    deals[args.score_col] = safe_numeric_series(deals[args.score_col])
    deals["SoldLabel"] = deals[args.id_col].isin(sold_ids).astype(int)
    deals["SoldLabelName"] = deals["SoldLabel"].map({0: "unsold", 1: "sold"})

    labeled_path = out_dir / "deals_ranked_labeled.csv"
    deals.to_csv(labeled_path, index=False)

    overall = sold_vs_unsold_summary(deals, args.score_col)
    overall.to_csv(out_dir / "overall_sold_vs_unsold_summary.csv", index=False)

    corr = correlation_summary(deals, args.score_col)
    corr.to_csv(out_dir / "correlation_summary.csv", index=False)

    deals["ScoreBin"] = quantile_binning(deals[args.score_col], args.bins)
    binned = (
        deals.dropna(subset=["ScoreBin"])
        .groupby("ScoreBin", observed=False)
        .agg(
            Count=("SoldLabel", "size"),
            SoldCount=("SoldLabel", "sum"),
            SoldRate=("SoldLabel", "mean"),
            MeanDealScore=(args.score_col, "mean"),
            MedianDealScore=(args.score_col, "median"),
            MinDealScore=(args.score_col, "min"),
            MaxDealScore=(args.score_col, "max"),
        )
        .reset_index()
    )
    binned["ScoreBin"] = binned["ScoreBin"].astype(str)
    binned.to_csv(out_dir / "sell_rate_by_score_bin.csv", index=False)

    k_values = [int(x) for x in str(args.topk).split(",") if str(x).strip()]
    topk_df = precision_at_k(deals.dropna(subset=[args.score_col]), k_values, args.score_col)
    topk_df.to_csv(out_dir / "precision_at_k.csv", index=False)

    if args.search_col in deals.columns:
        by_search = grouped_summary(deals, args.search_col, args.score_col)
        by_search.to_csv(out_dir / "summary_by_search.csv", index=False)

    if args.product_col in deals.columns:
        by_product = grouped_summary(deals, args.product_col, args.score_col)
        by_product.to_csv(out_dir / "summary_by_product.csv", index=False)

    if args.variant_col in deals.columns:
        by_variant = grouped_summary(deals, args.variant_col, args.score_col)
        by_variant.to_csv(out_dir / "summary_by_variant.csv", index=False)

    false_positives = deals[(deals[args.score_col] >= args.high_score_threshold) & (deals["SoldLabel"] == 0)].copy()
    false_positives = false_positives.sort_values(args.score_col, ascending=False)
    false_positives.to_csv(out_dir / "false_positives_high_score_unsold.csv", index=False)

    false_negatives = deals[(deals[args.score_col] <= args.low_score_threshold) & (deals["SoldLabel"] == 1)].copy()
    false_negatives = false_negatives.sort_values(args.score_col, ascending=True)
    false_negatives.to_csv(out_dir / "false_negatives_low_score_sold.csv", index=False)

    # plots
    save_hist_by_class(deals, args.score_col, out_dir / "dealscore_distribution_sold_vs_unsold.png")
    save_bar_plot(binned, "ScoreBin", "SoldRate", "Sold rate by DealScore quantile bin", out_dir / "sell_rate_by_score_bin.png")
    save_precision_plot(topk_df, out_dir / "precision_at_k.png")

    # brief report
    best_topk = topk_df.loc[topk_df["SoldRate"].idxmax()].to_dict() if not topk_df.empty else {}
    report = {
        "n_deals_rows": int(len(deals)),
        "n_unique_sold_ids": int(len(sold_ids)),
        "n_rows_labeled_sold": int(deals["SoldLabel"].sum()),
        "overall_sold_rate": float(deals["SoldLabel"].mean()),
        "mean_score_sold": float(deals.loc[deals["SoldLabel"] == 1, args.score_col].mean()) if (deals["SoldLabel"] == 1).any() else None,
        "mean_score_unsold": float(deals.loc[deals["SoldLabel"] == 0, args.score_col].mean()) if (deals["SoldLabel"] == 0).any() else None,
        "best_precision_at_k": best_topk,
        "output_files": sorted([p.name for p in out_dir.iterdir()]),
    }
    with open(out_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Done.")
    print(f"Labeled file: {labeled_path}")
    print(f"Overall sold rate: {report['overall_sold_rate']:.4f}")
    if report["mean_score_sold"] is not None and report["mean_score_unsold"] is not None:
        print(f"Mean DealScore sold:   {report['mean_score_sold']:.4f}")
        print(f"Mean DealScore unsold: {report['mean_score_unsold']:.4f}")
    if best_topk:
        print(f"Best Precision@K row: {best_topk}")


if __name__ == "__main__":
    main()
