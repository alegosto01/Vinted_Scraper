#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
ANALYSIS_DIR = SCRIPTS_DIR / "analysis_pipeline"
for path in (SCRIPTS_DIR, ANALYSIS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd

from analysis_pipeline.evaluation.evaluate_deal_score import dedupe_listings, normalize_id_series


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a balanced raw dataset using all sold rows plus an equal number of unsold rows.")
    ap.add_argument("--big_raw", required=True)
    ap.add_argument("--sold", required=True)
    ap.add_argument("--sold_eventually", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--id_col", default="Dataid")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    big_raw = pd.read_csv(args.big_raw)
    sold = pd.read_csv(args.sold)
    sold_eventually = pd.read_csv(args.sold_eventually) if args.sold_eventually else pd.DataFrame(columns=[args.id_col])

    for df in (big_raw, sold, sold_eventually):
        if args.id_col in df.columns:
            df[args.id_col] = normalize_id_series(df[args.id_col])

    big_raw = dedupe_listings(big_raw, args.id_col)
    sold = dedupe_listings(sold, args.id_col)
    if not sold_eventually.empty:
        sold_eventually = dedupe_listings(sold_eventually, args.id_col)

    sold_ids = set(sold[args.id_col].dropna().tolist())
    sold_eventually_ids = set(sold_eventually[args.id_col].dropna().tolist())
    positive_ids = sold_ids | sold_eventually_ids

    positives = big_raw[big_raw[args.id_col].isin(positive_ids)].copy()
    negatives_pool = big_raw[~big_raw[args.id_col].isin(positive_ids)].copy()
    negatives = negatives_pool.sample(n=min(len(positives), len(negatives_pool)), random_state=args.seed)

    positives["BalancedLabel"] = "sold"
    negatives["BalancedLabel"] = "unsold"
    balanced = pd.concat([positives, negatives], ignore_index=True)

    if "SearchDate" in balanced.columns:
        balanced = balanced.sort_values("SearchDate", ascending=False)
    balanced = balanced.reset_index(drop=True)
    balanced["SoldLabel"] = (balanced["BalancedLabel"] == "sold").astype(int)

    out_csv = out_dir / "balanced_raw.csv"
    balanced.to_csv(out_csv, index=False)

    summary = {
        "big_raw_csv": args.big_raw,
        "sold_csv": args.sold,
        "n_big_raw_rows": int(len(big_raw)),
        "n_positive_rows": int(len(positives)),
        "n_negative_pool_rows": int(len(negatives_pool)),
        "n_sampled_negatives": int(len(negatives)),
        "n_balanced_rows": int(len(balanced)),
        "balanced_csv": str(out_csv),
    }
    (out_dir / "balanced_raw_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
