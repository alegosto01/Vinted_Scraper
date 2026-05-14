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

from analysis_pipeline.evaluation.evaluate_buy_decisions import add_sold_labels


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a balanced sold-vs-unsold dataset for buy-decision evaluation.")
    ap.add_argument("--input", required=True, help="Path to source CSV, typically deals_ranked.csv")
    ap.add_argument("--sold", required=True, help="Path to sold_df.csv")
    ap.add_argument("--sold_eventually", default=None, help="Optional path to sold_eventually.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--id_col", default="Dataid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_dedupe", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(args.input)
    sold = pd.read_csv(args.sold)
    sold_eventually = pd.read_csv(args.sold_eventually) if args.sold_eventually else pd.DataFrame(columns=[args.id_col])

    labeled = add_sold_labels(source, sold, sold_eventually, args.id_col, args.no_dedupe)
    positives = labeled[labeled["SoldLabel"] == 1].copy()
    negatives = labeled[labeled["SoldLabel"] == 0].copy()
    sampled_negatives = negatives.sample(n=min(len(positives), len(negatives)), random_state=args.seed) if len(positives) else negatives.head(0)

    balanced = pd.concat([positives, sampled_negatives], ignore_index=True)
    if "DealScore" in balanced.columns:
        balanced = balanced.sort_values("DealScore", ascending=False)
    balanced = balanced.reset_index(drop=True)

    balanced_path = out_dir / "balanced_buy_eval.csv"
    balanced.to_csv(balanced_path, index=False)

    summary = {
        "input_csv": args.input,
        "n_source_rows": int(len(source)),
        "n_positive_rows": int(len(positives)),
        "n_negative_rows": int(len(negatives)),
        "n_sampled_negatives": int(len(sampled_negatives)),
        "n_balanced_rows": int(len(balanced)),
        "balanced_csv": str(balanced_path),
    }
    (out_dir / "balanced_buy_eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
