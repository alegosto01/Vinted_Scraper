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


PIPELINE_COLUMNS = [
    "ProductId",
    "VariantId",
    "VariantCount",
    "VariantPriceMedian",
    "VariantPriceMAD",
    "VariantPriceMADRatio",
    "ProductVariantSilhouette",
    "VariantCentroidSim",
    "DealScoreRaw",
    "DealConfidence",
    "DealEligible",
    "DealScore",
    "DealNotes",
    "ExpectedResalePrice",
    "ExpectedNetProceeds",
    "ExpectedProfit",
    "ExpectedProfitMargin",
    "ResaleSafetyScore",
    "VariantClusterSize",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare a balanced evaluation input by merging raw rows, local image paths, and pipeline scores.")
    ap.add_argument("--balanced_raw", required=True)
    ap.add_argument("--local_images_csv", required=True)
    ap.add_argument("--pipeline_items", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--id_col", default="Dataid")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.local_images_csv)
    pipeline = pd.read_csv(args.pipeline_items)

    raw[args.id_col] = normalize_id_series(raw[args.id_col])
    pipeline[args.id_col] = normalize_id_series(pipeline[args.id_col])
    pipeline = dedupe_listings(pipeline, args.id_col)

    keep = [args.id_col] + [col for col in PIPELINE_COLUMNS if col in pipeline.columns]
    merged = raw.merge(pipeline[keep], on=args.id_col, how="left")

    out_csv = out_dir / "balanced_visual_eval_input.csv"
    merged.to_csv(out_csv, index=False)

    summary = {
        "balanced_raw_csv": args.balanced_raw,
        "local_images_csv": args.local_images_csv,
        "pipeline_items_csv": args.pipeline_items,
        "n_rows": int(len(merged)),
        "n_rows_with_pipeline_metrics": int(merged["DealScore"].notna().sum()) if "DealScore" in merged.columns else 0,
        "output_csv": str(out_csv),
    }
    (out_dir / "balanced_visual_eval_input_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
