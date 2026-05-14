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

from analysis_pipeline.scoring.final_buy_filter import (
    apply_visual_rerank,
    compute_buy_decision,
    parse_named_float_map,
    resolve_component_weights,
    resolve_min_buy_score,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Apply visual reranking to an existing buy_candidates_enriched.csv file.")
    ap.add_argument("--input", required=True, help="Path to an existing buy_candidates_enriched.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--min_buy_score", type=float, default=0.75)
    ap.add_argument("--min_seller_score", type=float, default=0.45)
    ap.add_argument("--category_min_buy_scores", default="")
    ap.add_argument("--component_weights", default="")
    ap.add_argument("--visual_penalty_scale", type=float, default=1.0)
    ap.add_argument("--visual_max_images", type=int, default=6)
    ap.add_argument("--visual_main_image_weight", type=float, default=0.55)
    ap.add_argument("--visual_timeout", type=float, default=8.0)
    ap.add_argument("--visual_enable_clip", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    category_thresholds = parse_named_float_map(args.category_min_buy_scores)
    component_weights = resolve_component_weights(args.component_weights)

    enriched = pd.read_csv(args.input)
    reranked = apply_visual_rerank(enriched, args)
    scores = reranked.apply(
        lambda row: compute_buy_decision(
            row.to_dict(),
            resolve_min_buy_score(row, args.min_buy_score, category_thresholds),
            args.min_seller_score,
            component_weights=component_weights,
            visual_penalty_scale=args.visual_penalty_scale,
        ),
        axis=1,
        result_type="expand",
    )
    scores.columns = ["BuyDecisionScore", "WorthBuying", "BuyDecisionNotes"]
    reranked[["BuyDecisionScore", "WorthBuying", "BuyDecisionNotes"]] = scores
    reranked = reranked.sort_values(
        [c for c in ["WorthBuying", "BuyDecisionScore", "ResaleSafetyScore", "ExpectedProfitMargin", "DealScore"] if c in reranked.columns],
        ascending=False,
    )

    reranked_path = out_dir / "buy_candidates_enriched_visual.csv"
    reranked.to_csv(reranked_path, index=False)
    reranked[reranked["WorthBuying"].fillna(False)].to_csv(out_dir / "buy_candidates_recommended_visual.csv", index=False)

    summary = {
        "n_rows": int(len(reranked)),
        "n_worth_buying": int(reranked["WorthBuying"].fillna(False).sum()) if "WorthBuying" in reranked.columns else 0,
        "mean_buy_decision_score": float(reranked["BuyDecisionScore"].mean()) if "BuyDecisionScore" in reranked.columns and not reranked.empty else None,
        "mean_visual_score": float(reranked["VisualScore"].mean()) if "VisualScore" in reranked.columns and not reranked.empty else None,
        "mean_visual_risk_penalty": float(reranked["VisualRiskPenalty"].mean()) if "VisualRiskPenalty" in reranked.columns and not reranked.empty else None,
        "output_file": reranked_path.name,
    }
    (out_dir / "visual_rerank_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
