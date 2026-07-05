#!/usr/bin/env python3
"""Compute per-search thresholds from a live scoring run's matured labels.

Usage:
    python calibrate_thresholds.py \
        --live-scored <path/to/live_scored_items.csv> \
        --model-dir   <path/to/live_trained_YYYYMMDD_HHMMSS>

Reads the scored+labeled CSV, finds the best per-search threshold meeting
MIN_PRECISION at MIN_COUNT, writes per_search_thresholds.json into the
model directory. The apply script loads this file automatically if present.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCORE_COL = "score__main_image_scores_live_trained"
DEFAULT_MIN_PRECISION = 0.60
DEFAULT_MIN_COUNT = 3
LABEL_WINDOW = "72h"
LABEL_COL = f"sold_within_{LABEL_WINDOW}"
EVAL_COL = f"evaluated_at_{LABEL_WINDOW}"


def calibrate_search(sdf: pd.DataFrame, global_thresh: float, min_precision: float, min_count: int) -> dict:
    evaluated = sdf[sdf[EVAL_COL].notna()].copy()
    if evaluated.empty:
        return {"threshold": global_thresh, "source": "global_fallback_no_evaluated", "precision": None, "count": 0}
    scores = pd.to_numeric(evaluated[SCORE_COL], errors="coerce").fillna(0.0)
    labels = pd.to_numeric(evaluated[LABEL_COL], errors="coerce").fillna(0).astype(int)
    candidates = sorted(set(np.round(scores.values, 4).tolist()), reverse=True)

    best_calibrated = None
    for t in candidates:
        mask = scores >= t
        cnt = int(mask.sum())
        if cnt < min_count:
            continue
        prec = float(labels[mask].sum() / cnt)
        if prec >= min_precision:
            if best_calibrated is None or prec > best_calibrated["precision"] or (
                prec == best_calibrated["precision"] and cnt > best_calibrated["count"]
            ):
                best_calibrated = {"threshold": float(t), "precision": round(prec, 4), "count": cnt}
    if best_calibrated:
        best_calibrated["source"] = "calibrated_72h"
        return best_calibrated

    best_fallback = None
    for t in candidates:
        mask = scores >= t
        cnt = int(mask.sum())
        if cnt < min_count:
            continue
        prec = float(labels[mask].sum() / cnt)
        if best_fallback is None or prec > best_fallback["precision"]:
            best_fallback = {"threshold": float(t), "precision": round(prec, 4), "count": cnt}
    if best_fallback:
        best_fallback["source"] = "best_achievable_72h"
        return best_fallback

    return {"threshold": global_thresh, "source": "global_fallback_insufficient_data", "precision": None, "count": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-scored", type=Path, required=True, help="live_scored_items.csv from an apply run")
    parser.add_argument("--model-dir", type=Path, required=True, help="live_trained_* model directory")
    parser.add_argument("--min-precision", type=float, default=DEFAULT_MIN_PRECISION)
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    args = parser.parse_args()

    results_path = args.model_dir / "results.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found", file=sys.stderr)
        sys.exit(1)
    global_thresh = float(json.loads(results_path.read_text())["results"]["main_image_scores"]["threshold"])

    df = pd.read_csv(args.live_scored, low_memory=False)
    if SCORE_COL not in df.columns:
        print(f"ERROR: {SCORE_COL} not in {args.live_scored}", file=sys.stderr)
        sys.exit(1)

    print(f"Calibrating per-search thresholds from {args.live_scored}")
    print(f"Label: {LABEL_COL}, global_fallback={global_thresh:.4f}, min_precision={args.min_precision}, min_count={args.min_count}\n")

    results: dict[str, dict] = {}
    for search, g in sorted(df.groupby("SearchName")):
        r = calibrate_search(g, global_thresh, args.min_precision, args.min_count)
        results[str(search)] = r
        prec_str = f"{r['precision']:.4f}" if r["precision"] is not None else "N/A"
        print(f"  {search:<32} thresh={r['threshold']:.4f}  prec={prec_str}  n={r['count']}  [{r['source']}]")

    out = args.model_dir / "per_search_thresholds.json"
    out.write_text(
        json.dumps({
            "model_dir": str(args.model_dir),
            "label_window": LABEL_WINDOW,
            "min_precision": args.min_precision,
            "min_count": args.min_count,
            "global_threshold": global_thresh,
            "calibration_source": str(args.live_scored),
            "thresholds": results,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
