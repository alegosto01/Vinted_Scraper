#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
ANALYSIS_DIR = SCRIPTS_DIR / "analysis_pipeline"
for path in (SCRIPTS_DIR, ANALYSIS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd

from analysis_pipeline.evaluation.evaluate_buy_decisions import add_sold_labels
from analysis_pipeline.scoring.final_buy_filter import DEFAULT_COMPONENT_WEIGHTS, extract_buy_components


SEARCHES = ("ps4", "gucci", "prada")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tune buy thresholds and lightweight score weights on full-enrichment visual-scored data.")
    ap.add_argument("--trials", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_seller_score", type=float, default=0.45)
    ap.add_argument("--out_dir", default="data/simple_scrape/tuning_reports/full_enrichment_buy_policy")
    return ap.parse_args()


def safe_float(value, default=0.0) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(num):
        return default
    return num


def load_labeled_rows(min_seller_score: float) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for search in SEARCHES:
        df = pd.read_csv(ROOT / f"data/simple_scrape/{search}/buy_decision_eval/visual_scored/buy_candidates_enriched_visual.csv")
        sold = pd.read_csv(ROOT / f"data/simple_scrape/{search}/sold_df.csv")
        sold_eventually_path = ROOT / f"data/simple_scrape/{search}/eventual_sale_check/sold_eventually.csv"
        sold_eventually = pd.read_csv(sold_eventually_path) if sold_eventually_path.exists() else pd.DataFrame(columns=["Dataid"])
        labeled = add_sold_labels(df, sold, sold_eventually, "Dataid", False)
        labeled["Search"] = search
        parts.append(labeled)

    all_rows = pd.concat(parts, ignore_index=True)
    component_rows = []
    for row in all_rows.to_dict(orient="records"):
        parts_map = extract_buy_components(row, min_seller_score=min_seller_score)
        component_rows.append(
            {
                "Search": row["Search"],
                "SoldLabel": int(row["SoldLabel"]),
                "Title": row.get("Title", ""),
                "BuyDecisionScore": safe_float(row.get("BuyDecisionScore"), 0.0),
                "resale": safe_float(parts_map["resale"]),
                "profit": safe_float(parts_map["profit"]),
                "margin": safe_float(parts_map["margin"]),
                "seller": safe_float(parts_map["seller"]),
                "demand": safe_float(parts_map["demand"]),
                "fresh": safe_float(parts_map["fresh"]),
                "condition": safe_float(parts_map["condition"]),
                "visual": safe_float(parts_map["visual"]),
                "visual_penalty": safe_float(parts_map["visual_penalty"]),
                "rule_penalty": safe_float(parts_map["rule_penalty"]),
                "hard_flag": bool(parts_map["hard_flags"]),
            }
        )
    return pd.DataFrame(component_rows)


def evaluate_policy(df: pd.DataFrame, scores: np.ndarray, thresholds: dict[str, float]) -> dict:
    threshold_series = df["Search"].map(thresholds).fillna(max(thresholds.values())).to_numpy(dtype=float)
    selected = (scores >= threshold_series) & (~df["hard_flag"].to_numpy(dtype=bool))
    sold = df["SoldLabel"].to_numpy(dtype=int)

    tp = int(((selected) & (sold == 1)).sum())
    fp = int(((selected) & (sold == 0)).sum())
    fn = int((~selected & (sold == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "n_buy": int(selected.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
    }


def score_from_weights(df: pd.DataFrame, weights: dict[str, float], visual_penalty_scale: float) -> np.ndarray:
    positive = sum(
        float(weights.get(name, 0.0)) * df[name].to_numpy(dtype=float)
        for name in ("resale", "profit", "margin", "seller", "demand", "fresh", "condition", "visual")
    )
    penalties = df["rule_penalty"].to_numpy(dtype=float) + float(visual_penalty_scale) * df["visual_penalty"].to_numpy(dtype=float)
    return np.clip(positive - penalties, 0.0, 1.0)


def objective(metrics: dict, min_selected: int) -> float:
    precision = metrics["precision"] or 0.0
    recall = metrics["recall"] or 0.0
    n_buy = metrics["n_buy"]
    tp = metrics["true_positives"]
    if n_buy < min_selected or tp == 0:
        return -1e6 + tp * 100 - abs(min_selected - n_buy) * 10
    return precision * 1000 + recall * 100 + tp * 10 - n_buy


def random_weight_config(rng: random.Random) -> tuple[dict[str, float], float]:
    names = list(DEFAULT_COMPONENT_WEIGHTS)
    raw = {}
    total = 0.0
    for name in names:
        base = DEFAULT_COMPONENT_WEIGHTS[name]
        if name == "visual":
            value = rng.uniform(0.0, 0.20)
        else:
            value = base * rng.uniform(0.35, 2.25)
        raw[name] = value
        total += value
    weights = {name: (value / total) for name, value in raw.items()}
    visual_penalty_scale = rng.uniform(0.35, 1.50)
    return weights, visual_penalty_scale


def threshold_only_grid(df: pd.DataFrame) -> dict:
    thresholds = np.arange(0.40, 0.91, 0.05)
    best: dict[str, dict] = {}
    base_scores = df["BuyDecisionScore"].to_numpy(dtype=float)
    for min_selected in (1, 2, 3):
        winner = None
        for ps4_thr in thresholds:
            for gucci_thr in thresholds:
                for prada_thr in thresholds:
                    cfg = {"ps4": float(ps4_thr), "gucci": float(gucci_thr), "prada": float(prada_thr)}
                    metrics = evaluate_policy(df, base_scores, cfg)
                    score = objective(metrics, min_selected)
                    if winner is None or score > winner["objective"]:
                        winner = {"objective": score, "thresholds": cfg, "metrics": metrics}
        best[f"min_selected_{min_selected}"] = winner
    return best


def random_search(df: pd.DataFrame, trials: int, seed: int) -> dict:
    rng = random.Random(seed)
    best: dict[str, dict] = {}
    for min_selected in (1, 2, 3):
        best[f"min_selected_{min_selected}"] = None

    for _ in range(int(trials)):
        weights, visual_penalty_scale = random_weight_config(rng)
        scores = score_from_weights(df, weights, visual_penalty_scale)
        thresholds = {
            "ps4": rng.uniform(0.55, 0.95),
            "gucci": rng.uniform(0.35, 0.90),
            "prada": rng.uniform(0.35, 0.90),
        }
        metrics = evaluate_policy(df, scores, thresholds)
        for min_selected in (1, 2, 3):
            key = f"min_selected_{min_selected}"
            score = objective(metrics, min_selected)
            winner = best[key]
            if winner is None or score > winner["objective"]:
                best[key] = {
                    "objective": score,
                    "thresholds": thresholds,
                    "weights": weights,
                    "visual_penalty_scale": visual_penalty_scale,
                    "metrics": metrics,
                }
    return best


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_labeled_rows(args.min_seller_score)
    report = {
        "n_rows": int(len(df)),
        "n_sold": int(df["SoldLabel"].sum()),
        "search_breakdown": {
            search: {
                "n_rows": int((df["Search"] == search).sum()),
                "n_sold": int(df.loc[df["Search"] == search, "SoldLabel"].sum()),
            }
            for search in SEARCHES
        },
        "threshold_only": threshold_only_grid(df),
        "weight_search": random_search(df, args.trials, args.seed),
    }

    out_path = out_dir / "full_enrichment_buy_policy_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
