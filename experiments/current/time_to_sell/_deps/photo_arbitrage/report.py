#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.current.time_to_sell._deps.photo_arbitrage.modeling import TRAINING_LABELS, normalize_label
from experiments.current.time_to_sell._deps.photo_arbitrage.paths import (
    FEATURES_DIR,
    LABELS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    ensure_experiment_dirs,
    ensure_project_imports,
    read_json,
    write_json,
    write_manifest,
)


METHOD_SCORE_COLUMNS = {
    "simple": "SimpleBadPhotoScore",
    "pyiqa": "PyiqaBadPhotoScore",
    "aesthetic": "AestheticBadPhotoScore",
    "fashionclip": "FashionClipBadPhotoScore",
    "dino": "DinoOutlierScore",
    "combined": "CombinedBadPhotoScore",
}


def safe_metric(value: object, digits: int = 4) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not np.isfinite(number):
        return None
    return round(number, digits)


def safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return safe_metric(numerator / denominator)


def binary_auc(labels: pd.Series, scores: pd.Series) -> float | None:
    work = pd.DataFrame({"label": labels, "score": scores}).dropna()
    if work.empty:
        return None
    positives = int((work["label"] == 1).sum())
    negatives = int((work["label"] == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = work["score"].rank(method="average")
    positive_rank_sum = float(ranks[work["label"] == 1].sum())
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return safe_metric(auc)


def fashionclip_pseudo_agreement(review: pd.DataFrame) -> dict[str, object]:
    if review.empty:
        return {"review_rows": 0, "manual_reviewed_rows": 0, "pending_manual_rows": 0}
    manual = review.get("manual_label", pd.Series(dtype=str)).reindex(review.index, fill_value="").map(normalize_label)
    pseudo = review.get("FashionClipPseudoLabel", pd.Series(dtype=str)).reindex(review.index, fill_value="").map(normalize_label)
    reviewed = manual.isin(TRAINING_LABELS)
    comparable = reviewed & pseudo.isin(TRAINING_LABELS)
    result: dict[str, object] = {
        "review_rows": int(len(review)),
        "manual_reviewed_rows": int(reviewed.sum()),
        "pending_manual_rows": int((manual == "").sum()),
        "comparable_rows": int(comparable.sum()),
    }
    if not bool(comparable.any()):
        return result
    manual_bad = manual[comparable].map(TRAINING_LABELS).astype(int)
    pseudo_bad = pseudo[comparable].map(TRAINING_LABELS).astype(int)
    tp = int(((pseudo_bad == 1) & (manual_bad == 1)).sum())
    fp = int(((pseudo_bad == 1) & (manual_bad == 0)).sum())
    tn = int(((pseudo_bad == 0) & (manual_bad == 0)).sum())
    fn = int(((pseudo_bad == 0) & (manual_bad == 1)).sum())
    total = int(len(manual_bad))
    result.update(
        {
            "agreement": safe_divide(tp + tn, total),
            "bad_precision": safe_divide(tp, tp + fp),
            "bad_recall": safe_divide(tp, tp + fn),
            "good_precision": safe_divide(tn, tn + fn),
            "good_recall": safe_divide(tn, tn + fp),
            "mismatches": int(fp + fn),
            "true_bad": tp,
            "false_bad": fp,
            "true_good": tn,
            "false_good": fn,
        }
    )
    return result


def quality_method_score_metrics(review: pd.DataFrame) -> pd.DataFrame:
    if review.empty or "manual_label" not in review.columns:
        return pd.DataFrame()
    manual = review["manual_label"].map(normalize_label)
    usable = manual.isin(TRAINING_LABELS)
    if not bool(usable.any()):
        return pd.DataFrame()
    target = manual.map(TRAINING_LABELS)
    rows = []
    for method, column in METHOD_SCORE_COLUMNS.items():
        if column not in review.columns:
            continue
        scores = pd.to_numeric(review[column], errors="coerce")
        valid = usable & scores.notna()
        if not bool(valid.any()):
            continue
        y = target[valid].astype(int)
        s = scores[valid].astype(float)
        bad_scores = s[y == 1]
        good_scores = s[y == 0]
        predicted_bad = s >= 0.5
        tp = int(((predicted_bad == 1) & (y == 1)).sum())
        fp = int(((predicted_bad == 1) & (y == 0)).sum())
        tn = int(((predicted_bad == 0) & (y == 0)).sum())
        fn = int(((predicted_bad == 0) & (y == 1)).sum())
        rows.append(
            {
                "method": method,
                "score_column": column,
                "rows": int(len(s)),
                "bad_rows": int((y == 1).sum()),
                "good_rows": int((y == 0).sum()),
                "auc_bad_vs_good": binary_auc(y, s),
                "mean_bad_score": safe_metric(bad_scores.mean()),
                "mean_good_score": safe_metric(good_scores.mean()),
                "mean_delta_bad_minus_good": safe_metric(bad_scores.mean() - good_scores.mean()),
                "threshold_0_5_accuracy": safe_divide(tp + tn, len(s)),
                "threshold_0_5_bad_precision": safe_divide(tp, tp + fp),
                "threshold_0_5_bad_recall": safe_divide(tp, tp + fn),
            }
        )
    return pd.DataFrame(rows)


def fashionclip_failure_examples(review: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    if review.empty or "manual_label" not in review.columns or "FashionClipPseudoLabel" not in review.columns:
        return pd.DataFrame()
    work = review.copy()
    work["manual_label"] = work["manual_label"].map(normalize_label)
    work["FashionClipPseudoLabel"] = work["FashionClipPseudoLabel"].map(normalize_label)
    comparable = work["manual_label"].isin(TRAINING_LABELS) & work["FashionClipPseudoLabel"].isin(TRAINING_LABELS)
    mismatches = work[comparable & (work["manual_label"] != work["FashionClipPseudoLabel"])].copy()
    if mismatches.empty:
        return mismatches
    if "FashionClipPseudoConfidence" in mismatches.columns:
        mismatches["_confidence"] = pd.to_numeric(mismatches["FashionClipPseudoConfidence"], errors="coerce")
        mismatches = mismatches.sort_values("_confidence", ascending=False, kind="stable")
    cols = [
        "SearchName",
        "Title",
        "Price",
        "manual_label",
        "FashionClipPseudoLabel",
        "FashionClipPseudoConfidence",
        "FashionClipGoodScore",
        "FashionClipBadScore",
        "CombinedBadPhotoScore",
        "manual_notes",
    ]
    return mismatches[[col for col in cols if col in mismatches.columns]].head(int(limit)).copy()


def usefulness_sentence(metrics: dict[str, object]) -> str:
    agreement = metrics.get("agreement")
    comparable = int(metrics.get("comparable_rows") or 0)
    if comparable == 0:
        return "Manual validation pending; FashionCLIP usefulness cannot be judged yet."
    if agreement is None:
        return "Manual validation is present, but agreement could not be computed."
    score = float(agreement)
    if score >= 0.8:
        return "FashionCLIP pseudo-labels look useful as a seed signal, pending a larger held-out test."
    if score >= 0.65:
        return "FashionCLIP pseudo-labels look mixed: useful for triage, risky as automatic truth."
    return "FashionCLIP pseudo-labels look weak against manual labels; use them only for exploration."


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            text = str(row.get(col, "")).replace("\n", " ").replace("|", "/")
            if len(text) > 80:
                text = text[:77] + "..."
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a compact photo-improvement experiment report.")
    parser.add_argument("--scores", default=str(FEATURES_DIR / "latest_scored_candidates.csv"))
    parser.add_argument("--labels", default=str(LABELS_DIR / "photo_quality_label_sheet.csv"))
    parser.add_argument("--fashionclip-review", default=str(REPORTS_DIR / "fashionclip_pseudo_label_review_queue.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_experiment_dirs()
    scores_path = Path(args.scores)
    labels_path = Path(args.labels)
    fashionclip_review_path = Path(args.fashionclip_review)
    scored = pd.read_csv(scores_path, low_memory=False) if scores_path.exists() else pd.DataFrame()
    labels = pd.read_csv(labels_path, low_memory=False) if labels_path.exists() else pd.DataFrame()
    fashionclip_review = pd.read_csv(fashionclip_review_path, low_memory=False) if fashionclip_review_path.exists() else pd.DataFrame()
    metadata = read_json(MODELS_DIR / "photo_quality_v1_latest_metadata.json", default={})
    fashionclip_metrics = fashionclip_pseudo_agreement(fashionclip_review)
    method_metrics = quality_method_score_metrics(fashionclip_review)
    failure_examples = fashionclip_failure_examples(fashionclip_review)

    summary = {
        "scores_path": str(scores_path) if scores_path.exists() else "",
        "labels_path": str(labels_path) if labels_path.exists() else "",
        "fashionclip_review_path": str(fashionclip_review_path) if fashionclip_review_path.exists() else "",
        "scored_rows": int(len(scored)),
        "label_rows": int(len(labels)),
        "fashionclip_review_rows": int(len(fashionclip_review)),
        "model_status": metadata.get("status", "missing"),
        "model_version": metadata.get("model_version", "heuristic_v0"),
        "fashionclip_pseudo_agreement": fashionclip_metrics,
    }
    if not scored.empty:
        summary["top_searches"] = scored["SearchName"].value_counts(dropna=False).head(10).to_dict() if "SearchName" in scored else {}
        summary["mean_bad_photo_probability"] = float(pd.to_numeric(scored.get("BadPhotoProbability"), errors="coerce").mean())
        summary["mean_photo_opportunity_score"] = float(pd.to_numeric(scored.get("PhotoOpportunityScore"), errors="coerce").mean())
    if not labels.empty and "manual_label" in labels:
        summary["label_counts"] = labels["manual_label"].map(normalize_label).value_counts(dropna=False).to_dict()
    if not fashionclip_review.empty:
        summary["fashionclip_pseudo_label_counts"] = (
            fashionclip_review.get("FashionClipPseudoLabel", pd.Series(dtype=str)).map(normalize_label).value_counts(dropna=False).to_dict()
        )
        summary["fashionclip_manual_label_counts"] = (
            fashionclip_review.get("manual_label", pd.Series(dtype=str)).map(normalize_label).value_counts(dropna=False).to_dict()
        )
    if not method_metrics.empty:
        summary["quality_method_metrics"] = method_metrics.to_dict(orient="records")

    lines = [
        "# Photo-Improvement Arbitrage Report",
        "",
        f"- Scored rows: {summary['scored_rows']}",
        f"- Label rows: {summary['label_rows']}",
        f"- Model version: {summary['model_version']}",
        f"- Model status: {summary['model_status']}",
    ]
    if "mean_bad_photo_probability" in summary:
        lines.extend(
            [
                f"- Mean bad-photo probability: {summary['mean_bad_photo_probability']:.4f}",
                f"- Mean opportunity score: {summary['mean_photo_opportunity_score']:.4f}",
            ]
        )
    if summary.get("label_counts"):
        lines.append(f"- Label counts: {summary['label_counts']}")
    if summary.get("top_searches"):
        lines.append(f"- Top searches in scored rows: {summary['top_searches']}")
    lines.extend(
        [
            "",
            "## FashionCLIP Pseudo-Label Review",
            "",
            f"- Review rows: {fashionclip_metrics.get('review_rows', 0)}",
            f"- Manual reviewed rows: {fashionclip_metrics.get('manual_reviewed_rows', 0)}",
            f"- Comparable good/bad rows: {fashionclip_metrics.get('comparable_rows', 0)}",
            f"- Pending manual rows: {fashionclip_metrics.get('pending_manual_rows', 0)}",
            f"- Usefulness read: {usefulness_sentence(fashionclip_metrics)}",
        ]
    )
    if summary.get("fashionclip_pseudo_label_counts"):
        lines.append(f"- FashionCLIP pseudo label counts: {summary['fashionclip_pseudo_label_counts']}")
    if summary.get("fashionclip_manual_label_counts"):
        lines.append(f"- Manual label counts in FashionCLIP queue: {summary['fashionclip_manual_label_counts']}")
    if fashionclip_metrics.get("agreement") is not None:
        lines.extend(
            [
                f"- Pseudo-label agreement: {fashionclip_metrics['agreement']}",
                f"- Bad-photo precision: {fashionclip_metrics.get('bad_precision')}",
                f"- Bad-photo recall: {fashionclip_metrics.get('bad_recall')}",
                f"- Pseudo/manual mismatches: {fashionclip_metrics.get('mismatches')}",
            ]
        )
    if not method_metrics.empty:
        lines.extend(["", "## Method Comparison Against Manual Review", "", markdown_table(method_metrics)])
    if not failure_examples.empty:
        lines.extend(["", "## FashionCLIP Failure Examples", "", markdown_table(failure_examples)])
    lines.extend(
        [
            "",
            "## Generalization Gap",
            "",
            "Current evidence is Vinted/fashion listing data only. Before using this as a non-fashion photo-quality model, collect a separate manually labeled holdout set from non-fashion marketplace categories and report the same pseudo-label agreement and method metrics there.",
        ]
    )
    if not scored.empty:
        lines.extend(["", "## Top Review Candidates", ""])
        preview_cols = [col for col in ["SearchName", "Title", "Price", "Brand", "BadPhotoProbability", "PhotoOpportunityScore", "PhotoOpportunityNotes"] if col in scored.columns]
        preview = markdown_table(scored.head(20)[preview_cols])
        lines.append(preview)

    report_path = REPORTS_DIR / "photo_arbitrage_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(REPORTS_DIR / "photo_arbitrage_summary.json", summary)
    write_manifest(
        REPORTS_DIR / "photo_arbitrage_report_manifest.json",
        command=" ".join(sys.argv),
        extra={"report_path": str(report_path), "summary": summary},
    )
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
