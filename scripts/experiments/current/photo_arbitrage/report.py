#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.photo_arbitrage.modeling import normalize_label
from experiments.photo_arbitrage.paths import (
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_experiment_dirs()
    scores_path = Path(args.scores)
    labels_path = Path(args.labels)
    scored = pd.read_csv(scores_path, low_memory=False) if scores_path.exists() else pd.DataFrame()
    labels = pd.read_csv(labels_path, low_memory=False) if labels_path.exists() else pd.DataFrame()
    metadata = read_json(MODELS_DIR / "photo_quality_v1_latest_metadata.json", default={})

    summary = {
        "scores_path": str(scores_path) if scores_path.exists() else "",
        "labels_path": str(labels_path) if labels_path.exists() else "",
        "scored_rows": int(len(scored)),
        "label_rows": int(len(labels)),
        "model_status": metadata.get("status", "missing"),
        "model_version": metadata.get("model_version", "heuristic_v0"),
    }
    if not scored.empty:
        summary["top_searches"] = scored["SearchName"].value_counts(dropna=False).head(10).to_dict() if "SearchName" in scored else {}
        summary["mean_bad_photo_probability"] = float(pd.to_numeric(scored.get("BadPhotoProbability"), errors="coerce").mean())
        summary["mean_photo_opportunity_score"] = float(pd.to_numeric(scored.get("PhotoOpportunityScore"), errors="coerce").mean())
    if not labels.empty and "manual_label" in labels:
        summary["label_counts"] = labels["manual_label"].map(normalize_label).value_counts(dropna=False).to_dict()

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
