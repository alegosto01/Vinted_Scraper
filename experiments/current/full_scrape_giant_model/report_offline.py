from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.current.full_scrape_giant_model.paths import REPORTS_DIR, assert_experiment_path


def _fmt(v, d=3) -> str:
    """Format value as string; return '-' for None/NaN, otherwise float with d decimals."""
    if v is None or pd.isna(v):
        return "-"
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return str(v)


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Generate markdown table lines: header row, separator, data rows."""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_offline_report(run_dir: Path) -> Path:
    """
    Generate offline evaluation report for full_scrape_giant_model.

    Reads CSVs from run_dir, builds markdown document, writes to:
    - run_dir / "offline_report.md"
    - REPORTS_DIR / "{run_dir.name}_offline_report.md"

    Returns path to run_dir/offline_report.md.
    """
    run_dir = Path(run_dir)

    # Load CSVs; treat missing files as empty DataFrames
    metrics_long_path = run_dir / "metrics_long.csv"
    metrics_long = pd.read_csv(metrics_long_path) if metrics_long_path.exists() else pd.DataFrame()

    per_search_metrics_path = run_dir / "per_search_metrics.csv"
    per_search_metrics = pd.read_csv(per_search_metrics_path) if per_search_metrics_path.exists() else pd.DataFrame()

    image_filter_summary_path = run_dir / "image_filter_summary.csv"
    image_filter_summary = pd.read_csv(image_filter_summary_path) if image_filter_summary_path.exists() else pd.DataFrame()

    dataset_summary_path = run_dir / "dataset_summary.csv"
    dataset_summary = pd.read_csv(dataset_summary_path) if dataset_summary_path.exists() else pd.DataFrame()

    # Build markdown document
    lines = []

    # Section 1: Title and run folder
    lines.append("# Full-Scrape Giant Model — Offline Report")
    lines.append(f"Run folder: `{run_dir}`")
    lines.append("")

    # Section 2: Main-Image Filtering
    lines.append("## Main-Image Filtering")
    if not image_filter_summary.empty:
        headers = list(image_filter_summary.columns)
        rows = []
        for _, row in image_filter_summary.iterrows():
            rows.append([_fmt(v, d=0) if isinstance(v, (int, float)) else str(v) for v in row])
        lines.extend(_table(headers, rows))

        # Calculate totals using __TOTAL__ row if present
        total_row = image_filter_summary[image_filter_summary["search_name"] == "__TOTAL__"]
        if not total_row.empty:
            before = int(total_row.iloc[0]["total_rows"])
            after = int(total_row.iloc[0]["kept_rows"])
            lines.append(f"Total: {before} rows before filtering → {after} rows after filtering.")
        lines.append("")
    else:
        lines.append("_(no data)_")
        lines.append("")

    # Section 3: Dataset Summary
    lines.append("## Dataset Summary (after filtering)")
    if not dataset_summary.empty:
        headers = ["SearchName", "rows", "positive_rows", "base_rate"]
        rows = []
        for _, row in dataset_summary.iterrows():
            rows.append([
                str(row.get("SearchName", "-")),
                _fmt(row.get("rows", None), d=0),
                _fmt(row.get("positive_rows", None), d=0),
                _fmt(row.get("base_rate", None), d=3),
            ])
        lines.extend(_table(headers, rows))
        lines.append("")
    else:
        lines.append("_(no data)_")
        lines.append("")

    # Section 4: Model Metrics (held-out test)
    lines.append("## Model Metrics (held-out test)")
    if not metrics_long.empty:
        trained = metrics_long[metrics_long["status"] == "trained"].sort_values("test_roc_auc", ascending=False)
        if not trained.empty:
            headers = [
                "approach", "origin", "AUC", "PR-AUC", "precision", "recall",
                "coverage", "selected", "P@10", "P@25", "P@50", "lift", "Brier", "ECE", "fit_s"
            ]
            rows = []
            for _, row in trained.iterrows():
                rows.append([
                    str(row.get("approach", "-")),
                    str(row.get("origin", "-")),
                    _fmt(row.get("test_roc_auc", None)),
                    _fmt(row.get("test_pr_auc", None)),
                    _fmt(row.get("test_precision", None)),
                    _fmt(row.get("test_recall", None)),
                    _fmt(row.get("test_coverage", None)),
                    _fmt(row.get("test_selected_count", None), d=0),
                    _fmt(row.get("test_precision_at_10", None)),
                    _fmt(row.get("test_precision_at_25", None)),
                    _fmt(row.get("test_precision_at_50", None)),
                    _fmt(row.get("test_lift", None)),
                    _fmt(row.get("brier", None)),
                    _fmt(row.get("ece", None)),
                    _fmt(row.get("fit_seconds", None), d=0),
                ])
            lines.extend(_table(headers, rows))
            lines.append("")
        else:
            lines.append("_(no trained models)_")
            lines.append("")
    else:
        lines.append("_(no data)_")
        lines.append("")

    # Section 5: Best Model Overall
    lines.append("## Best Model Overall")
    if not metrics_long.empty:
        trained = metrics_long[metrics_long["status"] == "trained"]
        auc_ok = trained.dropna(subset=["test_roc_auc"]) if "test_roc_auc" in trained else trained.iloc[0:0]
        if not auc_ok.empty:
            best_auc_row = auc_ok.loc[auc_ok["test_roc_auc"].idxmax()]
            lines.append(f"- Best by AUC: {best_auc_row.get('approach', '-')} ({_fmt(best_auc_row.get('test_roc_auc'))})")
        p25_ok = trained.dropna(subset=["test_precision_at_25"]) if "test_precision_at_25" in trained else trained.iloc[0:0]
        if not p25_ok.empty:
            best_p25_row = p25_ok.loc[p25_ok["test_precision_at_25"].idxmax()]
            lines.append(f"- Best by P@25: {best_p25_row.get('approach', '-')} ({_fmt(best_p25_row.get('test_precision_at_25'))})")
        lines.append("")

    # Section 6: Best Model Per Search
    lines.append("## Best Model Per Search")
    ps_ok = per_search_metrics.dropna(subset=["roc_auc"]) if "roc_auc" in per_search_metrics else per_search_metrics.iloc[0:0]
    if not ps_ok.empty:
        best_idx = ps_ok.groupby("search_name")["roc_auc"].idxmax()
        best_per_search = ps_ok.loc[best_idx].sort_values("search_name")

        headers = ["search", "best approach", "AUC", "PR-AUC", "precision", "P@25", "lift", "base_rate"]
        rows = []
        for _, row in best_per_search.iterrows():
            rows.append([
                str(row.get("search_name", "-")),
                str(row.get("approach", "-")),
                _fmt(row.get("roc_auc", None)),
                _fmt(row.get("pr_auc", None)),
                _fmt(row.get("precision", None)),
                _fmt(row.get("precision_at_25", None)),
                _fmt(row.get("lift", None)),
                _fmt(row.get("base_rate", None)),
            ])
        lines.extend(_table(headers, rows))
        lines.append("")
    else:
        lines.append("_(no data)_")
        lines.append("")

    # Section 7: Errors
    lines.append("## Errors")
    if not metrics_long.empty:
        errors = metrics_long[metrics_long["status"] != "trained"]
        if not errors.empty:
            for _, row in errors.iterrows():
                approach = str(row.get("approach", "-"))
                reason = str(row.get("reason", "unknown error")) if "reason" in row and pd.notna(row["reason"]) else "unknown error"
                lines.append(f"- {approach}: {reason}")
        else:
            lines.append("_(no errors)_")
        lines.append("")

    # Section 8: Notes
    lines.append("## Notes")
    lines.append("- Telegram policy was NOT changed: this is offline training/evaluation only; no Telegram messages were sent and no Telegram code was modified.")
    lines.append("- Threshold selected globally on the validation split (>=80% precision, >=20 selected).")
    lines.append("")

    markdown_content = "\n".join(lines)

    # Write to run_dir/offline_report.md
    report_path_1 = run_dir / "offline_report.md"
    report_path_1.parent.mkdir(parents=True, exist_ok=True)
    assert_experiment_path(report_path_1)
    report_path_1.write_text(markdown_content)

    # Write to REPORTS_DIR/{run_dir.name}_offline_report.md
    report_path_2 = REPORTS_DIR / f"{run_dir.name}_offline_report.md"
    report_path_2.parent.mkdir(parents=True, exist_ok=True)
    assert_experiment_path(report_path_2)
    report_path_2.write_text(markdown_content)

    return report_path_1
