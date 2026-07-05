#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.old.basic_plus_visual._deps.deal_finder.modeling import latest_offline_run
from experiments.old.basic_plus_visual._deps.deal_finder.paper_trading import latest_live_run
from experiments.old.basic_plus_visual._deps.deal_finder.paths import REPORTS_DIR, ensure_experiment_dirs, read_json, utc_now_iso, write_manifest


def build_report(out_path: Path | None = None) -> Path:
    ensure_experiment_dirs()
    offline = latest_offline_run()
    live = latest_live_run()
    out_path = out_path or REPORTS_DIR / "latest_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Deal Finder Report", "", f"Generated at: `{utc_now_iso()}`", ""]
    if offline:
        summary = read_json(offline / "metrics_summary.json", {})
        lines += [
            "## Offline",
            "",
            f"Run: `{offline}`",
            f"Models trained: `{summary.get('model_count', 0)}`",
            f"Qualified searches: `{summary.get('qualified_count', 0)}`",
            "",
        ]
        metrics_csv = offline / "metrics_summary.csv"
        if metrics_csv.exists():
            metrics = pd.read_csv(metrics_csv)
            cols = [c for c in ["search_name", "status", "qualified_for_paper_trading", "test_precision", "test_count", "reason"] if c in metrics.columns]
            lines += ["```text", metrics[cols].to_string(index=False), "```", ""]
    else:
        lines += ["## Offline", "", "No offline run found.", ""]

    if live:
        lines += ["## Paper Trading", "", f"Latest live run: `{live}`", ""]
        tracked_path = live / "tracked_items.csv"
        if tracked_path.exists():
            tracked = pd.read_csv(tracked_path)
            lines += [
                f"Tracked items: `{len(tracked)}`",
                f"Sold within 2h known positives: `{int(tracked.get('sold_within_2h', pd.Series(dtype=object)).fillna(False).astype(bool).sum())}`",
                f"Sold within 12h known positives: `{int(tracked.get('sold_within_12h', pd.Series(dtype=object)).fillna(False).astype(bool).sum())}`",
                f"Sold within 2d known positives: `{int(tracked.get('sold_within_2d', pd.Series(dtype=object)).fillna(False).astype(bool).sum())}`",
                f"Sold within 7d known positives: `{int(tracked.get('sold_within_7d', pd.Series(dtype=object)).fillna(False).astype(bool).sum())}`",
                "",
            ]
    else:
        lines += ["## Paper Trading", "", "No paper-trading run found.", ""]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    write_manifest(REPORTS_DIR / "latest_report_manifest.json", command="report", extra={"report_path": str(out_path)})
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a deal-finder experiment report.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    print(build_report(Path(args.out) if args.out else None))


if __name__ == "__main__":
    main()
