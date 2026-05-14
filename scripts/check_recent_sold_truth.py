#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPTS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scraping_options import update_eventual_sale_labels_for_csv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Re-check the most recent sold listings in every sold_df.csv and summarize which are still sold."
    )
    ap.add_argument("--root", default="data", help="Root folder to search for sold_df.csv files")
    ap.add_argument("--limit", type=int, default=100, help="How many most recent rows to check from each sold_df.csv")
    ap.add_argument("--max_workers", type=int, default=4, help="Parallel workers per sold_df.csv run")
    ap.add_argument("--fetch_sleep", type=float, default=0.5, help="Sleep before each item-page fetch")
    ap.add_argument("--fetch_max_attempts", type=int, default=1, help="Max item-page fetch attempts")
    ap.add_argument("--no_residential", action="store_true", help="Use only the datacenter proxy path")
    ap.add_argument("--allow_residential_fallback", action="store_true", help="Allow residential fallback")
    ap.add_argument("--output_root", default=None, help="Optional output directory. Defaults under data/")
    return ap.parse_args()


def sold_csv_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("sold_df.csv") if path.is_file())


def output_root_path(args: argparse.Namespace) -> Path:
    if args.output_root:
        return Path(args.output_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "data" / f"recent_sold_truth_check_last{args.limit}_{stamp}"


def safe_slug(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return "__".join(rel.with_suffix("").parts)


def summarize_output_paths(labeled_path: Path, sold_path: Path, active_path: Path) -> dict[str, int]:
    if not labeled_path.exists():
        return {"n_checked": 0, "n_sold": 0, "n_active": 0, "n_fetch_failed": 0}

    labeled_df = pd.read_csv(labeled_path)
    sold_df = pd.read_csv(sold_path) if sold_path.exists() else pd.DataFrame()
    active_df = pd.read_csv(active_path) if active_path.exists() else pd.DataFrame()
    status_counts = labeled_df.get("LastCheckStatus", pd.Series(dtype="object")).fillna("<NA>").value_counts(dropna=False)
    return {
        "n_checked": int(len(labeled_df)),
        "n_sold": int(len(sold_df)),
        "n_active": int(len(active_df)),
        "n_fetch_failed": int(status_counts.get("FetchFailed", 0)),
    }


def run_one_csv(path: Path, root: Path, out_root: Path, args: argparse.Namespace) -> dict[str, object]:
    df = pd.read_csv(path)
    total_rows = int(len(df))
    if df.empty:
        return {
            "source_csv": str(path),
            "total_rows": 0,
            "rows_selected": 0,
            "skipped": True,
            "reason": "empty_csv",
        }

    recent_df = df.tail(int(args.limit)).copy()
    csv_output_dir = out_root / safe_slug(path, root)
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = csv_output_dir / "input_last_rows.csv"
    recent_df.to_csv(subset_path, index=False)

    result = update_eventual_sale_labels_for_csv(
        str(subset_path),
        out_dir=str(csv_output_dir),
        max_workers=args.max_workers,
        delay=0.0,
        allow_residential_fallback=args.allow_residential_fallback,
        initial_delay=0.0,
        fetch_sleep=args.fetch_sleep,
        fetch_max_attempts=args.fetch_max_attempts,
        no_residential=args.no_residential,
        recheck_sold_rows=True,
    )
    metrics = summarize_output_paths(
        Path(result["labeled_path"]),
        Path(result["sold_path"]),
        Path(result["active_path"]),
    )
    return {
        "source_csv": str(path),
        "total_rows": total_rows,
        "rows_selected": int(len(recent_df)),
        "skipped": False,
        "result_dir": str(csv_output_dir),
        "labeled_path": result["labeled_path"],
        "sold_path": result["sold_path"],
        "active_path": result["active_path"],
        **metrics,
    }


def main() -> int:
    args = parse_args()
    root = (ROOT / args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root).resolve()
    out_root = output_root_path(args).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for path in sold_csv_paths(root):
        rows.append(run_one_csv(path.resolve(), root, out_root, args))

    summary_df = pd.DataFrame(rows)
    summary_csv = out_root / "summary.csv"
    summary_json = out_root / "summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    payload = {
        "output_root": str(out_root),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "files_processed": int(len(rows)),
        "files_skipped": int(sum(1 for row in rows if row.get("skipped"))),
        "rows_selected_total": int(sum(int(row.get("rows_selected", 0) or 0) for row in rows)),
        "n_sold_total": int(sum(int(row.get("n_sold", 0) or 0) for row in rows)),
        "n_active_total": int(sum(int(row.get("n_active", 0) or 0) for row in rows)),
        "n_fetch_failed_total": int(sum(int(row.get("n_fetch_failed", 0) or 0) for row in rows)),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
