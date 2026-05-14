#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.project_config import settings
try:
    from config.search_loader import load_searches
except Exception:
    load_searches = None
from full_scraper import Full_Scraper


REPORT_DIR = ROOT / "data" / "experiments" / "deal_finder" / "reports"


def normalize_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def dedupe_sold_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "Dataid" in out.columns:
        out["_identity"] = out["Dataid"].map(normalize_id)
    else:
        out["_identity"] = ""
    if "Link" in out.columns:
        links = out["Link"].fillna("").astype(str).str.strip()
        out["_identity"] = out["_identity"].where(out["_identity"].astype(str).str.len() > 0, links)
    out = out[out["_identity"].astype(str).str.len() > 0].copy()
    if out.empty:
        return out.drop(columns=["_identity"], errors="ignore")
    out = out.drop_duplicates(subset=["_identity"], keep="last")
    return out.drop(columns=["_identity"], errors="ignore").reset_index(drop=True)


def load_search_config_by_folder() -> dict[str, object]:
    if load_searches is None:
        return {}
    try:
        searches = load_searches(str(settings.paths.searches_yaml))
    except Exception:
        return {}
    return {str(search.folder): search for search in searches.values()}


def search_dirs(args: argparse.Namespace) -> list[Path]:
    root = Path(str(settings.paths.simple_scrape_dir))
    if args.all_searches:
        paths = [path for path in root.iterdir() if path.is_dir() and (path / "sold_df.csv").exists()]
    else:
        wanted = set(args.search or [])
        paths = [root / name for name in wanted if (root / name / "sold_df.csv").exists()]
    return sorted(paths, key=lambda path: path.name.lower())


def pending_rows_for_search(scraper: Full_Scraper, search_dir: Path, *, limit: int | None) -> tuple[pd.DataFrame, dict[str, int]]:
    sold_path = search_dir / "sold_df.csv"
    try:
        sold_df = pd.read_csv(sold_path)
    except Exception:
        sold_df = pd.DataFrame()
    deduped = dedupe_sold_rows(sold_df)
    if limit is not None:
        deduped = deduped.head(int(limit)).copy()

    paths = scraper._full_scrape_paths(search_dir.name)
    existing = scraper._identity_keys_for_frame(scraper._read_csv_or_empty(paths["items"]))
    pending = []
    skipped_existing = 0
    for _, row in deduped.iterrows():
        if scraper._identity_key(row) in existing:
            skipped_existing += 1
            continue
        pending.append(row.to_dict())
    pending_df = pd.DataFrame(pending)
    stats = {
        "sold_rows": int(len(sold_df)),
        "deduped_rows": int(len(deduped)),
        "pending_rows": int(len(pending_df)),
        "skipped_existing": int(skipped_existing),
    }
    return pending_df, stats


def write_summary(summary: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"sold_history_full_scrape_{ts}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    latest = REPORT_DIR / "sold_history_full_scrape_latest.json"
    latest.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> int:
    scraper = Full_Scraper()
    configs = load_search_config_by_folder()
    selected_dirs = search_dirs(args)
    if not selected_dirs:
        print("No per-search sold_df.csv files found.")
        return 1

    summary: dict[str, Any] = {
        "dry_run": bool(args.dry_run),
        "batch_size": int(args.batch_size),
        "max_workers": int(args.max_workers),
        "limit_per_search": args.limit_per_search,
        "image_mode": args.image_mode,
        "searches": {},
    }

    for search_dir in selected_dirs:
        search_name = search_dir.name
        pending_df, stats = pending_rows_for_search(scraper, search_dir, limit=args.limit_per_search)
        search_config = configs.get(search_name)
        no_residential = bool(getattr(search_config, "no_residential", True))
        stats.update({
            "no_residential": bool(no_residential),
            "batches": 0,
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
        })

        print(
            f"{search_name}: sold_rows={stats['sold_rows']} deduped={stats['deduped_rows']} "
            f"pending={stats['pending_rows']} skipped_existing={stats['skipped_existing']}"
        )

        if args.dry_run or pending_df.empty:
            summary["searches"][search_name] = stats
            continue

        batch_size = max(1, int(args.batch_size))
        for start in range(0, len(pending_df), batch_size):
            batch = pending_df.iloc[start:start + batch_size].copy()
            result = scraper.collect_and_store_full_items(
                batch,
                search_name=search_name,
                reason="sold_backfill",
                max_workers=int(args.max_workers),
                no_residential=no_residential,
                image_mode=args.image_mode,
                skip_existing=True,
            )
            stats["batches"] += 1
            stats["processed"] += int(result.get("processed", 0) or 0)
            stats["succeeded"] += int(result.get("succeeded", 0) or 0)
            stats["failed"] += int(result.get("failed", 0) or 0)
            print(
                f"{search_name}: batch {stats['batches']} processed={result.get('processed', 0)} "
                f"succeeded={result.get('succeeded', 0)} failed={result.get('failed', 0)}"
            )
            if args.pause_seconds > 0 and start + batch_size < len(pending_df):
                time.sleep(float(args.pause_seconds))

        summary["searches"][search_name] = stats

    report_path = write_summary(summary)
    print(f"Summary written to {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill full item/seller details for per-search sold_df.csv rows.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all-searches", action="store_true", help="Process every search folder with its own sold_df.csv.")
    group.add_argument("--search", action="append", help="Process one search folder; can be repeated.")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--limit-per-search", type=int, default=None)
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    parser.add_argument("--image-mode", choices=["html", "selenium"], default="html")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
