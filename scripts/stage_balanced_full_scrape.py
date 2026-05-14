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
from full_scrape_sold_history import load_search_config_by_folder, normalize_id
from full_scraper import Full_Scraper


REPORT_DIR = ROOT / "data" / "experiments" / "deal_finder" / "reports"
DEFAULT_OUTPUT_SUBDIR = "full_scrape_stage_resume"
DEFAULT_SOLD_MAX_AGE_DAYS_BY_SEARCH = {
    "griffati_donna_all": 21,
    "griffati_uomo_all": 30,
    "gucci": 28,
    "nike": 28,
}


def dedupe_rows(df: pd.DataFrame) -> pd.DataFrame:
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


def search_dirs(args: argparse.Namespace) -> list[Path]:
    root = Path(str(settings.paths.simple_scrape_dir))
    if args.all_searches:
        paths = [path for path in root.iterdir() if path.is_dir() and (path / "sold_df.csv").exists()]
    else:
        wanted = set(args.search or [])
        paths = [root / name for name in wanted if (root / name / "sold_df.csv").exists()]
    return sorted(paths, key=lambda path: path.name.lower())


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def combined_items(scraper: Full_Scraper, search_name: str, output_subdir: str) -> pd.DataFrame:
    live_paths = scraper._full_scrape_paths(search_name)
    staged_paths = scraper._full_scrape_paths(search_name, output_subdir=output_subdir)
    frames = [read_csv_or_empty(live_paths["items"]), read_csv_or_empty(staged_paths["items"])]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return scraper._dedupe_by_identity(combined, keep="last")


def sort_newest_first(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    order_columns: list[str] = []
    for source_col in (
        "SearchDate",
        "DealFinderScoredAt",
        "PriorityQueueEnqueuedAt",
        "PriorityQueueLastCheckedAt",
        "Upload_date",
    ):
        if source_col not in out.columns:
            continue
        sort_col = f"__sort_{source_col}"
        out[sort_col] = pd.to_datetime(out[source_col], errors="coerce", utc=True, dayfirst=True)
        if out[sort_col].notna().any():
            order_columns.append(sort_col)
    if not order_columns:
        return out.reset_index(drop=True)
    out = out.sort_values(order_columns, ascending=[False] * len(order_columns), kind="stable")
    return out.drop(columns=order_columns, errors="ignore").reset_index(drop=True)


def filter_sold_rows_by_age(df: pd.DataFrame, max_age_days: int | None) -> tuple[pd.DataFrame, int]:
    if df.empty or max_age_days is None or "SearchDate" not in df.columns:
        return df.copy(), 0
    out = df.copy()
    search_dates = pd.to_datetime(out["SearchDate"], errors="coerce", utc=True, dayfirst=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(max_age_days))
    keep_mask = search_dates.notna() & (search_dates >= cutoff)
    filtered_out = int((~keep_mask).sum())
    return out.loc[keep_mask].reset_index(drop=True), filtered_out


def rows_with_market_status(df: pd.DataFrame, status: str) -> pd.DataFrame:
    if df.empty or "MarketStatus" not in df.columns:
        return df.iloc[0:0].copy() if not df.empty else pd.DataFrame()
    norm = df["MarketStatus"].fillna("").astype(str).str.strip().str.lower()
    wanted = status.strip().lower()
    alt = {"on sale": {"on sale", "onsale"}, "sold": {"sold"}}.get(wanted, {wanted})
    return df[norm.isin(alt)].copy()


def sold_pending_rows(
    scraper: Full_Scraper,
    search_dir: Path,
    *,
    output_subdir: str,
    limit: int | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    sold_df = dedupe_rows(read_csv_or_empty(search_dir / "sold_df.csv"))
    if limit is not None:
        sold_df = sold_df.head(int(limit)).copy()
    existing_sold = rows_with_market_status(combined_items(scraper, search_dir.name, output_subdir), "sold")
    existing_keys = scraper._identity_keys_for_frame(existing_sold)
    pending = [row.to_dict() for _, row in sold_df.iterrows() if scraper._identity_key(row) not in existing_keys]
    pending_df = pd.DataFrame(pending)
    sold_max_age_days = DEFAULT_SOLD_MAX_AGE_DAYS_BY_SEARCH.get(search_dir.name)
    pending_df, sold_filtered_by_age = filter_sold_rows_by_age(pending_df, sold_max_age_days)
    stats = {
        "sold_rows": int(len(sold_df)),
        "sold_already_scraped": int(len(existing_sold)),
        "sold_pending": int(len(pending_df)),
        "sold_filtered_by_age": sold_filtered_by_age,
        "sold_max_age_days": int(sold_max_age_days) if sold_max_age_days is not None else None,
    }
    return pending_df, stats


def unsold_rows_to_balance(
    scraper: Full_Scraper,
    search_dir: Path,
    *,
    output_subdir: str,
    limit: int | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    existing = combined_items(scraper, search_dir.name, output_subdir)
    sold_total = int(len(rows_with_market_status(existing, "sold")))
    unsold_existing = rows_with_market_status(existing, "on sale")
    unsold_existing_count = int(len(unsold_existing))
    needed = max(0, sold_total - unsold_existing_count)

    unsold_df = dedupe_rows(read_csv_or_empty(search_dir / "unsold_df.csv"))
    existing_keys = scraper._identity_keys_for_frame(existing)
    candidates = [row.to_dict() for _, row in unsold_df.iterrows() if scraper._identity_key(row) not in existing_keys]
    candidate_df = sort_newest_first(pd.DataFrame(candidates))
    selected_count = min(int(len(candidate_df)), int(needed))
    if limit is not None:
        selected_count = min(selected_count, int(limit))
    selected = candidate_df.head(selected_count).copy()
    stats = {
        "sold_total_after_stage": sold_total,
        "unsold_existing": unsold_existing_count,
        "unsold_needed": int(needed),
        "unsold_candidates": int(len(candidate_df)),
        "unsold_selected": int(len(selected)),
    }
    return selected, stats


def write_summary(summary: dict[str, Any], output_subdir: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    safe_name = output_subdir.replace("/", "_")
    path = REPORT_DIR / f"balanced_full_scrape_{safe_name}_{ts}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    latest = REPORT_DIR / f"balanced_full_scrape_{safe_name}_latest.json"
    latest.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def process_batches(
    scraper: Full_Scraper,
    df: pd.DataFrame,
    *,
    search_name: str,
    reason: str,
    batch_size: int,
    max_workers: int,
    no_residential: bool,
    image_mode: str,
    output_subdir: str,
    pause_seconds: float,
) -> dict[str, int]:
    stats = {"batches": 0, "processed": 0, "succeeded": 0, "failed": 0}
    if df.empty:
        return stats
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size].copy()
        result = scraper.collect_and_store_full_items(
            batch,
            search_name=search_name,
            reason=reason,
            max_workers=max_workers,
            no_residential=no_residential,
            image_mode=image_mode,
            skip_existing=True,
            output_subdir=output_subdir,
        )
        stats["batches"] += 1
        stats["processed"] += int(result.get("processed", 0) or 0)
        stats["succeeded"] += int(result.get("succeeded", 0) or 0)
        stats["failed"] += int(result.get("failed", 0) or 0)
        print(
            f"{search_name}: {reason} batch {stats['batches']} processed={result.get('processed', 0)} "
            f"succeeded={result.get('succeeded', 0)} failed={result.get('failed', 0)}"
        )
        if pause_seconds > 0 and start + batch_size < len(df):
            time.sleep(float(pause_seconds))
    return stats


def run(args: argparse.Namespace) -> int:
    scraper = Full_Scraper()
    configs = load_search_config_by_folder()
    selected_dirs = search_dirs(args)
    if not selected_dirs:
        print("No per-search sold_df.csv files found.")
        return 1

    output_subdir = str(args.output_subdir or DEFAULT_OUTPUT_SUBDIR).strip() or DEFAULT_OUTPUT_SUBDIR
    summary: dict[str, Any] = {
        "output_subdir": output_subdir,
        "dry_run": bool(args.dry_run),
        "batch_size": int(args.batch_size),
        "max_workers": int(args.max_workers),
        "sold_limit_per_search": args.sold_limit_per_search,
        "unsold_limit_per_search": args.unsold_limit_per_search,
        "image_mode": args.image_mode,
        "searches": {},
    }

    for search_dir in selected_dirs:
        search_name = search_dir.name
        search_config = configs.get(search_name)
        no_residential = bool(getattr(search_config, "no_residential", True))

        sold_pending, sold_stats = sold_pending_rows(
            scraper,
            search_dir,
            output_subdir=output_subdir,
            limit=args.sold_limit_per_search,
        )
        print(
            f"{search_name}: sold_rows={sold_stats['sold_rows']} already_sold={sold_stats['sold_already_scraped']} "
            f"sold_pending={sold_stats['sold_pending']} sold_filtered_by_age={sold_stats['sold_filtered_by_age']} "
            f"sold_max_age_days={sold_stats['sold_max_age_days']}"
        )
        sold_run_stats = {"batches": 0, "processed": 0, "succeeded": 0, "failed": 0}
        if not args.dry_run and not sold_pending.empty:
            sold_run_stats = process_batches(
                scraper,
                sold_pending,
                search_name=search_name,
                reason="sold_backfill_stage",
                batch_size=max(1, int(args.batch_size)),
                max_workers=int(args.max_workers),
                no_residential=no_residential,
                image_mode=args.image_mode,
                output_subdir=output_subdir,
                pause_seconds=float(args.pause_seconds),
            )

        unsold_selected, unsold_stats = unsold_rows_to_balance(
            scraper,
            search_dir,
            output_subdir=output_subdir,
            limit=args.unsold_limit_per_search,
        )
        print(
            f"{search_name}: sold_total_after_stage={unsold_stats['sold_total_after_stage']} "
            f"unsold_existing={unsold_stats['unsold_existing']} unsold_needed={unsold_stats['unsold_needed']} "
            f"unsold_selected={unsold_stats['unsold_selected']}"
        )
        unsold_run_stats = {"batches": 0, "processed": 0, "succeeded": 0, "failed": 0}
        if not args.dry_run and not unsold_selected.empty:
            unsold_run_stats = process_batches(
                scraper,
                unsold_selected,
                search_name=search_name,
                reason="unsold_balance_stage",
                batch_size=max(1, int(args.batch_size)),
                max_workers=int(args.max_workers),
                no_residential=no_residential,
                image_mode=args.image_mode,
                output_subdir=output_subdir,
                pause_seconds=float(args.pause_seconds),
            )

        final_existing = combined_items(scraper, search_name, output_subdir)
        summary["searches"][search_name] = {
            **sold_stats,
            **{f"sold_run_{k}": v for k, v in sold_run_stats.items()},
            **unsold_stats,
            **{f"unsold_run_{k}": v for k, v in unsold_run_stats.items()},
            "final_sold_scraped": int(len(rows_with_market_status(final_existing, "sold"))),
            "final_unsold_scraped": int(len(rows_with_market_status(final_existing, "on sale"))),
        }

    report_path = write_summary(summary, output_subdir)
    print(f"Summary written to {report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume sold full scraping into a staging folder and balance each search with the same number of unsold full scrapes."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all-searches", action="store_true", help="Process every search folder with its own sold_df.csv.")
    group.add_argument("--search", action="append", help="Process one search folder; can be repeated.")
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    parser.add_argument("--sold-limit-per-search", type=int, default=None)
    parser.add_argument("--unsold-limit-per-search", type=int, default=None)
    parser.add_argument("--image-mode", choices=["html", "selenium"], default="html")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))