#!/usr/bin/env python3
"""Merge a search's full_scrape + balanced staging results into a labeled
items_enriched.csv (with MergedStatusBinary) and append it to the global
backfill_sold_unsold_all_searches.csv used by the giant-model dataset loader.

Mirrors the layout produced for griffati_donna_all/griffati_uomo_all/gucci/
nike/prada/ps4 under full_scrape_merged_stage_resume_20260513/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[3]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config.project_config import settings  # noqa: E402
from full_scraper import Full_Scraper  # noqa: E402

SIMPLE_SCRAPE_DIR = Path(str(settings.paths.simple_scrape_dir))
GLOBAL_BACKFILL_PATH = (
    SIMPLE_SCRAPE_DIR / "full_scrape_merged_stage_resume_20260513" / "backfill_sold_unsold_all_searches.csv"
)


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def status_to_binary(market_status: pd.Series) -> pd.Series:
    norm = market_status.fillna("").astype(str).str.strip().str.lower()
    return norm.map({"sold": "sold", "on sale": "not_sold", "onsale": "not_sold"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", required=True, help="Search folder name, e.g. telefoni")
    parser.add_argument("--stage-subdir", required=True, help="Balanced staging subdir, e.g. full_scrape_stage_resume_20260611")
    parser.add_argument("--merged-subdir", default=None, help="Output merged subdir name (default: full_scrape_merged_<stage-subdir>)")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing any files")
    args = parser.parse_args()

    search = args.search
    stage_subdir = args.stage_subdir
    merged_subdir = args.merged_subdir or f"full_scrape_merged_{stage_subdir}"

    scraper = Full_Scraper()
    search_dir = SIMPLE_SCRAPE_DIR / search

    base = read_csv_or_empty(search_dir / "full_scrape" / "items_enriched.csv")
    staged = read_csv_or_empty(search_dir / stage_subdir / "items_enriched.csv")

    if base.empty and staged.empty:
        print(f"No full_scrape data found for {search}")
        return 1

    if not base.empty:
        base["MergedSourceSubdir"] = "full_scrape"
    if not staged.empty:
        staged["MergedSourceSubdir"] = stage_subdir

    frames = [f for f in (base, staged) if not f.empty]
    combined = pd.concat(frames, ignore_index=True)
    # staged results (sold_confirmed_live / unsold_balance_stage) win on conflicts
    combined = scraper._dedupe_by_identity(combined, keep="last")

    combined["MergedStatusBinary"] = status_to_binary(combined["MarketStatus"])
    before = len(combined)
    combined = combined[combined["MergedStatusBinary"].notna()].reset_index(drop=True)
    dropped = before - len(combined)

    counts = combined["MergedStatusBinary"].value_counts().to_dict()
    print(f"{search}: merged rows={len(combined)} (dropped {dropped} with unrecognized MarketStatus) -> {counts}")

    if args.dry_run:
        return 0

    out_dir = search_dir / merged_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "items_enriched.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    global_df = read_csv_or_empty(GLOBAL_BACKFILL_PATH)
    new_rows = combined.copy()
    new_rows["SourceSearch"] = search

    if not global_df.empty:
        global_df = global_df[global_df["SourceSearch"] != search].copy()

    updated = pd.concat([global_df, new_rows], ignore_index=True)
    updated.to_csv(GLOBAL_BACKFILL_PATH, index=False)
    print(f"Updated {GLOBAL_BACKFILL_PATH} (total rows={len(updated)}, {search} rows={len(new_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
