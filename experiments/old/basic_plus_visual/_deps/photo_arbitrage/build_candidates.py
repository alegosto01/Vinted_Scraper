#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.old.basic_plus_visual._deps.photo_arbitrage.dataset import build_candidate_dataset
from experiments.old.basic_plus_visual._deps.photo_arbitrage.paths import (
    CANDIDATES_DIR,
    ensure_experiment_dirs,
    ensure_project_imports,
    run_id,
    write_csv,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local photo-improvement candidate rows.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all-searches", action="store_true")
    group.add_argument("--search", action="append")
    parser.add_argument("--limit-per-search", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_experiment_dirs()
    candidates = build_candidate_dataset(
        all_searches=bool(args.all_searches),
        searches=args.search,
        limit_per_search=args.limit_per_search,
    )
    stem = run_id("candidates")
    out_path = write_csv(candidates, CANDIDATES_DIR / f"{stem}.csv")
    latest_path = write_csv(candidates, CANDIDATES_DIR / "latest_candidates.csv")
    write_manifest(
        CANDIDATES_DIR / f"{stem}_manifest.json",
        command=" ".join(sys.argv),
        extra={
            "rows": int(len(candidates)),
            "searches": sorted(candidates["SearchName"].dropna().astype(str).unique().tolist()) if "SearchName" in candidates else [],
            "output": str(out_path),
            "latest_output": str(latest_path),
        },
    )
    print(f"Wrote {len(candidates)} candidate rows to {out_path}")
    print(f"Latest candidate file: {latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
