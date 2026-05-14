#!/usr/bin/env python3
from __future__ import annotations

import argparse

from utils_lib.download_tracker import summarize_downloads


def format_bytes(value: int) -> str:
    size = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"


def print_group(title: str, rows: dict[str, dict[str, int]], limit: int | None = None) -> None:
    print(title)
    if not rows:
        print("  none")
        return
    for index, (name, stats) in enumerate(rows.items(), start=1):
        if limit is not None and index > limit:
            break
        label = name or "<empty>"
        print(
            f"  {label}: events={stats['events']} "
            f"bytes={stats['bytes']} ({format_bytes(stats['bytes'])})"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize tracked download bytes for the scraper.")
    ap.add_argument("--path", default=None, help="Optional path to a download_stats.jsonl file")
    ap.add_argument("--top-hosts", type=int, default=10, help="How many hosts to show")
    args = ap.parse_args()

    summary = summarize_downloads(args.path)
    print(f"path: {summary['path']}")
    print(f"events: {summary['events']}")
    print(f"total_bytes: {summary['total_bytes']} ({format_bytes(summary['total_bytes'])})")
    print_group("By transport:", summary["by_transport"])
    print_group("By kind:", summary["by_kind"])
    print_group("Top hosts:", summary["by_host"], limit=args.top_hosts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
