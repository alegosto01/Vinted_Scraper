#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.old.basic_plus_visual._deps.deal_finder.dataset import normalize_id_series
from experiments.old.basic_plus_visual._deps.deal_finder.paths import (
    REPORTS_DIR,
    SIMPLE_SCRAPE_DIR,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_json,
    write_manifest,
)


IDENTITY_COLUMNS = ("Dataid", "Link")


def search_folder_for(path: Path, base_dir: Path) -> str:
    rel = path.relative_to(base_dir)
    return rel.parts[0] if len(rel.parts) > 1 else "."


def read_identity_columns(path: Path) -> tuple[pd.DataFrame, str | None]:
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception as exc:
        return pd.DataFrame(), f"header_read_error:{type(exc).__name__}"
    usecols = [col for col in IDENTITY_COLUMNS if col in header.columns]
    if not usecols:
        return pd.DataFrame(), "no_identity_columns"
    try:
        return pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False), None
    except Exception as exc:
        return pd.DataFrame(), f"read_error:{type(exc).__name__}"


def add_item_identity(df: pd.DataFrame) -> pd.Series:
    if "Dataid" in df.columns:
        item_id = normalize_id_series(df["Dataid"])
    else:
        item_id = pd.Series([""] * len(df), index=df.index)
    if "Link" in df.columns:
        links = df["Link"].fillna("").astype(str).str.strip()
        item_id = item_id.where(item_id.astype(str).str.len() > 0, links)
    return item_id.fillna("").astype(str).str.strip()


def audit_csv(path: Path, base_dir: Path, *, max_groups_per_file: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    df, error = read_identity_columns(path)
    rel = str(path.relative_to(base_dir))
    folder = search_folder_for(path, base_dir)
    if error is not None:
        return (
            {
                "search_folder": folder,
                "file": rel,
                "rows": 0,
                "identity_rows": 0,
                "unique_identities": 0,
                "duplicate_groups": 0,
                "duplicate_extra_rows": 0,
                "max_group_size": 0,
                "status": error,
            },
            [],
        )

    item_id = add_item_identity(df)
    non_empty = item_id[item_id.str.len() > 0]
    counts = non_empty.value_counts(dropna=False)
    dup_counts = counts[counts > 1]
    summary = {
        "search_folder": folder,
        "file": rel,
        "rows": int(len(df)),
        "identity_rows": int(len(non_empty)),
        "unique_identities": int(counts.shape[0]),
        "duplicate_groups": int(dup_counts.shape[0]),
        "duplicate_extra_rows": int((dup_counts - 1).sum()) if not dup_counts.empty else 0,
        "max_group_size": int(dup_counts.max()) if not dup_counts.empty else 0,
        "status": "ok",
    }
    groups = [
        {
            "search_folder": folder,
            "file": rel,
            "item_id": str(item_id_value),
            "count": int(count),
        }
        for item_id_value, count in dup_counts.head(max_groups_per_file).items()
    ]
    return summary, groups


def write_markdown(summary: pd.DataFrame, groups: pd.DataFrame, path: Path) -> None:
    path = assert_experiment_path(path)
    duplicate_files = summary[summary["duplicate_extra_rows"] > 0].copy()
    lines = [
        "# Duplicate CSV Audit",
        "",
        "Identity rule: `Dataid` first, `Link` as fallback.",
        "",
        f"- CSV files checked: {len(summary)}",
        f"- Files with duplicate identities: {len(duplicate_files)}",
        f"- Duplicate extra rows: {int(duplicate_files['duplicate_extra_rows'].sum()) if not duplicate_files.empty else 0}",
        "",
    ]
    if duplicate_files.empty:
        lines.append("No duplicate item identities were found.")
    else:
        lines.extend(["## Files With Duplicates", ""])
        top = duplicate_files.sort_values(["duplicate_extra_rows", "duplicate_groups"], ascending=False).head(80)
        lines.append("| search | file | duplicate groups | extra rows | max group |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for _, row in top.iterrows():
            lines.append(
                f"| {row['search_folder']} | `{row['file']}` | {int(row['duplicate_groups'])} | "
                f"{int(row['duplicate_extra_rows'])} | {int(row['max_group_size'])} |"
            )
        if len(duplicate_files) > len(top):
            lines.append("")
            lines.append(f"Only the top {len(top)} duplicate-heavy files are shown here; see the CSV report for the full audit.")
    if not groups.empty:
        lines.extend(["", "## Largest Duplicate Groups", ""])
        top_groups = groups.sort_values("count", ascending=False).head(80)
        lines.append("| search | file | item id | count |")
        lines.append("| --- | --- | --- | ---: |")
        for _, row in top_groups.iterrows():
            lines.append(f"| {row['search_folder']} | `{row['file']}` | `{row['item_id']}` | {int(row['count'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_duplicates(*, base_dir: Path = SIMPLE_SCRAPE_DIR, max_groups_per_file: int = 25) -> dict[str, Any]:
    ensure_experiment_dirs()
    run_name = run_id("duplicate_audit")
    csv_paths = sorted(base_dir.rglob("*.csv"))
    summaries = []
    groups = []
    for path in csv_paths:
        summary, file_groups = audit_csv(path, base_dir, max_groups_per_file=max_groups_per_file)
        summaries.append(summary)
        groups.extend(file_groups)

    summary_df = pd.DataFrame(summaries)
    groups_df = pd.DataFrame(groups)
    summary_path = assert_experiment_path(REPORTS_DIR / f"{run_name}.csv")
    groups_path = assert_experiment_path(REPORTS_DIR / f"{run_name}_groups.csv")
    markdown_path = assert_experiment_path(REPORTS_DIR / "duplicate_audit_latest.md")
    summary_df.to_csv(summary_path, index=False)
    groups_df.to_csv(groups_path, index=False)
    write_markdown(summary_df, groups_df, markdown_path)

    payload = {
        "csv_files_checked": int(len(summary_df)),
        "files_with_duplicates": int((summary_df["duplicate_extra_rows"] > 0).sum()) if not summary_df.empty else 0,
        "duplicate_extra_rows": int(summary_df["duplicate_extra_rows"].sum()) if not summary_df.empty else 0,
        "summary_path": str(summary_path),
        "groups_path": str(groups_path),
        "markdown_path": str(markdown_path),
    }
    write_json(REPORTS_DIR / f"{run_name}.json", payload)
    write_manifest(REPORTS_DIR / f"{run_name}_manifest.json", command="audit_duplicates", extra=payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit data/simple_scrape CSV files for duplicate item identities.")
    parser.add_argument("--max-groups-per-file", type=int, default=25)
    args = parser.parse_args()
    result = audit_duplicates(max_groups_per_file=args.max_groups_per_file)
    print(result)


if __name__ == "__main__":
    main()
