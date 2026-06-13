#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[3]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.time_to_sell.paths import (
    EXPERIMENT_ROOT,
    ROOT,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    utc_now_iso,
    write_manifest,
)


DEFAULT_CASCADE_RUN = (
    ROOT
    / "data"
    / "experiments"
    / "benchmark_basic_to_full"
    / "live_runs"
    / "cascade_scheduled_20260518_212856"
)
DEFAULT_WINDOWS = (2, 4, 6, 12, 24, 48, 72)
SOLD_STATUS_VALUES = {"sold", "venduto", "vendu"}
ON_SALE_STATUS_VALUES = {"onsale", "on sale", "available", "disponibile", "active"}
TIMING_START_COLUMNS = (
    "first_stage1_pass_at",
    "SearchDate",
    "DealFinderScoredAt",
    "PriorityQueueEnqueuedAt",
    "QueuedAt",
    "snapshot_at",
)
TIMING_CHECK_COLUMNS = (
    "sold_at",
    "PriorityQueueLastCheckedAt",
    "LastCheckedAt",
    "CheckedAt",
    "last_checked_at",
    "rechecked_at",
    "last_rechecked_at",
    "evaluated_at_72h",
)
STATUS_COLUMNS = (
    "last_recheck_status",
    "LastCheckStatus",
    "PriorityQueueLastStatus",
    "MarketStatus",
    "status",
    "_status",
)
BASIC5_COLUMNS = (
    "tracking_key",
    "item_id",
    "SearchName",
    "Link",
    "Title",
    "Brand",
    "Size",
    "Price",
    "Likes",
    "Stage1Model",
    "Stage1Score",
    "Stage1Threshold",
    "Stage2Model",
    "Stage2Score",
    "Stage2Threshold",
    "Stage2Passed",
    "Stage2Status",
    "FullScrapeStatus",
    "QualityMethodStatus",
)


@dataclass(frozen=True)
class DatasetPaths:
    output_dir: Path
    basic5: Path
    full_visual: Path
    labels: Path
    historical_audit: Path
    historical_candidates: Path
    summary_json: Path
    summary_md: Path
    manifest: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build time-to-sell label datasets from live cascade and historical recheck files."
    )
    parser.add_argument("--cascade-run", type=Path, default=DEFAULT_CASCADE_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--simple-scrape-dir", type=Path, default=ROOT / "data" / "simple_scrape")
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=list(DEFAULT_WINDOWS),
        help="Hour windows to expose as sold-before labels.",
    )
    parser.add_argument(
        "--all-live-windows",
        action="store_true",
        help="Use all sold_within_*h windows present in tracked_items.csv.",
    )
    return parser.parse_args()


def parse_datetime(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.to_datetime(series, errors="coerce", utc=True)
    text = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    slash_date = text.str.contains(r"^\d{1,2}/\d{1,2}/\d{4}", regex=True, na=False)
    if slash_date.any():
        parsed.loc[slash_date] = pd.to_datetime(text.loc[slash_date], errors="coerce", utc=True, dayfirst=True)
    if (~slash_date).any():
        parsed.loc[~slash_date] = pd.to_datetime(text.loc[~slash_date], errors="coerce", utc=True)
    return parsed


def first_existing(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    result = pd.Series(pd.NA, index=frame.index, dtype="object")
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column]
        missing = result.isna() | (result.astype(str).str.strip() == "")
        result.loc[missing] = values.loc[missing]
    return result


def normalize_status(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def boolish(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return pd.Series(
        np.select(
            [
                text.isin({"true", "1", "1.0", "yes", "y"}),
                text.isin({"false", "0", "0.0", "no", "n"}),
            ],
            [True, False],
            default=pd.NA,
        ),
        index=series.index,
        dtype="boolean",
    )


def derive_key(frame: pd.DataFrame) -> pd.Series:
    if "tracking_key" in frame.columns:
        key = frame["tracking_key"].astype(str)
        if key.str.contains("::", regex=False, na=False).any():
            return key
    item = first_existing(frame, ("item_id", "Dataid", "ItemId", "id")).astype(str).str.replace(r"\.0$", "", regex=True)
    search = first_existing(frame, ("SearchName", "search_name", "folder", "Folder")).astype(str)
    search = search.replace({"<NA>": "", "nan": "", "None": ""})
    return search + "::" + item


def available_live_windows(frame: pd.DataFrame) -> list[int]:
    windows: list[int] = []
    for column in frame.columns:
        match = re.fullmatch(r"sold_within_(\d+)h", column)
        if match:
            windows.append(int(match.group(1)))
    return sorted(set(windows))


def prepare_live_labels(tracked: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    labels = tracked.copy()
    labels["tracking_key"] = derive_key(labels)
    labels["item_id"] = first_existing(labels, ("item_id", "Dataid")).astype(str).str.replace(r"\.0$", "", regex=True)
    labels["observation_time"] = parse_datetime(first_existing(labels, ("first_stage1_pass_at", "last_seen_at")))
    labels["sold_detected_at"] = parse_datetime(first_existing(labels, ("sold_at",)))
    labels["label_quality"] = "exact_live_cascade"
    labels["dataset_source"] = "cascade_live"
    labels["sold_elapsed_hours"] = (
        labels["sold_detected_at"] - labels["observation_time"]
    ).dt.total_seconds() / 3600.0
    labels["eventually_sold_detected"] = labels["sold_detected_at"].notna()
    for window in windows:
        sold_col = f"sold_within_{window}h"
        eval_col = f"evaluated_at_{window}h"
        status_col = f"status_at_{window}h"
        label_col = f"label_sold_within_{window}h"
        evaluated_col = f"label_evaluated_{window}h"
        labels[evaluated_col] = labels[eval_col].notna() if eval_col in labels.columns else False
        if sold_col in labels.columns:
            parsed = boolish(labels[sold_col])
            labels[label_col] = parsed.where(labels[evaluated_col], pd.NA)
        else:
            labels[label_col] = pd.NA
        if status_col in labels.columns:
            labels[f"status_at_label_{window}h"] = labels[status_col]
    return labels


def label_columns(windows: list[int]) -> list[str]:
    columns = [
        "tracking_key",
        "dataset_source",
        "label_quality",
        "observation_time",
        "sold_detected_at",
        "sold_elapsed_hours",
        "eventually_sold_detected",
    ]
    for window in windows:
        columns.extend(
            [
                f"label_sold_within_{window}h",
                f"label_evaluated_{window}h",
            ]
        )
    return columns


def build_basic5(labels: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    columns = [column for column in BASIC5_COLUMNS if column in labels.columns]
    columns.extend(column for column in label_columns(windows) if column in labels.columns and column not in columns)
    return labels[columns].copy()


def read_csv_if_useful(path: Path) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path, low_memory=False)
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        return None
    except Exception:
        return None
    return None if frame.empty else frame


def sort_columns_for_latest(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in ("snapshot_at", "FullScrapedAt", "SearchDate") if column in frame.columns]
    if not sort_columns:
        return frame
    sorted_frame = frame.copy()
    for column in sort_columns:
        sorted_frame[f"__sort_{column}"] = parse_datetime(sorted_frame[column])
    return sorted_frame.sort_values([f"__sort_{column}" for column in sort_columns])


def load_visual_features(cascade_run: Path) -> pd.DataFrame:
    visual_dir = cascade_run / "visual_features"
    frames: list[pd.DataFrame] = []
    if visual_dir.exists():
        for path in sorted(visual_dir.glob("*.csv")):
            frame = read_csv_if_useful(path)
            if frame is None:
                continue
            frame["visual_feature_source_file"] = str(path)
            frames.append(frame)
    if frames:
        visual = pd.concat(frames, ignore_index=True, sort=False)
    else:
        enriched_path = cascade_run / "full_items" / "items_enriched.csv"
        frame = read_csv_if_useful(enriched_path)
        visual = frame if frame is not None else pd.DataFrame()
    if visual.empty:
        return visual
    visual["tracking_key"] = derive_key(visual)
    visual = sort_columns_for_latest(visual)
    drop_sort = [column for column in visual.columns if column.startswith("__sort_")]
    visual = visual.drop_duplicates("tracking_key", keep="last").drop(columns=drop_sort, errors="ignore")
    return visual


def build_full_visual(labels: pd.DataFrame, windows: list[int], cascade_run: Path) -> pd.DataFrame:
    visual = load_visual_features(cascade_run)
    if visual.empty:
        return pd.DataFrame()
    keep = [column for column in label_columns(windows) if column in labels.columns]
    label_frame = labels[keep].copy()
    merged = visual.merge(label_frame, on="tracking_key", how="inner", suffixes=("", "_label"))
    return merged


def infer_search_from_path(path: Path, simple_scrape_dir: Path) -> str:
    try:
        relative = path.relative_to(simple_scrape_dir)
    except ValueError:
        return ""
    return relative.parts[0] if relative.parts else ""


def collect_historical_csvs(simple_scrape_dir: Path) -> list[Path]:
    if not simple_scrape_dir.exists():
        return []
    paths: list[Path] = []
    for event_dir in simple_scrape_dir.glob("*/eventual_sale_check"):
        if not event_dir.is_dir():
            continue
        for name in (
            "big_raw_eventual_sale_labeled.csv",
            "sold_eventually.csv",
            "not_sold_yet.csv",
            "priority_check_queue.csv",
        ):
            path = event_dir / name
            if path.exists():
                paths.append(path)
    return sorted(paths)


def status_series(frame: pd.DataFrame) -> pd.Series:
    combined = first_existing(frame, STATUS_COLUMNS)
    if combined.isna().all() and "sold" in frame.columns:
        sold = boolish(frame["sold"])
        combined = sold.map({True: "sold", False: "on_sale"}).astype("object")
    return combined.astype(str)


def row_quality(frame: pd.DataFrame, path: Path) -> pd.Series:
    statuses = status_series(frame).map(normalize_status)
    sold = statuses.isin(SOLD_STATUS_VALUES)
    on_sale = statuses.isin({normalize_status(value) for value in ON_SALE_STATUS_VALUES})
    if path.name == "sold_eventually.csv":
        sold = sold | (~on_sale & statuses.ne("") & statuses.ne("nan"))
    start_time = parse_datetime(first_existing(frame, TIMING_START_COLUMNS))
    check_time = parse_datetime(first_existing(frame, TIMING_CHECK_COLUMNS))
    has_upload_age = pd.to_numeric(first_existing(frame, ("Upload_date_days", "upload_date_days")), errors="coerce").notna()
    quality = pd.Series("unusable_or_unlabeled", index=frame.index, dtype="object")
    quality.loc[sold & start_time.notna() & check_time.notna()] = "approximate_eventual_check"
    quality.loc[sold & start_time.notna() & check_time.isna() & has_upload_age] = "upload_age_proxy"
    quality.loc[(sold | on_sale) & (quality == "unusable_or_unlabeled")] = "weak_binary_only"
    return quality


def audit_historical(simple_scrape_dir: Path, windows: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows: list[dict[str, Any]] = []
    candidate_frames: list[pd.DataFrame] = []
    for path in collect_historical_csvs(simple_scrape_dir):
        frame = read_csv_if_useful(path)
        if frame is None:
            audit_rows.append(
                {
                    "search_name": infer_search_from_path(path, simple_scrape_dir),
                    "file_name": path.name,
                    "path": str(path),
                    "rows": 0,
                    "label_quality": "empty_or_unreadable",
                }
            )
            continue
        search = infer_search_from_path(path, simple_scrape_dir)
        quality = row_quality(frame, path)
        counts = quality.value_counts(dropna=False)
        for label_quality, rows in counts.items():
            audit_rows.append(
                {
                    "search_name": search,
                    "file_name": path.name,
                    "path": str(path),
                    "rows": int(rows),
                    "label_quality": str(label_quality),
                }
            )
        approximate = quality == "approximate_eventual_check"
        if approximate.any():
            candidates = frame.loc[approximate].copy()
            candidates["tracking_key"] = derive_key(candidates)
            candidates["SearchName"] = first_existing(candidates, ("SearchName", "search_name")).fillna(search)
            candidates["item_id"] = first_existing(candidates, ("item_id", "Dataid")).astype(str).str.replace(
                r"\.0$", "", regex=True
            )
            candidates["dataset_source"] = "historical_eventual_sale"
            candidates["label_quality"] = "approximate_eventual_check"
            candidates["source_file"] = str(path)
            candidates["observation_time"] = parse_datetime(first_existing(candidates, TIMING_START_COLUMNS))
            candidates["sold_detected_at"] = parse_datetime(first_existing(candidates, TIMING_CHECK_COLUMNS))
            candidates["sold_elapsed_hours"] = (
                candidates["sold_detected_at"] - candidates["observation_time"]
            ).dt.total_seconds() / 3600.0
            candidates["eventually_sold_detected"] = True
            for window in windows:
                candidates[f"label_sold_within_{window}h"] = candidates["sold_elapsed_hours"] <= float(window)
                candidates[f"label_evaluated_{window}h"] = True
            candidate_frames.append(candidates)
    audit = pd.DataFrame(audit_rows)
    if candidate_frames:
        candidates = pd.concat(candidate_frames, ignore_index=True, sort=False)
    else:
        candidate_columns = [
            "tracking_key",
            "SearchName",
            "item_id",
            "dataset_source",
            "label_quality",
            "source_file",
            "observation_time",
            "sold_detected_at",
            "sold_elapsed_hours",
            "eventually_sold_detected",
        ]
        for window in windows:
            candidate_columns.extend([f"label_sold_within_{window}h", f"label_evaluated_{window}h"])
        candidates = pd.DataFrame(columns=candidate_columns)
    return audit, candidates


def bool_rate(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    values = boolish(frame[column]).dropna()
    if values.empty:
        return float("nan")
    return float(values.astype(bool).mean())


def window_summary(frame: pd.DataFrame, windows: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in windows:
        label_col = f"label_sold_within_{window}h"
        eval_col = f"label_evaluated_{window}h"
        evaluated = boolish(frame[eval_col]).fillna(False) if eval_col in frame.columns else pd.Series(False, index=frame.index)
        rows.append(
            {
                "window_h": window,
                "evaluated_rows": int(evaluated.sum()),
                "positive_rows": int(boolish(frame.loc[evaluated, label_col]).fillna(False).sum()) if label_col in frame else 0,
                "positive_rate": bool_rate(frame.loc[evaluated], label_col) if label_col in frame else float("nan"),
            }
        )
    return rows


def live_per_search_summary(frame: pd.DataFrame, windows: list[int]) -> list[dict[str, Any]]:
    if "SearchName" not in frame.columns:
        return []
    rows: list[dict[str, Any]] = []
    for search, group in frame.groupby("SearchName", dropna=False):
        row: dict[str, Any] = {
            "search_name": str(search),
            "rows": int(len(group)),
            "sold_detected_rows": int(group["eventually_sold_detected"].sum()),
        }
        for window in windows:
            label_col = f"label_sold_within_{window}h"
            eval_col = f"label_evaluated_{window}h"
            if label_col not in group.columns or eval_col not in group.columns:
                continue
            evaluated = boolish(group[eval_col]).fillna(False)
            row[f"evaluated_{window}h_rows"] = int(evaluated.sum())
            row[f"sold_within_{window}h_rows"] = int(boolish(group.loc[evaluated, label_col]).fillna(False).sum())
        rows.append(row)
    return sorted(rows, key=lambda item: item["search_name"])


def write_markdown_report(paths: DatasetPaths, summary: dict[str, Any]) -> None:
    lines = [
        "# Time-to-sell dataset build",
        "",
        f"Created at: `{summary['created_at']}`",
        f"Cascade run: `{summary['cascade_run']}`",
        "",
        "## Output files",
        "",
        f"- Basic5 speed dataset: `{paths.basic5}`",
        f"- Full + visual speed dataset: `{paths.full_visual}`",
        f"- Live labels only: `{paths.labels}`",
        f"- Historical timing audit: `{paths.historical_audit}`",
        f"- Approximate historical speed candidates: `{paths.historical_candidates}`",
        "",
        "## Live cascade labels",
        "",
        f"- Tracked rows: {summary['live_rows']}",
        f"- Basic5 rows: {summary['basic5_rows']}",
        f"- Full + visual rows: {summary['full_visual_rows']}",
        f"- Detected sold rows: {summary['live_sold_detected_rows']}",
        "",
        "### By search",
        "",
        "| Search | Basic5 Rows | Full+Visual Rows | Sold Detected | Sold <= 24h | Sold <= 48h | Sold <= 72h |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    full_visual_by_search = {
        row["search_name"]: row["rows"] for row in summary.get("full_visual_per_search_summary", [])
    }
    for row in summary.get("live_per_search_summary", []):
        search = row["search_name"]
        lines.append(
            "| "
            + " | ".join(
                [
                    search,
                    str(row["rows"]),
                    str(full_visual_by_search.get(search, 0)),
                    str(row["sold_detected_rows"]),
                    str(row.get("sold_within_24h_rows", "")),
                    str(row.get("sold_within_48h_rows", "")),
                    str(row.get("sold_within_72h_rows", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### By window",
            "",
        "| Window | Evaluated | Sold Within Window | Positive Rate |",
        "|---:|---:|---:|---:|",
        ]
    )
    for row in summary["live_window_summary"]:
        rate = row["positive_rate"]
        rate_text = "" if pd.isna(rate) else f"{rate:.3f}"
        lines.append(
            f"| {row['window_h']}h | {row['evaluated_rows']} | {row['positive_rows']} | {rate_text} |"
        )
    lines.extend(["", "## Historical timing audit", ""])
    if summary["historical_quality_counts"]:
        lines.extend(["| Label quality | Rows |", "|---|---:|"])
        for row in summary["historical_quality_counts"]:
            lines.append(f"| {row['label_quality']} | {row['rows']} |")
    else:
        lines.append("No historical eventual-sale files were found.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use `exact_live_cascade` rows as the clean dataset for sold-before-X-hour models.",
            "Use `approximate_eventual_check` rows only as weaker extra data, because the sale time is the last observed check time, not the real transaction time.",
            "Do not mix `weak_binary_only` rows into speed labels unless we later recover a timestamp.",
            "",
        ]
    )
    paths.summary_md.write_text("\n".join(lines), encoding="utf-8")


def make_paths(output_dir: Path) -> DatasetPaths:
    output_dir = assert_experiment_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return DatasetPaths(
        output_dir=output_dir,
        basic5=output_dir / "basic5_speed_dataset.csv",
        full_visual=output_dir / "full_visual_speed_dataset.csv",
        labels=output_dir / "live_cascade_speed_labels.csv",
        historical_audit=output_dir / "historical_timing_audit.csv",
        historical_candidates=output_dir / "historical_approx_speed_candidates.csv",
        summary_json=output_dir / "summary.json",
        summary_md=output_dir / "README.md",
        manifest=output_dir / "manifest.json",
    )


def main() -> None:
    args = parse_args()
    ensure_experiment_dirs()
    output_dir = args.output_dir or (EXPERIMENT_ROOT / run_id("speed_labels"))
    paths = make_paths(output_dir)
    cascade_run = args.cascade_run.resolve()
    tracked_path = cascade_run / "tracked_items.csv"
    if not tracked_path.exists():
        raise FileNotFoundError(f"Missing tracked_items.csv: {tracked_path}")

    tracked = pd.read_csv(tracked_path)
    windows = available_live_windows(tracked) if args.all_live_windows else sorted(set(args.windows))
    labels = prepare_live_labels(tracked, windows)
    basic5 = build_basic5(labels, windows)
    full_visual = build_full_visual(labels, windows, cascade_run)
    historical_audit, historical_candidates = audit_historical(args.simple_scrape_dir, windows)

    labels.to_csv(paths.labels, index=False)
    basic5.to_csv(paths.basic5, index=False)
    full_visual.to_csv(paths.full_visual, index=False)
    historical_audit.to_csv(paths.historical_audit, index=False)
    historical_candidates.to_csv(paths.historical_candidates, index=False)

    quality_counts: list[dict[str, Any]] = []
    if not historical_audit.empty:
        grouped = (
            historical_audit.groupby("label_quality", dropna=False)["rows"]
            .sum()
            .reset_index()
            .sort_values(["rows", "label_quality"], ascending=[False, True])
        )
        quality_counts = grouped.to_dict(orient="records")

    summary: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "cascade_run": str(cascade_run),
        "windows_h": windows,
        "live_rows": int(len(labels)),
        "basic5_rows": int(len(basic5)),
        "full_visual_rows": int(len(full_visual)),
        "live_sold_detected_rows": int(labels["eventually_sold_detected"].sum()),
        "live_window_summary": window_summary(labels, windows),
        "live_per_search_summary": live_per_search_summary(labels, windows),
        "full_visual_per_search_summary": live_per_search_summary(full_visual, windows) if not full_visual.empty else [],
        "historical_files_audited": int(historical_audit["path"].nunique()) if not historical_audit.empty else 0,
        "historical_quality_counts": quality_counts,
        "historical_approx_candidate_rows": int(len(historical_candidates)),
        "outputs": {
            "basic5_speed_dataset": str(paths.basic5),
            "full_visual_speed_dataset": str(paths.full_visual),
            "live_cascade_speed_labels": str(paths.labels),
            "historical_timing_audit": str(paths.historical_audit),
            "historical_approx_speed_candidates": str(paths.historical_candidates),
            "summary_markdown": str(paths.summary_md),
        },
    }
    paths.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(paths, summary)
    write_manifest(
        paths.manifest,
        command=" ".join(sys.argv),
        extra={
            "cascade_run": str(cascade_run),
            "windows_h": windows,
            "output_dir": str(paths.output_dir),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
