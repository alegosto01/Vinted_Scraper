from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.deal_finder.paths import (
    EXPERIMENT_ROOT,
    OFFLINE_RUNS_DIR,
    SIMPLE_SCRAPE_DIR,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_json,
    write_manifest,
)


NON_SEARCH_DIRS = {"tuning_reports", "experiments"}
FUTURE_ONLY_PREFIXES = ("PriorityQueue",)
FUTURE_ONLY_COLUMNS = {
    "MarketStatus",
    "LastCheckStatus",
    "SoldLabel",
    "SoldEventuallyLabel",
    "offline_sold_label",
    "offline_label_eligible",
    "fast_sale_2d",
    "fast_sale_7d",
    "eventual_sale",
    "label_quality",
    "label_source",
    "primary_label_eligible",
    "secondary_label_eligible",
    "sold_detected_at",
    "last_known_active_at",
    "sale_delay_hours",
    "sale_delay_days",
    "outcome_source_file",
}


@dataclass(frozen=True)
class SearchDatasetResult:
    search_name: str
    path: Path
    rows: int
    primary_eligible_rows: int
    primary_positive_rows: int
    weak_or_unlabeled_rows: int
    input_paths: dict[str, str]


def normalize_id_value(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def normalize_id_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_id_value)


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    text = series.astype("object").where(series.notna(), "").astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    iso_mask = text.str.match(r"^\d{4}-\d{2}-\d{2}")
    if iso_mask.any():
        parsed.loc[iso_mask] = pd.to_datetime(text.loc[iso_mask], errors="coerce", dayfirst=False, utc=True)
    if (~iso_mask).any():
        parsed.loc[~iso_mask] = pd.to_datetime(text.loc[~iso_mask], errors="coerce", dayfirst=True, utc=True)
    return parsed


def parse_timestamp_value(value) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    text = str(value).strip()
    dayfirst = not bool(pd.Series([text]).str.match(r"^\d{4}-\d{2}-\d{2}").iloc[0])
    parsed = pd.to_datetime(pd.Series([text]), errors="coerce", dayfirst=dayfirst, utc=True).iloc[0]
    return parsed


def list_search_dirs(base_dir: Path = SIMPLE_SCRAPE_DIR) -> list[Path]:
    if not base_dir.exists():
        return []
    out = []
    for path in base_dir.iterdir():
        if not path.is_dir() or path.name in NON_SEARCH_DIRS or path.name.startswith("."):
            continue
        if any((path / name).exists() for name in ("big_raw.csv", "old_df.csv", "sold_df.csv")):
            out.append(path)
    return sorted(out, key=lambda p: p.name.lower())


def add_identity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Dataid" in out.columns:
        out["item_id"] = normalize_id_series(out["Dataid"])
    else:
        out["item_id"] = ""
    if "Link" in out.columns:
        link_ids = out["Link"].fillna("").astype(str).str.strip()
        out["item_id"] = out["item_id"].where(out["item_id"].astype(str).str.len() > 0, link_ids)
    return out


def dedupe_by_identity(df: pd.DataFrame, *, keep: str = "first") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = add_identity(df)
    out = out[out["item_id"].astype(str).str.len() > 0].copy()
    if out.empty:
        return out
    out["_row_order"] = range(len(out))
    if "SearchDate" in out.columns:
        out["_obs_ts_sort"] = parse_timestamp_series(out["SearchDate"])
    else:
        out["_obs_ts_sort"] = pd.NaT
    out = out.sort_values(["_obs_ts_sort", "_row_order"], kind="stable", na_position="last")
    out = out.drop_duplicates(subset=["item_id"], keep=keep)
    return out.drop(columns=["_row_order", "_obs_ts_sort"], errors="ignore").reset_index(drop=True)


def choose_first_present(row: pd.Series, columns: Iterable[str]):
    for col in columns:
        if col not in row.index:
            continue
        value = row.get(col)
        if pd.notna(value) and str(value).strip() != "":
            return value
    return pd.NA


def choose_detection(row: pd.Series, *, fallback_file_mtime: pd.Timestamp | None = None) -> tuple[pd.Timestamp, str]:
    exact_cols = ("PriorityQueueLastCheckedAt", "CheckedAt", "CheckedDate", "SoldDate")
    approx_cols = ("PriorityQueueEnqueuedAt",)
    for col in exact_cols:
        if col in row.index:
            ts = parse_timestamp_value(row.get(col))
            if pd.notna(ts):
                return ts, "exact"
    for col in approx_cols:
        if col in row.index:
            ts = parse_timestamp_value(row.get(col))
            if pd.notna(ts):
                return ts, "approximate"
    if fallback_file_mtime is not None and pd.notna(fallback_file_mtime):
        return fallback_file_mtime, "weak"
    return pd.NaT, "weak"


def file_mtime_ts(path: Path) -> pd.Timestamp:
    if not path.exists():
        return pd.NaT
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")


def outcome_maps(search_dir: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    sold_map: dict[str, dict] = {}
    active_map: dict[str, dict] = {}

    sold_sources = [
        (search_dir / "sold_df.csv", "sold_df", None),
        (
            search_dir / "eventual_sale_check" / "sold_eventually.csv",
            "sold_eventually",
            file_mtime_ts(search_dir / "eventual_sale_check" / "sold_eventually.csv"),
        ),
    ]
    for path, label_source, fallback_mtime in sold_sources:
        df = dedupe_by_identity(read_csv_or_empty(path), keep="last")
        if df.empty:
            continue
        for _, row in df.iterrows():
            item_id = str(row.get("item_id", "")).strip()
            if not item_id:
                continue
            detected_at, quality = choose_detection(row, fallback_file_mtime=fallback_mtime)
            sold_map[item_id] = {
                "sold_detected_at": detected_at,
                "label_quality": quality,
                "label_source": label_source,
                "outcome_source_file": str(path),
            }

    active_path = search_dir / "eventual_sale_check" / "not_sold_yet.csv"
    active = dedupe_by_identity(read_csv_or_empty(active_path), keep="last")
    fallback_active_mtime = file_mtime_ts(active_path)
    if not active.empty:
        for _, row in active.iterrows():
            item_id = str(row.get("item_id", "")).strip()
            if not item_id:
                continue
            checked_at, quality = choose_detection(row, fallback_file_mtime=fallback_active_mtime)
            active_map[item_id] = {
                "last_known_active_at": checked_at,
                "label_quality": "approximate" if quality == "weak" and pd.notna(checked_at) else quality,
                "label_source": "not_sold_yet",
                "outcome_source_file": str(active_path),
            }
    return sold_map, active_map


def merge_deal_features(base: pd.DataFrame, search_dir: Path) -> pd.DataFrame:
    deals_path = search_dir / "pipeline_out" / "deals_ranked.csv"
    deals = dedupe_by_identity(read_csv_or_empty(deals_path), keep="last")
    if base.empty or deals.empty:
        return base.copy()
    keep_cols = ["item_id"] + [c for c in deals.columns if c != "item_id" and c not in base.columns]
    out = base.merge(deals[keep_cols], on="item_id", how="left")
    return out


def build_search_dataset(search_dir: Path) -> pd.DataFrame:
    source_path = search_dir / "big_raw.csv"
    if not source_path.exists():
        source_path = search_dir / "old_df.csv"
    base = dedupe_by_identity(read_csv_or_empty(source_path), keep="first")
    if base.empty:
        return pd.DataFrame()
    base["SearchName"] = search_dir.name
    base["observation_ts"] = parse_timestamp_series(base["SearchDate"]) if "SearchDate" in base.columns else pd.NaT
    base = merge_deal_features(base, search_dir)

    sold_map, active_map = outcome_maps(search_dir)
    rows = []
    for _, row in base.iterrows():
        item_id = str(row.get("item_id", "")).strip()
        obs_ts = row.get("observation_ts")
        outcome = {
            "fast_sale_2d": 0,
            "fast_sale_7d": 0,
            "eventual_sale": 0,
            "offline_sold_label": 0,
            "offline_label_eligible": False,
            "label_quality": "unlabeled",
            "label_source": "",
            "primary_label_eligible": False,
            "secondary_label_eligible": False,
            "sold_detected_at": pd.NaT,
            "last_known_active_at": pd.NaT,
            "sale_delay_hours": np.nan,
            "sale_delay_days": np.nan,
            "outcome_source_file": "",
        }
        if item_id in sold_map:
            sold = sold_map[item_id]
            detected_at = sold["sold_detected_at"]
            quality = sold["label_quality"]
            outcome.update(
                {
                    "eventual_sale": 1,
                    "offline_sold_label": 1,
                    "offline_label_eligible": True,
                    "label_quality": quality,
                    "label_source": sold["label_source"],
                    "sold_detected_at": detected_at,
                    "outcome_source_file": sold["outcome_source_file"],
                }
            )
            if pd.notna(obs_ts) and pd.notna(detected_at):
                delay_hours = (detected_at - obs_ts).total_seconds() / 3600.0
                outcome["sale_delay_hours"] = delay_hours
                outcome["sale_delay_days"] = delay_hours / 24.0
                if delay_hours >= 0:
                    outcome["fast_sale_2d"] = int(delay_hours <= 48.0)
                    outcome["fast_sale_7d"] = int(delay_hours <= 168.0)
                    outcome["primary_label_eligible"] = quality in {"exact", "approximate"}
                    outcome["secondary_label_eligible"] = quality in {"exact", "approximate"}
        elif item_id in active_map:
            active = active_map[item_id]
            checked_at = active["last_known_active_at"]
            quality = active["label_quality"]
            outcome.update(
                {
                    "label_quality": quality,
                    "label_source": active["label_source"],
                    "offline_sold_label": 0,
                    "offline_label_eligible": True,
                    "last_known_active_at": checked_at,
                    "outcome_source_file": active["outcome_source_file"],
                }
            )
            if pd.notna(obs_ts) and pd.notna(checked_at):
                active_hours = (checked_at - obs_ts).total_seconds() / 3600.0
                if active_hours >= 48.0:
                    outcome["primary_label_eligible"] = quality in {"exact", "approximate"}
                if active_hours >= 168.0:
                    outcome["secondary_label_eligible"] = quality in {"exact", "approximate"}
        rows.append(outcome)

    label_df = pd.DataFrame(rows, index=base.index)
    out = pd.concat([base.reset_index(drop=True), label_df.reset_index(drop=True)], axis=1)
    return out


def safe_feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = set(FUTURE_ONLY_COLUMNS)
    blocked.update(c for c in df.columns if c.startswith(FUTURE_ONLY_PREFIXES))
    blocked.update(c for c in df.columns if c.lower().endswith("_at") and c not in {"created_at"})
    blocked.update(c for c in df.columns if "status" in c.lower())
    blocked.update(c for c in df.columns if "sold" in c.lower())
    blocked.update({"item_id", "Dataid", "Link", "Images", "LocalImagePaths", "LocalPrimaryImagePath"})
    return [c for c in df.columns if c not in blocked]


def build_datasets(
    *,
    searches: list[str] | None = None,
    out_dir: Path | None = None,
    limit_rows: int | None = None,
) -> list[SearchDatasetResult]:
    ensure_experiment_dirs()
    if out_dir is None:
        out_dir = OFFLINE_RUNS_DIR / run_id("dataset")
    out_dir = assert_experiment_path(out_dir)
    datasets_dir = out_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(searches or [])
    results = []
    for search_dir in list_search_dirs():
        if wanted and search_dir.name not in wanted:
            continue
        df = build_search_dataset(search_dir)
        if limit_rows is not None and limit_rows > 0:
            df = df.head(limit_rows).copy()
        path = datasets_dir / f"{search_dir.name}.csv"
        df.to_csv(path, index=False)
        input_paths = {
            name: str(search_dir / name)
            for name in (
                "big_raw.csv",
                "old_df.csv",
                "sold_df.csv",
                "eventual_sale_check/sold_eventually.csv",
                "eventual_sale_check/not_sold_yet.csv",
                "pipeline_out/deals_ranked.csv",
            )
            if (search_dir / name).exists()
        }
        result = SearchDatasetResult(
            search_name=search_dir.name,
            path=path,
            rows=int(len(df)),
            primary_eligible_rows=int(df["offline_label_eligible"].sum()) if "offline_label_eligible" in df else 0,
            primary_positive_rows=int(df["offline_sold_label"].sum()) if "offline_sold_label" in df else 0,
            weak_or_unlabeled_rows=int(df["label_quality"].isin(["weak", "unlabeled"]).sum()) if "label_quality" in df else int(len(df)),
            input_paths=input_paths,
        )
        results.append(result)

    summary = [r.__dict__ | {"path": str(r.path)} for r in results]
    write_json(out_dir / "dataset_summary.json", {"datasets": summary})
    write_manifest(
        out_dir / "manifest.json",
        command="build_dataset",
        extra={"dataset_count": len(results), "datasets": summary},
    )
    return results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build normalized deal-finder datasets from saved Vinted data.")
    ap.add_argument("--all-searches", action="store_true", help="Build datasets for all search folders.")
    ap.add_argument("--search", action="append", default=[], help="Search folder to include. Can be repeated.")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit-rows", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all_searches and not args.search:
        raise SystemExit("Use --all-searches or at least one --search.")
    out_dir = Path(args.out_dir) if args.out_dir else None
    results = build_datasets(
        searches=None if args.all_searches else args.search,
        out_dir=out_dir,
        limit_rows=args.limit_rows,
    )
    for result in results:
        print(
            f"{result.search_name}: rows={result.rows} "
            f"offline_eligible={result.primary_eligible_rows} "
            f"sold={result.primary_positive_rows} path={result.path}"
        )
    print(f"Output root: {out_dir or EXPERIMENT_ROOT}")


if __name__ == "__main__":
    main()
