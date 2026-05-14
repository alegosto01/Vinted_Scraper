#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import __main__
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.deal_finder.dataset import add_identity
from experiments.deal_finder.model_sweep import (
    IMAGE_FEATURES,
    RulePriceScorer,
    add_engineered_snapshot_features,
    compute_basic_image_features,
    normalize_image_sources,
)
from experiments.deal_finder.modeling import load_pickle, score_with_model
from experiments.deal_finder.paper_trading import (
    checkpoint_due_mask,
    ensure_outcome_columns,
    load_search_config_by_folder,
    update_outcome_windows,
)
from experiments.deal_finder.paths import (
    LIVE_RUNS_DIR,
    MODELS_DIR,
    assert_experiment_path,
    ensure_experiment_dirs,
    ensure_project_imports,
    read_json,
    run_id,
    utc_now_iso,
    write_json,
    write_manifest,
)

setattr(__main__, "RulePriceScorer", RulePriceScorer)


DEFAULT_OUT_DIR = LIVE_RUNS_DIR / "hourly_all_models_benchmark"
DEFAULT_SWEEP_RUN = "sweep_20260510_222252"
BENCHMARK_TRACKED_FILE = "tracked_model_threshold_items.csv"
BENCHMARK_EVENTS_FILE = "events.jsonl"
SKIPPED_MODELS_FILE = "skipped_model_scores.csv"
THRESHOLD_LABELS = ("strict", "medium", "loose")
IMAGE_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.vinted.it/",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path = assert_experiment_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def safe_timestamp_for_path(timestamp: str) -> str:
    return timestamp.replace(":", "").replace("+", "Z")


def latest_sweep_run() -> Path:
    runs = sorted((MODELS_DIR.parent / "offline_runs").glob("sweep_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError("No sweep_* offline run found under data/experiments/deal_finder/offline_runs.")
    return runs[-1]


def resolve_sweep_run(value: str | None) -> Path:
    if not value:
        return latest_sweep_run()
    path = Path(value)
    if path.exists():
        return path
    candidate = MODELS_DIR.parent / "offline_runs" / value
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Sweep run not found: {value}")


def clamp_probability(value: float) -> float:
    if not np.isfinite(value):
        value = 0.9
    return float(min(0.9999, max(0.01, round(float(value), 4))))


def threshold_variants(base_threshold: float) -> list[dict[str, Any]]:
    strict = clamp_probability(base_threshold)
    medium = clamp_probability(strict - min(0.05, strict * 0.25))
    loose = clamp_probability(strict - min(0.10, strict * 0.50))
    variants = [
        {"threshold_label": "strict", "threshold": strict, "threshold_delta": 0.0},
        {"threshold_label": "medium", "threshold": medium, "threshold_delta": round(medium - strict, 4)},
        {"threshold_label": "loose", "threshold": loose, "threshold_delta": round(loose - strict, 4)},
    ]
    seen: set[tuple[str, float]] = set()
    out = []
    for row in variants:
        key = (row["threshold_label"], row["threshold"])
        if key not in seen:
            out.append(row)
            seen.add(key)
    return out


def load_sweep_model_metadata(sweep_run: Path) -> list[dict[str, Any]]:
    prefix = sweep_run.name
    metadata_rows: list[dict[str, Any]] = []
    for path in sorted(MODELS_DIR.glob(f"{prefix}_*_metadata.json")):
        metadata = read_json(path, {})
        if not isinstance(metadata, dict):
            continue
        artifact_path = Path(str(metadata.get("artifact_path", "")))
        if not artifact_path.exists():
            continue
        metadata = dict(metadata)
        metadata["metadata_path"] = str(path)
        metadata["artifact_path"] = str(artifact_path)
        metadata_rows.append(metadata)
    if not metadata_rows:
        raise FileNotFoundError(f"No model metadata found for sweep run {sweep_run.name}.")
    return metadata_rows


def filter_model_metadata(
    metadata_rows: list[dict[str, Any]],
    searches: list[str] | None,
) -> list[dict[str, Any]]:
    if not searches:
        return sorted(metadata_rows, key=lambda row: (str(row.get("search_name", "")).lower(), str(row.get("approach", ""))))
    wanted = {search.lower() for search in searches}
    return [
        row
        for row in sorted(metadata_rows, key=lambda item: (str(item.get("search_name", "")).lower(), str(item.get("approach", ""))))
        if str(row.get("search_name", "")).lower() in wanted
    ]


def model_searches(metadata_rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row.get("search_name")) for row in metadata_rows if str(row.get("search_name", "")).strip()}, key=str.lower)


def prepare_live_features(candidates: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    out = add_engineered_snapshot_features(candidates)
    numeric_features = list(metadata.get("numeric_features") or [])
    text_features = list(metadata.get("text_features") or [])
    for col in numeric_features:
        if col in out.columns:
            continue
        if col == "has_basic_image_features":
            out[col] = 0.0
        elif col in IMAGE_FEATURES or col.startswith("image_"):
            out[col] = np.nan
        else:
            out[col] = np.nan
    for col in text_features:
        if col not in out.columns:
            out[col] = ""
    if "has_basic_image_features" in out.columns:
        out["has_basic_image_features"] = pd.to_numeric(out["has_basic_image_features"], errors="coerce").fillna(0.0)
    return out


def infer_image_extension(url: str, response: requests.Response) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
        return suffix
    content_type = (response.headers.get("content-type") or "").lower()
    if "webp" in content_type:
        return ".webp"
    if "png" in content_type:
        return ".png"
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    return ".img"


def primary_image_url(row: pd.Series) -> str:
    for column in ("Images", "Image", "image_url"):
        if column not in row.index:
            continue
        for source in normalize_image_sources(row.get(column)):
            if source.startswith(("http://", "https://")):
                return source
    return ""


def download_primary_image(
    *,
    url: str,
    cache_root: Path,
    search_name: str,
    item_id: object,
    timeout: float,
) -> str:
    item_text = str(item_id or "unknown").strip() or "unknown"
    base_path = cache_root / search_name / item_text / "primary"
    for existing in sorted(base_path.parent.glob(base_path.name + ".*")):
        return str(existing.resolve())
    response = requests.get(url, headers=IMAGE_REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    suffix = infer_image_extension(url, response)
    final_path = base_path.with_suffix(suffix)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_bytes(response.content)
    return str(final_path.resolve())


def add_live_image_features(
    candidates: pd.DataFrame,
    *,
    out_dir: Path,
    search_name: str,
    timeout: float = 8.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = candidates.copy()
    cache_root = assert_experiment_path(out_dir / "image_cache")
    rows = []
    local_paths = []
    empty = {name: np.nan for name in IMAGE_FEATURES}
    attempted = 0
    downloaded_or_cached = 0
    failures = 0
    for _, row in out.iterrows():
        url = primary_image_url(row)
        item_id = row.get("item_id") or row.get("Dataid")
        if not url:
            rows.append(empty.copy())
            local_paths.append("")
            continue
        attempted += 1
        try:
            local_path = download_primary_image(
                url=url,
                cache_root=cache_root,
                search_name=search_name,
                item_id=item_id,
                timeout=timeout,
            )
            features = compute_basic_image_features(local_path)
            downloaded_or_cached += 1
        except Exception:
            local_path = ""
            features = empty.copy()
            failures += 1
        rows.append(features)
        local_paths.append(local_path)

    image_df = pd.DataFrame(rows, index=out.index)
    for col in IMAGE_FEATURES:
        out[col] = image_df[col] if col in image_df else np.nan
    out["has_basic_image_features"] = out[IMAGE_FEATURES].notna().any(axis=1).astype(int)
    out["LivePrimaryImagePath"] = local_paths
    stats = {
        "search_name": search_name,
        "attempted_images": int(attempted),
        "feature_rows": int(out["has_basic_image_features"].sum()),
        "missing_or_failed_images": int(failures + max(0, len(out) - attempted)),
        "downloaded_or_cached_images": int(downloaded_or_cached),
        "cache_root": str(cache_root),
    }
    return out, stats


def image_feature_status(candidates: pd.DataFrame, metadata: dict[str, Any]) -> str:
    if not bool(metadata.get("requires_images")):
        return "not_required"
    required = [col for col in metadata.get("numeric_features", []) if col in IMAGE_FEATURES or str(col).startswith("image_")]
    if required and all(col in candidates.columns for col in required):
        return "available"
    return "missing"


def score_one_model(candidates: pd.DataFrame, metadata: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    status = image_feature_status(candidates, metadata)
    if status == "missing":
        skipped = {
            "search_name": metadata.get("search_name"),
            "approach": metadata.get("approach"),
            "reason": "image-derived numeric features are not present in this live snapshot",
            "requires_images": bool(metadata.get("requires_images")),
            "artifact_path": metadata.get("artifact_path"),
            "metadata_path": metadata.get("metadata_path"),
        }
        return pd.DataFrame(), skipped

    work = prepare_live_features(candidates, metadata)
    model = load_pickle(Path(str(metadata["artifact_path"])))
    scores = score_with_model(model, work)

    base = candidates.copy()
    base["model_search_name"] = metadata.get("search_name")
    base["approach"] = metadata.get("approach")
    base["model_kind"] = metadata.get("model_kind")
    base["feature_policy"] = metadata.get("feature_policy")
    base["model_seed"] = metadata.get("seed")
    base["model_artifact"] = metadata.get("artifact_path")
    base["model_metadata"] = metadata.get("metadata_path")
    base["base_threshold"] = clamp_probability(float(metadata.get("threshold", 0.9)))
    base["model_probability"] = np.asarray(scores, dtype=float)
    base["rank"] = base["model_probability"].rank(ascending=False, method="first").astype(int)
    base["image_feature_status"] = status

    expanded = []
    for variant in threshold_variants(float(metadata.get("threshold", 0.9))):
        variant_frame = base.copy()
        variant_frame["threshold_label"] = variant["threshold_label"]
        variant_frame["threshold"] = variant["threshold"]
        variant_frame["threshold_delta"] = variant["threshold_delta"]
        variant_frame["above_threshold"] = variant_frame["model_probability"] >= variant["threshold"]
        expanded.append(variant_frame)
    return pd.concat(expanded, ignore_index=True), None


def collect_search_snapshot(search_name: str, search_config: object) -> pd.DataFrame:
    ensure_project_imports()
    from simple_scraper import Simple_scraper

    scraper = Simple_scraper()
    search_count = int(pd.Timestamp.now(tz="UTC").timestamp())
    rows = scraper.scrape_products_serial(search_config, search_count, pages_to_scrape=1, get_images=False)
    candidates = add_identity(pd.DataFrame(rows))
    if candidates.empty:
        return candidates
    candidates["SearchName"] = search_name
    candidates["snapshot_at"] = utc_now_iso()
    return candidates


def has_collectable_search_settings(search_config: object) -> bool:
    search_text = str(getattr(search_config, "search", "") or "").strip()
    category_text = str(getattr(search_config, "category", "") or "").strip()
    return bool(search_text or category_text)


def normalized_search_config(search_config: object) -> object:
    # Keep production settings untouched; normalize only the in-memory object used by this benchmark pass.
    config = copy.copy(search_config)
    for attr in ("search", "prezzoDa", "prezzoA", "condition", "colore", "brands", "category"):
        value = getattr(config, attr, " ")
        if value is None or str(value).strip() == "":
            setattr(config, attr, " ")
    if not getattr(config, "sort", None):
        setattr(config, "sort", "newest_first")
    if not hasattr(config, "no_residential"):
        setattr(config, "no_residential", True)
    return config


def make_benchmark_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["SearchName"].fillna("").astype(str)
        + "|"
        + frame["item_id"].fillna("").astype(str)
        + "|"
        + frame["approach"].fillna("").astype(str)
        + "|"
        + frame["threshold_label"].fillna("").astype(str)
    )


def append_tracked(out_dir: Path, selected: pd.DataFrame) -> dict[str, Any]:
    tracked_path = out_dir / BENCHMARK_TRACKED_FILE
    if selected.empty:
        if not tracked_path.exists():
            pd.DataFrame().to_csv(tracked_path, index=False)
        return {"tracked_path": str(tracked_path), "new_selected_rows": 0, "total_tracked_rows": int(pd.read_csv(tracked_path).shape[0]) if tracked_path.exists() else 0}

    tracked = selected.copy()
    tracked["tracking_reason"] = "above_threshold"
    tracked["first_tracked_at"] = tracked.get("snapshot_at", utc_now_iso())
    tracked["last_rechecked_at"] = pd.NA
    tracked["last_recheck_status"] = pd.NA
    tracked["benchmark_key"] = make_benchmark_key(tracked)
    tracked = ensure_outcome_columns(tracked)

    if tracked_path.exists():
        existing = pd.read_csv(tracked_path)
        existing = ensure_outcome_columns(existing)
        if "benchmark_key" not in existing.columns and not existing.empty:
            existing = add_identity(existing)
            existing["benchmark_key"] = make_benchmark_key(existing)
        tracked = pd.concat([existing, tracked], ignore_index=True)

    tracked = ensure_outcome_columns(tracked)
    if not tracked.empty:
        tracked = tracked.drop_duplicates(subset=["benchmark_key"], keep="first")
    tracked.to_csv(tracked_path, index=False)
    return {
        "tracked_path": str(tracked_path),
        "new_selected_rows": int(len(selected)),
        "total_tracked_rows": int(len(tracked)),
    }


def recheck_benchmark_due(
    *,
    out_dir: Path,
    due_hours: float = 1.0,
    max_workers: int = 3,
    dry_run: bool = False,
) -> dict[str, Any]:
    tracked_path = out_dir / BENCHMARK_TRACKED_FILE
    if not tracked_path.exists():
        return {"status": "skipped", "reason": f"{BENCHMARK_TRACKED_FILE} not found", "live_run": str(out_dir)}

    tracked = ensure_outcome_columns(pd.read_csv(tracked_path))
    if tracked.empty:
        return {"status": "skipped", "reason": f"{BENCHMARK_TRACKED_FILE} is empty", "live_run": str(out_dir)}

    tracked = add_identity(tracked)
    now = pd.Timestamp.now(tz="UTC")
    last = pd.to_datetime(tracked.get("last_rechecked_at"), errors="coerce", utc=True)
    first = pd.to_datetime(tracked.get("first_tracked_at"), errors="coerce", utc=True)
    first_elapsed_hours = (now - first).dt.total_seconds() / 3600.0
    first_check_due = last.isna() & first.notna() & (first_elapsed_hours >= due_hours)
    repeat_check_due = last.notna() & ((now - last).dt.total_seconds() >= due_hours * 3600.0)
    due_mask = first_check_due | repeat_check_due | checkpoint_due_mask(tracked, now)
    due = tracked.loc[due_mask].copy()
    due_unique = due.drop_duplicates(subset=["item_id"], keep="first").copy()
    result = {
        "live_run": str(out_dir),
        "due_rows": int(len(due)),
        "due_unique_items": int(len(due_unique)),
        "dry_run": bool(dry_run),
        "due_hours": float(due_hours),
        "max_workers": int(max_workers),
    }
    if dry_run or due_unique.empty:
        write_json(out_dir / "benchmark_recheck_plan.json", result)
        tracked.to_csv(tracked_path, index=False)
        return result

    ensure_project_imports()
    from scraping_options import _update_market_status_for_df

    checked, _sold = _update_market_status_for_df(
        due_unique,
        max_workers=max(1, int(max_workers)),
        delay=0.0,
        allow_residential_fallback=False,
        fetch_sleep=0.0,
        fetch_max_attempts=1,
        no_residential=True,
    )
    checked["rechecked_at"] = utc_now_iso()
    checked_path = out_dir / f"benchmark_recheck_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    checked.to_csv(checked_path, index=False)

    checked_by_id = add_identity(checked).drop_duplicates(subset=["item_id"], keep="first").set_index("item_id")
    for idx, row in tracked.iterrows():
        item_id = str(row.get("item_id", "")).strip()
        if not item_id or item_id not in checked_by_id.index:
            continue
        checked_row = checked_by_id.loc[item_id]
        tracked.at[idx, "last_rechecked_at"] = checked_row.get("rechecked_at")
        status = checked_row.get("LastCheckStatus", checked_row.get("MarketStatus"))
        tracked.at[idx, "last_recheck_status"] = status
        first_ts = pd.to_datetime(row.get("first_tracked_at"), errors="coerce", utc=True)
        recheck_ts = pd.to_datetime(checked_row.get("rechecked_at"), errors="coerce", utc=True)
        update_outcome_windows(tracked, idx, status=status, first_ts=first_ts, recheck_ts=recheck_ts)
    tracked.to_csv(tracked_path, index=False)
    result |= {"status": "checked", "checked_path": str(checked_path)}
    write_json(out_dir / "benchmark_recheck_summary.json", result)
    return result


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    grouped = scores.groupby(["SearchName", "approach", "threshold_label", "threshold"], dropna=False)
    return grouped.agg(
        candidates=("item_id", "count"),
        selected=("above_threshold", "sum"),
        max_probability=("model_probability", "max"),
        median_probability=("model_probability", "median"),
    ).reset_index()


def collect_and_score_once(
    *,
    sweep_run: Path,
    searches: list[str] | None,
    out_dir: Path,
    dry_run: bool = False,
    recheck_due_hours: float = 1.0,
    recheck_max_workers: int = 3,
    recheck_after_collect: bool = True,
    enable_live_image_features: bool = False,
    image_timeout: float = 8.0,
) -> dict[str, Any]:
    ensure_experiment_dirs()
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_snapshots").mkdir(parents=True, exist_ok=True)
    (out_dir / "scores").mkdir(parents=True, exist_ok=True)
    (out_dir / "summaries").mkdir(parents=True, exist_ok=True)

    all_metadata = filter_model_metadata(load_sweep_model_metadata(sweep_run), searches)
    selected_searches = model_searches(all_metadata)
    metadata_by_search = {
        search: [row for row in all_metadata if str(row.get("search_name")) == search]
        for search in selected_searches
    }
    plan = {
        "run_dir": str(out_dir),
        "sweep_run": str(sweep_run),
        "dry_run": bool(dry_run),
        "searches": selected_searches,
        "models_total": int(len(all_metadata)),
        "threshold_labels": list(THRESHOLD_LABELS),
        "collection_interval_hours": 1.0,
        "recheck_due_hours": float(recheck_due_hours),
        "recheck_max_workers": int(recheck_max_workers),
        "enable_live_image_features": bool(enable_live_image_features),
        "image_timeout": float(image_timeout),
    }
    if dry_run:
        plan["models_by_search"] = {search: len(rows) for search, rows in metadata_by_search.items()}
        write_json(out_dir / "benchmark_collect_plan.json", plan)
        write_manifest(out_dir / "manifest.json", command="paper_trade_model_benchmark --dry-run", extra=plan)
        return plan

    config_by_folder = load_search_config_by_folder(selected_searches)
    timestamp = utc_now_iso()
    safe_ts = safe_timestamp_for_path(timestamp)
    all_scores = []
    skipped = []
    snapshots = []
    image_stats = []

    for search_name in selected_searches:
        search_config = config_by_folder.get(search_name)
        if search_config is None:
            skipped.append({"search_name": search_name, "approach": "", "reason": "search config not found"})
            continue
        if not has_collectable_search_settings(search_config):
            skipped.append({"search_name": search_name, "approach": "", "reason": "no active search/category settings found"})
            continue
        search_config = normalized_search_config(search_config)
        candidates = collect_search_snapshot(search_name, search_config)
        if candidates.empty:
            snapshots.append({"search_name": search_name, "rows": 0, "path": ""})
            continue
        if enable_live_image_features:
            candidates, stats = add_live_image_features(
                candidates,
                out_dir=out_dir,
                search_name=search_name,
                timeout=image_timeout,
            )
            image_stats.append(stats)

        raw_path = out_dir / "raw_snapshots" / f"{search_name}_snapshot_{safe_ts}.csv"
        candidates.to_csv(raw_path, index=False)
        snapshots.append({"search_name": search_name, "rows": int(len(candidates)), "path": str(raw_path)})

        for metadata in metadata_by_search.get(search_name, []):
            try:
                scored, skipped_row = score_one_model(candidates, metadata)
            except Exception as exc:
                scored = pd.DataFrame()
                skipped_row = {
                    "search_name": search_name,
                    "approach": metadata.get("approach"),
                    "reason": f"{type(exc).__name__}: {exc}",
                    "requires_images": bool(metadata.get("requires_images")),
                    "artifact_path": metadata.get("artifact_path"),
                    "metadata_path": metadata.get("metadata_path"),
                }
            if skipped_row:
                skipped.append(skipped_row)
                continue
            all_scores.append(scored)

    if all_scores:
        scores = pd.concat(all_scores, ignore_index=True)
    else:
        scores = pd.DataFrame()

    score_path = out_dir / "scores" / f"model_threshold_scores_{safe_ts}.csv"
    scores.to_csv(score_path, index=False)
    summary = summarize_scores(scores)
    summary_path = out_dir / "summaries" / f"selection_summary_{safe_ts}.csv"
    summary.to_csv(summary_path, index=False)

    skipped_path = out_dir / SKIPPED_MODELS_FILE
    skipped_df = pd.DataFrame(skipped)
    if skipped_path.exists() and not skipped_df.empty:
        skipped_df = pd.concat([pd.read_csv(skipped_path), skipped_df], ignore_index=True)
    skipped_df.to_csv(skipped_path, index=False)

    selected = scores.loc[scores.get("above_threshold", pd.Series(False, index=scores.index)).fillna(False)].copy() if not scores.empty else pd.DataFrame()
    tracking = append_tracked(out_dir, selected)
    result = plan | {
        "snapshot_at": timestamp,
        "snapshots": snapshots,
        "score_path": str(score_path),
        "summary_path": str(summary_path),
        "scored_rows": int(len(scores)),
        "selected_rows": int(len(selected)),
        "skipped_model_rows": int(len(skipped)),
        "image_stats": image_stats,
        "tracking": tracking,
    }
    if recheck_after_collect:
        result["recheck"] = recheck_benchmark_due(
            out_dir=out_dir,
            due_hours=recheck_due_hours,
            max_workers=recheck_max_workers,
            dry_run=dry_run,
        )
    write_json(out_dir / "latest_status.json", {"status": "ok", "result": result, "finished_at": utc_now_iso()})
    write_manifest(out_dir / "manifest.json", command="paper_trade_model_benchmark", extra=result)
    return result


def run_hourly_benchmark(
    *,
    sweep_run: Path,
    searches: list[str] | None,
    out_dir: Path,
    interval_hours: float,
    iterations: int | None,
    dry_run: bool,
    recheck_due_hours: float,
    recheck_max_workers: int,
    recheck_after_collect: bool,
    enable_live_image_features: bool,
    image_timeout: float,
) -> dict[str, Any]:
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_dir": str(out_dir),
        "sweep_run": str(sweep_run),
        "searches": searches or "all",
        "interval_hours": float(interval_hours),
        "iterations": iterations,
        "dry_run": bool(dry_run),
        "recheck_due_hours": float(recheck_due_hours),
        "recheck_max_workers": int(recheck_max_workers),
        "recheck_after_collect": bool(recheck_after_collect),
        "enable_live_image_features": bool(enable_live_image_features),
        "image_timeout": float(image_timeout),
        "started_at": utc_now_iso(),
    }
    write_manifest(out_dir / "manifest.json", command="paper_trade_model_benchmark hourly", extra=config)
    event_log = out_dir / BENCHMARK_EVENTS_FILE

    iteration = 0
    while iterations is None or iteration < iterations:
        iteration += 1
        append_jsonl(event_log, {"iteration": iteration, "started_at": utc_now_iso(), "status": "started"})
        try:
            result = collect_and_score_once(
                sweep_run=sweep_run,
                searches=searches,
                out_dir=out_dir,
                dry_run=dry_run,
                recheck_due_hours=recheck_due_hours,
                recheck_max_workers=recheck_max_workers,
                recheck_after_collect=recheck_after_collect,
                enable_live_image_features=enable_live_image_features,
                image_timeout=image_timeout,
            )
            event = {"iteration": iteration, "finished_at": utc_now_iso(), "status": "ok", "result": result}
        except Exception as exc:
            event = {
                "iteration": iteration,
                "finished_at": utc_now_iso(),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(out_dir / "latest_status.json", event)
        append_jsonl(event_log, event)
        print(json.dumps(event, sort_keys=True), flush=True)

        if iterations is not None and iteration >= iterations:
            break
        time.sleep(max(1.0, float(interval_hours) * 3600.0))
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark all saved deal-finder models on live first-page snapshots.")
    parser.add_argument("--sweep-run", default=DEFAULT_SWEEP_RUN)
    parser.add_argument("--all-searches", action="store_true", help="Use all searches found in the sweep model metadata.")
    parser.add_argument("--search", action="append", default=[], help="Restrict to one search folder. Can be repeated.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--interval-hours", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=1, help="0 means keep running in this process.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-recheck-after-collect", action="store_true")
    parser.add_argument("--recheck-due-hours", type=float, default=1.0)
    parser.add_argument("--recheck-max-workers", type=int, default=3)
    parser.add_argument("--enable-live-image-features", action="store_true")
    parser.add_argument("--image-timeout", type=float, default=8.0)
    args = parser.parse_args()

    if not args.all_searches and not args.search:
        raise SystemExit("Use --all-searches or at least one --search.")

    run_hourly_benchmark(
        sweep_run=resolve_sweep_run(args.sweep_run),
        searches=None if args.all_searches else args.search,
        out_dir=Path(args.out_dir),
        interval_hours=args.interval_hours,
        iterations=args.iterations or None,
        dry_run=args.dry_run,
        recheck_due_hours=args.recheck_due_hours,
        recheck_max_workers=args.recheck_max_workers,
        recheck_after_collect=not args.no_recheck_after_collect,
        enable_live_image_features=args.enable_live_image_features,
        image_timeout=args.image_timeout,
    )


if __name__ == "__main__":
    main()
