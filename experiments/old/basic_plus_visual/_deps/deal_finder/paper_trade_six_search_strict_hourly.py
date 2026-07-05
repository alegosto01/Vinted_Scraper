#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.old.basic_plus_visual._deps.deal_finder.dataset import add_identity
from experiments.old.basic_plus_visual._deps.deal_finder.paper_trade_model_benchmark import (
    add_live_image_features,
    load_sweep_model_metadata,
    resolve_sweep_run,
    score_one_model,
)
from experiments.old.basic_plus_visual._deps.deal_finder.paper_trading import load_search_config_by_folder
from experiments.old.basic_plus_visual._deps.deal_finder.paths import (
    LIVE_RUNS_DIR,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    utc_now_iso,
    write_json,
    write_manifest,
)


DEFAULT_SWEEP_RUN = "sweep_20260510_222252"
DEFAULT_EXCLUDED_SEARCHES = ("Borse_Griffate", "Scarpe_Griffate")
DEFAULT_EXPECTED_SEARCH_COUNT = 6
DEFAULT_RECHECK_INTERVAL_HOURS = 1.0
STATE_FILE = "tracked_state.csv"
HISTORY_FILE = "hourly_history.csv"
EVENTS_FILE = "events.jsonl"
THRESHOLD_FILE = "threshold_snapshot.json"
PLAN_FILE = "search_plan.json"


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path = assert_experiment_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def safe_timestamp_for_path(timestamp: str) -> str:
    return timestamp.replace(":", "").replace("+", "Z")


def parse_truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_threshold_overrides(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    source = raw.get("threshold_overrides", raw) if isinstance(raw, dict) else {}
    overrides: dict[str, float] = {}
    for search_name, value in source.items():
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= threshold <= 1.0:
            overrides[str(search_name).lower()] = threshold
    return overrides


def resolve_six_search_models(
    *,
    sweep_run: Path,
    excluded_searches: tuple[str, ...],
    expected_count: int,
    threshold_overrides: dict[str, float],
) -> list[dict[str, Any]]:
    best_path = sweep_run / "best_by_search.csv"
    if not best_path.exists():
        raise FileNotFoundError(f"best_by_search.csv not found under {sweep_run}")
    best = pd.read_csv(best_path)
    filtered = best.loc[
        best.get("status", pd.Series(index=best.index, dtype=object)).fillna("").eq("trained")
        & best.get("live_ready", pd.Series(index=best.index, dtype=object)).map(parse_truthy)
        & ~best["search_name"].isin(excluded_searches)
    ].copy()
    filtered = filtered.sort_values("search_name", kind="stable").reset_index(drop=True)
    if filtered["search_name"].nunique() != expected_count:
        raise ValueError(
            f"Expected {expected_count} live searches after exclusions, found {filtered['search_name'].nunique()}: "
            f"{sorted(filtered['search_name'].astype(str).unique().tolist())}"
        )

    metadata_rows = load_sweep_model_metadata(sweep_run)
    metadata_index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for metadata in metadata_rows:
        key = (
            str(metadata.get("search_name", "")).lower(),
            str(metadata.get("approach", "")),
            int(metadata.get("seed", 0)),
        )
        metadata_index[key] = metadata

    plan_rows: list[dict[str, Any]] = []
    for row in filtered.to_dict(orient="records"):
        key = (str(row.get("search_name", "")).lower(), str(row.get("approach", "")), int(row.get("seed", 0)))
        if key not in metadata_index:
            raise KeyError(f"Model metadata not found for {key}")
        metadata = dict(metadata_index[key])
        override = threshold_overrides.get(str(row.get("search_name", "")).lower())
        if override is not None:
            metadata["offline_threshold"] = metadata.get("threshold")
            metadata["threshold"] = float(override)
            metadata["threshold_override_source"] = str(path.name) if (path := None) else "cli"
        plan_rows.append(
            {
                "search_name": row["search_name"],
                "approach": row["approach"],
                "seed": int(row.get("seed", 0)),
                "threshold": float(metadata.get("threshold", row.get("threshold", 1.0))),
                "model_metadata": metadata,
            }
        )
    return plan_rows


def ensure_state_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    defaults = {
        "tracking_key": "",
        "item_id": "",
        "SearchName": "",
        "Link": pd.NA,
        "Title": pd.NA,
        "Brand": pd.NA,
        "Size": pd.NA,
        "Price": pd.NA,
        "Likes": pd.NA,
        "model_version": pd.NA,
        "model_threshold": pd.NA,
        "first_seen_at": pd.NA,
        "first_above_threshold_at": pd.NA,
        "last_scored_at": pd.NA,
        "last_snapshot_path": pd.NA,
        "last_rechecked_at": pd.NA,
        "current_market_status": pd.NA,
        "last_recheck_status": pd.NA,
        "sold_at": pd.NA,
        "current_score": pd.NA,
        "max_score": pd.NA,
        "score_observations": 0,
        "times_above_threshold": 0,
    }
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value
    object_cols = [
        "tracking_key",
        "item_id",
        "SearchName",
        "Link",
        "Title",
        "Brand",
        "Size",
        "Price",
        "Likes",
        "model_version",
        "model_threshold",
        "first_seen_at",
        "first_above_threshold_at",
        "last_scored_at",
        "last_snapshot_path",
        "last_rechecked_at",
        "current_market_status",
        "last_recheck_status",
        "sold_at",
    ]
    for col in object_cols:
        if col in out.columns:
            out[col] = out[col].astype("object")
    return out


def tracking_key(search_name: object, item_id: object) -> str:
    return f"{str(search_name).strip().lower()}::{str(item_id).strip()}"


def merge_snapshot_selection(
    tracked: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    observed_at: str,
    snapshot_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tracked = ensure_state_columns(tracked)
    selected = selected.copy()
    if "item_id" not in selected.columns:
        selected = add_identity(selected)
    else:
        selected["item_id"] = selected["item_id"].astype(str).str.strip()
    if "tracking_key" not in selected.columns:
        selected["tracking_key"] = selected.apply(lambda row: tracking_key(row.get("SearchName"), row.get("item_id")), axis=1)
    tracked = tracked.set_index("tracking_key", drop=False) if not tracked.empty else ensure_state_columns(pd.DataFrame()).set_index(pd.Index([], name="tracking_key"))
    history_rows: list[dict[str, Any]] = []
    for row in selected.to_dict(orient="records"):
        key = row["tracking_key"]
        score = float(row.get("model_probability", 0.0))
        status = row.get("MarketStatus")
        payload = {
            "tracking_key": key,
            "item_id": row.get("item_id"),
            "SearchName": row.get("SearchName"),
            "Link": row.get("Link"),
            "Title": row.get("Title"),
            "Brand": row.get("Brand"),
            "Size": row.get("Size"),
            "Price": row.get("Price"),
            "Likes": row.get("Likes"),
            "model_version": row.get("model_version"),
            "model_threshold": row.get("model_threshold"),
            "last_scored_at": observed_at,
            "last_snapshot_path": str(snapshot_path),
            "current_market_status": status,
            "current_score": score,
        }
        if key in tracked.index:
            tracked.at[key, "last_scored_at"] = observed_at
            tracked.at[key, "last_snapshot_path"] = str(snapshot_path)
            tracked.at[key, "current_market_status"] = status
            tracked.at[key, "current_score"] = score
            tracked.at[key, "max_score"] = max(float(tracked.at[key, "max_score"] or score), score)
            tracked.at[key, "score_observations"] = int(tracked.at[key, "score_observations"] or 0) + 1
            tracked.at[key, "times_above_threshold"] = int(tracked.at[key, "times_above_threshold"] or 0) + 1
        else:
            payload.update(
                {
                    "first_seen_at": observed_at,
                    "first_above_threshold_at": observed_at,
                    "max_score": score,
                    "score_observations": 1,
                    "times_above_threshold": 1,
                }
            )
            tracked.loc[key] = {col: payload.get(col, pd.NA) for col in ensure_state_columns(pd.DataFrame()).columns}
            for col, value in payload.items():
                tracked.at[key, col] = value
        history_rows.append(
            {
                "tracking_key": key,
                "item_id": row.get("item_id"),
                "SearchName": row.get("SearchName"),
                "event_type": "threshold_hit",
                "observed_at": observed_at,
                "market_status": status,
                "model_probability": score,
                "model_threshold": row.get("model_threshold"),
                "snapshot_path": str(snapshot_path),
            }
        )
    tracked = tracked.reset_index(drop=True)
    return ensure_state_columns(tracked), pd.DataFrame(history_rows)


def due_recheck_mask(tracked: pd.DataFrame, now: pd.Timestamp, interval_hours: float) -> pd.Series:
    if tracked.empty:
        return pd.Series(False, index=tracked.index)
    first_seen = pd.to_datetime(tracked.get("first_seen_at"), errors="coerce", utc=True)
    last_rechecked = pd.to_datetime(tracked.get("last_rechecked_at"), errors="coerce", utc=True)
    sold_at = pd.to_datetime(tracked.get("sold_at"), errors="coerce", utc=True)
    never_checked_due = last_rechecked.isna() & ((now - first_seen).dt.total_seconds() >= interval_hours * 3600.0)
    checked_due = last_rechecked.notna() & ((now - last_rechecked).dt.total_seconds() >= interval_hours * 3600.0)
    return sold_at.isna() & (never_checked_due | checked_due)


def apply_recheck_results(tracked: pd.DataFrame, checked: pd.DataFrame, *, checked_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tracked = ensure_state_columns(tracked)
    if checked.empty or tracked.empty:
        return tracked, pd.DataFrame()
    checked = checked.copy()
    if "item_id" not in checked.columns or checked["item_id"].astype(str).str.strip().eq("").all():
        checked = add_identity(checked)
    else:
        checked["item_id"] = checked["item_id"].astype(str).str.strip()
    if "tracking_key" not in checked.columns or checked["tracking_key"].astype(str).str.strip().eq("").all():
        checked["tracking_key"] = checked.apply(lambda row: tracking_key(row.get("SearchName"), row.get("item_id")), axis=1)
    tracked = tracked.set_index("tracking_key", drop=False)
    history_rows: list[dict[str, Any]] = []
    for row in checked.to_dict(orient="records"):
        key = row.get("tracking_key")
        if key not in tracked.index:
            continue
        rechecked_at = row.get("rechecked_at") or utc_now_iso()
        market_status = row.get("MarketStatus")
        last_check_status = row.get("LastCheckStatus")
        status = market_status if pd.notna(market_status) and str(market_status).strip() else last_check_status
        tracked.at[key, "last_rechecked_at"] = rechecked_at
        tracked.at[key, "last_recheck_status"] = last_check_status
        tracked.at[key, "current_market_status"] = status
        if str(status or "").strip().lower() == "sold" and pd.isna(tracked.at[key, "sold_at"]):
            tracked.at[key, "sold_at"] = rechecked_at
        history_rows.append(
            {
                "tracking_key": key,
                "item_id": row.get("item_id"),
                "SearchName": row.get("SearchName"),
                "event_type": "recheck",
                "observed_at": rechecked_at,
                "market_status": status,
                "model_probability": pd.NA,
                "model_threshold": tracked.at[key, "model_threshold"],
                "snapshot_path": str(checked_path),
            }
        )
    tracked = tracked.reset_index(drop=True)
    return ensure_state_columns(tracked), pd.DataFrame(history_rows)


def run_iteration(
    *,
    out_dir: Path,
    plan_rows: list[dict[str, Any]],
    pages_to_scrape: int,
    recheck_interval_hours: float,
    enable_live_image_features: bool,
    image_timeout: float,
    dry_run: bool,
) -> dict[str, Any]:
    ensure_experiment_dirs()
    out_dir = assert_experiment_path(out_dir)
    raw_dir = out_dir / "raw_snapshots"
    scored_dir = out_dir / "scored_snapshots"
    rechecks_dir = out_dir / "rechecks"
    reports_dir = out_dir / "reports"
    plots_dir = out_dir / "plots"
    for path in (raw_dir, scored_dir, rechecks_dir, reports_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    plan_payload = [
        {
            "search_name": row["search_name"],
            "approach": row["approach"],
            "seed": row["seed"],
            "threshold": row["threshold"],
            "artifact_path": row["model_metadata"].get("artifact_path"),
        }
        for row in plan_rows
    ]
    write_json(out_dir / PLAN_FILE, {"searches": plan_payload})
    write_json(out_dir / THRESHOLD_FILE, {"thresholds": {row["search_name"]: row["threshold"] for row in plan_rows}})

    result = {
        "run_dir": str(out_dir),
        "search_count": len(plan_rows),
        "dry_run": bool(dry_run),
        "pages_to_scrape": int(pages_to_scrape),
        "recheck_interval_hours": float(recheck_interval_hours),
        "enable_live_image_features": bool(enable_live_image_features),
        "image_timeout": float(image_timeout),
        "snapshot_order": "collect_then_recheck",
        "snapshots": [],
        "image_stats": [],
    }
    if dry_run:
        return result

    tracked_path = out_dir / STATE_FILE
    history_path = out_dir / HISTORY_FILE
    tracked = ensure_state_columns(pd.read_csv(tracked_path)) if tracked_path.exists() else ensure_state_columns(pd.DataFrame())
    history_chunks: list[pd.DataFrame] = []

    config_by_folder = load_search_config_by_folder([row["search_name"] for row in plan_rows])
    from simple_scraper import Simple_scraper
    from scraping_options import _update_market_status_for_df

    for plan_row in plan_rows:
        search_name = plan_row["search_name"]
        search = config_by_folder.get(search_name)
        if search is None:
            continue
        timestamp = utc_now_iso()
        search_count = int(pd.Timestamp.now(tz="UTC").timestamp())
        scraper = Simple_scraper()
        rows = scraper.scrape_products_serial(search, search_count, pages_to_scrape=pages_to_scrape, get_images=False)
        candidates = pd.DataFrame(rows)
        if candidates.empty:
            continue
        candidates["SearchName"] = search_name
        candidates["snapshot_at"] = timestamp
        candidates = add_identity(candidates)
        if enable_live_image_features:
            candidates, image_stats = add_live_image_features(
                candidates,
                out_dir=out_dir,
                search_name=search_name,
                timeout=image_timeout,
            )
            result["image_stats"].append(image_stats)
        raw_path = raw_dir / f"{search_name}_raw_{safe_timestamp_for_path(timestamp)}.csv"
        candidates.to_csv(raw_path, index=False)

        scored_all_variants, skipped = score_one_model(candidates, plan_row["model_metadata"])
        if skipped or scored_all_variants.empty:
            result["snapshots"].append(
                {
                    "search_name": search_name,
                    "raw_path": str(raw_path),
                    "scored_path": "",
                    "raw_rows": int(len(candidates)),
                    "selected_rows": 0,
                    "skipped_reason": (skipped or {}).get("reason", "scoring returned no rows"),
                }
            )
            continue
        scored = scored_all_variants.loc[scored_all_variants["threshold_label"] == "strict"].copy()
        scored["tracking_key"] = scored.apply(lambda row: tracking_key(row.get("SearchName"), row.get("item_id")), axis=1)
        scored_path = scored_dir / f"{search_name}_scored_{safe_timestamp_for_path(timestamp)}.csv"
        scored.to_csv(scored_path, index=False)
        selected = scored.loc[scored["above_threshold"]].copy()
        if not selected.empty:
            tracked, selection_history = merge_snapshot_selection(tracked, selected, observed_at=timestamp, snapshot_path=scored_path)
            if not selection_history.empty:
                history_chunks.append(selection_history)
        result["snapshots"].append(
            {
                "search_name": search_name,
                "raw_path": str(raw_path),
                "scored_path": str(scored_path),
                "raw_rows": int(len(candidates)),
                "selected_rows": int(len(selected)),
            }
        )

    now = pd.Timestamp.now(tz="UTC")
    recheck_mask = due_recheck_mask(tracked, now, recheck_interval_hours)
    due = tracked.loc[recheck_mask].copy()
    result["due_rechecks"] = int(len(due))
    if not due.empty:
        checked, _sold = _update_market_status_for_df(
            due,
            max_workers=1,
            delay=0.0,
            fetch_sleep=0.0,
            fetch_max_attempts=1,
        )
        checked["rechecked_at"] = utc_now_iso()
        checked_path = rechecks_dir / f"recheck_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}.csv"
        checked.to_csv(checked_path, index=False)
        tracked, recheck_history = apply_recheck_results(tracked, checked, checked_path=checked_path)
        if not recheck_history.empty:
            history_chunks.append(recheck_history)
        result["checked_path"] = str(checked_path)

    tracked = tracked.sort_values(["SearchName", "first_seen_at", "item_id"], kind="stable").reset_index(drop=True)
    tracked.to_csv(tracked_path, index=False)
    if history_chunks:
        new_history = pd.concat(history_chunks, ignore_index=True)
        if history_path.exists():
            existing_history = pd.read_csv(history_path)
            new_history = pd.concat([existing_history, new_history], ignore_index=True)
        new_history.to_csv(history_path, index=False)
    elif not history_path.exists():
        pd.DataFrame(columns=["tracking_key", "item_id", "SearchName", "event_type", "observed_at", "market_status", "model_probability", "model_threshold", "snapshot_path"]).to_csv(history_path, index=False)

    result["tracked_state_path"] = str(tracked_path)
    result["hourly_history_path"] = str(history_path)
    result["tracked_count"] = int(len(tracked))
    result["sold_count"] = int(pd.to_datetime(tracked.get("sold_at"), errors="coerce", utc=True).notna().sum())
    return result


def run_hourly_experiment(
    *,
    sweep_run: Path,
    out_dir: Path | None,
    threshold_overrides_path: Path | None,
    interval_hours: float,
    recheck_interval_hours: float,
    enable_live_image_features: bool,
    image_timeout: float,
    iterations: int | None,
    pages_to_scrape: int,
    excluded_searches: tuple[str, ...],
    expected_search_count: int,
    dry_run: bool,
) -> dict[str, Any]:
    ensure_experiment_dirs()
    if out_dir is None:
        out_dir = LIVE_RUNS_DIR / run_id("six_search_strict_hourly_loop")
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold_overrides = load_threshold_overrides(threshold_overrides_path)
    plan_rows = resolve_six_search_models(
        sweep_run=sweep_run,
        excluded_searches=excluded_searches,
        expected_count=expected_search_count,
        threshold_overrides=threshold_overrides,
    )
    manifest = {
        "run_dir": str(out_dir),
        "sweep_run": str(sweep_run),
        "iterations": iterations,
        "interval_hours": float(interval_hours),
        "recheck_interval_hours": float(recheck_interval_hours),
        "enable_live_image_features": bool(enable_live_image_features),
        "image_timeout": float(image_timeout),
        "pages_to_scrape": int(pages_to_scrape),
        "excluded_searches": list(excluded_searches),
        "expected_search_count": int(expected_search_count),
        "searches": [row["search_name"] for row in plan_rows],
        "threshold_overrides_path": str(threshold_overrides_path) if threshold_overrides_path else None,
        "thresholds": {row["search_name"]: row["threshold"] for row in plan_rows},
        "snapshot_order": "collect_then_recheck",
        "storage_layout": {
            "raw_snapshots": "raw_snapshots/",
            "scored_snapshots": "scored_snapshots/",
            "tracked_state": STATE_FILE,
            "hourly_history": HISTORY_FILE,
            "rechecks": "rechecks/",
            "plots": "plots/",
            "reports": "reports/",
        },
    }
    write_manifest(out_dir / "manifest.json", command="paper_trade_six_search_strict_hourly", extra=manifest)

    iteration = 0
    event_log = out_dir / EVENTS_FILE
    while iterations is None or iteration < iterations:
        iteration += 1
        event = {"iteration": iteration, "started_at": utc_now_iso(), "status": "started"}
        append_jsonl(event_log, event)
        try:
            result = run_iteration(
                out_dir=out_dir,
                plan_rows=plan_rows,
                pages_to_scrape=pages_to_scrape,
                recheck_interval_hours=recheck_interval_hours,
                enable_live_image_features=enable_live_image_features,
                image_timeout=image_timeout,
                dry_run=dry_run,
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
        append_jsonl(event_log, event)
        write_json(out_dir / "latest_status.json", event)
        print(json.dumps(event, sort_keys=True), flush=True)
        if iterations is not None and iteration >= iterations:
            break
        time.sleep(max(1.0, interval_hours * 3600.0))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the six-search strict hourly live experiment in a separate experiment folder.")
    ap.add_argument("--sweep-run", default=DEFAULT_SWEEP_RUN)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--threshold-overrides-json", default=None)
    ap.add_argument("--interval-hours", type=float, default=1.0)
    ap.add_argument("--recheck-interval-hours", type=float, default=DEFAULT_RECHECK_INTERVAL_HOURS)
    ap.add_argument("--enable-live-image-features", action="store_true")
    ap.add_argument("--image-timeout", type=float, default=8.0)
    ap.add_argument("--iterations", type=int, default=0, help="0 means run until stopped.")
    ap.add_argument("--pages-to-scrape", type=int, default=1)
    ap.add_argument("--expected-search-count", type=int, default=DEFAULT_EXPECTED_SEARCH_COUNT)
    ap.add_argument("--exclude-search", action="append", default=list(DEFAULT_EXCLUDED_SEARCHES))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run_hourly_experiment(
        sweep_run=resolve_sweep_run(args.sweep_run),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        threshold_overrides_path=Path(args.threshold_overrides_json) if args.threshold_overrides_json else None,
        interval_hours=args.interval_hours,
        recheck_interval_hours=args.recheck_interval_hours,
        enable_live_image_features=args.enable_live_image_features,
        image_timeout=args.image_timeout,
        iterations=args.iterations or None,
        pages_to_scrape=args.pages_to_scrape,
        excluded_searches=tuple(args.exclude_search),
        expected_search_count=args.expected_search_count,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()