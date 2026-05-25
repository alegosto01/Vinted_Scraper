#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.benchmark_basic_to_full import cascade_runner as cascade
from experiments.deal_finder.dataset import add_identity
from experiments.deal_finder.model_sweep import normalize_image_sources
from experiments.deal_finder.paper_trade_model_benchmark import (
    download_primary_image,
    safe_timestamp_for_path,
)
from experiments.deal_finder.paper_trading import load_search_config_by_folder
from experiments.full_scrape_model.compare_feature_modalities import add_dino_embedding_columns
from experiments.photo_arbitrage.features import add_photo_features
from experiments.photo_arbitrage.quality_methods import (
    DEFAULT_AESTHETIC_MODEL,
    DEFAULT_DINO_MODEL,
    DEFAULT_PYIQA_MODEL,
    MethodConfig,
    add_quality_method_scores,
)
from experiments.time_to_sell.paths import (
    EXPERIMENT_ROOT,
    ROOT,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    utc_now_iso,
    write_json,
    write_manifest,
)


DEFAULT_SEARCHES = (
    "nike",
    "ps4",
    "gucci",
    "prada",
    "griffati_uomo_all",
    "griffati_donna_all",
)
SEARCH_ALIASES = {
    "griffati_uomo": "griffati_uomo_all",
    "griffati_donna": "griffati_donna_all",
}
DEFAULT_BALANCE_WINDOWS = (1, 2, 3, 6, 12, 18, 24, 36, 39, 42, 45, 48, 60, 72)
STATE_FILE = cascade.STATE_FILE
EVENTS_FILE = cascade.EVENTS_FILE
LIVE_RUNS_DIR = EXPERIMENT_ROOT / "live_runs"
COLLECTOR_COLUMNS = {
    "cohort": pd.NA,
    "score_band": pd.NA,
    "selection_reason": pd.NA,
    "selected_at": pd.NA,
    "collector_snapshot_at": pd.NA,
    "collector_stage1_rank": np.nan,
    "collector_full_scrape_attempted_at": pd.NA,
    "collector_full_scrape_finished_at": pd.NA,
    "collector_visual_features_path": pd.NA,
}
OBJECT_COLUMNS = (
    "cohort",
    "score_band",
    "selection_reason",
    "selected_at",
    "collector_snapshot_at",
    "collector_full_scrape_attempted_at",
    "collector_full_scrape_finished_at",
    "collector_visual_features_path",
)


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path = assert_experiment_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")


def append_rows(path: Path, rows: pd.DataFrame, *, dedupe_subset: list[str] | None = None) -> None:
    path = assert_experiment_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows.empty:
        return
    out = rows.copy()
    if path.exists():
        existing = read_csv_or_empty(path)
        if not existing.empty:
            out = pd.concat([existing, out], ignore_index=True)
    if dedupe_subset:
        cols = [col for col in dedupe_subset if col in out.columns]
        if cols:
            out = out.drop_duplicates(subset=cols, keep="last")
    out.to_csv(path, index=False)


def normalize_searches(searches: list[str] | None) -> list[str]:
    if not searches:
        return list(DEFAULT_SEARCHES)
    out: list[str] = []
    for search in searches:
        key = str(search).strip().lower()
        out.append(SEARCH_ALIASES.get(key, key))
    return out


def add_tracking_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_identity(frame)
    if "SearchName" not in out.columns:
        out["SearchName"] = ""
    out["tracking_key"] = out.apply(lambda row: cascade.tracking_key(row.get("SearchName"), row.get("item_id")), axis=1)
    return out


def ensure_collector_state(frame: pd.DataFrame) -> pd.DataFrame:
    out = cascade.ensure_state(frame.copy())
    missing = {col: value for col, value in COLLECTOR_COLUMNS.items() if col not in out.columns}
    if missing:
        out = pd.concat([out, pd.DataFrame(missing, index=out.index)], axis=1)
    for col in OBJECT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("object")
    if "tracking_key" in out.columns:
        out["tracking_key"] = out["tracking_key"].fillna("").astype(str)
    return out


def bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def existing_tracking_keys(tracked: pd.DataFrame) -> set[str]:
    if tracked.empty:
        return set()
    tracked = add_tracking_key(tracked)
    return set(tracked["tracking_key"].astype(str))


def resolve_collector_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    return cascade.resolve_cascade_plan(
        run_name=args.modality_run,
        stage1_source=args.stage1_source,
        student_run=args.student_run,
        student_objective=args.student_objective,
        student_recall_target=args.student_recall_target,
        student_precision_target=args.student_precision_target,
        student_binary_target=args.student_binary_target,
        searches=normalize_searches(args.search),
    )


def write_plan_csv(out_dir: Path, plan_rows: list[dict[str, Any]]) -> Path:
    plan_path = assert_experiment_path(out_dir / "collector_plan.csv")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "SearchName": row["search_name"],
                "Stage1Source": row["stage1_source"],
                "Stage1Model": row["stage1_approach"],
                "Stage1Threshold": row["stage1_threshold"],
                "StudentRun": row.get("student_run", ""),
                "StudentObjective": row.get("student_objective", ""),
                "StudentBinaryTarget": row.get("student_binary_target", np.nan),
                "Stage2ModelAvailableForReference": row.get("stage2_approach", ""),
                "Stage2ThresholdAvailableForReference": row.get("stage2_threshold", np.nan),
            }
            for row in plan_rows
        ]
    ).to_csv(plan_path, index=False)
    return plan_path


def search_control_deficit(tracked: pd.DataFrame, search_name: str, windows: list[int]) -> dict[str, Any]:
    tracked = ensure_collector_state(tracked)
    if tracked.empty:
        return {"max_deficit": 0, "total_high_sold": 0, "low_controls": 0, "window_deficits": {}}
    scope = tracked[tracked["SearchName"].astype(str) == str(search_name)].copy()
    high = scope[scope["cohort"].astype(str) == "high_score_candidate"].copy()
    low = scope[scope["cohort"].astype(str) == "low_score_control"].copy()
    window_deficits: dict[str, int] = {}
    total_high_sold = 0
    max_deficit = 0
    for hours in windows:
        outcome = cascade.outcome_col(hours)
        evaluated = cascade.evaluated_col(hours)
        if outcome not in scope.columns or evaluated not in scope.columns:
            continue
        high_eval = high[evaluated].notna()
        low_eval = low[evaluated].notna()
        high_sold = int(bool_series(high.loc[high_eval, outcome]).sum()) if not high.empty else 0
        low_unsold = int((~bool_series(low.loc[low_eval, outcome])).sum()) if not low.empty else 0
        deficit = max(0, high_sold - low_unsold)
        window_deficits[cascade.window_label(hours)] = deficit
        total_high_sold += high_sold
        max_deficit = max(max_deficit, deficit)
    return {
        "max_deficit": int(max_deficit),
        "total_high_sold": int(total_high_sold),
        "low_controls": int(len(low)),
        "window_deficits": window_deficits,
    }


def select_scored_items(
    stage1: pd.DataFrame,
    *,
    tracked: pd.DataFrame,
    search_name: str,
    observed_at: str,
    high_per_search: int,
    low_per_search: int,
    low_floor_per_search: int,
    bootstrap_low_per_search: int,
    balance_windows: list[int],
) -> pd.DataFrame:
    if stage1.empty:
        return pd.DataFrame()
    scored = add_tracking_key(stage1)
    scored["Stage1Score"] = pd.to_numeric(scored.get("Stage1Score"), errors="coerce")
    scored = scored[scored["tracking_key"].astype(str).str.len() > 2].copy()
    scored = scored.drop_duplicates("tracking_key", keep="first")
    scored_ranked_high = scored.sort_values("Stage1Score", ascending=False, kind="stable")
    high_pool = scored_ranked_high.head(max(0, int(high_per_search))).copy()
    high_pool_keys = set(high_pool["tracking_key"].astype(str))
    low_pool_base = scored[~scored["tracking_key"].astype(str).isin(high_pool_keys)].copy()
    seen_keys = existing_tracking_keys(tracked)
    high = high_pool[~high_pool["tracking_key"].astype(str).isin(seen_keys)].copy()

    deficit = search_control_deficit(tracked, search_name, balance_windows)
    if deficit["total_high_sold"] == 0 and deficit["low_controls"] < int(bootstrap_low_per_search):
        adaptive_low = int(bootstrap_low_per_search) - int(deficit["low_controls"])
    else:
        adaptive_low = int(deficit["max_deficit"])
    low_take = min(max(0, int(low_per_search)), max(int(low_floor_per_search), adaptive_low))
    low_pool = low_pool_base.sort_values("Stage1Score", ascending=True, kind="stable").head(low_take).copy()
    low = low_pool[~low_pool["tracking_key"].astype(str).isin(seen_keys)].copy()

    selected: list[pd.DataFrame] = []
    if not high.empty:
        high["cohort"] = "high_score_candidate"
        high["score_band"] = "high"
        high["selection_reason"] = f"top_{int(high_per_search)}_stage1_score"
        selected.append(high)
    if not low.empty:
        low["cohort"] = "low_score_control"
        low["score_band"] = "low"
        low["selection_reason"] = (
            f"bottom_stage1_score_low_floor_{int(low_floor_per_search)}_deficit_{int(deficit['max_deficit'])}"
        )
        selected.append(low)
    if not selected:
        return pd.DataFrame()
    out = pd.concat(selected, ignore_index=True)
    out["selected_at"] = observed_at
    out["collector_snapshot_at"] = observed_at
    out["collector_stage1_rank"] = out.get("Stage1Rank", np.nan)
    out["Stage2Status"] = "not_applicable_dataset_collector"
    out["FullScrapeStatus"] = "pending"
    out["QualityMethodStatus"] = "pending"
    return out


def merge_selected(tracked: pd.DataFrame, selected: pd.DataFrame, *, observed_at: str) -> pd.DataFrame:
    tracked = ensure_collector_state(tracked)
    if selected.empty:
        return tracked
    selected = add_tracking_key(selected)
    base_cols = list(ensure_collector_state(pd.DataFrame()).columns)
    missing_selected_cols = {col: pd.NA for col in selected.columns if col not in tracked.columns}
    if missing_selected_cols:
        tracked = pd.concat([tracked, pd.DataFrame(missing_selected_cols, index=tracked.index)], axis=1)
    tracked = ensure_collector_state(tracked)
    indexed = tracked.set_index("tracking_key", drop=False) if not tracked.empty else tracked.set_index(pd.Index([], name="tracking_key"))
    for row in selected.to_dict(orient="records"):
        key = str(row.get("tracking_key") or "")
        if not key:
            continue
        if key not in indexed.index:
            payload = {col: pd.NA for col in base_cols}
            payload.update(row)
            payload["tracking_key"] = key
            payload["first_stage1_pass_at"] = observed_at
            payload["last_seen_at"] = observed_at
            payload["selected_at"] = row.get("selected_at") or observed_at
            indexed.loc[key] = payload
        else:
            indexed.at[key, "last_seen_at"] = observed_at
    return ensure_collector_state(indexed.reset_index(drop=True))


def mark_full_scrape_attempted(tracked: pd.DataFrame, selected: pd.DataFrame, *, attempted_at: str) -> pd.DataFrame:
    tracked = ensure_collector_state(add_tracking_key(tracked))
    selected = add_tracking_key(selected)
    if tracked.empty or selected.empty:
        return tracked
    indexed = tracked.set_index("tracking_key", drop=False)
    for key in selected["tracking_key"].astype(str):
        if key not in indexed.index:
            continue
        existing = indexed.at[key, "collector_full_scrape_attempted_at"]
        if pd.isna(existing) or str(existing).strip() == "":
            indexed.at[key, "collector_full_scrape_attempted_at"] = attempted_at
        status = indexed.at[key, "FullScrapeStatus"]
        if pd.isna(status) or str(status).strip() == "":
            indexed.at[key, "FullScrapeStatus"] = "pending"
    return ensure_collector_state(indexed.reset_index(drop=True))


def primary_image_url(row: pd.Series | dict) -> str:
    getter = row.get
    for column in ("PrimaryImageUrl", "FullImageUrls", "Images", "ImageUrls"):
        for source in normalize_image_sources(getter(column)):
            if str(source).startswith(("http://", "https://")):
                return str(source)
    return ""


def ensure_local_primary_images(rows: pd.DataFrame, *, out_dir: Path, timeout: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = rows.copy()
    cache_root = assert_experiment_path(out_dir / "image_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    local_paths: list[str] = []
    attempted = 0
    ok = 0
    failed = 0
    for _, row in out.iterrows():
        existing = str(row.get("LocalPrimaryImagePath") or "").strip()
        if existing and Path(existing).exists():
            local_paths.append(existing)
            ok += 1
            continue
        url = primary_image_url(row)
        if not url:
            local_paths.append("")
            failed += 1
            continue
        attempted += 1
        try:
            local_path = download_primary_image(
                url=url,
                cache_root=cache_root,
                search_name=str(row.get("SearchName") or "unknown"),
                item_id=row.get("item_id") or row.get("Dataid"),
                timeout=timeout,
            )
            local_paths.append(local_path)
            ok += 1
        except Exception:
            local_paths.append("")
            failed += 1
    out["LocalPrimaryImagePath"] = local_paths
    out["LocalImagePaths"] = out["LocalPrimaryImagePath"].map(
        lambda value: json.dumps([value], ensure_ascii=True) if value else "[]"
    )
    return out, {
        "attempted": int(attempted),
        "local_image_rows": int(ok),
        "failed_or_missing": int(failed),
        "cache_root": str(cache_root),
    }


def add_live_visual_features(
    rows: pd.DataFrame,
    *,
    out_dir: Path,
    image_timeout: float,
    methods: str,
    device: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    with_images, image_stats = ensure_local_primary_images(rows, out_dir=out_dir, timeout=image_timeout)
    featured = add_photo_features(with_images)
    config = MethodConfig(
        methods=tuple(part.strip() for part in methods.split(",") if part.strip()) if methods else ("simple",),
        pyiqa_model=DEFAULT_PYIQA_MODEL,
        aesthetic_model=DEFAULT_AESTHETIC_MODEL,
        dino_model=DEFAULT_DINO_MODEL,
        max_images_per_item=1,
        device=device,
    )
    scored = add_quality_method_scores(featured, config=config)
    scored, embedding_cols = add_dino_embedding_columns(scored)
    stats = image_stats | {"methods": methods, "dino_embedding_cols": int(len(embedding_cols))}
    return scored, stats


def update_full_scrape_state(
    tracked: pd.DataFrame,
    *,
    selected_for_search: pd.DataFrame,
    successes: pd.DataFrame,
    failures: pd.DataFrame,
    visual_path: Path | None,
    finished_at: str,
) -> pd.DataFrame:
    tracked = ensure_collector_state(add_tracking_key(tracked))
    selected = add_tracking_key(selected_for_search)
    success_keys = set(add_tracking_key(successes)["tracking_key"].astype(str)) if not successes.empty else set()
    failure_keys = set(add_tracking_key(failures)["tracking_key"].astype(str)) if not failures.empty else set()
    indexed = tracked.set_index("tracking_key", drop=False)
    for key in selected["tracking_key"].astype(str):
        if key not in indexed.index:
            continue
        attempted_at = indexed.at[key, "collector_full_scrape_attempted_at"]
        if pd.isna(attempted_at) or str(attempted_at).strip() == "":
            indexed.at[key, "collector_full_scrape_attempted_at"] = finished_at
        indexed.at[key, "collector_full_scrape_finished_at"] = finished_at
        if key in success_keys:
            indexed.at[key, "FullScrapeStatus"] = "success"
            indexed.at[key, "QualityMethodStatus"] = "success" if visual_path is not None else "not_requested"
            if visual_path is not None:
                indexed.at[key, "collector_visual_features_path"] = str(visual_path)
        elif key in failure_keys:
            indexed.at[key, "FullScrapeStatus"] = "failed"
            indexed.at[key, "QualityMethodStatus"] = "not_attempted_full_scrape_failed"
        else:
            indexed.at[key, "FullScrapeStatus"] = "missing_result"
            indexed.at[key, "QualityMethodStatus"] = "not_attempted_missing_full_scrape"
    return ensure_collector_state(indexed.reset_index(drop=True))


def write_balance_reports(out_dir: Path, *, balance_windows: list[int]) -> dict[str, Any]:
    out_dir = assert_experiment_path(out_dir)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    tracked = ensure_collector_state(read_csv_or_empty(out_dir / STATE_FILE))
    if tracked.empty:
        result = {"status": "skipped", "reason": "no tracked items", "run_dir": str(out_dir)}
        write_json(reports_dir / "collector_report_summary.json", result)
        return result

    rows: list[dict[str, Any]] = []
    for search in sorted(tracked["SearchName"].dropna().astype(str).unique(), key=str.lower):
        scope = tracked[tracked["SearchName"].astype(str) == search].copy()
        high = scope[scope["cohort"].astype(str) == "high_score_candidate"].copy()
        low = scope[scope["cohort"].astype(str) == "low_score_control"].copy()
        for hours in balance_windows:
            outcome = cascade.outcome_col(hours)
            evaluated = cascade.evaluated_col(hours)
            if outcome not in scope.columns or evaluated not in scope.columns:
                continue
            high_eval = high[evaluated].notna()
            low_eval = low[evaluated].notna()
            high_sold = int(bool_series(high.loc[high_eval, outcome]).sum()) if not high.empty else 0
            high_unsold = int((~bool_series(high.loc[high_eval, outcome])).sum()) if not high.empty else 0
            low_sold = int(bool_series(low.loc[low_eval, outcome]).sum()) if not low.empty else 0
            low_unsold = int((~bool_series(low.loc[low_eval, outcome])).sum()) if not low.empty else 0
            rows.append(
                {
                    "SearchName": search,
                    "window": cascade.window_label(hours),
                    "hours": int(hours),
                    "high_tracked": int(len(high)),
                    "low_tracked": int(len(low)),
                    "high_evaluated": int(high_eval.sum()),
                    "high_sold": high_sold,
                    "high_unsold": high_unsold,
                    "low_evaluated": int(low_eval.sum()),
                    "low_sold": low_sold,
                    "low_unsold": low_unsold,
                    "balanced_ready_pairs": int(min(high_sold, low_unsold)),
                    "control_deficit": int(max(0, high_sold - low_unsold)),
                }
            )
    balance = pd.DataFrame(rows)
    balance_path = reports_dir / "bin_balance.csv"
    balance.to_csv(balance_path, index=False)
    sold_detected = int(pd.to_datetime(tracked.get("sold_at"), errors="coerce", utc=True).notna().sum())
    summary = {
        "status": "reported",
        "run_dir": str(out_dir),
        "tracked_rows": int(len(tracked)),
        "high_score_rows": int((tracked["cohort"].astype(str) == "high_score_candidate").sum()),
        "low_score_rows": int((tracked["cohort"].astype(str) == "low_score_control").sum()),
        "sold_detected_rows": sold_detected,
        "balance_path": str(balance_path),
    }
    write_json(reports_dir / "collector_report_summary.json", summary)
    lines = [
        "# Live Time-To-Sell Bin Collector",
        "",
        f"Run folder: `{out_dir}`",
        "",
        f"- Tracked rows: `{summary['tracked_rows']}`",
        f"- High-score candidates: `{summary['high_score_rows']}`",
        f"- Low-score controls: `{summary['low_score_rows']}`",
        f"- Sold detected so far: `{summary['sold_detected_rows']}`",
        "",
        "## Balanced Rows Ready",
        "",
        "| search | window | high sold | low unsold | balanced pairs | deficit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not balance.empty:
        for _, row in balance.iterrows():
            if int(row.get("hours", 0)) not in {12, 24, 48, 72}:
                continue
            lines.append(
                f"| {row['SearchName']} | {row['window']} | {int(row['high_sold'])} | "
                f"{int(row['low_unsold'])} | {int(row['balanced_ready_pairs'])} | {int(row['control_deficit'])} |"
            )
    report_path = reports_dir / "collector_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["report_path"] = str(report_path)
    return summary


def run_collect_once(args: argparse.Namespace, *, out_dir: Path) -> dict[str, Any]:
    ensure_experiment_dirs()
    out_dir = assert_experiment_path(out_dir)
    for name in ("raw_snapshots", "stage1_scores", "full_items", "visual_features", "rechecks", "reports", "image_cache"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    searches = normalize_searches(args.search)
    plan_rows = resolve_collector_plan(args)
    write_plan_csv(out_dir, plan_rows)
    write_manifest(
        out_dir / "manifest.json",
        command="time_to_sell live_bin_collector",
        extra={
            "collector": "live_bin_collector",
            "searches": searches,
            "stage1_source": args.stage1_source,
            "modality_run": args.modality_run,
            "high_per_search": int(args.high_per_search),
            "low_per_search": int(args.low_per_search),
            "low_floor_per_search": int(args.low_floor_per_search),
            "balance_windows": [int(h) for h in args.balance_windows],
            "collect_every_hours": float(getattr(args, "collect_every_hours", 2.0)),
            "max_recheck_age_hours": args.max_recheck_age_hours,
        },
    )
    if getattr(args, "dry_run", False):
        result = {"status": "dry_run_planned", "run_dir": str(out_dir), "searches": searches}
        write_json(out_dir / "latest_status.json", result)
        return result

    config_by_search = load_search_config_by_folder([row["search_name"] for row in plan_rows])
    observed_at = utc_now_iso()
    safe_ts = safe_timestamp_for_path(observed_at)
    tracked = ensure_collector_state(read_csv_or_empty(out_dir / STATE_FILE))
    summary_rows: list[dict[str, Any]] = []
    selected_by_search: dict[str, pd.DataFrame] = {}

    for plan in plan_rows:
        search = str(plan["search_name"])
        search_config = config_by_search.get(search)
        if search_config is None or not cascade.has_collectable_search_settings(search_config):
            summary_rows.append({"SearchName": search, "status": "skipped_no_search_config"})
            continue
        search_config = cascade.normalized_search_config(search_config)
        raw = cascade.collect_search_snapshot(search, search_config)
        raw_path = assert_experiment_path(out_dir / "raw_snapshots" / f"{search}_{safe_ts}.csv")
        raw.to_csv(raw_path, index=False)
        if raw.empty:
            summary_rows.append({"SearchName": search, "status": "empty_snapshot", "raw_rows": 0})
            continue
        stage1 = cascade.score_stage1(raw, plan["stage1_metadata"], float(plan["stage1_threshold"]))
        stage1_path = assert_experiment_path(out_dir / "stage1_scores" / f"{search}_{safe_ts}.csv")
        stage1.to_csv(stage1_path, index=False)
        selected = select_scored_items(
            stage1,
            tracked=tracked,
            search_name=search,
            observed_at=observed_at,
            high_per_search=args.high_per_search,
            low_per_search=args.low_per_search,
            low_floor_per_search=args.low_floor_per_search,
            bootstrap_low_per_search=args.bootstrap_low_per_search,
            balance_windows=list(args.balance_windows),
        )
        tracked = merge_selected(tracked, selected, observed_at=observed_at)
        selected_by_search[search] = selected
        summary_rows.append(
            {
                "SearchName": search,
                "status": "scored",
                "raw_rows": int(len(raw)),
                "new_high_selected": int((selected.get("cohort", pd.Series(dtype=object)).astype(str) == "high_score_candidate").sum()),
                "new_low_selected": int((selected.get("cohort", pd.Series(dtype=object)).astype(str) == "low_score_control").sum()),
                "stage1_score_min": float(pd.to_numeric(stage1["Stage1Score"], errors="coerce").min()),
                "stage1_score_max": float(pd.to_numeric(stage1["Stage1Score"], errors="coerce").max()),
            }
        )

    tracked.to_csv(out_dir / STATE_FILE, index=False)

    for plan in plan_rows:
        search = str(plan["search_name"])
        selected = selected_by_search.get(search, pd.DataFrame()).copy()
        if selected.empty:
            continue
        selected["collector_full_scrape_attempted_at"] = observed_at
        tracked = mark_full_scrape_attempted(tracked, selected, attempted_at=observed_at)
        tracked.to_csv(out_dir / STATE_FILE, index=False)
        successes, failures = cascade.collect_full_payloads(selected, search_name=search, image_mode=args.full_scrape_image_mode)
        append_rows(out_dir / "full_items" / "items_enriched.csv", add_tracking_key(successes), dedupe_subset=["tracking_key"])
        append_rows(out_dir / "full_items" / "full_scrape_failures.csv", add_tracking_key(failures), dedupe_subset=["tracking_key"])
        visual_path: Path | None = None
        visual_stats: dict[str, Any] = {}
        if not successes.empty and args.quality_methods.strip().lower() not in {"", "none", "off"}:
            visual, visual_stats = add_live_visual_features(
                add_identity(successes),
                out_dir=out_dir,
                image_timeout=args.image_timeout,
                methods=args.quality_methods,
                device=args.quality_device,
            )
            visual = add_tracking_key(visual)
            visual_path = assert_experiment_path(out_dir / "visual_features" / f"{search}_{safe_ts}.csv")
            visual.to_csv(visual_path, index=False)
        finished_at = utc_now_iso()
        tracked = update_full_scrape_state(
            tracked,
            selected_for_search=selected,
            successes=successes,
            failures=failures,
            visual_path=visual_path,
            finished_at=finished_at,
        )
        for row in summary_rows:
            if row.get("SearchName") == search:
                row.update(
                    {
                        "full_requested": int(len(selected)),
                        "full_success": int(len(successes)),
                        "full_failures": int(len(failures)),
                        **{f"visual_{k}": v for k, v in visual_stats.items()},
                    }
                )
        append_jsonl(
            out_dir / EVENTS_FILE,
            {
                "event": "collect_search",
                "at": observed_at,
                "SearchName": search,
                "selected_rows": int(len(selected)),
                "full_success": int(len(successes)),
                "full_failures": int(len(failures)),
            },
        )

    tracked = ensure_collector_state(tracked)
    tracked.to_csv(out_dir / STATE_FILE, index=False)
    summary = pd.DataFrame(summary_rows)
    summary_path = assert_experiment_path(out_dir / "reports" / f"collect_summary_{safe_ts}.csv")
    summary.to_csv(summary_path, index=False)
    report = write_balance_reports(out_dir, balance_windows=list(args.balance_windows))
    result = {
        "status": "collected",
        "run_dir": str(out_dir),
        "snapshot_at": observed_at,
        "summary_path": str(summary_path),
        "tracked_path": str(out_dir / STATE_FILE),
        "tracked_rows": int(len(tracked)),
        "summary": summary_rows,
        "balance_report": report,
    }
    write_json(out_dir / "latest_status.json", result)
    return result


def run_recheck_due(args: argparse.Namespace, *, out_dir: Path) -> dict[str, Any]:
    out_dir = assert_experiment_path(out_dir)
    tracked_path = out_dir / STATE_FILE
    tracked = ensure_collector_state(read_csv_or_empty(tracked_path))
    if tracked.empty:
        result = {"status": "skipped", "reason": "no tracked items", "run_dir": str(out_dir)}
        write_json(out_dir / "latest_status.json", result)
        return result
    now = pd.Timestamp.now(tz="UTC")
    due = tracked[cascade.due_recheck_mask(tracked, now, max_age_hours=args.max_recheck_age_hours)].copy()
    result = {
        "status": "planned",
        "run_dir": str(out_dir),
        "due_rows": int(len(due)),
        "dry_run": bool(getattr(args, "dry_run", False)),
        "max_recheck_age_hours": None if args.max_recheck_age_hours is None else float(args.max_recheck_age_hours),
    }
    if getattr(args, "dry_run", False) or due.empty:
        write_json(out_dir / "recheck_plan.json", result)
        return result

    cascade.ensure_project_imports()
    from scraping_options import _update_market_status_for_df

    due_unique = add_identity(due).drop_duplicates("item_id", keep="first")
    checked, _sold = _update_market_status_for_df(
        due_unique,
        max_workers=max(1, int(args.max_workers)),
        delay=0.0,
        fetch_sleep=0.0,
        fetch_max_attempts=1,
    )
    checked["rechecked_at"] = utc_now_iso()
    checked = add_tracking_key(checked)
    checked_path = assert_experiment_path(out_dir / "rechecks" / f"recheck_{pd.Timestamp.now('UTC').strftime('%Y%m%d_%H%M%S')}.csv")
    checked.to_csv(checked_path, index=False)
    checked_by_key = checked.drop_duplicates("tracking_key", keep="last").set_index("tracking_key")
    tracked = add_tracking_key(tracked)
    for idx, row in tracked.iterrows():
        key = str(row.get("tracking_key") or "")
        if key not in checked_by_key.index:
            continue
        checked_row = checked_by_key.loc[key]
        rechecked_at = checked_row.get("rechecked_at")
        status = checked_row.get("LastCheckStatus", checked_row.get("MarketStatus"))
        tracked.at[idx, "last_rechecked_at"] = rechecked_at
        tracked.at[idx, "last_recheck_status"] = status
        if str(status or "").strip().casefold() == "sold" and pd.isna(tracked.at[idx, "sold_at"]):
            tracked.at[idx, "sold_at"] = rechecked_at
        first_ts = pd.to_datetime(row.get("first_stage1_pass_at"), errors="coerce", utc=True)
        recheck_ts = pd.to_datetime(rechecked_at, errors="coerce", utc=True)
        cascade.update_outcome_windows(tracked, idx, status=status, first_ts=first_ts, recheck_ts=recheck_ts)
    tracked = ensure_collector_state(tracked)
    tracked.to_csv(tracked_path, index=False)
    report = write_balance_reports(out_dir, balance_windows=list(args.balance_windows))
    result |= {
        "status": "checked",
        "checked_path": str(checked_path),
        "checked_rows": int(len(checked)),
        "balance_report": report,
    }
    write_json(out_dir / "latest_status.json", result)
    append_jsonl(out_dir / EVENTS_FILE, {"event": "recheck", "at": utc_now_iso(), **result})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect live time-to-sell data with high-score and low-score cohorts.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out-dir", type=Path, default=None)
        p.add_argument("--modality-run", default=cascade.DEFAULT_MODALITY_RUN)
        p.add_argument("--stage1-source", choices=["teacher_student", "basic_5"], default="teacher_student")
        p.add_argument("--student-run", default=cascade.DEFAULT_STUDENT_RUN)
        p.add_argument("--student-objective", choices=list(cascade.STUDENT_OBJECTIVES), default=cascade.DEFAULT_STUDENT_OBJECTIVE)
        p.add_argument("--student-recall-target", type=float, default=cascade.DEFAULT_STUDENT_RECALL_TARGET)
        p.add_argument("--student-precision-target", type=float, default=cascade.DEFAULT_STUDENT_PRECISION_TARGET)
        p.add_argument("--student-binary-target", type=int, default=cascade.DEFAULT_STUDENT_BINARY_TARGET)
        p.add_argument("--search", action="append", default=[])
        p.add_argument("--balance-windows", type=int, nargs="+", default=list(DEFAULT_BALANCE_WINDOWS))

    collect = sub.add_parser("collect-once")
    add_common(collect)
    collect.add_argument("--dry-run", action="store_true")
    collect.add_argument("--high-per-search", type=int, default=5)
    collect.add_argument("--low-per-search", type=int, default=5)
    collect.add_argument("--low-floor-per-search", type=int, default=5)
    collect.add_argument("--bootstrap-low-per-search", type=int, default=5)
    collect.add_argument("--full-scrape-image-mode", default="html")
    collect.add_argument("--image-timeout", type=float, default=8.0)
    collect.add_argument("--quality-methods", default="simple,aesthetic,dino")
    collect.add_argument("--quality-device", default="auto")
    collect.add_argument("--max-recheck-age-hours", type=float, default=72.0)
    collect.add_argument("--collect-every-hours", type=float, default=2.0)

    recheck = sub.add_parser("recheck-due")
    add_common(recheck)
    recheck.add_argument("--dry-run", action="store_true")
    recheck.add_argument("--max-workers", type=int, default=3)
    recheck.add_argument("--max-recheck-age-hours", type=float, default=72.0)

    report = sub.add_parser("report")
    add_common(report)

    loop = sub.add_parser("run-loop")
    add_common(loop)
    loop.add_argument("--collect-every-hours", type=float, default=2.0)
    loop.add_argument("--sleep-seconds", type=float, default=300.0)
    loop.add_argument("--high-per-search", type=int, default=5)
    loop.add_argument("--low-per-search", type=int, default=5)
    loop.add_argument("--low-floor-per-search", type=int, default=5)
    loop.add_argument("--bootstrap-low-per-search", type=int, default=5)
    loop.add_argument("--full-scrape-image-mode", default="html")
    loop.add_argument("--image-timeout", type=float, default=8.0)
    loop.add_argument("--quality-methods", default="simple,aesthetic,dino")
    loop.add_argument("--quality-device", default="auto")
    loop.add_argument("--max-workers", type=int, default=3)
    loop.add_argument("--max-recheck-age-hours", type=float, default=72.0)

    daemon = sub.add_parser("start-daemon")
    add_common(daemon)
    daemon.add_argument("--collect-every-hours", type=float, default=2.0)
    daemon.add_argument("--sleep-seconds", type=float, default=300.0)
    daemon.add_argument("--high-per-search", type=int, default=5)
    daemon.add_argument("--low-per-search", type=int, default=5)
    daemon.add_argument("--low-floor-per-search", type=int, default=5)
    daemon.add_argument("--bootstrap-low-per-search", type=int, default=5)
    daemon.add_argument("--full-scrape-image-mode", default="html")
    daemon.add_argument("--image-timeout", type=float, default=8.0)
    daemon.add_argument("--quality-methods", default="simple,aesthetic,dino")
    daemon.add_argument("--quality-device", default="auto")
    daemon.add_argument("--max-workers", type=int, default=3)
    daemon.add_argument("--max-recheck-age-hours", type=float, default=72.0)
    daemon.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def resolve_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        return assert_experiment_path(args.out_dir)
    return assert_experiment_path(LIVE_RUNS_DIR / run_id("bin_collector"))


def daemon_command(args: argparse.Namespace, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run-loop",
        "--out-dir",
        str(out_dir),
        "--modality-run",
        str(args.modality_run),
        "--stage1-source",
        str(args.stage1_source),
        "--student-run",
        str(args.student_run),
        "--student-objective",
        str(args.student_objective),
        "--student-recall-target",
        str(args.student_recall_target),
        "--student-precision-target",
        str(args.student_precision_target),
        "--student-binary-target",
        str(args.student_binary_target),
        "--collect-every-hours",
        str(args.collect_every_hours),
        "--sleep-seconds",
        str(args.sleep_seconds),
        "--high-per-search",
        str(args.high_per_search),
        "--low-per-search",
        str(args.low_per_search),
        "--low-floor-per-search",
        str(args.low_floor_per_search),
        "--bootstrap-low-per-search",
        str(args.bootstrap_low_per_search),
        "--full-scrape-image-mode",
        str(args.full_scrape_image_mode),
        "--image-timeout",
        str(args.image_timeout),
        "--quality-methods",
        str(args.quality_methods),
        "--quality-device",
        str(args.quality_device),
        "--max-workers",
        str(args.max_workers),
        "--max-recheck-age-hours",
        str(args.max_recheck_age_hours),
        "--balance-windows",
        *[str(hours) for hours in args.balance_windows],
    ]
    for search in args.search:
        cmd.extend(["--search", str(search)])
    return cmd


def start_daemon(args: argparse.Namespace, *, out_dir: Path) -> dict[str, Any]:
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = assert_experiment_path(args.log_file if args.log_file else out_dir / "collector.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = daemon_command(args, out_dir)
    log_handle = log_file.open("ab", buffering=0)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
        close_fds=True,
    )
    result = {
        "status": "started",
        "pid": int(proc.pid),
        "run_dir": str(out_dir),
        "log_file": str(log_file),
        "command": cmd,
        "started_at": utc_now_iso(),
    }
    write_json(out_dir / "daemon.json", result)
    write_json(out_dir / "latest_status.json", result)
    return result


def main() -> int:
    args = parse_args()
    ensure_experiment_dirs()
    if args.cmd == "collect-once":
        result = run_collect_once(args, out_dir=resolve_out_dir(args))
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
        return 0
    if args.cmd == "recheck-due":
        result = run_recheck_due(args, out_dir=resolve_out_dir(args))
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
        return 0
    if args.cmd == "report":
        result = write_balance_reports(resolve_out_dir(args), balance_windows=list(args.balance_windows))
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
        return 0
    if args.cmd == "run-loop":
        out_dir = resolve_out_dir(args)
        last_collect: pd.Timestamp | None = None
        while True:
            now = pd.Timestamp.now(tz="UTC")
            should_collect = last_collect is None or (now - last_collect).total_seconds() >= float(args.collect_every_hours) * 3600.0
            if should_collect:
                run_collect_once(args, out_dir=out_dir)
                last_collect = pd.Timestamp.now(tz="UTC")
            run_recheck_due(args, out_dir=out_dir)
            write_balance_reports(out_dir, balance_windows=list(args.balance_windows))
            time.sleep(max(float(args.sleep_seconds), 5.0))
    if args.cmd == "start-daemon":
        result = start_daemon(args, out_dir=resolve_out_dir(args))
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
