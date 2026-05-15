#!/usr/bin/env python3
from __future__ import annotations

import argparse
import __main__
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.benchmark_basic_to_full.paths import (
    LIVE_RUNS_DIR,
    ROOT,
    append_jsonl,
    assert_experiment_path,
    ensure_experiment_dirs,
    ensure_project_imports,
    run_id,
    utc_now_iso,
    write_json,
    write_manifest,
)
from experiments.deal_finder.dataset import add_identity
from experiments.deal_finder.model_sweep import RulePriceScorer, normalize_image_sources
from experiments.deal_finder.modeling import load_pickle, score_with_model
from experiments.deal_finder.paper_trade_model_benchmark import (
    download_primary_image,
    safe_timestamp_for_path,
)
from experiments.deal_finder.paper_trading import load_search_config_by_folder
from experiments.full_scrape_model.compare_feature_modalities import (
    DEFAULT_EXCLUDED_SEARCHES,
    add_dino_embedding_columns,
    add_full_engineered_features,
)
from experiments.full_scrape_model.paths import MODELS_DIR as FULL_MODEL_DIR
from experiments.photo_arbitrage.features import add_photo_features
from experiments.photo_arbitrage.quality_methods import (
    DEFAULT_AESTHETIC_MODEL,
    DEFAULT_DINO_MODEL,
    DEFAULT_PYIQA_MODEL,
    MethodConfig,
    add_quality_method_scores,
)


setattr(__main__, "RulePriceScorer", RulePriceScorer)


DEFAULT_MODALITY_RUN = "sold_status_feature_modalities_20260515_full_visual"
DEFAULT_STUDENT_RUN = "student_fullvisual_score_20260515_154011"
DEFAULT_STUDENT_RECALL_TARGET = 0.95
STATE_FILE = "tracked_items.csv"
EVENTS_FILE = "events.jsonl"
WINDOW_HOURS = [*range(1, 25), *range(27, 49, 3), *range(60, 169, 12)]


def window_label(hours: int | float) -> str:
    value = float(hours)
    if value.is_integer():
        return f"{int(value)}h"
    return f"{value:g}h"


def outcome_col(hours: int | float) -> str:
    return f"sold_within_{window_label(hours)}"


def evaluated_col(hours: int | float) -> str:
    return f"evaluated_at_{window_label(hours)}"


def status_col(hours: int | float) -> str:
    return f"status_at_{window_label(hours)}"


def current_recheck_interval_hours(first_seen_at: object, now: pd.Timestamp | None = None) -> float:
    now = now or pd.Timestamp.now(tz="UTC")
    first = pd.to_datetime(first_seen_at, errors="coerce", utc=True)
    if pd.isna(first):
        return 1.0
    age_hours = (now - first).total_seconds() / 3600.0
    if age_hours < 24.0:
        return 1.0
    if age_hours < 48.0:
        return 3.0
    return 12.0


def ensure_outcome_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    missing: dict[str, object] = {}
    for hours in WINDOW_HOURS:
        for col in (outcome_col(hours), evaluated_col(hours), status_col(hours)):
            if col not in out.columns:
                missing[col] = pd.NA
    if missing:
        out = pd.concat([out, pd.DataFrame(missing, index=out.index)], axis=1)
    for hours in WINDOW_HOURS:
        for col in (outcome_col(hours), evaluated_col(hours), status_col(hours)):
            out[col] = out[col].astype("object")
    return out


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def metadata_rows_for_run(run_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(FULL_MODEL_DIR.glob(f"{run_name}_*_metadata.json")):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        artifact = Path(str(metadata.get("artifact_path", "")))
        if not artifact.exists():
            continue
        metadata["metadata_path"] = str(path)
        metadata["artifact_path"] = str(artifact)
        rows.append(metadata)
    if not rows:
        raise FileNotFoundError(f"No full-scrape feature-modality metadata found for run {run_name}")
    return rows


def best_mode_table(run_name: str) -> pd.DataFrame:
    path = ROOT / "data" / "experiments" / "full_scrape_model" / "offline_runs" / run_name / "best_by_search_mode.csv"
    if not path.exists():
        raise FileNotFoundError(f"best_by_search_mode.csv not found: {path}")
    return pd.read_csv(path)


def teacher_student_run_dir(run_name: str) -> Path:
    root = ROOT / "data" / "experiments" / "teacher_student_basic_filter" / "offline_runs"
    if run_name == "latest":
        candidates = sorted(root.glob("student_fullvisual_score_*"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No teacher-student runs found under {root}")
        return candidates[0]
    path = root / run_name
    if not path.exists():
        raise FileNotFoundError(f"Teacher-student run not found: {path}")
    return path


def teacher_student_plan_table(run_name: str, target_recall: float) -> pd.DataFrame:
    run_dir = teacher_student_run_dir(run_name)
    tradeoff_path = run_dir / "target_tradeoff_by_search.csv"
    best_path = run_dir / "best_student_by_search.csv"
    if tradeoff_path.exists():
        table = pd.read_csv(tradeoff_path)
        matches = table[np.isclose(pd.to_numeric(table["target_teacher_recall"], errors="coerce"), float(target_recall))]
        if not matches.empty:
            return matches.copy()
    if best_path.exists():
        table = pd.read_csv(best_path)
        return table.copy()
    raise FileNotFoundError(f"No teacher-student plan table found in {run_dir}")


def load_teacher_student_metadata(path: object) -> dict[str, Any]:
    metadata_path = Path(str(path))
    if metadata_path.suffix == ".pkl":
        metadata_path = metadata_path.with_name(metadata_path.stem + "_metadata.json")
    if not metadata_path.exists():
        metadata_path = Path(str(path).replace(".pkl", "_metadata.json"))
    if not metadata_path.exists():
        raise FileNotFoundError(f"Teacher-student metadata not found for {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifact = Path(str(metadata.get("artifact_path", "")))
    if not artifact.exists():
        raise FileNotFoundError(f"Teacher-student artifact not found: {artifact}")
    metadata["metadata_path"] = str(metadata_path)
    metadata["artifact_path"] = str(artifact)
    metadata["approach"] = metadata.get("student_name") or metadata.get("approach") or metadata_path.stem
    metadata["feature_mode"] = "teacher_student_basic"
    metadata["score_kind"] = "regression_score"
    return metadata


def resolve_cascade_plan(
    *,
    run_name: str,
    stage1_source: str = "teacher_student",
    student_run: str = DEFAULT_STUDENT_RUN,
    student_recall_target: float = DEFAULT_STUDENT_RECALL_TARGET,
    searches: list[str] | None = None,
    include_excluded: bool = False,
) -> list[dict[str, Any]]:
    best = best_mode_table(run_name)
    if not include_excluded:
        best = best.loc[~best["search_name"].astype(str).isin(DEFAULT_EXCLUDED_SEARCHES)].copy()
    if searches:
        wanted = {search.lower() for search in searches}
        best = best.loc[best["search_name"].astype(str).str.lower().isin(wanted)].copy()
    student_plan = pd.DataFrame()
    if stage1_source == "teacher_student":
        student_plan = teacher_student_plan_table(student_run, student_recall_target)
        if not include_excluded:
            student_plan = student_plan.loc[~student_plan["search_name"].astype(str).isin(DEFAULT_EXCLUDED_SEARCHES)].copy()
        if searches:
            wanted = {search.lower() for search in searches}
            student_plan = student_plan.loc[student_plan["search_name"].astype(str).str.lower().isin(wanted)].copy()
    metadata_rows = metadata_rows_for_run(run_name)
    metadata_index = {
        (
            str(row.get("search_name", "")).lower(),
            str(row.get("feature_mode", "")),
            str(row.get("approach", "")),
            int(row.get("seed", 0)),
        ): row
        for row in metadata_rows
    }
    plan_rows: list[dict[str, Any]] = []
    for search in sorted(best["search_name"].dropna().astype(str).unique(), key=str.lower):
        stage2 = best[(best["search_name"].astype(str) == search) & (best["feature_mode"].astype(str) == "full_scrape_plus_visual")]
        if stage2.empty:
            continue
        stage2_row = stage2.iloc[0].to_dict()
        stage2_key = (search.lower(), "full_scrape_plus_visual", str(stage2_row.get("approach")), int(stage2_row.get("seed", 42)))
        if stage2_key not in metadata_index:
            raise KeyError(f"Missing stage metadata for {search}: {stage2_key}")
        if stage1_source == "teacher_student":
            stage1 = student_plan[student_plan["search_name"].astype(str) == search]
            if stage1.empty:
                continue
            stage1_row = stage1.iloc[0].to_dict()
            stage1_metadata = load_teacher_student_metadata(stage1_row.get("artifact_path"))
            stage1_approach = stage1_row.get("student_model")
            stage1_threshold = float(stage1_row.get("student_threshold"))
        elif stage1_source == "basic_5":
            stage1 = best[(best["search_name"].astype(str) == search) & (best["feature_mode"].astype(str) == "basic_5")]
            if stage1.empty:
                continue
            stage1_row = stage1.iloc[0].to_dict()
            stage1_key = (search.lower(), "basic_5", str(stage1_row.get("approach")), int(stage1_row.get("seed", 42)))
            if stage1_key not in metadata_index:
                raise KeyError(f"Missing stage metadata for {search}: {stage1_key}")
            stage1_metadata = metadata_index[stage1_key]
            stage1_approach = stage1_row.get("approach")
            stage1_threshold = float(stage1_row.get("threshold"))
        else:
            raise ValueError(f"Unknown stage1_source: {stage1_source}")
        plan_rows.append(
            {
                "search_name": search,
                "stage1_source": stage1_source,
                "stage1_approach": stage1_approach,
                "stage1_threshold": stage1_threshold,
                "stage1_metadata": stage1_metadata,
                "student_run": student_run if stage1_source == "teacher_student" else "",
                "student_recall_target": float(student_recall_target) if stage1_source == "teacher_student" else np.nan,
                "stage2_approach": stage2_row.get("approach"),
                "stage2_threshold": float(stage2_row.get("threshold")),
                "stage2_metadata": metadata_index[stage2_key],
            }
        )
    return plan_rows


def normalized_search_config(search_config: object) -> object:
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


def has_collectable_search_settings(search_config: object) -> bool:
    return bool(str(getattr(search_config, "search", "") or "").strip() or str(getattr(search_config, "category", "") or "").strip())


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


def prepare_for_metadata(frame: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    out = add_full_engineered_features(frame)
    for col in metadata.get("numeric_features", []) or []:
        if col not in out.columns:
            out[col] = np.nan
    for col in metadata.get("text_features", []) or []:
        if col not in out.columns:
            out[col] = ""
    return out


def score_with_metadata(frame: pd.DataFrame, metadata: dict[str, Any]) -> np.ndarray:
    model = load_pickle(Path(str(metadata["artifact_path"])))
    work = prepare_for_metadata(frame, metadata)
    if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
        scores = score_with_model(model, work)
    else:
        scores = model.predict(work)
    return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)


def score_stage1(candidates: pd.DataFrame, metadata: dict[str, Any], threshold: float) -> pd.DataFrame:
    out = candidates.copy()
    scores = score_with_metadata(out, metadata)
    out["Stage1Model"] = metadata.get("approach")
    out["Stage1Score"] = np.asarray(scores, dtype=float)
    out["Stage1Threshold"] = float(threshold)
    out["Stage1Passed"] = out["Stage1Score"] >= float(threshold)
    out["Stage1Rank"] = out["Stage1Score"].rank(ascending=False, method="first").astype(int)
    return out


def parse_sources(value: object) -> list[str]:
    return normalize_image_sources(value)


def primary_image_url(row: pd.Series | dict) -> str:
    getter = row.get
    for column in ("PrimaryImageUrl", "FullImageUrls", "Images", "ImageUrls"):
        for source in parse_sources(getter(column)):
            if str(source).startswith(("http://", "https://")):
                return str(source)
    return ""


def ensure_local_primary_images(rows: pd.DataFrame, *, out_dir: Path, timeout: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = rows.copy()
    cache_root = assert_experiment_path(out_dir / "image_cache")
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
    out["LocalImagePaths"] = out["LocalPrimaryImagePath"].map(lambda value: json.dumps([value], ensure_ascii=True) if value else "[]")
    return out, {"attempted": int(attempted), "local_image_rows": int(ok), "failed_or_missing": int(failed), "cache_root": str(cache_root)}


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


def collect_full_payloads(rows: pd.DataFrame, *, search_name: str, no_residential: bool, image_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_project_imports()
    from full_scraper import Full_Scraper

    scraper = Full_Scraper()
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        success, failure = scraper.collect_full_item_payload(
            row.to_dict(),
            search_name=search_name,
            reason="stage1_pass",
            no_residential=no_residential,
            image_mode=image_mode,
        )
        if success is not None:
            successes.append(success)
        if failure is not None:
            failures.append(failure)
    return pd.DataFrame(successes), pd.DataFrame(failures)


def score_stage2(full_visual_rows: pd.DataFrame, metadata: dict[str, Any], threshold: float) -> pd.DataFrame:
    out = full_visual_rows.copy()
    scores = score_with_metadata(out, metadata)
    out["Stage2Model"] = metadata.get("approach")
    out["Stage2Score"] = np.asarray(scores, dtype=float)
    out["Stage2Threshold"] = float(threshold)
    out["Stage2Passed"] = out["Stage2Score"] >= float(threshold)
    out["Stage2Rank"] = out["Stage2Score"].rank(ascending=False, method="first").astype(int)
    return out


def tracking_key(search_name: object, item_id: object) -> str:
    return f"{str(search_name).strip().lower()}::{str(item_id).strip()}"


def ensure_state(frame: pd.DataFrame) -> pd.DataFrame:
    out = ensure_outcome_columns(frame.copy())
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
        "first_stage1_pass_at": pd.NA,
        "last_seen_at": pd.NA,
        "last_rechecked_at": pd.NA,
        "last_recheck_status": pd.NA,
        "sold_at": pd.NA,
        "Stage1Model": pd.NA,
        "Stage1Score": np.nan,
        "Stage1Threshold": np.nan,
        "Stage2Model": pd.NA,
        "Stage2Score": np.nan,
        "Stage2Threshold": np.nan,
        "Stage2Passed": False,
        "Stage2Status": "pending",
        "FullScrapeStatus": pd.NA,
        "QualityMethodStatus": pd.NA,
    }
    missing: dict[str, object] = {}
    for col, value in defaults.items():
        if col not in out.columns:
            missing[col] = value
    if missing:
        out = pd.concat([out, pd.DataFrame(missing, index=out.index)], axis=1)
    return out


def merge_tracked(existing: pd.DataFrame, updates: pd.DataFrame, *, observed_at: str) -> pd.DataFrame:
    existing = ensure_state(existing)
    if updates.empty:
        return existing
    updates = add_identity(updates)
    updates["tracking_key"] = updates.apply(lambda row: tracking_key(row.get("SearchName"), row.get("item_id")), axis=1)
    if existing.empty:
        existing = ensure_state(pd.DataFrame())
    existing = existing.set_index("tracking_key", drop=False) if not existing.empty else existing.set_index(pd.Index([], name="tracking_key"))
    for row in updates.to_dict(orient="records"):
        key = row["tracking_key"]
        if key not in existing.index:
            payload = {col: pd.NA for col in ensure_state(pd.DataFrame()).columns}
            payload.update(row)
            payload["first_stage1_pass_at"] = observed_at
            payload["last_seen_at"] = observed_at
            existing.loc[key] = payload
        else:
            for col, value in row.items():
                if col in existing.columns and pd.notna(value):
                    existing.at[key, col] = value
            existing.at[key, "last_seen_at"] = observed_at
    return ensure_state(existing.reset_index(drop=True))


def update_outcome_windows(tracked: pd.DataFrame, idx: int, *, status: object, first_ts: pd.Timestamp, recheck_ts: pd.Timestamp) -> None:
    if pd.isna(first_ts) or pd.isna(recheck_ts):
        return
    status_text = str(status or "").strip()
    is_sold = status_text.casefold() == "sold"
    elapsed = (recheck_ts - first_ts).total_seconds() / 3600.0
    for hours in WINDOW_HOURS:
        out_col = outcome_col(hours)
        eval_col = evaluated_col(hours)
        st_col = status_col(hours)
        if pd.notna(tracked.at[idx, out_col]):
            continue
        if is_sold and elapsed <= float(hours) + 0.25:
            tracked.at[idx, out_col] = True
            tracked.at[idx, eval_col] = recheck_ts.isoformat()
            tracked.at[idx, st_col] = status_text
        elif elapsed >= float(hours):
            tracked.at[idx, eval_col] = recheck_ts.isoformat()
            tracked.at[idx, st_col] = status_text
            if not is_sold:
                tracked.at[idx, out_col] = False


def due_recheck_mask(tracked: pd.DataFrame, now: pd.Timestamp) -> pd.Series:
    if tracked.empty:
        return pd.Series(False, index=tracked.index)
    first = pd.to_datetime(tracked.get("first_stage1_pass_at"), errors="coerce", utc=True)
    last = pd.to_datetime(tracked.get("last_rechecked_at"), errors="coerce", utc=True)
    sold = pd.to_datetime(tracked.get("sold_at"), errors="coerce", utc=True)
    age = (now - first).dt.total_seconds() / 3600.0
    interval = first.map(lambda ts: current_recheck_interval_hours(ts, now=now))
    no_longer_due = age.gt(max(WINDOW_HOURS))
    due_by_interval = last.isna() | ((now - last).dt.total_seconds() / 3600.0 >= interval)
    due_by_window = pd.Series(False, index=tracked.index)
    for hours in WINDOW_HOURS:
        due_by_window |= first.notna() & tracked.get(evaluated_col(hours), pd.Series(pd.NA, index=tracked.index)).isna() & age.ge(float(hours))
    return first.notna() & sold.isna() & ~no_longer_due & (due_by_interval | due_by_window)


def append_rows(path: Path, rows: pd.DataFrame) -> None:
    path = assert_experiment_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows.empty:
        return
    if path.exists():
        existing = read_csv_or_empty(path)
        if not existing.empty:
            rows = pd.concat([existing, rows], ignore_index=True)
    rows.to_csv(path, index=False)


def run_collect_once(
    *,
    out_dir: Path,
    modality_run: str,
    stage1_source: str,
    student_run: str,
    student_recall_target: float,
    searches: list[str] | None,
    dry_run: bool,
    max_stage1_items_per_search: int | None,
    max_full_items_per_search: int | None,
    image_timeout: float,
    quality_methods: str,
    quality_device: str,
) -> dict[str, Any]:
    ensure_experiment_dirs()
    out_dir = assert_experiment_path(out_dir)
    for name in ("raw_snapshots", "stage1_scores", "full_items", "visual_features", "stage2_scores", "rechecks", "reports"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    plan_rows = resolve_cascade_plan(
        run_name=modality_run,
        stage1_source=stage1_source,
        student_run=student_run,
        student_recall_target=student_recall_target,
        searches=searches,
    )
    plan_path = out_dir / "cascade_plan.csv"
    pd.DataFrame(
        [
            {
                "SearchName": row["search_name"],
                "Stage1Source": row["stage1_source"],
                "Stage1Model": row["stage1_approach"],
                "Stage1Threshold": row["stage1_threshold"],
                "StudentRun": row.get("student_run", ""),
                "StudentRecallTarget": row.get("student_recall_target", np.nan),
                "Stage2Model": row["stage2_approach"],
                "Stage2Threshold": row["stage2_threshold"],
            }
            for row in plan_rows
        ]
    ).to_csv(plan_path, index=False)
    result: dict[str, Any] = {
        "run_dir": str(out_dir),
        "modality_run": modality_run,
        "stage1_source": stage1_source,
        "student_run": student_run if stage1_source == "teacher_student" else "",
        "student_recall_target": float(student_recall_target) if stage1_source == "teacher_student" else None,
        "searches": [row["search_name"] for row in plan_rows],
        "dry_run": bool(dry_run),
        "recheck_schedule": "1h until 24h, 3h until 48h, 12h until 7d",
    }
    write_manifest(out_dir / "manifest.json", command="benchmark_basic_to_full collect", extra=result)
    if dry_run:
        write_json(out_dir / "latest_status.json", result | {"status": "dry_run_planned"})
        return result

    ensure_project_imports()
    config_by_search = load_search_config_by_folder([row["search_name"] for row in plan_rows])
    observed_at = utc_now_iso()
    safe_ts = safe_timestamp_for_path(observed_at)
    tracked = read_csv_or_empty(out_dir / STATE_FILE)
    summary_rows: list[dict[str, Any]] = []
    for plan in plan_rows:
        search = plan["search_name"]
        search_config = config_by_search.get(search)
        if search_config is None or not has_collectable_search_settings(search_config):
            summary_rows.append({"SearchName": search, "status": "skipped_no_search_config"})
            continue
        search_config = normalized_search_config(search_config)
        raw = collect_search_snapshot(search, search_config)
        raw_path = out_dir / "raw_snapshots" / f"{search}_{safe_ts}.csv"
        raw.to_csv(raw_path, index=False)
        if raw.empty:
            summary_rows.append({"SearchName": search, "status": "empty_snapshot", "raw_rows": 0})
            continue
        stage1 = score_stage1(raw, plan["stage1_metadata"], plan["stage1_threshold"])
        stage1_path = out_dir / "stage1_scores" / f"{search}_{safe_ts}.csv"
        stage1.to_csv(stage1_path, index=False)
        selected = stage1[stage1["Stage1Passed"]].copy()
        selected = selected.sort_values("Stage1Score", ascending=False, kind="stable")
        if max_stage1_items_per_search is not None and max_stage1_items_per_search > 0:
            selected = selected.head(max_stage1_items_per_search).copy()
        to_full = selected
        if max_full_items_per_search is not None and max_full_items_per_search > 0:
            to_full = selected.head(max_full_items_per_search).copy()
        successes, failures = collect_full_payloads(
            to_full,
            search_name=search,
            no_residential=bool(getattr(search_config, "no_residential", True)),
            image_mode="html",
        )
        append_rows(out_dir / "full_items" / "items_enriched.csv", successes)
        append_rows(out_dir / "full_items" / "full_scrape_failures.csv", failures)
        if not successes.empty:
            successes = add_identity(successes)
            visual, visual_stats = add_live_visual_features(
                successes,
                out_dir=out_dir,
                image_timeout=image_timeout,
                methods=quality_methods,
                device=quality_device,
            )
            visual_path = out_dir / "visual_features" / f"{search}_{safe_ts}.csv"
            visual.to_csv(visual_path, index=False)
            stage2 = score_stage2(visual, plan["stage2_metadata"], plan["stage2_threshold"])
        else:
            visual_stats = {}
            stage2 = pd.DataFrame()
            visual_path = out_dir / "visual_features" / f"{search}_{safe_ts}.csv"
            stage2 = pd.DataFrame()
        stage2_path = out_dir / "stage2_scores" / f"{search}_{safe_ts}.csv"
        stage2.to_csv(stage2_path, index=False)

        selected_updates = selected.copy()
        selected_updates["Stage2Status"] = "not_full_scraped"
        if not failures.empty:
            failures = add_identity(failures)
            selected_updates.loc[selected_updates["item_id"].astype(str).isin(failures["item_id"].astype(str)), "Stage2Status"] = "full_scrape_failed"
        if not stage2.empty:
            stage2_small = stage2[
                [
                    col
                    for col in [
                        "item_id",
                        "Stage2Model",
                        "Stage2Score",
                        "Stage2Threshold",
                        "Stage2Passed",
                        "FullScrapeStatus",
                        "QualityMethodStatus",
                    ]
                    if col in stage2.columns
                ]
            ].drop_duplicates("item_id", keep="last")
            selected_updates = selected_updates.merge(stage2_small, on="item_id", how="left", suffixes=("", "_stage2"))
            selected_updates["Stage2Status"] = np.where(selected_updates["Stage2Score"].notna(), "scored", selected_updates["Stage2Status"])
        tracked = merge_tracked(tracked, selected_updates, observed_at=observed_at)
        summary_rows.append(
            {
                "SearchName": search,
                "status": "ok",
                "raw_rows": int(len(raw)),
                "stage1_pass_rows": int(len(selected)),
                "full_requested": int(len(to_full)),
                "full_success": int(len(successes)),
                "full_failures": int(len(failures)),
                "stage2_scored": int(len(stage2)),
                "stage2_pass_rows": int(stage2["Stage2Passed"].sum()) if "Stage2Passed" in stage2.columns else 0,
                **{f"visual_{k}": v for k, v in visual_stats.items()},
            }
        )
        append_jsonl(out_dir / EVENTS_FILE, {"event": "collect_search", "at": observed_at, **summary_rows[-1]})

    tracked = ensure_state(tracked)
    tracked.to_csv(out_dir / STATE_FILE, index=False)
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "reports" / f"collect_summary_{safe_ts}.csv"
    summary.to_csv(summary_path, index=False)
    result |= {
        "status": "collected",
        "snapshot_at": observed_at,
        "summary_path": str(summary_path),
        "tracked_path": str(out_dir / STATE_FILE),
        "tracked_rows": int(len(tracked)),
        "final_selected_rows": int(pd.Series(tracked.get("Stage2Passed", [])).fillna(False).astype(str).str.lower().isin(["true", "1"]).sum()),
        "summary": summary_rows,
    }
    write_json(out_dir / "latest_status.json", result)
    return result


def run_recheck_due(*, out_dir: Path, dry_run: bool, max_workers: int = 3) -> dict[str, Any]:
    ensure_experiment_dirs()
    out_dir = assert_experiment_path(out_dir)
    tracked_path = out_dir / STATE_FILE
    tracked = ensure_state(read_csv_or_empty(tracked_path))
    if tracked.empty:
        result = {"status": "skipped", "reason": "no tracked items", "run_dir": str(out_dir)}
        write_json(out_dir / "latest_status.json", result)
        return result
    now = pd.Timestamp.now(tz="UTC")
    due = tracked[due_recheck_mask(tracked, now)].copy()
    result = {"run_dir": str(out_dir), "due_rows": int(len(due)), "dry_run": bool(dry_run)}
    if dry_run or due.empty:
        write_json(out_dir / "recheck_plan.json", result)
        return result

    ensure_project_imports()
    from scraping_options import _update_market_status_for_df

    due_unique = add_identity(due).drop_duplicates("item_id", keep="first")
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
    checked = add_identity(checked)
    checked["tracking_key"] = checked.apply(lambda row: tracking_key(row.get("SearchName"), row.get("item_id")), axis=1)
    checked_path = out_dir / "rechecks" / f"recheck_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    checked.to_csv(checked_path, index=False)
    checked_by_key = checked.drop_duplicates("tracking_key", keep="last").set_index("tracking_key")
    tracked["tracking_key"] = tracked.apply(lambda row: tracking_key(row.get("SearchName"), row.get("item_id")), axis=1)
    for idx, row in tracked.iterrows():
        key = row.get("tracking_key")
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
        update_outcome_windows(tracked, idx, status=status, first_ts=first_ts, recheck_ts=recheck_ts)
    tracked.to_csv(tracked_path, index=False)
    result |= {"status": "checked", "checked_path": str(checked_path), "checked_rows": int(len(checked))}
    write_json(out_dir / "latest_status.json", result)
    append_jsonl(out_dir / EVENTS_FILE, {"event": "recheck", "at": utc_now_iso(), **result})
    return result


def bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def generate_report(*, out_dir: Path) -> dict[str, Any]:
    out_dir = assert_experiment_path(out_dir)
    tracked = ensure_state(read_csv_or_empty(out_dir / STATE_FILE))
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    if tracked.empty:
        result = {"status": "skipped", "reason": "no tracked items", "run_dir": str(out_dir)}
        write_json(reports_dir / "report_summary.json", result)
        return result
    tracked = ensure_outcome_columns(tracked)
    stage2_pass = bool_series(tracked.get("Stage2Passed", pd.Series(False, index=tracked.index)))
    rows: list[dict[str, Any]] = []
    fp_frames = []
    fn_frames = []
    for hours in WINDOW_HOURS:
        oc = outcome_col(hours)
        ec = evaluated_col(hours)
        evaluated = tracked[ec].notna()
        for group_name, mask in [
            ("final_stage2_pass", stage2_pass),
            ("stage1_pass_stage2_reject", ~stage2_pass),
            ("all_stage1_pass", pd.Series(True, index=tracked.index)),
        ]:
            scope = tracked[evaluated & mask].copy()
            if scope.empty:
                rows.append({"window": window_label(hours), "group": group_name, "evaluated_count": 0, "sold_count": 0, "precision": np.nan})
                continue
            sold = bool_series(scope[oc])
            rows.append(
                {
                    "window": window_label(hours),
                    "hours": float(hours),
                    "group": group_name,
                    "evaluated_count": int(len(scope)),
                    "sold_count": int(sold.sum()),
                    "precision": float(sold.mean()) if len(scope) else np.nan,
                }
            )
            if group_name == "final_stage2_pass":
                fp_frames.append(scope.loc[~sold].assign(FalsePositiveWindow=window_label(hours)))
            if group_name == "stage1_pass_stage2_reject":
                fn_frames.append(scope.loc[sold].assign(FalseNegativeWindow=window_label(hours)))
    precision = pd.DataFrame(rows)
    precision_path = reports_dir / "precision_by_window.csv"
    precision.to_csv(precision_path, index=False)
    false_positives = pd.concat(fp_frames, ignore_index=True) if fp_frames else pd.DataFrame()
    false_negatives = pd.concat(fn_frames, ignore_index=True) if fn_frames else pd.DataFrame()
    false_positives.to_csv(reports_dir / "false_positives.csv", index=False)
    false_negatives.to_csv(reports_dir / "false_negatives.csv", index=False)
    lines = [
        "# Basic To Full Cascade Benchmark",
        "",
        f"Run folder: `{out_dir}`",
        "",
        "Final precision is computed on items that passed both stages.",
        "False negatives are stage-1 pass items rejected by stage 2 that later sold.",
        "",
        "## Current Counts",
        "",
        f"- Stage-1 tracked items: `{len(tracked)}`",
        f"- Stage-2 pass items: `{int(stage2_pass.sum())}`",
        f"- Sold detected so far: `{int(pd.to_datetime(tracked.get('sold_at'), errors='coerce', utc=True).notna().sum())}`",
        "",
        "## Precision Windows",
        "",
        "| window | group | evaluated | sold | precision |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in precision.iterrows():
        precision_value = row.get("precision")
        precision_text = "" if pd.isna(precision_value) else f"{float(precision_value):.3f}"
        lines.append(
            f"| {row.get('window')} | {row.get('group')} | {int(row.get('evaluated_count', 0))} | "
            f"{int(row.get('sold_count', 0))} | {precision_text} |"
        )
    report_path = reports_dir / "benchmark_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "status": "reported",
        "run_dir": str(out_dir),
        "tracked_rows": int(len(tracked)),
        "stage2_pass_rows": int(stage2_pass.sum()),
        "precision_path": str(precision_path),
        "false_positives_path": str(reports_dir / "false_positives.csv"),
        "false_negatives_path": str(reports_dir / "false_negatives.csv"),
        "report_path": str(report_path),
    }
    write_json(reports_dir / "report_summary.json", result)
    return result


def latest_run() -> Path | None:
    if not LIVE_RUNS_DIR.exists():
        return None
    runs = [path for path in LIVE_RUNS_DIR.iterdir() if path.is_dir()]
    return max(runs, key=lambda path: path.stat().st_mtime) if runs else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the basic-to-full+visual cascade benchmark.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    collect = sub.add_parser("collect-once")
    collect.add_argument("--out-dir", default=None)
    collect.add_argument("--modality-run", default=DEFAULT_MODALITY_RUN)
    collect.add_argument("--stage1-source", choices=["teacher_student", "basic_5"], default="teacher_student")
    collect.add_argument("--student-run", default=DEFAULT_STUDENT_RUN)
    collect.add_argument("--student-recall-target", type=float, default=DEFAULT_STUDENT_RECALL_TARGET)
    collect.add_argument("--search", action="append", default=[])
    collect.add_argument("--dry-run", action="store_true")
    collect.add_argument("--max-stage1-items-per-search", type=int, default=None)
    collect.add_argument("--max-full-items-per-search", type=int, default=None)
    collect.add_argument("--image-timeout", type=float, default=8.0)
    collect.add_argument("--quality-methods", default="simple,pyiqa,aesthetic,dino")
    collect.add_argument("--quality-device", default="auto")

    recheck = sub.add_parser("recheck-due")
    recheck.add_argument("--out-dir", default=None)
    recheck.add_argument("--dry-run", action="store_true")
    recheck.add_argument("--max-workers", type=int, default=3)

    report = sub.add_parser("report")
    report.add_argument("--out-dir", default=None)

    loop = sub.add_parser("run-loop")
    loop.add_argument("--out-dir", default=None)
    loop.add_argument("--modality-run", default=DEFAULT_MODALITY_RUN)
    loop.add_argument("--stage1-source", choices=["teacher_student", "basic_5"], default="teacher_student")
    loop.add_argument("--student-run", default=DEFAULT_STUDENT_RUN)
    loop.add_argument("--student-recall-target", type=float, default=DEFAULT_STUDENT_RECALL_TARGET)
    loop.add_argument("--search", action="append", default=[])
    loop.add_argument("--collect-every-hours", type=float, default=1.0)
    loop.add_argument("--sleep-seconds", type=float, default=60.0)
    loop.add_argument("--max-stage1-items-per-search", type=int, default=None)
    loop.add_argument("--max-full-items-per-search", type=int, default=None)
    loop.add_argument("--image-timeout", type=float, default=8.0)
    loop.add_argument("--quality-methods", default="simple,pyiqa,aesthetic,dino")
    loop.add_argument("--quality-device", default="auto")
    return parser.parse_args()


def resolve_out_dir(value: str | None) -> Path:
    if value:
        return assert_experiment_path(Path(value))
    current = latest_run()
    if current is not None:
        return assert_experiment_path(current)
    return assert_experiment_path(LIVE_RUNS_DIR / run_id("cascade"))


def main() -> int:
    args = parse_args()
    ensure_experiment_dirs()
    if args.cmd == "collect-once":
        out_dir = assert_experiment_path(Path(args.out_dir) if args.out_dir else LIVE_RUNS_DIR / run_id("cascade"))
        result = run_collect_once(
            out_dir=out_dir,
            modality_run=args.modality_run,
            stage1_source=args.stage1_source,
            student_run=args.student_run,
            student_recall_target=args.student_recall_target,
            searches=args.search or None,
            dry_run=bool(args.dry_run),
            max_stage1_items_per_search=args.max_stage1_items_per_search,
            max_full_items_per_search=args.max_full_items_per_search,
            image_timeout=args.image_timeout,
            quality_methods=args.quality_methods,
            quality_device=args.quality_device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.cmd == "recheck-due":
        result = run_recheck_due(out_dir=resolve_out_dir(args.out_dir), dry_run=bool(args.dry_run), max_workers=args.max_workers)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.cmd == "report":
        result = generate_report(out_dir=resolve_out_dir(args.out_dir))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.cmd == "run-loop":
        out_dir = assert_experiment_path(Path(args.out_dir) if args.out_dir else LIVE_RUNS_DIR / run_id("cascade_scheduled"))
        last_collect: pd.Timestamp | None = None
        while True:
            now = pd.Timestamp.now(tz="UTC")
            should_collect = last_collect is None or (now - last_collect).total_seconds() >= args.collect_every_hours * 3600.0
            if should_collect:
                run_collect_once(
                    out_dir=out_dir,
                    modality_run=args.modality_run,
                    stage1_source=args.stage1_source,
                    student_run=args.student_run,
                    student_recall_target=args.student_recall_target,
                    searches=args.search or None,
                    dry_run=False,
                    max_stage1_items_per_search=args.max_stage1_items_per_search,
                    max_full_items_per_search=args.max_full_items_per_search,
                    image_timeout=args.image_timeout,
                    quality_methods=args.quality_methods,
                    quality_device=args.quality_device,
                )
                last_collect = now
            run_recheck_due(out_dir=out_dir, dry_run=False)
            generate_report(out_dir=out_dir)
            time.sleep(max(float(args.sleep_seconds), 5.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
