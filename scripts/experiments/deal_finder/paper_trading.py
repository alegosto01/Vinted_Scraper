from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from experiments.deal_finder.dataset import add_identity
from experiments.deal_finder.modeling import load_qualified_searches, score_candidates
from experiments.deal_finder.paths import (
    LIVE_RUNS_DIR,
    ROOT,
    assert_experiment_path,
    ensure_experiment_dirs,
    ensure_project_imports,
    read_json,
    run_id,
    utc_now_iso,
    write_json,
    write_manifest,
)


TRACK_TOP_K = (10, 25, 50)
THRESHOLD_OVERRIDES_FILE = "threshold_overrides.json"
OUTCOME_WINDOWS = (
    ("2h", 2.0),
    ("12h", 12.0),
    ("2d", 48.0),
    ("7d", 168.0),
)
CHECKPOINT_GRACE_HOURS = 1.0
TRACKED_TEXT_COLUMNS = (
    "first_tracked_at",
    "last_rechecked_at",
    "last_recheck_status",
)


def outcome_col(label: str) -> str:
    return f"sold_within_{label}"


def evaluated_col(label: str) -> str:
    return f"evaluated_{label}_at"


def status_col(label: str) -> str:
    return f"status_at_{label}"


def ensure_outcome_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in TRACKED_TEXT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("object")
    for label, _hours in OUTCOME_WINDOWS:
        outcome = outcome_col(label)
        evaluated = evaluated_col(label)
        status = status_col(label)
        if outcome not in out.columns:
            out[outcome] = pd.NA
        if evaluated not in out.columns:
            out[evaluated] = pd.NA
        if status not in out.columns:
            out[status] = pd.NA
        out[outcome] = out[outcome].astype("object")
        out[evaluated] = out[evaluated].astype("object")
        out[status] = out[status].astype("object")
    return out


def parse_boolish(series: pd.Series) -> pd.Series:
    if str(series.dtype).lower() in {"bool", "boolean"}:
        return series.fillna(False).astype(bool)
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def checkpoint_due_mask(tracked: pd.DataFrame, now: pd.Timestamp) -> pd.Series:
    first = pd.to_datetime(tracked.get("first_tracked_at"), errors="coerce", utc=True)
    due = pd.Series(False, index=tracked.index)
    for label, hours in OUTCOME_WINDOWS:
        evaluated = pd.to_datetime(tracked.get(evaluated_col(label)), errors="coerce", utc=True)
        elapsed_hours = (now - first).dt.total_seconds() / 3600.0
        due |= first.notna() & evaluated.isna() & (elapsed_hours >= hours)
    return due


def update_outcome_windows(
    tracked: pd.DataFrame,
    idx: int,
    *,
    status: object,
    first_ts: pd.Timestamp,
    recheck_ts: pd.Timestamp,
) -> None:
    if pd.isna(first_ts) or pd.isna(recheck_ts):
        return
    status_text = str(status or "").strip()
    is_sold = status_text.casefold() == "sold"
    elapsed_hours = (recheck_ts - first_ts).total_seconds() / 3600.0
    for label, window_hours in OUTCOME_WINDOWS:
        existing = tracked.at[idx, outcome_col(label)] if outcome_col(label) in tracked.columns else pd.NA
        if pd.notna(existing):
            continue
        if is_sold and elapsed_hours <= window_hours + CHECKPOINT_GRACE_HOURS:
            tracked.at[idx, outcome_col(label)] = True
            tracked.at[idx, evaluated_col(label)] = recheck_ts.isoformat()
            tracked.at[idx, status_col(label)] = status_text
        elif elapsed_hours >= window_hours:
            tracked.at[idx, evaluated_col(label)] = recheck_ts.isoformat()
            tracked.at[idx, status_col(label)] = status_text
            if not is_sold:
                tracked.at[idx, outcome_col(label)] = False


def _parse_simple_yaml_value(value: str):
    text = value.strip().strip('"').strip("'")
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [part.strip().strip('"').strip("'") for part in inner.split(",") if part.strip()]
    return text


def _load_searches_yaml_fallback(path: Path) -> dict[str, object]:
    defaults = {
        "search": "",
        "prezzoDa": "",
        "prezzoA": "",
        "condition": "",
        "colore": "",
        "brands": "",
        "sort": "newest_first",
        "category": "",
        "enabled": True,
        "tags": [],
        "wrong_words": [],
    }
    searches: dict[str, object] = {}
    current_name = None
    current: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if not raw_line.startswith(" ") and raw_line.rstrip().endswith(":"):
            if current_name and not str(current_name).startswith("#"):
                data = defaults | current
                data.setdefault("folder", current_name)
                searches[str(data.get("folder") or current_name)] = SimpleNamespace(**data)
            current_name = raw_line.strip()[:-1]
            current = {}
            continue
        if current_name and ":" in raw_line:
            key, value = raw_line.split(":", 1)
            current[key.strip()] = _parse_simple_yaml_value(value)
    if current_name:
        data = defaults | current
        data.setdefault("folder", current_name)
        searches[str(data.get("folder") or current_name)] = SimpleNamespace(**data)
    return searches


def load_search_config_by_folder(extra_folders: list[str] | None = None) -> dict[str, object]:
    ensure_project_imports()
    search_config_class = None
    try:
        from config.project_config import settings
        from config.search_loader import SearchConfig, load_searches

        search_config_class = SearchConfig
        searches = load_searches(str(settings.paths.searches_yaml))
        out = {search.folder: search for search in searches.values()}
    except ModuleNotFoundError:
        out = _load_searches_yaml_fallback(ROOT / "data" / "searches.yaml")
    for folder in extra_folders or []:
        if folder and folder not in out:
            if search_config_class is not None:
                out[folder] = search_config_class(search="", folder=folder, enabled=True)
            else:
                out[folder] = SimpleNamespace(search="", folder=folder, enabled=True)
    return out


def latest_live_run() -> Path | None:
    if not LIVE_RUNS_DIR.exists():
        return None
    runs = [p for p in LIVE_RUNS_DIR.iterdir() if p.is_dir()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def load_threshold_overrides(out_dir: Path) -> dict[str, float]:
    path = out_dir / THRESHOLD_OVERRIDES_FILE
    data = read_json(path, {})
    raw = data.get("threshold_overrides", data) if isinstance(data, dict) else {}
    overrides: dict[str, float] = {}
    for search_name, value in raw.items():
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= threshold <= 1.0:
            overrides[str(search_name).lower()] = threshold
    return overrides


def apply_threshold_override(metadata: dict, search_name: str, threshold_overrides: dict[str, float]) -> dict:
    metadata = dict(metadata)
    key = str(search_name).lower()
    if key not in threshold_overrides:
        return metadata
    metadata["offline_threshold"] = metadata.get("threshold")
    metadata["threshold"] = float(threshold_overrides[key])
    metadata["threshold_override_source"] = THRESHOLD_OVERRIDES_FILE
    return metadata


def collect_snapshot(
    *,
    max_searches: int = 3,
    searches: list[str] | None = None,
    offline_run: Path | None = None,
    out_dir: Path | None = None,
    dry_run: bool = False,
) -> dict:
    ensure_experiment_dirs()
    qualified = load_qualified_searches(offline_run)
    if searches:
        wanted = {search.lower() for search in searches}
        qualified = [row for row in qualified if str(row.get("search_name", "")).lower() in wanted]
    qualified = qualified[:max_searches]
    if out_dir is None:
        out_dir = LIVE_RUNS_DIR / run_id("paper")
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "run_dir": str(out_dir),
        "dry_run": bool(dry_run),
        "qualified_count": len(qualified),
        "searches": [q.get("search_name") for q in qualified],
        "snapshot_interval_hours": 1,
    }
    if dry_run or not qualified:
        write_json(out_dir / "collect_plan.json", plan)
        command = "paper_trade_collect --dry-run" if dry_run else "paper_trade_collect --qualified"
        write_manifest(out_dir / "manifest.json", command=command, extra=plan)
        return plan

    threshold_overrides = load_threshold_overrides(out_dir)
    if threshold_overrides:
        plan["threshold_overrides"] = threshold_overrides

    config_by_folder = load_search_config_by_folder([q.get("search_name") for q in qualified])
    ensure_project_imports()
    from simple_scraper import Simple_scraper

    all_tracked = []
    snapshots = []
    for qualified_row in qualified:
        search_name = qualified_row["search_name"]
        search = config_by_folder.get(search_name)
        if search is None:
            continue
        metadata = apply_threshold_override(qualified_row["model_metadata"], search_name, threshold_overrides)
        scraper = Simple_scraper()
        timestamp = utc_now_iso()
        search_count = int(pd.Timestamp.now('UTC').timestamp())
        rows = scraper.scrape_products_serial(search, search_count, pages_to_scrape=1, get_images=False)
        candidates = pd.DataFrame(rows)
        if candidates.empty:
            continue
        candidates["SearchName"] = search_name
        candidates["snapshot_at"] = timestamp
        candidates = add_identity(candidates)
        scored = score_candidates(candidates, metadata)
        snapshot_path = out_dir / f"{search_name}_snapshot_{timestamp.replace(':', '').replace('+', 'Z')}.csv"
        scored.to_csv(snapshot_path, index=False)
        track_mask = scored["above_threshold"].copy()
        for k in TRACK_TOP_K:
            track_mask |= scored["rank"] <= k
        tracked = scored.loc[track_mask].copy()
        tracked["tracking_reason"] = tracked.apply(
            lambda row: "above_threshold" if bool(row["above_threshold"]) else f"top_{int(row['rank'])}",
            axis=1,
        )
        tracked["first_tracked_at"] = timestamp
        tracked["last_rechecked_at"] = pd.NA
        tracked["last_recheck_status"] = pd.NA
        tracked = ensure_outcome_columns(tracked)
        all_tracked.append(tracked)
        snapshots.append({"search_name": search_name, "rows": int(len(scored)), "path": str(snapshot_path)})

    tracked_path = out_dir / "tracked_items.csv"
    if all_tracked:
        tracked_df = pd.concat(all_tracked, ignore_index=True)
    else:
        tracked_df = pd.DataFrame()
    if tracked_path.exists():
        existing_tracked = pd.read_csv(tracked_path)
        existing_tracked = ensure_outcome_columns(existing_tracked)
        if not existing_tracked.empty:
            tracked_df = pd.concat([existing_tracked, tracked_df], ignore_index=True)
    if not tracked_df.empty:
        tracked_df = ensure_outcome_columns(tracked_df)
        tracked_df = tracked_df.drop_duplicates(subset=["item_id"], keep="first")
    tracked_df.to_csv(tracked_path, index=False)
    result = plan | {"snapshots": snapshots, "tracked_path": str(tracked_path), "tracked_count": int(len(tracked_df))}
    write_json(out_dir / "collect_summary.json", result)
    write_manifest(out_dir / "manifest.json", command="paper_trade_collect", extra=result)
    return result


def recheck_due(
    *,
    live_run: Path | None = None,
    due_hours: float = 12.0,
    above_threshold_only: bool = False,
    dry_run: bool = False,
) -> dict:
    ensure_experiment_dirs()
    live_run = live_run or latest_live_run()
    if live_run is None:
        return {"status": "skipped", "reason": "no live run found"}
    live_run = assert_experiment_path(live_run)
    tracked_path = live_run / "tracked_items.csv"
    if not tracked_path.exists():
        return {"status": "skipped", "reason": "tracked_items.csv not found", "live_run": str(live_run)}

    tracked = ensure_outcome_columns(pd.read_csv(tracked_path))
    if tracked.empty:
        return {"status": "skipped", "reason": "tracked_items.csv is empty", "live_run": str(live_run)}
    now = pd.Timestamp.now(tz="UTC")
    last = pd.to_datetime(tracked.get("last_rechecked_at"), errors="coerce", utc=True)
    due_mask = last.isna() | ((now - last).dt.total_seconds() >= due_hours * 3600.0) | checkpoint_due_mask(tracked, now)
    if above_threshold_only:
        due_mask &= parse_boolish(tracked.get("above_threshold", pd.Series(False, index=tracked.index)))
    due = tracked.loc[due_mask].copy()
    result = {
        "live_run": str(live_run),
        "due_count": int(len(due)),
        "dry_run": bool(dry_run),
        "due_hours": float(due_hours),
        "above_threshold_only": bool(above_threshold_only),
    }
    if dry_run or due.empty:
        write_json(live_run / "recheck_plan.json", result)
        tracked.to_csv(tracked_path, index=False)
        return result

    ensure_project_imports()
    from scraping_options import _update_market_status_for_df

    checked, _sold = _update_market_status_for_df(
        due,
        max_workers=1,
        delay=0.0,
        fetch_sleep=0.0,
        fetch_max_attempts=1,
    )
    checked["rechecked_at"] = utc_now_iso()
    checked_path = live_run / f"recheck_{pd.Timestamp.now('UTC').strftime('%Y%m%d_%H%M%S')}.csv"
    checked.to_csv(checked_path, index=False)

    checked_by_id = add_identity(checked).set_index("item_id")
    tracked = add_identity(tracked)
    for idx, row in tracked.iterrows():
        item_id = row.get("item_id")
        if item_id not in checked_by_id.index:
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
    write_json(live_run / "recheck_summary.json", result)
    return result


def collect_main() -> None:
    ap = argparse.ArgumentParser(description="Collect paper-trading first-page snapshots for qualified searches.")
    ap.add_argument("--qualified", action="store_true", help="Use latest qualified offline searches.")
    ap.add_argument("--max-searches", type=int, default=3)
    ap.add_argument("--search", action="append", default=[], help="Restrict collection to a qualified search name.")
    ap.add_argument("--offline-run", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.qualified:
        raise SystemExit("Use --qualified to collect from qualified offline searches.")
    result = collect_snapshot(
        max_searches=args.max_searches,
        searches=args.search or None,
        offline_run=Path(args.offline_run) if args.offline_run else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        dry_run=args.dry_run,
    )
    print(result)


def recheck_main() -> None:
    ap = argparse.ArgumentParser(description="Recheck due paper-trading tracked items.")
    ap.add_argument("--due", action="store_true")
    ap.add_argument("--live-run", default=None)
    ap.add_argument("--due-hours", type=float, default=12.0)
    ap.add_argument("--above-threshold-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.due:
        raise SystemExit("Use --due to recheck due tracked items.")
    result = recheck_due(
        live_run=Path(args.live_run) if args.live_run else None,
        due_hours=args.due_hours,
        above_threshold_only=args.above_threshold_only,
        dry_run=args.dry_run,
    )
    print(result)
