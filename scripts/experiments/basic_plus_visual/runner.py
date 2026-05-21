"""Live runner for basic_plus_visual experiment.

Every hour:
1. Collect basic snapshot per search
2. Download primary images
3. Extract visual features
4. Score with trained sold-predictor model
5. Track items, select top N by predicted sold probability
6. Periodically recheck sold status
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import requests
from urllib.parse import urlparse

from experiments.basic_plus_visual.paths import (
    ensure_experiment_dirs,
    EXPERIMENT_ROOT,
    LIVE_RUNS_DIR,
    MODELS_DIR,
    run_id,
    utc_now_iso,
    write_json,
    write_manifest,
)

# Reuse cascade utilities for search loading / snapshot collection.
# We intentionally do NOT reuse cascade's image-download helper because it
# enforces an assert_experiment_path check that refuses writes outside the
# benchmark_basic_to_full directory. basic+visual has its own image_cache.
sys.path.insert(0, str(SCRIPTS_DIR / "experiments" / "benchmark_basic_to_full"))
from cascade_runner import (
    collect_search_snapshot,
    has_collectable_search_settings,
    load_search_config_by_folder,
    normalized_search_config,
    primary_image_url,
)
from experiments.full_scrape_model.compare_feature_modalities import add_dino_embedding_columns

STATE_FILE = "tracked_items.csv"
EVENTS_FILE = "events.jsonl"


IMAGE_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def infer_image_extension(url: str, response: requests.Response) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
        return suffix
    ct = response.headers.get("Content-Type", "").lower()
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    if "gif" in ct:
        return ".gif"
    return ".jpg"


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


def _ensure_local_primary_images(rows: pd.DataFrame, *, out_dir: Path, timeout: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = rows.copy()
    cache_root = out_dir / "image_cache"
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
    out["LocalImagePaths"] = out["LocalPrimaryImagePath"].map(lambda value: json.dumps([value], ensure_ascii=True) if value else "[]")
    return out, {"attempted": int(attempted), "local_image_rows": int(ok), "failed_or_missing": int(failed), "cache_root": str(cache_root)}


def to_numeric_frame(X):
    return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")


def text_values_to_str(X):
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, 0].fillna("").astype(str)
    return pd.Series(X).fillna("").astype(str)


def load_trained_models() -> dict[str, dict[str, Any]]:
    """Load all trained basic_plus_visual models from MODELS_DIR."""
    models: dict[str, dict[str, Any]] = {}
    for meta_path in sorted(MODELS_DIR.glob("*extra_trees.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        search = meta.get("search_name")
        if not search:
            continue
        pkl_path = Path(meta["artifact_path"])
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
        models[search] = {
            "model": model,
            "threshold": float(meta.get("threshold", 0.5)),
            "numeric_features": meta.get("numeric_features", []),
            "text_features": meta.get("text_features", []),
            "metadata": meta,
        }
    print(f"Loaded {len(models)} models: {list(models.keys())}")
    return models


def score_items(rows: pd.DataFrame, model_dict: dict[str, Any]) -> pd.DataFrame:
    """Score a dataframe of items with the trained model."""
    out = rows.copy()
    model = model_dict["model"]
    numeric = model_dict["numeric_features"]
    text = model_dict["text_features"]
    feature_cols = [c for c in text + numeric if c in out.columns]

    if not feature_cols or out.empty:
        out["SoldProba"] = np.nan
        out["SoldPred"] = False
        return out

    proba = model.predict_proba(out[feature_cols])[:, 1]
    out["SoldProba"] = proba
    out["SoldPred"] = proba >= model_dict["threshold"]
    out["SoldThreshold"] = model_dict["threshold"]
    return out


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def merge_tracked(existing: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    if updates.empty:
        return existing
    combined = pd.concat([existing, updates], ignore_index=True)
    # Deduplicate by Dataid + SearchName, keep latest
    if "Dataid" in combined.columns and "SearchName" in combined.columns:
        combined = combined.sort_values("_tracked_at", ascending=False).drop_duplicates(
            subset=["Dataid", "SearchName"], keep="first"
        )
    return combined.reset_index(drop=True)


def run_collect_once(
    out_dir: Path,
    models: dict[str, dict[str, Any]],
    max_items_total: int | None = 30,
    image_timeout: float = 8.0,
    quality_methods: str = "simple,pyiqa,aesthetic,dino",
    quality_device: str = "auto",
) -> dict[str, Any]:
    """One collection cycle."""
    from config.search_loader import load_searches
    from config.project_config import settings

    ensure_experiment_dirs()
    out_dir = Path(out_dir)
    for name in ("raw_snapshots", "scored_items", "image_cache", "reports"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    searches = load_searches(str(settings.paths.searches_yaml))
    search_names = [s.folder for s in searches.values() if s.enabled and getattr(s, "cascade_only", False) is not True]
    # Also include cascade_only searches since we want to test on all
    search_names = [s.folder for s in searches.values() if s.enabled]

    config_by_search = load_search_config_by_folder(search_names)
    observed_at = utc_now_iso()
    safe_ts = observed_at.replace(":", "").replace("-", "")

    tracked = read_csv_or_empty(out_dir / STATE_FILE)
    all_scored: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for search_name in search_names:
        if search_name not in models:
            continue
        model_dict = models[search_name]
        search_config = config_by_search.get(search_name)
        if search_config is None or not has_collectable_search_settings(search_config):
            continue
        search_config = normalized_search_config(search_config)

        # 1. Collect raw snapshot
        raw = collect_search_snapshot(search_name, search_config)
        raw_path = out_dir / "raw_snapshots" / f"{search_name}_{safe_ts}.csv"
        raw.to_csv(raw_path, index=False)
        if raw.empty:
            summary_rows.append({"SearchName": search_name, "status": "empty", "raw_rows": 0})
            continue

        # 2. Download images
        try:
            with_images, img_stats = _ensure_local_primary_images(
                raw, out_dir=out_dir, timeout=image_timeout
            )
        except Exception as exc:
            print(f"[{search_name}] Image download failed: {exc}", flush=True)
            summary_rows.append({"SearchName": search_name, "status": "image_failed", "raw_rows": len(raw)})
            continue

        # 3. Extract visual features + flatten DINO embedding into per-dim columns
        # (the trained model expects DinoEmbedding_0000..DinoEmbedding_0383, not a serialized DinoEmbedding column).
        try:
            from experiments.photo_arbitrage.features import add_photo_features
            from experiments.photo_arbitrage.quality_methods import add_quality_method_scores, MethodConfig

            featured = add_photo_features(with_images)
            config = MethodConfig(
                methods=tuple(part.strip() for part in quality_methods.split(",") if part.strip()) if quality_methods else ("simple",),
                device=quality_device,
                max_images_per_item=1,
            )
            scored = add_quality_method_scores(featured, config=config)
            scored, _embedding_cols = add_dino_embedding_columns(scored)
        except Exception as exc:
            print(f"[{search_name}] Visual feature extraction failed: {exc}", flush=True)
            summary_rows.append({"SearchName": search_name, "status": "visual_failed", "raw_rows": len(raw)})
            continue

        # 4. Score with model
        scored = score_items(scored, model_dict)
        scored_path = out_dir / "scored_items" / f"{search_name}_{safe_ts}.csv"
        scored.to_csv(scored_path, index=False)

        # 5. Track all items
        updates = scored.copy()
        updates["_tracked_at"] = observed_at
        updates["_status"] = "scored"
        all_scored.append(updates)

        passed = scored[scored["SoldPred"]]
        summary_rows.append({
            "SearchName": search_name,
            "status": "ok",
            "raw_rows": int(len(raw)),
            "images_ok": int(img_stats.get("local_image_rows", 0)),
            "images_failed": int(img_stats.get("failed_or_missing", 0)),
            "scored_rows": int(len(scored)),
            "passed_rows": int(len(passed)),
            "max_proba": float(scored["SoldProba"].max()) if not scored.empty else np.nan,
        })
        append_jsonl(out_dir / EVENTS_FILE, {"event": "collect_search", "at": observed_at, **summary_rows[-1]})

    # Global selection across searches, ranked by margin above each search's threshold.
    # SoldProba alone is not comparable across searches because each model has its own
    # threshold (e.g. nike=0.992, gucci=0.894). An item at 0.91 over a 0.80 threshold
    # is a stronger signal than 0.91 over a 0.90 threshold. So:
    #   1. Keep only items that passed their per-search threshold (SoldPred=True).
    #   2. Compute SoldMargin = SoldProba - SoldThreshold.
    #   3. Pool across all searches and take top-N by SoldMargin.
    if all_scored:
        combined = pd.concat(all_scored, ignore_index=True)
        combined = combined[combined["SoldPred"].astype(bool)].copy()
        combined["SoldMargin"] = combined["SoldProba"] - combined["SoldThreshold"]
        combined = combined.sort_values("SoldMargin", ascending=False)
        if max_items_total is not None and max_items_total > 0:
            selected = combined.head(max_items_total).copy()
        else:
            selected = combined.copy()
        selected["_status"] = "selected"
        tracked = merge_tracked(tracked, selected)

    tracked = tracked.reset_index(drop=True)
    tracked.to_csv(out_dir / STATE_FILE, index=False)

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "reports" / f"collect_summary_{safe_ts}.csv"
    summary.to_csv(summary_path, index=False)

    result = {
        "status": "collected",
        "snapshot_at": observed_at,
        "summary_path": str(summary_path),
        "tracked_rows": int(len(tracked)),
        "selected_rows": int(len(selected)) if all_scored else 0,
        "summary": summary_rows,
    }
    write_json(out_dir / "latest_status.json", result)
    return result


def run_recheck_due(out_dir: Path, max_workers: int = 3) -> dict[str, Any]:
    """Recheck tracked items for sold status."""
    from scraping_options import _update_market_status_for_df

    out_dir = Path(out_dir)
    tracked = read_csv_or_empty(out_dir / STATE_FILE)
    if tracked.empty or "Link" not in tracked.columns:
        return {"status": "no_tracked_items", "checked_rows": 0}

    # Check items that haven't been checked in the last hour
    now = pd.Timestamp.now(tz="UTC")
    if "_last_checked_at" in tracked.columns:
        tracked["_last_checked_at"] = pd.to_datetime(tracked["_last_checked_at"], errors="coerce", utc=True)
        due = tracked[
            tracked["_last_checked_at"].isna()
            | ((now - tracked["_last_checked_at"]).dt.total_seconds() >= 3600)
        ].copy()
    else:
        due = tracked.copy()

    if due.empty:
        return {"status": "nothing_due", "checked_rows": 0}

    # Limit to avoid overloading
    due = due.head(50).copy()

    checked, sold_df = _update_market_status_for_df(
        due,
        max_workers=max_workers,
        delay=2.0,
        initial_delay=0.0,
        fetch_sleep=1.0,
        fetch_max_attempts=1,
    )

    checked["_last_checked_at"] = utc_now_iso()
    # Update tracked with recheck results
    for _, row in checked.iterrows():
        mask = (tracked["Dataid"] == row["Dataid"]) & (tracked["SearchName"] == row["SearchName"])
        if mask.any():
            idx = tracked[mask].index[0]
            for col in ["LastCheckStatus", "MarketStatus", "ObservedPagePrice", "_last_checked_at"]:
                if col in row.index and pd.notna(row[col]):
                    tracked.at[idx, col] = row[col]

    tracked.to_csv(out_dir / STATE_FILE, index=False)
    recheck_path = out_dir / "reports" / f"recheck_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    checked.to_csv(recheck_path, index=False)

    result = {
        "status": "checked",
        "checked_rows": int(len(checked)),
        "sold_found": int(len(sold_df)),
        "recheck_path": str(recheck_path),
    }
    append_jsonl(out_dir / EVENTS_FILE, {"event": "recheck", **result})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="basic_plus_visual live runner")
    sub = parser.add_subparsers(dest="cmd")

    collect = sub.add_parser("collect-once")
    collect.add_argument("--out-dir", type=str, default=None)
    collect.add_argument("--max-items-total", type=int, default=30)
    collect.add_argument("--image-timeout", type=float, default=8.0)
    collect.add_argument("--quality-methods", default="simple,aesthetic,dino")
    collect.add_argument("--quality-device", default="auto")

    recheck = sub.add_parser("recheck-due")
    recheck.add_argument("--out-dir", type=str, default=None)
    recheck.add_argument("--max-workers", type=int, default=3)

    loop = sub.add_parser("run-loop")
    loop.add_argument("--out-dir", type=str, default=None)
    loop.add_argument("--collect-every-hours", type=float, default=1.0)
    loop.add_argument("--sleep-seconds", type=float, default=60.0)
    loop.add_argument("--max-items-total", type=int, default=30)
    loop.add_argument("--image-timeout", type=float, default=8.0)
    loop.add_argument("--quality-methods", default="simple,aesthetic,dino")
    loop.add_argument("--quality-device", default="auto")

    return parser.parse_args()


def resolve_out_dir(value: str | None) -> Path:
    if value:
        return Path(value)
    return LIVE_RUNS_DIR / run_id("basic_plus_visual_live")


def main() -> int:
    args = parse_args()
    ensure_experiment_dirs()

    if args.cmd == "collect-once":
        out_dir = resolve_out_dir(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        models = load_trained_models()
        result = run_collect_once(
            out_dir=out_dir,
            models=models,
            max_items_total=args.max_items_total,
            image_timeout=args.image_timeout,
            quality_methods=args.quality_methods,
            quality_device=args.quality_device,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.cmd == "recheck-due":
        out_dir = resolve_out_dir(args.out_dir)
        result = run_recheck_due(out_dir=out_dir, max_workers=args.max_workers)
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.cmd == "run-loop":
        out_dir = resolve_out_dir(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        models = load_trained_models()
        last_collect: pd.Timestamp | None = None
        while True:
            now = pd.Timestamp.now(tz="UTC")
            should_collect = last_collect is None or (now - last_collect).total_seconds() >= args.collect_every_hours * 3600.0
            if should_collect:
                print(f"[{utc_now_iso()}] Starting collection...")
                try:
                    run_collect_once(
                        out_dir=out_dir,
                        models=models,
                        max_items_total=args.max_items_total,
                        image_timeout=args.image_timeout,
                        quality_methods=args.quality_methods,
                        quality_device=args.quality_device,
                    )
                except Exception as exc:
                    print(f"Collection failed: {exc}")
                last_collect = now
            try:
                run_recheck_due(out_dir=out_dir)
            except Exception as exc:
                print(f"Recheck failed: {exc}")
            time.sleep(args.sleep_seconds)

    print("No command given. Use: collect-once, recheck-due, or run-loop")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
