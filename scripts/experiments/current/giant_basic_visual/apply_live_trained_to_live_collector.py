#!/usr/bin/env python3
"""Shadow-score live collector items with the live-trained main_image_scores model.

Pairs giant_basic_visual's live-trained visual model (see train_on_live.py,
which fixes the offline/live base-rate prior shift) with the same
tracked_items.csv that basic_5_giant_model/apply_to_live_collector.py scores,
so precision/coverage can be compared after a few days. Shadow-only: no
Telegram messages, no policy changes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))

from experiments.current.giant_basic_visual.features import (  # noqa: E402
    MAIN_IMAGE_FEATURES,
    main_image_features_for_path,
    resolve_main_image_path,
)
from experiments.current.giant_basic_visual.paths import (  # noqa: E402
    EXPERIMENT_ROOT,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_manifest,
)
from experiments.current.giant_basic_visual.train_on_live import (  # noqa: E402
    KNOWN_SEARCHES,
    SCORE_COLS,
    SEARCH_ONEHOTS,
    VISUAL_FEATURES,
    load_visual_source_for_items,
)
from experiments.deal_finder.modeling import score_with_model  # noqa: E402

MODEL_DIR = EXPERIMENT_ROOT / "live_trained" / "live_trained_20260613_200922"
MODEL_PATH = MODEL_DIR / "main_image_scores_hist_gradient_seed42.pkl"
RESULTS_PATH = MODEL_DIR / "results.json"

APPROACH_KEY = "main_image_scores_live_trained"
SCORE_PREFIX = "score__"
THRESHOLD_PREFIX = "threshold__"
PASS_PREFIX = "pass__"
SCORE_COL = f"{SCORE_PREFIX}{APPROACH_KEY}"
THRESHOLD_COL = f"{THRESHOLD_PREFIX}{APPROACH_KEY}"
PASS_COL = f"{PASS_PREFIX}{APPROACH_KEY}"

MODEL_SEARCHES = (
    "griffati_donna_all",
    "griffati_uomo_all",
    "gucci",
    "nike",
    "prada",
    "ps4",
    "donna_accessori_gioielli",
    "telefoni",
    "hobby_collezionismo",
)
ALL_SEARCHES = "__all__"

IMAGE_FEATURE_CACHE_PATH = EXPERIMENT_ROOT / "live_scoring" / "main_image_feature_cache.csv"
CACHE_COLUMNS = ["image_path", *MAIN_IMAGE_FEATURES]
SKIPPED_COLUMNS = ["SearchName", "item_id", "Dataid", "Link", "LocalPrimaryImagePath", "skip_reason"]


def load_threshold() -> float:
    with RESULTS_PATH.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    return float(results["results"]["main_image_scores"]["threshold"])


def load_live_tracked(live_run_dir: Path) -> pd.DataFrame:
    path = live_run_dir / "tracked_items.csv"
    if not path.exists():
        raise FileNotFoundError(f"No tracked_items.csv found in {live_run_dir}")
    df = pd.read_csv(path, low_memory=False)
    df["item_id"] = df["item_id"].astype(str)
    return df


def merge_visual_source(tracked: pd.DataFrame) -> pd.DataFrame:
    """Bring in LocalPrimaryImagePath + cached quality scores from per-search visual CSVs."""
    visual = load_visual_source_for_items(tracked)
    cols = [col for col in ["SearchName", "item_id", "LocalPrimaryImagePath", *SCORE_COLS] if col in visual.columns]
    return tracked.merge(visual[cols], on=["SearchName", "item_id"], how="left")


def load_image_feature_cache() -> pd.DataFrame:
    if IMAGE_FEATURE_CACHE_PATH.exists():
        return pd.read_csv(IMAGE_FEATURE_CACHE_PATH, low_memory=False)
    return pd.DataFrame(columns=CACHE_COLUMNS)


def save_image_feature_cache(cache: pd.DataFrame) -> None:
    path = assert_experiment_path(IMAGE_FEATURE_CACHE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(path, index=False)


def _skip_row(row: pd.Series, reason: str) -> dict[str, Any]:
    payload = {col: row.get(col, "") for col in SKIPPED_COLUMNS if col != "skip_reason"}
    payload["skip_reason"] = reason
    return payload


def add_main_image_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Resolve LocalPrimaryImagePath and attach MainImage* features.

    Only LocalPrimaryImagePath is ever inspected; rows without a readable main
    image are skipped. MainImage* features are computed once per unique image
    path and cached on disk so repeated run-loop passes are cheap.
    """
    cache = load_image_feature_cache()
    cache_map = {str(row["image_path"]): {col: row[col] for col in MAIN_IMAGE_FEATURES} for _, row in cache.iterrows()}
    new_entries: dict[str, dict[str, float]] = {}

    kept_rows: list[pd.Series] = []
    kept_features: list[dict[str, float]] = []
    skipped: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0

    for _, row in df.iterrows():
        path, reason = resolve_main_image_path(row.get("LocalPrimaryImagePath"))
        if reason:
            skipped.append(_skip_row(row, reason))
            continue
        assert path is not None
        cache_key = str(path.resolve())
        features = cache_map.get(cache_key) or new_entries.get(cache_key)
        if features is not None:
            cache_hits += 1
        else:
            try:
                features = main_image_features_for_path(path)
            except Exception as exc:
                skipped.append(_skip_row(row, f"main_image_unreadable:{type(exc).__name__}"))
                continue
            new_entries[cache_key] = features
            cache_misses += 1
        kept_rows.append(row)
        kept_features.append(features)

    if kept_rows:
        kept = pd.DataFrame(kept_rows).reset_index(drop=True)
        feature_df = pd.DataFrame(kept_features).reset_index(drop=True)
        kept = pd.concat([kept, feature_df], axis=1)
    else:
        kept = df.iloc[0:0].copy()
        for col in MAIN_IMAGE_FEATURES:
            kept[col] = pd.Series(dtype=float)

    if new_entries:
        new_rows = [{"image_path": path, **features} for path, features in new_entries.items()]
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache = cache.drop_duplicates(subset=["image_path"], keep="last")
        save_image_feature_cache(cache)

    skipped_df = pd.DataFrame(skipped, columns=SKIPPED_COLUMNS)
    stats = {
        "rows_in": int(len(df)),
        "rows_kept": int(len(kept)),
        "rows_skipped": int(len(skipped_df)),
        "image_cache_hits": cache_hits,
        "image_cache_misses": cache_misses,
        "image_cache_size_after": int(len(cache)),
    }
    return kept, skipped_df, stats


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SearchName"] = out.get("SearchName", "").fillna("").astype(str)
    for col in ["Price", "Likes"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    search_values = out["SearchName"]
    for col in SEARCH_ONEHOTS:
        search_key = col.replace("search__", "")
        out[col] = search_values.eq(search_key).astype(float)
    for col in VISUAL_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    return out


def apply_model(df: pd.DataFrame, model: Any, threshold: float) -> pd.DataFrame:
    out = df.copy()
    feature_frame = out[VISUAL_FEATURES].astype(float)
    probs = np.clip(np.asarray(score_with_model(model, feature_frame), dtype=float), 0.0, 1.0)
    out[SCORE_COL] = probs
    out[THRESHOLD_COL] = threshold
    out[PASS_COL] = out[SCORE_COL] >= threshold
    return out


def coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "yes", "sold"})


def outcome_windows(scored: pd.DataFrame) -> list[int]:
    return sorted(
        {
            int(match.group(1))
            for col in scored.columns
            for match in [re.fullmatch(r"sold_within_(\d+)h", col)]
            if match and f"evaluated_at_{match.group(1)}h" in scored.columns
        }
    )


def build_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for search in MODEL_SEARCHES:
        scope = scored[scored["SearchName"].astype(str).eq(search)]
        scores = pd.to_numeric(scope[SCORE_COL], errors="coerce")
        threshold = float(scope[THRESHOLD_COL].dropna().iloc[0]) if not scope.empty else np.nan
        passed = scope[scope[PASS_COL] == True]
        rows.append(
            {
                "approach": APPROACH_KEY,
                "search_name": search,
                "threshold": threshold,
                "in_training_scope": search in KNOWN_SEARCHES,
                "total_scored": int(len(scope)),
                "passed": int(len(passed)),
                "pass_rate": float(len(passed) / len(scope)) if len(scope) else 0.0,
                "mean_score": float(scores.mean()) if not scores.empty else np.nan,
                "p50_score": float(scores.quantile(0.50)) if not scores.empty else np.nan,
                "p90_score": float(scores.quantile(0.90)) if not scores.empty else np.nan,
                "p95_score": float(scores.quantile(0.95)) if not scores.empty else np.nan,
                "p99_score": float(scores.quantile(0.99)) if not scores.empty else np.nan,
                "max_score": float(scores.max()) if not scores.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def passed_window_performance(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    windows = outcome_windows(scored)
    scopes = (ALL_SEARCHES, *MODEL_SEARCHES)
    for search in scopes:
        if search == ALL_SEARCHES:
            search_scope = scored
        else:
            search_scope = scored[scored["SearchName"].astype(str).eq(search)]
        passed = search_scope[search_scope[PASS_COL] == True]
        for hours in windows:
            eval_col = f"evaluated_at_{hours}h"
            sold_col = f"sold_within_{hours}h"
            evaluated = passed[passed[eval_col].notna()]
            sold_count = int(coerce_bool(evaluated[sold_col]).sum()) if not evaluated.empty else 0
            evaluated_count = int(len(evaluated))
            rows.append(
                {
                    "approach": APPROACH_KEY,
                    "search_name": search,
                    "hours": int(hours),
                    "passed_items": int(len(passed)),
                    "evaluated": evaluated_count,
                    "sold": sold_count,
                    "precision": float(sold_count / evaluated_count) if evaluated_count else np.nan,
                    "coverage": float(evaluated_count / len(passed)) if len(passed) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    *,
    live_run_dir: Path,
    scored: pd.DataFrame,
    skipped: pd.DataFrame,
    summary: pd.DataFrame,
    performance: pd.DataFrame,
    image_stats: dict[str, Any],
    threshold: float,
) -> None:
    def fmt(value: object, decimals: int = 3) -> str:
        try:
            if value is None or pd.isna(value):
                return "-"
            return f"{float(value):.{decimals}f}"
        except Exception:
            return str(value)

    focus_hours = [h for h in (24, 48, 72) if h in set(performance["hours"].dropna().astype(int))]
    overall = performance[performance["search_name"].eq(ALL_SEARCHES)]

    lines = [
        "# Giant Basic Visual - Live-Trained Model - Live Collector Scoring",
        "",
        "Shadow-only run. No Telegram messages were sent and the basic5_giant Telegram policy was not changed.",
        "Model: live-trained `main_image_scores` (hist_gradient, seed 42), trained on basic5_giant's own "
        "live-scored history to fix the offline/live base-rate prior shift (see `train_on_live.py`, "
        "`data/experiments/giant_basic_visual/live_trained/live_trained_20260613_200922/results.json`).",
        f"Global threshold: `{threshold:.4f}` (validation precision 0.667 @ count=21; "
        "test precision 0.682 @ count=22 on the live-trained split).",
        "",
        f"Live collector: `{live_run_dir}`",
        "",
        "## Overall",
        "",
        f"- Rows scored: `{len(scored)}`",
        f"- Rows skipped (no readable main image): `{len(skipped)}`",
        f"- Total passed: `{int((scored[PASS_COL] == True).sum())}`",
    ]
    for hours in focus_hours:
        perf = overall[overall["hours"].eq(hours)]
        if not perf.empty:
            row = perf.iloc[0]
            lines.append(
                f"- {hours}h precision: `{fmt(row['precision'])}` "
                f"(sold {int(row['sold'])}/{int(row['evaluated'])} evaluated, "
                f"coverage {fmt(row['coverage'])}, passed {int(row['passed_items'])})"
            )

    lines.extend(
        [
            "",
            "## Per-Search Summary",
            "",
            "| search | in training scope | threshold | scored | passed | pass rate | mean score | p90 score | p99 score |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['search_name']} | {row['in_training_scope']} | {fmt(row['threshold'], 4)} | "
            f"{int(row['total_scored'])} | {int(row['passed'])} | {fmt(row['pass_rate'])} | "
            f"{fmt(row['mean_score'])} | {fmt(row['p90_score'])} | {fmt(row['p99_score'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-Search Window Performance",
            "",
            "| search | hours | passed | evaluated | sold | precision | coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in performance[performance["hours"].isin(focus_hours)].iterrows():
        lines.append(
            f"| {row['search_name']} | {int(row['hours'])} | {int(row['passed_items'])} | "
            f"{int(row['evaluated'])} | {int(row['sold'])} | {fmt(row['precision'])} | {fmt(row['coverage'])} |"
        )

    lines.extend(
        [
            "",
            "## Image Feature Cache",
            "",
            f"- `{json.dumps(image_stats, sort_keys=True)}`",
            "",
            "## Files",
            "",
            "- `live_scored_items.csv`: item-level scores, threshold, and pass flag.",
            "- `skipped_main_image_rows.csv`: rows without a readable `LocalPrimaryImagePath`, not scored.",
            "- `live_scored_summary.csv`: per-search score and threshold summary.",
            "- `live_scored_window_performance.csv`: sold precision by search and outcome window.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    out_dir: Path,
    *,
    live_run_dir: Path,
    scored: pd.DataFrame,
    skipped: pd.DataFrame,
    image_stats: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored_path = out_dir / "live_scored_items.csv"
    scored.to_csv(scored_path, index=False)

    skipped_path = out_dir / "skipped_main_image_rows.csv"
    skipped.to_csv(skipped_path, index=False)

    summary = build_summary(scored)
    summary_path = out_dir / "live_scored_summary.csv"
    summary.to_csv(summary_path, index=False)

    performance = passed_window_performance(scored)
    performance_path = out_dir / "live_scored_window_performance.csv"
    performance.to_csv(performance_path, index=False)

    report_path = out_dir / "live_scored_report.md"
    write_report(
        report_path,
        live_run_dir=live_run_dir,
        scored=scored,
        skipped=skipped,
        summary=summary,
        performance=performance,
        image_stats=image_stats,
        threshold=threshold,
    )

    manifest_extra = {
        "mode": "shadow_only",
        "telegram_policy_changed": False,
        "live_run_dir": str(live_run_dir),
        "model": {
            "feature_mode": "main_image_scores",
            "approach": APPROACH_KEY,
            "source": "live_trained_20260613_200922",
            "artifact_path": str(MODEL_PATH),
            "threshold": float(threshold),
        },
        "model_searches": list(MODEL_SEARCHES),
        "known_searches": sorted(KNOWN_SEARCHES),
        "image_feature_cache": image_stats,
        "outputs": {
            "scored": str(scored_path),
            "skipped": str(skipped_path),
            "summary": str(summary_path),
            "performance": str(performance_path),
            "report": str(report_path),
        },
    }
    write_manifest(out_dir / "manifest.json", command="giant_basic_visual.apply_live_trained_to_live_collector", extra=manifest_extra)
    return manifest_extra


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    live_run_dir = Path(args.live_run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else EXPERIMENT_ROOT / "live_scoring" / run_id("live_scoring")

    threshold = load_threshold()
    model = joblib.load(MODEL_PATH)

    print(f"[apply_live_trained] loading tracked items from {live_run_dir}", flush=True)
    tracked = load_live_tracked(live_run_dir)
    print(f"[apply_live_trained] {len(tracked)} tracked rows", flush=True)

    merged = merge_visual_source(tracked)
    print("[apply_live_trained] computing MainImage* features (cached)...", flush=True)
    kept, skipped, image_stats = add_main_image_features(merged)
    print(f"[apply_live_trained] {len(kept)} rows with readable main image, {len(skipped)} skipped", flush=True)
    print(f"[apply_live_trained] image cache: {json.dumps(image_stats, sort_keys=True)}", flush=True)

    prepared = prepare_feature_frame(kept)
    scored = apply_model(prepared, model, threshold)
    total_passed = int((scored[PASS_COL] == True).sum())
    print(f"[apply_live_trained] {APPROACH_KEY}: {total_passed} passed threshold {threshold:.4f}", flush=True)

    manifest = write_outputs(
        out_dir,
        live_run_dir=live_run_dir,
        scored=scored,
        skipped=skipped,
        image_stats=image_stats,
        threshold=threshold,
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Shadow-score live collector items with the live-trained giant_basic_visual main_image_scores model."
    )
    parser.add_argument("--live-run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-loop", action="store_true", help="Keep rescoring the live collector on a fixed cadence.")
    parser.add_argument("--sleep-seconds", type=float, default=7200.0, help="Sleep between run-loop scoring passes.")
    args = parser.parse_args()

    ensure_experiment_dirs()
    if args.run_loop and args.out_dir is not None:
        raise ValueError("--out-dir cannot be used with --run-loop because each loop pass writes a new live_scoring folder.")
    if args.run_loop:
        while True:
            run_once(args)
            print(f"[apply_live_trained] sleeping {args.sleep_seconds:.1f}s before next scoring pass", flush=True)
            time.sleep(max(1.0, float(args.sleep_seconds)))
    run_once(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
