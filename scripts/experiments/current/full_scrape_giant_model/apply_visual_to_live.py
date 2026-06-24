#!/usr/bin/env python3
"""Apply the offline-trained full_scrape_visual giant models to the live collector (read-only).

Loads the latest ``full_scrape_giant_visual_*`` model family (trained on the
offline backfill dataset + merged photo-quality/visual features), scores the
live ``tracked_items.csv`` snapshot, applies per-search thresholds from
``per_search_threshold_metrics.csv``, and reports sold-precision/coverage by
``sold_within_{24h,48h,72h}`` outcome window — in the same shape used by
basic_5_giant_model and giant_basic_visual, for a same-condition comparison.

STRICTLY READ-ONLY with respect to the live pipeline and Telegram: no Telegram
messages are sent and no Telegram code/policy is modified.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[3]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.deal_finder import model_sweep as base_sweep  # noqa: E402
from experiments.deal_finder.modeling import score_with_model  # noqa: E402
from experiments.current.full_scrape_giant_model.dataset import MAIN_SEARCHES  # noqa: E402
from experiments.current.full_scrape_giant_model.full_scrape_features import add_full_scrape_features  # noqa: E402
from experiments.current.full_scrape_giant_model.image_filter import live_image_filter  # noqa: E402
from experiments.current.full_scrape_giant_model.visual_features import load_live_visual_features  # noqa: E402
from experiments.current.full_scrape_giant_model.paths import (  # noqa: E402
    LIVE_RUNS_DIR,
    MODELS_DIR,
    OFFLINE_RUNS_DIR,
    REPORTS_DIR,
    ROOT,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_manifest,
)

LIVE_RUNS_ROOT = ROOT / "data" / "experiments" / "time_to_sell" / "live_runs"
WINDOWS = (24, 48, 72)
ALL_SEARCHES = "__all__"
TELEGRAM_NOTE = (
    "Telegram policy was NOT changed: this live evaluation is strictly read-only. "
    "No Telegram messages were sent and no Telegram code/policy was modified."
)


def latest_visual_run() -> Path:
    runs = sorted(
        (p for p in OFFLINE_RUNS_DIR.glob("full_scrape_giant_visual_*") if (p / "per_search_threshold_metrics.csv").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(
            f"No full_scrape_giant_visual run with per_search_threshold_metrics.csv under {OFFLINE_RUNS_DIR}"
        )
    return runs[0]


def latest_live_tracked() -> Path:
    runs = sorted(
        (p for p in LIVE_RUNS_ROOT.glob("bin_collector_*") if (p / "tracked_items.csv").exists()),
        key=lambda p: (p / "tracked_items.csv").stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No live tracked_items.csv under {LIVE_RUNS_ROOT}")
    return runs[0]


def load_models(run_dir: Path) -> list[dict[str, Any]]:
    """Load (approach, model, global threshold) for every model in a full_scrape_giant_visual run."""
    run_name = run_dir.name
    out: list[dict[str, Any]] = []
    for meta_path in sorted(MODELS_DIR.glob(f"{run_name}_full_scrape_visual_*_metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pkl_path = Path(meta["artifact_path"])
        if not pkl_path.exists():
            continue
        with pkl_path.open("rb") as handle:
            model = pickle.load(handle)
        out.append({"approach": meta["approach"], "model": model, "threshold": float(meta["threshold"])})
    return out


def load_threshold_table(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "per_search_threshold_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing per-search threshold table: {path}")
    return pd.read_csv(path, low_memory=False)


def thresholds_for_approach(approach: str, threshold_table: pd.DataFrame, fallback: float, searches: tuple[str, ...]) -> dict[str, float]:
    """Per-search thresholds for ``approach``, falling back to the model's global threshold
    for searches without a qualified per-search threshold."""
    thresholds: dict[str, float] = {}
    scoped = threshold_table[
        threshold_table["approach"].astype(str).eq(approach)
        & threshold_table["validation_threshold_qualified"].astype(bool)
    ]
    for _, row in scoped.iterrows():
        thresholds[str(row["search_name"])] = float(row["per_search_threshold"])
    for search in searches:
        thresholds.setdefault(search, fallback)
    return thresholds


def build_live_features(live_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(live_dir / "tracked_items.csv", low_memory=False)
    raw["SearchName"] = raw.get("SearchName", "").fillna("").astype(str)
    raw["item_id"] = raw.get("item_id", "").astype(str).str.strip()
    kept, image_acc = live_image_filter(raw)
    kept = kept.reset_index(drop=True)

    visual = load_live_visual_features(live_dir)
    merged = kept.merge(visual, on="item_id", how="left", suffixes=("", "_vis"))

    merged = base_sweep.add_engineered_snapshot_features(merged)
    merged = add_full_scrape_features(merged)
    return merged, image_acc


def score_items(feats: pd.DataFrame, models: list[dict[str, Any]], run_dir: Path) -> pd.DataFrame:
    scored = feats.copy()
    search_values = scored["SearchName"].astype(str)
    threshold_table = load_threshold_table(run_dir)
    for m in models:
        approach = str(m["approach"])
        score_col = f"score__{approach}"
        threshold_col = f"threshold__{approach}"
        pass_col = f"pass__{approach}"

        scores = np.clip(np.asarray(score_with_model(m["model"], feats), dtype=float), 0.0, 1.0)
        scored[score_col] = scores

        thresholds = thresholds_for_approach(approach, threshold_table, m["threshold"], MAIN_SEARCHES)
        scored[threshold_col] = search_values.map(thresholds).astype(float)
        scored[pass_col] = scored[score_col] >= scored[threshold_col]
    return scored


def coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "yes", "sold"})


def passed_window_performance(scored: pd.DataFrame, models: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = (ALL_SEARCHES, *MAIN_SEARCHES)
    for m in models:
        approach = str(m["approach"])
        pass_col = f"pass__{approach}"
        if pass_col not in scored.columns:
            continue
        for search in scopes:
            if search == ALL_SEARCHES:
                scope = scored
            else:
                scope = scored[scored["SearchName"].astype(str).eq(search)]
            passed = scope[scope[pass_col] == True]  # noqa: E712
            for hours in WINDOWS:
                eval_col = f"evaluated_at_{hours}h"
                sold_col = f"sold_within_{hours}h"
                if eval_col not in passed.columns or sold_col not in passed.columns:
                    continue
                evaluated = passed[passed[eval_col].notna()]
                sold_count = int(coerce_bool(evaluated[sold_col]).sum()) if not evaluated.empty else 0
                evaluated_count = int(len(evaluated))
                rows.append(
                    {
                        "approach": approach,
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


def _fmt(value: object, decimals: int = 3) -> str:
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def write_report(
    path: Path,
    *,
    run_dir: Path,
    live_tracked: Path,
    image_acc: pd.DataFrame,
    performance: pd.DataFrame,
) -> None:
    lines = [
        "# Full-Scrape Visual Giant Model - Live Collector Scoring",
        "",
        f"Offline run: `{run_dir.name}`",
        f"Live source: `{live_tracked}`",
        "",
        "## Main-Image Filtering (live)",
        "",
        "| search | total | kept | skipped | skipped_no_image |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, r in image_acc.iterrows():
        lines.append(
            f"| {r['search_name']} | {int(r['total_rows'])} | {int(r['kept_rows'])} | "
            f"{int(r['skipped_rows'])} | {int(r.get('skipped_no_image', 0))} |"
        )

    overall = performance[performance["search_name"].eq(ALL_SEARCHES)].copy()
    lines.extend(
        [
            "",
            "## Model Overview (all searches)",
            "",
            "| approach | passed (48h) | 24h precision | 48h precision | 72h precision | 24h coverage | 48h coverage | 72h coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for approach, grp in overall.groupby("approach", sort=False):
        by_hours = {int(r["hours"]): r for _, r in grp.iterrows()}
        passed_48 = int(by_hours[48]["passed_items"]) if 48 in by_hours else 0
        lines.append(
            f"| {approach} | {passed_48} | "
            f"{_fmt(by_hours.get(24, {}).get('precision'))} | {_fmt(by_hours.get(48, {}).get('precision'))} | "
            f"{_fmt(by_hours.get(72, {}).get('precision'))} | "
            f"{_fmt(by_hours.get(24, {}).get('coverage'))} | {_fmt(by_hours.get(48, {}).get('coverage'))} | "
            f"{_fmt(by_hours.get(72, {}).get('coverage'))} |"
        )

    per_search = performance[~performance["search_name"].eq(ALL_SEARCHES) & performance["hours"].eq(48)]
    if not per_search.empty:
        best = (
            per_search.sort_values(["search_name", "precision", "evaluated"], ascending=[True, False, False], na_position="last")
            .drop_duplicates("search_name", keep="first")
        )
        lines.extend(
            [
                "",
                "## Best Approach Per Search (48h window)",
                "",
                "| search | best approach | passed | evaluated | sold | precision | coverage |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in best.sort_values("search_name").iterrows():
            lines.append(
                f"| {row['search_name']} | {row['approach']} | {int(row['passed_items'])} | "
                f"{int(row['evaluated'])} | {int(row['sold'])} | {_fmt(row['precision'])} | {_fmt(row['coverage'])} |"
            )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `live_visual_window_performance.csv`: passed/evaluated/sold/precision/coverage by approach, search, and outcome window.",
            "",
            "## Notes",
            "",
            f"- {TELEGRAM_NOTE}",
            "- Per-search thresholds come from `per_search_threshold_metrics.csv` (validation-qualified rows); "
            "unqualified searches fall back to each model's global offline threshold.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_live(run_dir: Path, live_tracked: Path) -> dict[str, Any]:
    ensure_experiment_dirs()
    feats, image_acc = build_live_features(live_tracked.parent)
    models = load_models(run_dir)
    if not models:
        raise RuntimeError(f"No model artifacts found for run {run_dir.name}")

    scored = score_items(feats, models, run_dir)
    performance = passed_window_performance(scored, models)

    out_dir = assert_experiment_path(LIVE_RUNS_DIR / run_id("live_visual"))
    out_dir.mkdir(parents=True, exist_ok=True)
    performance_path = out_dir / "live_visual_window_performance.csv"
    performance.to_csv(performance_path, index=False)

    report_path = out_dir / "live_visual_report.md"
    write_report(report_path, run_dir=run_dir, live_tracked=live_tracked, image_acc=image_acc, performance=performance)
    mirror = assert_experiment_path(REPORTS_DIR / f"{out_dir.name}_live_visual_report.md")
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    write_manifest(
        out_dir / "manifest.json",
        command="full_scrape_giant_model.apply_visual_to_live",
        extra=base_sweep.to_builtin(
            {
                "offline_run": str(run_dir),
                "live_tracked": str(live_tracked),
                "approaches": [m["approach"] for m in models],
                "windows": list(WINDOWS),
                "read_only": True,
                "telegram_policy_changed": False,
            }
        ),
    )
    print(f"[apply_visual_live] report: {report_path}", flush=True)
    return {"run_dir": str(out_dir), "report": str(report_path), "performance": str(performance_path)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply full_scrape_visual giant models to live-collector data (read-only).")
    p.add_argument("--run-dir", default=None, help="full_scrape_giant_visual_* offline run dir; defaults to latest.")
    p.add_argument("--live-csv", default=None, help="Path to a live tracked_items.csv; defaults to the latest.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_visual_run()
    live_csv = Path(args.live_csv) if args.live_csv else (latest_live_tracked() / "tracked_items.csv")
    result = evaluate_live(run_dir, live_csv)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
