#!/usr/bin/env python3
"""3-way live comparison: full_scrape_visual vs basic5 vs basic5_visual (read-only).

Scores the same live ``tracked_items.csv`` snapshot with three model families
under the same conditions:

- ``full_scrape_visual``: this experiment's offline-trained giant model family
  (full-scrape + offline visual/photo-quality features), per-search thresholds.
- ``basic5``: ``basic_5_giant_model``'s flagship ``xgboost_basic_v1``, the
  live Telegram model, per-search thresholds.
- ``basic5_visual``: ``giant_basic_visual``'s live-trained
  ``main_image_scores_live_trained`` shadow model, global threshold.

All three are compared on ``sold_within_{24h,48h,72h}`` precision/coverage,
focal window 48h. STRICTLY READ-ONLY: no Telegram messages are sent and no
Telegram code/policy is modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[3]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.deal_finder import model_sweep as base_sweep  # noqa: E402
from experiments.current.full_scrape_giant_model.apply_visual_to_live import (  # noqa: E402
    ALL_SEARCHES,
    WINDOWS,
    build_live_features,
    latest_live_tracked,
    latest_visual_run,
    load_models as load_visual_models,
    passed_window_performance as full_scrape_window_performance,
    score_items as score_full_scrape_visual,
)
from experiments.current.full_scrape_giant_model.dataset import MAIN_SEARCHES  # noqa: E402
from experiments.current.full_scrape_giant_model.paths import (  # noqa: E402
    LIVE_RUNS_DIR,
    REPORTS_DIR,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_manifest,
)
from experiments.current.basic_5_giant_model.apply_to_live_collector import (  # noqa: E402
    apply_models as apply_basic5_models,
    load_live_tracked as load_basic5_tracked,
    load_model_records as load_basic5_records,
    passed_window_performance as basic5_window_performance,
)
from experiments.current.giant_basic_visual.apply_live_trained_to_live_collector import (  # noqa: E402
    add_main_image_features,
    apply_model as apply_basic5_visual_model,
    load_threshold as load_basic5_visual_threshold,
    merge_visual_source as merge_basic5_visual_source,
    passed_window_performance as basic5_visual_window_performance,
    prepare_feature_frame as prepare_basic5_visual_frame,
    MODEL_PATH as BASIC5_VISUAL_MODEL_PATH,
)

FOCUS_HOURS = 48
APPENDIX_HOURS = (24, 72)
FAMILY_LABELS = {
    "full_scrape_visual": "full_scrape_visual",
    "basic5": "basic5 (xgboost_basic_v1)",
    "basic5_visual": "basic5_visual (main_image_scores_live_trained)",
}
TELEGRAM_NOTE = (
    "Telegram policy was NOT changed: this comparison is strictly read-only. "
    "No Telegram messages were sent and no Telegram code/policy was modified."
)
MIN_EVALUATED_FOR_VERDICT = 5


def score_full_scrape_visual_family(live_dir: Path) -> pd.DataFrame:
    run_dir = latest_visual_run()
    feats, _ = build_live_features(live_dir)
    models = load_visual_models(run_dir)
    if not models:
        raise RuntimeError(f"No model artifacts found for run {run_dir.name}")
    scored = score_full_scrape_visual(feats, models, run_dir)
    return full_scrape_window_performance(scored, models)


def score_basic5_family(live_run_dir: Path) -> pd.DataFrame:
    records = load_basic5_records(("xgboost_basic_v1",))
    tracked = load_basic5_tracked(live_run_dir)
    scored = apply_basic5_models(tracked, records)
    return basic5_window_performance(scored, records)


def score_basic5_visual_family(live_run_dir: Path) -> pd.DataFrame:
    import joblib

    threshold = load_basic5_visual_threshold()
    model = joblib.load(BASIC5_VISUAL_MODEL_PATH)
    tracked = load_basic5_tracked(live_run_dir)
    tracked["item_id"] = tracked["item_id"].astype(str)
    merged = merge_basic5_visual_source(tracked)
    kept, _skipped, _stats = add_main_image_features(merged)
    prepared = prepare_basic5_visual_frame(kept)
    scored = apply_basic5_visual_model(prepared, model, threshold)
    return basic5_visual_window_performance(scored)


def build_comparison(
    full_scrape_perf: pd.DataFrame,
    basic5_perf: pd.DataFrame,
    basic5_visual_perf: pd.DataFrame,
) -> pd.DataFrame:
    """Pick one focus approach per family and merge into one (search, hours) table."""
    fs = full_scrape_perf[full_scrape_perf["approach"].eq(FULL_SCRAPE_FOCUS_APPROACH)].copy()
    fs["family"] = "full_scrape_visual"

    b5 = basic5_perf.copy()
    b5["family"] = "basic5"

    b5v = basic5_visual_perf.copy()
    b5v["family"] = "basic5_visual"

    combined = pd.concat([fs, b5, b5v], ignore_index=True)
    combined = combined[combined["hours"].isin((FOCUS_HOURS, *APPENDIX_HOURS))]

    searches = [s for s in MAIN_SEARCHES if s != "Borse_Griffate"]
    combined = combined[combined["search_name"].isin((ALL_SEARCHES, *searches))]
    return combined


def _fmt(value: object, decimals: int = 3) -> str:
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def _table_for_hours(combined: pd.DataFrame, hours: int, searches: list[str]) -> list[str]:
    lines = [
        f"### {hours}h window",
        "",
        "| search | family | passed | evaluated | sold | precision | coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    sub = combined[combined["hours"].eq(hours)]
    for search in searches:
        for family in ("full_scrape_visual", "basic5", "basic5_visual"):
            row = sub[sub["search_name"].eq(search) & sub["family"].eq(family)]
            if row.empty:
                continue
            r = row.iloc[0]
            label = "**ALL**" if search == ALL_SEARCHES else search
            lines.append(
                f"| {label} | {FAMILY_LABELS[family]} | {int(r['passed_items'])} | "
                f"{int(r['evaluated'])} | {int(r['sold'])} | {_fmt(r['precision'])} | {_fmt(r['coverage'])} |"
            )
    return lines


def _verdict_rows(combined: pd.DataFrame, searches: list[str]) -> list[str]:
    lines = [
        "## Best Family Per Search (48h window)",
        "",
        "| search | best family | precision | evaluated | runner-up | runner-up precision |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    sub = combined[combined["hours"].eq(FOCUS_HOURS)]
    for search in searches:
        rows = sub[sub["search_name"].eq(search)]
        eligible = rows[rows["evaluated"] >= MIN_EVALUATED_FOR_VERDICT].sort_values("precision", ascending=False)
        label = "**ALL**" if search == ALL_SEARCHES else search
        if eligible.empty:
            lines.append(f"| {label} | - (insufficient data) | - | - | - | - |")
            continue
        best = eligible.iloc[0]
        runner = eligible.iloc[1] if len(eligible) > 1 else None
        runner_label = FAMILY_LABELS[runner["family"]] if runner is not None else "-"
        runner_prec = _fmt(runner["precision"]) if runner is not None else "-"
        lines.append(
            f"| {label} | {FAMILY_LABELS[best['family']]} | {_fmt(best['precision'])} | "
            f"{int(best['evaluated'])} | {runner_label} | {runner_prec} |"
        )
    return lines


def write_report(path: Path, *, live_run_dir: Path, combined: pd.DataFrame) -> None:
    searches = [ALL_SEARCHES] + [s for s in MAIN_SEARCHES if s != "Borse_Griffate"]

    lines = [
        "# Three-Way Live Comparison: full_scrape_visual vs basic5 vs basic5_visual",
        "",
        f"Live source: `{live_run_dir / 'tracked_items.csv'}`",
        "",
        f"Focus approach for full_scrape_visual: `{FULL_SCRAPE_FOCUS_APPROACH}` "
        "(picked by most validation-qualified per-search thresholds in `per_search_threshold_metrics.csv`).",
        "Each family uses its own real candidate-selection policy: per-search thresholds for "
        "full_scrape_visual and basic5, a global threshold for basic5_visual.",
        "",
        "## Focal Window (48h)",
        "",
    ]
    lines.extend(_table_for_hours(combined, FOCUS_HOURS, searches))
    lines.append("")
    lines.extend(_verdict_rows(combined, searches))

    if APPENDIX_HOURS:
        lines.append("")
        lines.append("## Appendix")
    for hours in APPENDIX_HOURS:
        lines.append("")
        lines.extend(_table_for_hours(combined, hours, searches))

    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- {TELEGRAM_NOTE}",
            f"- 'Best family' requires at least {MIN_EVALUATED_FOR_VERDICT} evaluated outcomes in the 48h window; "
            "searches without enough evaluated items are marked insufficient.",
            "- basic5_visual uses a single global threshold (no per-search tuning wired in for this family).",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


FULL_SCRAPE_FOCUS_APPROACH = "catboost_full_v1"


def determine_focus_approach() -> str:
    run_dir = latest_visual_run()
    metrics_path = run_dir / "per_search_threshold_metrics.csv"
    df = pd.read_csv(metrics_path, low_memory=False)
    grouped = df.groupby("approach").agg(
        qualified=("validation_threshold_qualified", "sum"),
        mean_precision=("threshold_precision", "mean"),
    )
    grouped = grouped.sort_values(["qualified", "mean_precision"], ascending=[False, False])
    return str(grouped.index[0])


def main() -> None:
    global FULL_SCRAPE_FOCUS_APPROACH
    ensure_experiment_dirs()
    FULL_SCRAPE_FOCUS_APPROACH = determine_focus_approach()

    live_run_dir = latest_live_tracked()
    print(f"[compare3] live source: {live_run_dir / 'tracked_items.csv'}", flush=True)
    print(f"[compare3] full_scrape_visual focus approach: {FULL_SCRAPE_FOCUS_APPROACH}", flush=True)

    print("[compare3] scoring full_scrape_visual...", flush=True)
    full_scrape_perf = score_full_scrape_visual_family(live_run_dir)

    print("[compare3] scoring basic5 (xgboost_basic_v1)...", flush=True)
    basic5_perf = score_basic5_family(live_run_dir)

    print("[compare3] scoring basic5_visual (main_image_scores_live_trained)...", flush=True)
    basic5_visual_perf = score_basic5_visual_family(live_run_dir)

    combined = build_comparison(full_scrape_perf, basic5_perf, basic5_visual_perf)

    out_dir = assert_experiment_path(LIVE_RUNS_DIR / run_id("compare_three_families"))
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "three_family_comparison.csv"
    combined.to_csv(combined_path, index=False)

    report_path = out_dir / "three_family_report.md"
    write_report(report_path, live_run_dir=live_run_dir, combined=combined)
    mirror = assert_experiment_path(REPORTS_DIR / f"{out_dir.name}_three_family_report.md")
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    write_manifest(
        out_dir / "manifest.json",
        command="full_scrape_giant_model.compare_three_families",
        extra=base_sweep.to_builtin(
            {
                "live_run_dir": str(live_run_dir),
                "full_scrape_focus_approach": FULL_SCRAPE_FOCUS_APPROACH,
                "families": list(FAMILY_LABELS.keys()),
                "windows": [FOCUS_HOURS, *APPENDIX_HOURS],
                "read_only": True,
                "telegram_policy_changed": False,
            }
        ),
    )
    print(f"[compare3] report: {report_path}", flush=True)
    print(json.dumps({"report": str(report_path), "comparison": str(combined_path)}, indent=2))


if __name__ == "__main__":
    main()
