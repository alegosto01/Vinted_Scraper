#!/usr/bin/env python3
"""Apply trained full-scrape giant models to the live-collector data (read-only).

This scores the items the time-to-sell live collector has gathered so far with
each trained full-scrape giant model, then measures held-out performance against
the collector's ``sold_within_{24h,48h,72h}`` outcome labels.

STRICTLY READ-ONLY with respect to the live pipeline and Telegram:
- it never sends Telegram messages and never modifies any Telegram code/policy;
- it only reads the live ``tracked_items.csv`` and writes its own report under
  ``experiments/current/full_scrape_giant_model/data/live_runs/``.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.current.full_scrape_giant_model._deps.deal_finder import model_sweep as base_sweep  # noqa: E402
from experiments.current.full_scrape_giant_model._deps.deal_finder.modeling import score_with_model  # noqa: E402
from experiments.current.full_scrape_giant_model.full_scrape_features import add_full_scrape_features  # noqa: E402
from experiments.current.full_scrape_giant_model import metrics as fm  # noqa: E402
from experiments.current.full_scrape_giant_model.image_filter import live_image_filter  # noqa: E402
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

WINDOWS = ("24h", "48h", "72h")
LIVE_RUNS_ROOT = ROOT / "experiments" / "current" / "time_to_sell" / "data" / "live_runs"
TELEGRAM_NOTE = (
    "Telegram policy was NOT changed: this live evaluation is strictly read-only. "
    "No Telegram messages were sent and no Telegram code/policy was modified."
)


def latest_offline_run() -> Path:
    runs = sorted(
        (p for p in OFFLINE_RUNS_DIR.glob("full_scrape_giant_*") if (p / "metrics_long.csv").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No trained offline runs under {OFFLINE_RUNS_DIR}; run run.py first.")
    return runs[0]


def latest_live_tracked() -> Path:
    runs = sorted(
        (p for p in LIVE_RUNS_ROOT.glob("bin_collector_*") if (p / "tracked_items.csv").exists()),
        key=lambda p: (p / "tracked_items.csv").stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No live tracked_items.csv under {LIVE_RUNS_ROOT}")
    return runs[0] / "tracked_items.csv"


def load_models(offline_run: Path) -> list[dict[str, Any]]:
    """Load (approach, model, threshold) for every model trained in offline_run."""
    run_name = offline_run.name
    import json

    out: list[dict[str, Any]] = []
    for meta_path in sorted(MODELS_DIR.glob(f"{run_name}_full_scrape_*_metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pkl_path = Path(meta["artifact_path"])
        if not pkl_path.exists():
            continue
        with pkl_path.open("rb") as handle:
            model = pickle.load(handle)
        out.append({"approach": meta["approach"], "model": model, "threshold": float(meta["threshold"])})
    return out


def window_label(frame: pd.DataFrame, window: str) -> pd.Series | None:
    col = f"sold_within_{window}"
    if col not in frame.columns:
        return None
    return base_sweep.parse_bool_series(frame[col]).astype(int)


def evaluate_live(offline_run: Path, live_csv: Path, *, limit_rows: int | None = None) -> dict[str, Any]:
    ensure_experiment_dirs()
    raw = pd.read_csv(live_csv, low_memory=False)
    if "SearchName" not in raw.columns:
        raw["SearchName"] = "__ALL__"
    raw["SearchName"] = raw["SearchName"].fillna("").astype(str)
    if limit_rows:
        raw = raw.sample(min(limit_rows, len(raw)), random_state=base_sweep.DEFAULT_SEED).reset_index(drop=True)

    total_before = int(len(raw))
    kept, image_acc = live_image_filter(raw)
    kept = kept.reset_index(drop=True)
    # Build features exactly as offline training did so every model's
    # ColumnTransformer finds its columns (engineered snapshot numerics +
    # full-scrape item/seller metadata + text).
    feats = base_sweep.add_engineered_snapshot_features(kept)
    feats = add_full_scrape_features(feats)
    total_after = int(len(feats))

    models = load_models(offline_run)
    if not models:
        raise RuntimeError(f"No model artifacts found for offline run {offline_run.name}")

    # Score once per model, reuse across windows.
    scored: dict[str, np.ndarray] = {}
    for m in models:
        scored[m["approach"]] = np.clip(np.asarray(score_with_model(m["model"], feats), dtype=float), 0.0, 1.0)

    metric_rows: list[dict[str, Any]] = []
    per_search_rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        y = window_label(feats, window)
        if y is None:
            continue
        y_arr = y.to_numpy()
        for m in models:
            scores = scored[m["approach"]]
            summary = fm.summarize(y_arr, scores, m["threshold"])
            metric_rows.append({"window": window, "approach": m["approach"], "threshold": m["threshold"], **summary})
            for search, grp in feats.groupby("SearchName", sort=True):
                idx = grp.index.to_numpy(dtype=int)
                s = fm.summarize(y_arr[idx], scores[idx], m["threshold"])
                per_search_rows.append({"window": window, "search_name": str(search), "approach": m["approach"], **s})

    out_dir = assert_experiment_path(LIVE_RUNS_DIR / run_id("live_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    image_acc.to_csv(assert_experiment_path(out_dir / "live_image_filter_summary.csv"), index=False)
    metrics_df = pd.DataFrame(metric_rows)
    per_search_df = pd.DataFrame(per_search_rows)
    metrics_df.to_csv(assert_experiment_path(out_dir / "live_metrics_long.csv"), index=False)
    per_search_df.to_csv(assert_experiment_path(out_dir / "live_per_search_metrics.csv"), index=False)

    report_path = write_live_report(
        out_dir,
        offline_run=offline_run,
        live_csv=live_csv,
        image_acc=image_acc,
        metrics_df=metrics_df,
        per_search_df=per_search_df,
        total_before=total_before,
        total_after=total_after,
    )
    write_manifest(
        out_dir / "manifest.json",
        command="full_scrape_giant_model.apply_to_live",
        extra=base_sweep.to_builtin(
            {
                "offline_run": str(offline_run),
                "live_csv": str(live_csv),
                "rows_before_image_filter": total_before,
                "rows_after_image_filter": total_after,
                "windows": list(WINDOWS),
                "models": [m["approach"] for m in models],
                "read_only": True,
                "telegram_policy_changed": False,
            }
        ),
    )
    print(f"[live] report: {report_path}", flush=True)
    return {"run_dir": str(out_dir), "report": str(report_path), "rows_before": total_before, "rows_after": total_after}


def _fmt(v: Any, d: int = 3) -> str:
    try:
        if v is None or pd.isna(v):
            return "-"
        return f"{float(v):.{d}f}"
    except Exception:
        return str(v)


def write_live_report(
    out_dir: Path,
    *,
    offline_run: Path,
    live_csv: Path,
    image_acc: pd.DataFrame,
    metrics_df: pd.DataFrame,
    per_search_df: pd.DataFrame,
    total_before: int,
    total_after: int,
) -> Path:
    lines = [
        "# Full-Scrape Giant Model — Live-Collector Evaluation",
        "",
        f"Offline models: `{offline_run.name}`",
        f"Live source: `{live_csv}`",
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
    lines.append("")
    lines.append(f"Total live rows before image filtering → after: **{total_before} → {total_after}**.")
    lines.append("")
    lines.append(
        "> Note: the live tracker does not carry picture-count columns, so the full-scrape "
        "picture features are NaN-imputed (training medians) when scoring live items."
    )

    for window in WINDOWS:
        wdf = metrics_df[metrics_df["window"] == window].copy() if not metrics_df.empty else pd.DataFrame()
        if wdf.empty:
            continue
        wdf = wdf.sort_values("roc_auc", ascending=False, na_position="last")
        base_rate = float(wdf["base_rate"].iloc[0]) if "base_rate" in wdf else float("nan")
        lines.extend(
            [
                "",
                f"## Window: sold within {window} (base rate {_fmt(base_rate)})",
                "",
                "| approach | AUC | PR-AUC | precision | recall | coverage | selected | P@10 | P@25 | P@50 | lift |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, r in wdf.iterrows():
            lines.append(
                f"| {r['approach']} | {_fmt(r['roc_auc'])} | {_fmt(r['pr_auc'])} | {_fmt(r['precision'])} | "
                f"{_fmt(r['recall'])} | {_fmt(r['coverage'])} | {int(r['selected_count'])} | "
                f"{_fmt(r['precision_at_10'])} | {_fmt(r['precision_at_25'])} | {_fmt(r['precision_at_50'])} | {_fmt(r['lift'])} |"
            )
        best = wdf.iloc[0]
        lines.append("")
        lines.append(f"- Best by AUC: **{best['approach']}** ({_fmt(best['roc_auc'])}).")

    # Best model per search (use the 48h window if present, else first available).
    if not per_search_df.empty:
        pivot_window = "48h" if (per_search_df["window"] == "48h").any() else per_search_df["window"].iloc[0]
        sub = per_search_df[per_search_df["window"] == pivot_window]
        best_ps = sub.sort_values("roc_auc", ascending=False, na_position="last").drop_duplicates("search_name", keep="first")
        lines.extend(["", f"## Best Model Per Search (window {pivot_window})", "",
                      "| search | best approach | AUC | precision | P@25 | lift | base_rate |",
                      "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
        for _, r in best_ps.sort_values("search_name").iterrows():
            lines.append(
                f"| {r['search_name']} | {r['approach']} | {_fmt(r['roc_auc'])} | {_fmt(r['precision'])} | "
                f"{_fmt(r['precision_at_25'])} | {_fmt(r['lift'])} | {_fmt(r['base_rate'])} |"
            )

    lines.extend(["", "## Notes", "", f"- {TELEGRAM_NOTE}",
                  "- Outcome labels come from the live collector's `sold_within_{24h,48h,72h}` flags; "
                  "very recently collected items may not have fully matured within a window."])

    body = "\n".join(lines) + "\n"
    primary = assert_experiment_path(out_dir / "live_report.md")
    primary.write_text(body, encoding="utf-8")
    mirror = assert_experiment_path(REPORTS_DIR / f"{out_dir.name}_live_report.md")
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(body, encoding="utf-8")
    return primary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply full-scrape giant models to live-collector data (read-only).")
    p.add_argument("--offline-run", default=None, help="Offline run dir name or path; defaults to the latest.")
    p.add_argument("--live-csv", default=None, help="Path to a live tracked_items.csv; defaults to the latest.")
    p.add_argument("--limit-rows", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    offline_run = (
        Path(args.offline_run) if args.offline_run and Path(args.offline_run).exists()
        else (OFFLINE_RUNS_DIR / args.offline_run) if args.offline_run
        else latest_offline_run()
    )
    live_csv = Path(args.live_csv) if args.live_csv else latest_live_tracked()
    evaluate_live(offline_run, live_csv, limit_rows=args.limit_rows)


if __name__ == "__main__":
    main()
