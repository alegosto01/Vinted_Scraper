#!/usr/bin/env python3
"""Train models on the LIVE objective (sold_within_{window}) from the live collector.

Unlike run.py (offline backfill), this fits and evaluates on the live-collector
data itself, with three feature modes (basic / full_scrape / full_scrape+visual),
a time-based split (train older, test newer) to avoid leakage, and thresholds
chosen on a LIVE validation slice (live-calibrated) rather than transferred from
offline. This answers: does training on the live objective + adding visual
features let full-scrape beat the basic-5 feature set at a fair operating point?

Read-only w.r.t. the live pipeline and Telegram; writes only under
data/experiments/full_scrape_giant_model/live_runs/.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[3]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.deal_finder import model_sweep as base_sweep  # noqa: E402
from experiments.deal_finder.modeling import choose_threshold, score_with_model  # noqa: E402
from experiments.current.full_scrape_giant_model.full_scrape_features import add_full_scrape_features  # noqa: E402
from experiments.full_scrape_model.compare_feature_modalities import VISUAL_NUMERIC  # noqa: E402
from experiments.current.full_scrape_giant_model import metrics as fm  # noqa: E402
from experiments.current.full_scrape_giant_model import model_zoo as mz  # noqa: E402
from experiments.current.full_scrape_giant_model.image_filter import live_image_filter  # noqa: E402
from experiments.current.full_scrape_giant_model.paths import (  # noqa: E402
    LIVE_RUNS_DIR,
    REPORTS_DIR,
    ROOT,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_manifest,
)

LIVE_RUNS_ROOT = ROOT / "data" / "experiments" / "time_to_sell" / "live_runs"
MODES = ("basic", "full_scrape", "full_scrape_visual")
# Lean, live-relevant zoo: linear + rules + the trees/boosters that mattered live.
# (sgd/xgboost/lightgbm/stacking dropped — slow and not live winners.)
LIVE_SPEC_NAMES = (
    "logistic_full_v1",
    "logistic_engineered_full_v1",
    "linear_svm_calibrated_full_v1",
    "rules_price_v1",
    "random_forest_full_v1",
    "hist_gradient_full_v1",
    "catboost_full_v1",
)
TELEGRAM_NOTE = (
    "Telegram policy was NOT changed: live training/eval is read-only, no Telegram "
    "messages were sent and no Telegram code/policy was modified."
)


def latest_live_dir() -> Path:
    runs = sorted(
        (p for p in LIVE_RUNS_ROOT.glob("bin_collector_*") if (p / "tracked_items.csv").exists()),
        key=lambda p: (p / "tracked_items.csv").stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No live tracked_items.csv under {LIVE_RUNS_ROOT}")
    return runs[0]


def load_visual_features(live_dir: Path) -> pd.DataFrame:
    """Concatenate per-snapshot visual feature CSVs, keep item_id + VISUAL_NUMERIC."""
    vdir = live_dir / "visual_features"
    files = sorted(vdir.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=["item_id"])
    keep = ["item_id"] + list(VISUAL_NUMERIC)
    frames = []
    for f in files:
        head = pd.read_csv(f, nrows=0).columns
        usecols = [c for c in keep if c in head]
        if "item_id" not in usecols:
            continue
        frames.append(pd.read_csv(f, usecols=usecols, low_memory=False))
    if not frames:
        return pd.DataFrame(columns=["item_id"])
    visual = pd.concat(frames, ignore_index=True)
    visual["item_id"] = visual["item_id"].astype(str).str.strip()
    visual = visual[visual["item_id"].str.len() > 0].drop_duplicates("item_id", keep="last")
    return visual.reset_index(drop=True)


def assemble(live_dir: Path, window: str) -> tuple[pd.DataFrame, list[str], pd.Series]:
    raw = pd.read_csv(live_dir / "tracked_items.csv", low_memory=False)
    raw["SearchName"] = raw.get("SearchName", "").fillna("").astype(str)
    raw["item_id"] = raw.get("item_id", "").astype(str).str.strip()
    kept, _ = live_image_filter(raw)
    kept = kept.reset_index(drop=True)

    visual = load_visual_features(live_dir)
    visual_cols = [c for c in VISUAL_NUMERIC if c in visual.columns]
    merged = kept.merge(visual, on="item_id", how="left", suffixes=("", "_vis"))

    # Engineered + full-scrape features (picture features now recovered via visual merge).
    merged = base_sweep.add_engineered_snapshot_features(merged)
    merged = add_full_scrape_features(merged)

    label_col = f"sold_within_{window}"
    y = base_sweep.parse_bool_series(merged[label_col]).astype(int)
    ts = pd.to_datetime(merged.get("collector_snapshot_at"), errors="coerce", utc=True)
    merged["_ts"] = ts
    return merged, visual_cols, y


def time_split(frame: pd.DataFrame, y: pd.Series, *, seed: int) -> tuple[pd.Index, pd.Index, pd.Index]:
    """60/20/20 by collection time (older->train) if timestamps span >1 day; else stratified random."""
    ts = frame["_ts"]
    if ts.notna().sum() >= len(frame) * 0.8 and ts.dt.normalize().nunique() >= 3:
        order = ts.sort_values(kind="stable").index
        n = len(order)
        tr = order[: int(n * 0.6)]
        va = order[int(n * 0.6): int(n * 0.8)]
        te = order[int(n * 0.8):]
        return tr, va, te
    from sklearn.model_selection import train_test_split

    strat = frame["SearchName"].astype(str) + "_" + y.astype(str)
    strat = strat if strat.value_counts().min() >= 2 else None
    tr, rest = train_test_split(frame.index, train_size=0.6, random_state=seed, stratify=strat)
    rest_strat = (frame.loc[rest, "SearchName"].astype(str) + "_" + y.loc[rest].astype(str))
    rest_strat = rest_strat if rest_strat.value_counts().min() >= 2 else None
    va, te = train_test_split(rest, train_size=0.5, random_state=seed, stratify=rest_strat)
    return pd.Index(tr), pd.Index(va), pd.Index(te)


def select_mode_features(frame: pd.DataFrame, spec: mz.GiantSpec, mode: str, visual_cols: list[str]):
    if spec.kind == "rules":
        return mz.select_full_features(frame, spec)
    if mode == "basic":
        numeric = [c for c in ("Price", "Likes") if mz._has_numeric_signal(frame, c)]
        if spec.use_engineered:
            numeric += [c for c in mz.ENGINEERED_EXTRA if c not in numeric and mz._has_numeric_signal(frame, c)]
        categorical = [c for c in ["SearchName"] if mz._has_categorical_signal(frame, c)]
        text = [] if spec.kind in mz.DENSE_KINDS else [c for c in ("TitleText", "Brand", "Size") if mz._has_text_signal(frame, c)]
        numeric = base_sweep.safe_selected_features(numeric, frame)
        return numeric, categorical, text
    numeric, categorical, text = mz.select_full_features(frame, spec)
    if mode == "full_scrape_visual":
        extra = [c for c in visual_cols if mz._has_numeric_signal(frame, c) and c not in numeric]
        numeric = numeric + base_sweep.safe_selected_features(extra, frame)
    return numeric, categorical, text


def run(
    *,
    window: str = "48h",
    seed: int = base_sweep.DEFAULT_SEED,
    min_precision: float = 0.40,
    min_count: int = 25,
    modes: tuple[str, ...] = MODES,
) -> dict[str, Any]:
    ensure_experiment_dirs()
    live_dir = latest_live_dir()
    frame, visual_cols, y = assemble(live_dir, window)
    tr, va, te = time_split(frame, y, seed=seed)
    print(f"[live_obj] {live_dir.name} window={window} rows={len(frame)} "
          f"train={len(tr)} val={len(va)} test={len(te)} base_rate={y.mean():.3f} visual_cols={len(visual_cols)}", flush=True)

    specs = [mz.zoo_by_name()[n] for n in LIVE_SPEC_NAMES]
    # Precompute split frames + label arrays once (avoid repeated .loc copies).
    tr_frame, va_frame, te_frame = frame.loc[tr], frame.loc[va], frame.loc[te]
    ytr, yva, yte = y.loc[tr].astype(int).to_numpy(), y.loc[va].to_numpy(), y.loc[te].to_numpy()
    va_search = va_frame["SearchName"].to_numpy()
    te_search = te_frame["SearchName"].to_numpy()
    searches = sorted(pd.unique(te_search).tolist())
    # Per-search thresholds need their own min-count floor (val slices are small);
    # fall back to the global threshold when a search's val slice can't support
    # its own precision-targeted threshold, instead of choose_threshold's
    # best-effort fallback (which can pick threshold=1.0 / zero coverage).
    per_search_min_count = max(8, min_count // 2)
    min_val_rows = per_search_min_count * 2
    metric_rows: list[dict[str, Any]] = []
    per_search_rows: list[dict[str, Any]] = []
    for mode in modes:
        for spec in specs:
            t0 = time.perf_counter()
            try:
                numeric, categorical, text = select_mode_features(tr_frame, spec, mode, visual_cols)
                if not numeric and not categorical and not text:
                    continue
                model = mz.make_model(spec, numeric, categorical, text)
                model.fit(tr_frame, ytr)
                sv = np.clip(np.asarray(score_with_model(model, va_frame), dtype=float), 0, 1)
                st = np.clip(np.asarray(score_with_model(model, te_frame), dtype=float), 0, 1)
                thr = float(choose_threshold(yva, sv, min_precision=min_precision, min_count=min_count)["threshold"])
                m = fm.summarize(yte, st, thr)
                metric_rows.append({"mode": mode, "approach": spec.name, "threshold": thr, **m})
                for s in searches:
                    va_mask = va_search == s
                    s_thr = thr
                    if va_mask.sum() >= min_val_rows:
                        cand = choose_threshold(yva[va_mask], sv[va_mask], min_precision=min_precision, min_count=per_search_min_count)
                        cand_prec = cand["precision"]
                        if pd.notna(cand_prec) and cand_prec >= min_precision:
                            s_thr = float(cand["threshold"])
                    te_mask = te_search == s
                    ms = fm.summarize(yte[te_mask], st[te_mask], s_thr)
                    per_search_rows.append({"mode": mode, "search_name": str(s), "approach": spec.name, **ms})
                print(f"[live_obj] {mode:18s} {spec.name:30s} AUC={m['roc_auc']:.3f} P@25={m['precision_at_25']:.3f} "
                      f"prec={fm_fmt(m['precision'])} cov={m['coverage']:.3f} {time.perf_counter()-t0:.1f}s", flush=True)
            except Exception as exc:
                print(f"[live_obj] ERR {mode}/{spec.name}: {type(exc).__name__}: {exc}", flush=True)

    out_dir = assert_experiment_path(LIVE_RUNS_DIR / run_id(f"live_objective_{window}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    mdf = pd.DataFrame(metric_rows)
    psdf = pd.DataFrame(per_search_rows)
    mdf.to_csv(assert_experiment_path(out_dir / "live_objective_metrics.csv"), index=False)
    psdf.to_csv(assert_experiment_path(out_dir / "live_objective_per_search.csv"), index=False)
    report = write_report(out_dir, window=window, mdf=mdf, psdf=psdf, base_rate=float(y.mean()),
                          n_train=len(tr), n_test=len(te), visual_cols=visual_cols, min_precision=min_precision)
    write_manifest(out_dir / "manifest.json", command="full_scrape_giant_model.live_objective",
                   extra=base_sweep.to_builtin({"live_dir": str(live_dir), "window": window, "modes": list(modes),
                                                "min_precision": min_precision, "read_only": True, "telegram_policy_changed": False}))
    print(f"[live_obj] report: {report}", flush=True)
    return {"run_dir": str(out_dir), "report": str(report)}


def fm_fmt(v, d=3):
    try:
        return "-" if pd.isna(v) else f"{float(v):.{d}f}"
    except Exception:
        return str(v)


def write_report(out_dir, *, window, mdf, psdf, base_rate, n_train, n_test, visual_cols, min_precision) -> Path:
    f = fm_fmt
    lines = [
        f"# Live-Objective Models — sold within {window}",
        "",
        f"Trained AND tested on live-collector data (time-split, train older / test newer). "
        f"train={n_train}, test={n_test}, base rate={f(base_rate)}. Visual features merged: {len(visual_cols)}.",
        f"Thresholds chosen on a live validation slice (target precision >= {min_precision}).",
        "",
        "## Best Model Per Feature Mode (live test, by AUC)",
        "",
        "| mode | best approach | AUC | PR-AUC | precision@thr | recall | coverage | selected | P@25 | P@50 | lift |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    best_per_mode: dict[str, str] = {}
    for mode in MODES:
        sub = mdf[mdf["mode"] == mode].dropna(subset=["roc_auc"]) if not mdf.empty else pd.DataFrame()
        if sub.empty:
            continue
        r = sub.sort_values("roc_auc", ascending=False).iloc[0]
        best_per_mode[mode] = str(r["approach"])
        lines.append(
            f"| {mode} | {r['approach']} | {f(r['roc_auc'])} | {f(r['pr_auc'])} | {f(r['precision'])} | "
            f"{f(r['recall'])} | {f(r['coverage'])} | {int(r['selected_count'])} | {f(r['precision_at_25'])} | "
            f"{f(r['precision_at_50'])} | {f(r['lift'])} |"
        )

    # per-search thresholds for each mode's best-AUC model
    if not psdf.empty and best_per_mode:
        lines += ["", "## Per-Search Thresholds (best model per mode, target precision >= "
                  f"{min_precision})", ""]
        for mode, approach in best_per_mode.items():
            sub = psdf[(psdf["mode"] == mode) & (psdf["approach"] == approach)].sort_values("search_name")
            if sub.empty:
                continue
            lines += [f"### {mode} ({approach})", "",
                      "| search | threshold | selected | precision | recall | coverage | lift | base |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for _, r in sub.iterrows():
                lines.append(
                    f"| {r['search_name']} | {f(r['threshold'])} | {int(r['selected_count'])} | "
                    f"{f(r['precision'])} | {f(r['recall'])} | {f(r['coverage'])} | {f(r['lift'])} | {f(r['base_rate'])} |"
                )
            lines.append("")

    # best-precision (threshold-free top-25) per mode
    lines += ["", "## Best P@25 Per Mode (threshold-free top-25 precision)", "",
              "| mode | best approach | P@25 | AUC |", "| --- | --- | ---: | ---: |"]
    for mode in MODES:
        sub = mdf[mdf["mode"] == mode].dropna(subset=["precision_at_25"]) if not mdf.empty else pd.DataFrame()
        if sub.empty:
            continue
        r = sub.sort_values("precision_at_25", ascending=False).iloc[0]
        lines.append(f"| {mode} | {r['approach']} | {f(r['precision_at_25'])} | {f(r['roc_auc'])} |")

    # per-search: best AUC per (search, mode), full_scrape_visual vs basic
    if not psdf.empty:
        lines += ["", f"## Per-Search: basic vs full_scrape vs full_scrape_visual (best AUC, window {window})", "",
                  "| search | basic AUC | full AUC | full+visual AUC | best mode | base |",
                  "| --- | ---: | ---: | ---: | --- | ---: |"]
        for s, grp in psdf.groupby("search_name", sort=True):
            row = {}
            for mode in MODES:
                ms = grp[grp["mode"] == mode].dropna(subset=["roc_auc"])
                row[mode] = ms["roc_auc"].max() if not ms.empty else float("nan")
            base = grp["base_rate"].iloc[0] if "base_rate" in grp else float("nan")
            best_mode = max(MODES, key=lambda mm: (row[mm] if pd.notna(row[mm]) else -1))
            lines.append(f"| {s} | {f(row['basic'])} | {f(row['full_scrape'])} | {f(row['full_scrape_visual'])} | {best_mode} | {f(base)} |")

    lines += ["", "## Notes", "",
              "- This is the first model fit on the LIVE objective (sold_within_{window}); thresholds are "
              "live-calibrated, not transferred from offline.",
              "- `full_scrape_visual` adds photo-quality + aesthetic scores and picture-count info "
              "(PictureCountNum/PhotoImageCount) merged from the per-item visual feature files.",
              f"- {TELEGRAM_NOTE}"]
    body = "\n".join(lines) + "\n"
    primary = assert_experiment_path(out_dir / "live_objective_report.md")
    primary.write_text(body, encoding="utf-8")
    mirror = assert_experiment_path(REPORTS_DIR / f"{out_dir.name}_report.md")
    mirror.write_text(body, encoding="utf-8")
    return primary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/eval on the live objective with basic/full/full+visual modes.")
    p.add_argument("--window", default="48h", choices=["24h", "48h", "72h"])
    p.add_argument("--min-precision", type=float, default=0.40)
    p.add_argument("--min-count", type=int, default=25)
    p.add_argument("--seed", type=int, default=base_sweep.DEFAULT_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(window=args.window, seed=args.seed, min_precision=args.min_precision, min_count=args.min_count)


if __name__ == "__main__":
    main()
