#!/usr/bin/env python3
"""Compare the full-scrape giant models against the basic-5 giant models (offline).

Reads the latest offline run from each family and produces a side-by-side
markdown comparison: best overall and per-search AUC / PR-AUC / P@25 / precision,
with deltas and a win count. Read-only; writes only under the full_scrape_giant
reports directory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[3]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.current.full_scrape_giant_model.paths import (  # noqa: E402
    OFFLINE_RUNS_DIR,
    REPORTS_DIR,
    ROOT,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
)

BASIC5_OFFLINE = ROOT / "experiments" / "current" / "basic_5_giant_model" / "data" / "offline_runs"


def _latest(root: Path, glob: str, needs: str) -> Path:
    runs = sorted((p for p in root.glob(glob) if (p / needs).exists()), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No runs matching {glob} with {needs} under {root}")
    return runs[0]


def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    """Return the first present column among names, else an all-NaN series."""
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series([float("nan")] * len(df), index=df.index)


def _fmt(v: Any, d: int = 3) -> str:
    try:
        if v is None or pd.isna(v):
            return "-"
        return f"{float(v):.{d}f}"
    except Exception:
        return str(v)


def _best_overall(metrics_path: Path) -> dict[str, Any]:
    df = pd.read_csv(metrics_path)
    if "status" in df.columns:
        df = df[df["status"] == "trained"]
    if df.empty:
        return {}
    auc = _col(df, "test_roc_auc")
    df = df.assign(_auc=auc)
    top = df.sort_values("_auc", ascending=False, na_position="last").iloc[0]
    return {
        "approach": top.get("approach"),
        "auc": float(_col(df, "test_roc_auc").loc[top.name]),
        "pr_auc": float(_col(df, "test_pr_auc").loc[top.name]),
        "p25": float(_col(df, "test_precision_at_25").loc[top.name]),
        "precision": float(_col(df, "test_precision", "test_threshold_precision").loc[top.name]),
    }


def _per_search_best(per_search_path: Path) -> pd.DataFrame:
    df = pd.read_csv(per_search_path)
    df = df.assign(_auc=_col(df, "roc_auc"), _p25=_col(df, "precision_at_25"))
    best = df.sort_values("_auc", ascending=False, na_position="last").drop_duplicates("search_name", keep="first")
    return best[["search_name", "approach", "_auc", "_p25"]].rename(columns={"_auc": "auc", "_p25": "p25"}).reset_index(drop=True)


def compare(fs_run: Path, basic_run: Path) -> Path:
    ensure_experiment_dirs()
    fs_overall = _best_overall(fs_run / "metrics_long.csv")
    b5_overall = _best_overall(basic_run / "metrics_long.csv")
    fs_ps = _per_search_best(fs_run / "per_search_metrics.csv")
    b5_ps = _per_search_best(basic_run / "per_search_metrics.csv")

    merged = fs_ps.merge(b5_ps, on="search_name", how="outer", suffixes=("_fs", "_b5")).sort_values("search_name")
    # Win-count only over searches BOTH families actually trained+scored (non-NaN AUC).
    common = merged.dropna(subset=["auc_fs", "auc_b5"])
    auc_wins = int((common["auc_fs"] > common["auc_b5"]).sum())
    p25_common = common.dropna(subset=["p25_fs", "p25_b5"])
    p25_wins = int((p25_common["p25_fs"] > p25_common["p25_b5"]).sum())
    n_search = int(len(common))
    n_p25 = int(len(p25_common))

    lines = [
        "# Full-Scrape Giant vs Basic-5 Giant (offline)",
        "",
        f"- Full-scrape run: `{fs_run.name}`",
        f"- Basic-5 run: `{basic_run.name}`",
        "",
        "## Best Model Overall (held-out test)",
        "",
        "| family | best approach | AUC | PR-AUC | P@25 | threshold precision |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| full_scrape | {fs_overall.get('approach','-')} | {_fmt(fs_overall.get('auc'))} | "
        f"{_fmt(fs_overall.get('pr_auc'))} | {_fmt(fs_overall.get('p25'))} | {_fmt(fs_overall.get('precision'))} |",
        f"| basic_5 | {b5_overall.get('approach','-')} | {_fmt(b5_overall.get('auc'))} | "
        f"{_fmt(b5_overall.get('pr_auc'))} | {_fmt(b5_overall.get('p25'))} | {_fmt(b5_overall.get('precision'))} |",
        f"| **delta (fs - b5)** | | **{_fmt((fs_overall.get('auc') or 0) - (b5_overall.get('auc') or 0))}** | "
        f"**{_fmt((fs_overall.get('pr_auc') or 0) - (b5_overall.get('pr_auc') or 0))}** | "
        f"**{_fmt((fs_overall.get('p25') or 0) - (b5_overall.get('p25') or 0))}** | |",
        "",
        "## Best Per-Search AUC (full_scrape vs basic_5)",
        "",
        "| search | fs best | fs AUC | b5 best | b5 AUC | ΔAUC | fs P@25 | b5 P@25 | ΔP@25 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in merged.iterrows():
        d_auc = (r["auc_fs"] if pd.notna(r["auc_fs"]) else 0) - (r["auc_b5"] if pd.notna(r["auc_b5"]) else 0)
        d_p25 = (r["p25_fs"] if pd.notna(r["p25_fs"]) else 0) - (r["p25_b5"] if pd.notna(r["p25_b5"]) else 0)
        lines.append(
            f"| {r['search_name']} | {r.get('approach_fs','-')} | {_fmt(r['auc_fs'])} | "
            f"{r.get('approach_b5','-')} | {_fmt(r['auc_b5'])} | {_fmt(d_auc)} | "
            f"{_fmt(r['p25_fs'])} | {_fmt(r['p25_b5'])} | {_fmt(d_p25)} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Full-scrape beats basic-5 on AUC in **{auc_wins}/{n_search}** searches both families trained on.",
            f"- Full-scrape beats basic-5 on P@25 in **{p25_wins}/{n_p25}** searches both families trained on.",
            "- (Searches only one family trained on — e.g. Borse_Griffate, telefoni — are excluded from the win count.)",
            "",
            "## Notes",
            "",
            "- Both families train on the same offline backfill labels and the same stratified 60/20/20 "
            "split machinery, so the comparison is apples-to-apples. The full-scrape family reuses the nine "
            "basic-5 model kinds as baselines (renamed `*_full_v1`) and adds CatBoost, LightGBM, tuned "
            "ExtraTrees/HistGB and a stacking ensemble.",
            "- basic-5 uses Price/Likes/Title/Brand/Size + search one-hot; full-scrape adds item/seller "
            "metadata (reviews, stars, picture counts, description/title length, upload age, condition, "
            "brand, country) and Title/Description TF-IDF text.",
            "- Telegram policy was NOT changed by either this comparison or the full-scrape experiment.",
        ]
    )

    out = assert_experiment_path(REPORTS_DIR / f"{run_id('compare_basic5')}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[compare] AUC wins fs>{auc_wins}/{n_search}; P@25 wins fs>{p25_wins}/{n_search}")
    print(f"[compare] report: {out}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare full-scrape giant vs basic-5 giant (offline).")
    p.add_argument("--fs-run", default=None, help="Full-scrape offline run dir; defaults to latest.")
    p.add_argument("--basic5-run", default=None, help="Basic-5 offline run dir; defaults to latest.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fs_run = Path(args.fs_run) if args.fs_run else _latest(OFFLINE_RUNS_DIR, "full_scrape_giant_*", "metrics_long.csv")
    basic_run = Path(args.basic5_run) if args.basic5_run else _latest(BASIC5_OFFLINE, "basic_5_giant_*", "metrics_long.csv")
    compare(fs_run, basic_run)


if __name__ == "__main__":
    main()
