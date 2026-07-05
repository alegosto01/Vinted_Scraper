#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.old.basic_plus_visual._deps.deal_finder.paths import LIVE_RUNS_DIR, assert_experiment_path, utc_now_iso, write_manifest


def latest_run(prefix: str = "six_search_strict_hourly_loop") -> Path | None:
    if not LIVE_RUNS_DIR.exists():
        return None
    runs = [path for path in LIVE_RUNS_DIR.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    return max(runs, key=lambda path: path.stat().st_mtime) if runs else None


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_search_summary(tracked: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    if tracked.empty:
        summary = pd.DataFrame(columns=["SearchName", "tracked_count", "sold_count", "active_count", "mean_score", "max_score"])
        summary.to_csv(out_path, index=False)
        return summary
    sold_mask = pd.to_datetime(tracked.get("sold_at"), errors="coerce", utc=True).notna()
    grouped = tracked.assign(_sold=sold_mask).groupby("SearchName", dropna=False)
    summary = grouped.agg(
        tracked_count=("tracking_key", "count"),
        sold_count=("_sold", "sum"),
        mean_score=("current_score", "mean"),
        max_score=("max_score", "max"),
    ).reset_index()
    summary["active_count"] = summary["tracked_count"] - summary["sold_count"]
    summary = summary[["SearchName", "tracked_count", "sold_count", "active_count", "mean_score", "max_score"]]
    summary.to_csv(out_path, index=False)
    return summary


def build_markdown(run_dir: Path, tracked: pd.DataFrame, history: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = [
        "# Six-Search Strict Hourly Report",
        "",
        f"Generated at: `{utc_now_iso()}`",
        f"Run directory: `{run_dir}`",
        "",
    ]
    if tracked.empty:
        lines.extend(["No tracked state rows found yet.", ""])
    else:
        sold_count = int(pd.to_datetime(tracked.get("sold_at"), errors="coerce", utc=True).notna().sum())
        lines.extend(
            [
                f"Tracked items: `{len(tracked)}`",
                f"Sold items: `{sold_count}`",
                f"Hourly history rows: `{len(history)}`",
                "",
                "## Per Search",
                "",
                "```text",
                summary.to_string(index=False),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def build_plots(run_dir: Path, tracked: pd.DataFrame, history: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return created

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        chart = summary.set_index("SearchName")[["tracked_count", "sold_count", "active_count"]]
        chart.plot(kind="bar", ax=ax)
        ax.set_title("Tracked And Sold Items By Search")
        ax.set_ylabel("Items")
        ax.set_xlabel("Search")
        fig.tight_layout()
        path = plots_dir / "tracked_vs_sold_by_search.png"
        fig.savefig(path)
        plt.close(fig)
        created.append(str(path))

    if not tracked.empty and "current_score" in tracked.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        tracked["current_score"].dropna().astype(float).plot(kind="hist", bins=20, ax=ax)
        ax.set_title("Current Score Distribution")
        ax.set_xlabel("Model probability")
        fig.tight_layout()
        path = plots_dir / "current_score_distribution.png"
        fig.savefig(path)
        plt.close(fig)
        created.append(str(path))

    if not history.empty and "observed_at" in history.columns:
        hours = pd.to_datetime(history["observed_at"], errors="coerce", utc=True).dt.floor("h")
        counts = history.assign(observed_hour=hours).groupby(["observed_hour", "event_type"], dropna=False).size().unstack(fill_value=0)
        if not counts.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            counts.plot(ax=ax)
            ax.set_title("Hourly Event Volume")
            ax.set_ylabel("Rows")
            ax.set_xlabel("Observed hour")
            fig.tight_layout()
            path = plots_dir / "hourly_event_volume.png"
            fig.savefig(path)
            plt.close(fig)
            created.append(str(path))
    return created


def build_report(run_dir: Path | None) -> Path:
    chosen_run = latest_run() if run_dir is None else assert_experiment_path(run_dir)
    if chosen_run is None:
        raise FileNotFoundError("No six-search strict hourly run found.")
    reports_dir = chosen_run / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    tracked = load_csv(chosen_run / "tracked_state.csv")
    history = load_csv(chosen_run / "hourly_history.csv")
    summary = write_search_summary(tracked, reports_dir / "search_summary.csv")
    report_path = reports_dir / "latest_report.md"
    report_path.write_text(build_markdown(chosen_run, tracked, history, summary), encoding="utf-8")
    plots = build_plots(chosen_run, tracked, history, summary)
    write_manifest(reports_dir / "report_manifest.json", command="report_six_search_strict_hourly", extra={"run_dir": str(chosen_run), "report_path": str(report_path), "plot_paths": plots})
    return report_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate reports and plots for the six-search strict hourly experiment.")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else None
    print(build_report(run_dir))


if __name__ == "__main__":
    main()