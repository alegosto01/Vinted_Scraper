"""Vinted monitoring dashboard — live scraping, benchmark, eventual sales."""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
SIMPLE_SCRAPE = ROOT / "data" / "simple_scrape"

SCRAPE_LOG = SIMPLE_SCRAPE / "logs" / "usual_live_scrape.log"
EVENTUAL_LOG = SIMPLE_SCRAPE / "eventual_sale.log"

CASCADE_RUN = ROOT / "data" / "experiments" / "benchmark_basic_to_full" / "live_runs" / "cascade_live"
CASCADE_TRACKED = CASCADE_RUN / "tracked_items.csv"
CASCADE_STATUS = CASCADE_RUN / "latest_status.json"
CASCADE_LOG = CASCADE_RUN / "nohup.log"
CASCADE_LOOP_LOG = CASCADE_RUN / "cascade_loop.log"
CASCADE_PID = CASCADE_RUN / "runner.pid"
CASCADE_MANIFEST = CASCADE_RUN / "manifest.json"

CASCADE_RESULTS_RUNS = ROOT / "data" / "experiments" / "benchmark_basic_to_full" / "live_runs"
TIME_TO_SELL_LIVE_RUNS = ROOT / "data" / "experiments" / "time_to_sell" / "live_runs"

TELEGRAM_EVENTS = ROOT / "data" / "telegram_implementation" / "events.jsonl"
TELEGRAM_CACHE = ROOT / "data" / "telegram_implementation" / "pending_items"

# ── helpers ───────────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def file_age(path: Path) -> float:
    """Age of file in minutes. Returns inf if missing."""
    if not path.exists():
        return float("inf")
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (_now_utc() - mtime).total_seconds() / 60


def age_str(minutes: float) -> str:
    if minutes == float("inf"):
        return "—"
    if minutes < 2:
        return "just now"
    if minutes < 60:
        return f"{int(minutes)}m ago"
    if minutes < 24 * 60:
        return f"{int(minutes / 60)}h {int(minutes % 60)}m ago"
    return f"{int(minutes / 1440)}d ago"


def pid_alive(pid_file: Path) -> bool:
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError, ProcessLookupError):
        return False


def pid_number_alive(pid: object) -> bool:
    try:
        pid_int = int(pid)
        os.kill(pid_int, 0)
        return True
    except (OSError, ValueError, TypeError, ProcessLookupError):
        return False


def find_process_by_cmdline(*needles: str) -> int | None:
    proc_root = Path("/proc")
    if not proc_root.exists():
        return None
    for proc_dir in proc_root.iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            raw_cmdline = (proc_dir / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw_cmdline:
            continue
        cmdline = raw_cmdline.replace(b"\0", b" ").decode(errors="replace")
        if all(needle in cmdline for needle in needles):
            return int(proc_dir.name)
    return None


def tail_log(path: Path, n: int = 60) -> str:
    if not path.exists():
        return "(log file not found)"
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception as exc:
        return f"(error reading log: {exc})"


def status_row(label: str, running: bool, last_activity: str) -> None:
    col1, col2 = st.columns([1, 3])
    with col1:
        if running:
            st.success(f"● {label}")
        else:
            st.info(f"● {label}")
    with col2:
        st.caption(f"Last activity: {last_activity}")


def count_csv_rows(path: Path) -> str:
    """Count data rows in a CSV by counting newlines — no pandas, no dtype warnings."""
    if not path.exists():
        return "—"
    try:
        with open(path, "rb") as f:
            return str(sum(1 for _ in f) - 1)  # always str so the column type is consistent
    except Exception:
        return "—"


def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return None


def read_json_safe(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def list_live_runs(parent: Path, prefix: str | None = None) -> list[Path]:
    if not parent.exists():
        return []
    runs = [
        p for p in parent.iterdir()
        if p.is_dir() and (prefix is None or p.name.startswith(prefix))
    ]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def install_browser_autorefresh(seconds: int) -> None:
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {max(5, int(seconds)) * 1000});
        </script>
        """,
        height=0,
    )


def normalize_item_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def tracking_key(df: pd.DataFrame) -> pd.Series:
    search = (
        df["SearchName"].fillna("").astype(str)
        if "SearchName" in df.columns
        else pd.Series("", index=df.index)
    )
    if "tracking_key" in df.columns:
        existing = df["tracking_key"].fillna("").astype(str).str.strip()
        if existing.ne("").any():
            return existing

    id_col = next((c for c in ["item_id", "Dataid", "dataid", "ItemID", "ID"] if c in df.columns), None)
    if id_col is not None:
        ids = df[id_col].map(normalize_item_id)
    elif "Link" in df.columns:
        ids = df["Link"].fillna("").astype(str)
    else:
        ids = pd.Series(df.index.astype(str), index=df.index)
    return search + "::" + ids


def truthy_series(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index)
    return series.fillna("").astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes", "y"})


def to_utc(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(pd.NaT, index=index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(series, utc=True, errors="coerce")


def first_present_datetime(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for col in columns:
        if col in df.columns:
            out = out.fillna(to_utc(df[col], df.index))
    return out


def precision_cell(sold: int, denominator: int) -> str:
    if denominator <= 0:
        return "0/0 (—)"
    return f"{sold}/{denominator} ({100 * sold / denominator:.1f}%)"


def threshold_value(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def precision_by_hour(
    df: pd.DataFrame,
    start_col: str,
    sold_col: str,
    now: pd.Timestamp,
    hours: list[int],
) -> pd.DataFrame:
    rows = []
    valid = df[df[start_col].notna()].copy()
    elapsed_to_now = (now - valid[start_col]).dt.total_seconds() / 3600
    sold_elapsed = (valid[sold_col] - valid[start_col]).dt.total_seconds() / 3600

    for hour in hours:
        matured = valid[elapsed_to_now >= hour]
        sold = matured[matured[sold_col].notna() & (sold_elapsed.loc[matured.index] <= hour)]
        denominator = len(matured)
        rows.append({
            "Hour": hour,
            "Sold": int(len(sold)),
            "Matured items": int(denominator),
            "Precision": 0.0 if denominator == 0 else len(sold) / denominator,
            "Sold / denominator": precision_cell(int(len(sold)), int(denominator)),
        })
    return pd.DataFrame(rows)


def sold_hour_buckets(df: pd.DataFrame, start_col: str, sold_col: str) -> pd.DataFrame:
    sold = df[df[sold_col].notna() & df[start_col].notna()].copy()
    if sold.empty:
        return pd.DataFrame(columns=["Search", "Sold hour buckets"])

    elapsed = (sold[sold_col] - sold[start_col]).dt.total_seconds() / 3600
    sold = sold[elapsed >= 0].copy()
    sold["_bucket"] = elapsed.loc[sold.index].astype(int)
    rows = []
    for search, group in sold.groupby("SearchName", dropna=False):
        counts = group["_bucket"].value_counts().sort_index()
        bucket_text = ", ".join(f"{int(bucket)}-{int(bucket) + 1}h: {int(count)}" for bucket, count in counts.items())
        rows.append({"Search": search, "Sold hour buckets": bucket_text})
    return pd.DataFrame(rows).sort_values("Search").reset_index(drop=True)


def load_cascade_results(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    tracked_path = run_dir / "tracked_items.csv"
    if not tracked_path.exists():
        raise FileNotFoundError(f"{tracked_path} not found")

    usecols = lambda c: c in {
        "tracking_key", "item_id", "Dataid", "SearchName", "Link", "first_stage1_pass_at",
        "sold_at", "Stage1Threshold", "Stage2Threshold", "Stage2Score", "Stage2Passed",
    }
    df = pd.read_csv(tracked_path, low_memory=False, usecols=usecols)
    if df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, {"tracked_rows": 0, "updated_at": None}

    df["_tracking_key"] = tracking_key(df)
    df["_stage2_passed"] = truthy_series(df.get("Stage2Passed"), df.index)
    df["_stage2_score"] = pd.to_numeric(df.get("Stage2Score"), errors="coerce")
    df["_first_at_raw"] = first_present_datetime(df, ["first_stage1_pass_at"])
    df["_sold_at_raw"] = first_present_datetime(df, ["sold_at"])

    first_at_by_key = df.groupby("_tracking_key")["_first_at_raw"].min()
    sold_at_by_key = df.groupby("_tracking_key")["_sold_at_raw"].min()

    unique = (
        df.sort_values(["_tracking_key", "_stage2_passed", "_stage2_score"], ascending=[True, False, False])
        .drop_duplicates("_tracking_key", keep="first")
        .copy()
    )
    unique["_first_at"] = unique["_tracking_key"].map(first_at_by_key)
    unique["_sold_at"] = unique["_tracking_key"].map(sold_at_by_key)

    now = pd.Timestamp(_now_utc())
    checkpoints = [2, 4, 6, 12, 24, 48, 72]
    rows = []
    for search, group in unique.groupby("SearchName", dropna=False):
        passed = group[group["_stage2_passed"]]
        rejected = group[~group["_stage2_passed"]]
        row = {
            "Search": search,
            "S1 thr": threshold_value(group, "Stage1Threshold"),
            "S2 thr": threshold_value(group, "Stage2Threshold"),
            "S1 pass": int(len(group)),
            "S2 pass": int(len(passed)),
            "S2 sold": int(passed["_sold_at"].notna().sum()),
            "Reject sold": int(rejected["_sold_at"].notna().sum()),
        }

        elapsed_to_now = (now - passed["_first_at"]).dt.total_seconds() / 3600
        sold_elapsed = (passed["_sold_at"] - passed["_first_at"]).dt.total_seconds() / 3600
        for hour in checkpoints:
            matured = passed[passed["_first_at"].notna() & (elapsed_to_now >= hour)]
            sold = matured[matured["_sold_at"].notna() & (sold_elapsed.loc[matured.index] <= hour)]
            row[f"{hour}h"] = precision_cell(int(len(sold)), int(len(matured)))
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("Search").reset_index(drop=True)
    stage2_passed = unique[unique["_stage2_passed"]].copy()
    overall = precision_by_hour(
        stage2_passed,
        "_first_at",
        "_sold_at",
        now,
        [2, 4, 6, 12, 24, 48, 72],
    )
    buckets = sold_hour_buckets(stage2_passed, "_first_at", "_sold_at")

    meta = {
        "tracked_rows": len(df),
        "unique_stage1_pass": int(len(unique)),
        "unique_stage2_pass": int(stage2_passed.shape[0]),
        "sold_stage2_pass": int(stage2_passed["_sold_at"].notna().sum()),
        "updated_at": datetime.fromtimestamp(tracked_path.stat().st_mtime, tz=timezone.utc),
    }
    return summary, overall, buckets, meta


def show_precision_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No rows yet.")
        return
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Precision": st.column_config.NumberColumn("Precision", format="%.1%%"),
        },
    )


def _latest_run_with(base, pattern, required_file):
    """Newest run dir under `base` matching `pattern` that contains `required_file`."""
    if not base.exists():
        return None
    cands = [p for p in base.glob(pattern) if p.is_dir() and (p / required_file).exists()]
    if not cands:
        return None
    # newest by the required file's mtime — robust to non-date-sortable run names
    # (e.g. "..._new_searches_..." sorts after date-stamped runs lexicographically)
    return max(cands, key=lambda p: (p / required_file).stat().st_mtime)


def render_giant_live(live_runs_dir, key_prefix):
    """Live matured precision/coverage for a live_scored_* run dir (basic5 + visual)."""
    ALL = "__all__"
    run = _latest_run_with(live_runs_dir, "live_scoring_*", "manifest.json")
    if run is None:
        st.info(f"No live scoring runs found under `{live_runs_dir}`.")
        return
    manifest = read_json_safe(run / "manifest.json") or {}
    summary_path = run / "live_scored_summary.csv"
    performance_path = run / "live_scored_window_performance.csv"
    scored_path = run / "live_scored_items.csv"

    st.caption(f"Run: `{run.name}`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source", str(manifest.get("offline_run", manifest.get("model", "—")))[:22])
    c2.metric("Models", len(manifest.get("approaches", [])) or ("1" if manifest.get("model") else "—"))
    c3.metric("Scored rows", count_csv_rows(scored_path))
    c4.metric("Scored", age_str(file_age(scored_path)))

    summary_df = pd.read_csv(summary_path, low_memory=False) if summary_path.exists() else pd.DataFrame()
    performance_df = pd.read_csv(performance_path, low_memory=False) if performance_path.exists() else pd.DataFrame()
    if summary_df.empty and performance_df.empty:
        st.warning("Run has no scored summary/performance yet.")
        return

    windows = sorted(pd.to_numeric(performance_df.get("hours", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique())
    preferred = [h for h in [72, 48, 24, 12, 6, 4, 2, 1] if h in windows]
    window = st.selectbox("Outcome window (h)", options=preferred or windows or [72], index=0, key=f"{key_prefix}_window")

    if not performance_df.empty:
        if "passed" in summary_df.columns and "approach" in summary_df.columns:
            passed_tot = summary_df.groupby("approach", as_index=False)["passed"].sum().rename(columns={"passed": "passed_total"})
        else:
            passed_tot = pd.DataFrame(columns=["approach", "passed_total"])
        overall = performance_df[performance_df["search_name"].eq(ALL) & performance_df["hours"].eq(window)].merge(passed_tot, on="approach", how="left")
        st.subheader(f"Model comparison at {window}h (matured)")
        st.dataframe(
            overall[[c for c in ["approach", "passed_total", "evaluated", "sold", "precision", "coverage"] if c in overall.columns]]
            .sort_values([c for c in ["precision", "sold", "evaluated"] if c in overall.columns], ascending=False, na_position="last"),
            width="stretch", hide_index=True,
            column_config={"precision": st.column_config.NumberColumn(format="%.3f"), "coverage": st.column_config.NumberColumn(format="%.3f")},
        )

    if not performance_df.empty:
        bps = performance_df[~performance_df["search_name"].eq(ALL) & performance_df["hours"].eq(window)].copy()
        bps = bps[pd.to_numeric(bps.get("evaluated"), errors="coerce").fillna(0) > 0]
        if not bps.empty:
            bps = bps.sort_values(["precision", "sold", "coverage", "evaluated"], ascending=False, na_position="last")
            best = bps.drop_duplicates("search_name", keep="first")
            if "threshold" in summary_df.columns and "approach" in summary_df.columns:
                best = best.merge(summary_df[["approach", "search_name", "threshold"]].drop_duplicates(), on=["approach", "search_name"], how="left")
            st.subheader(f"Best model per search at {window}h")
            bcols = [c for c in ["search_name", "approach", "threshold", "evaluated", "sold", "precision", "coverage"] if c in best.columns]
            st.dataframe(
                best[bcols].sort_values("search_name"),
                width="stretch", hide_index=True,
                column_config={
                    "threshold": st.column_config.NumberColumn(format="%.4f"),
                    "precision": st.column_config.NumberColumn(format="%.3f"),
                    "coverage": st.column_config.NumberColumn(format="%.3f"),
                },
            )

    approaches = manifest.get("approaches") or sorted(summary_df.get("approach", pd.Series(dtype=str)).dropna().astype(str).unique())
    if not approaches:
        return
    approach = st.selectbox("Model", options=list(approaches), index=0, key=f"{key_prefix}_approach")
    st.subheader(f"{approach} — per-search matured precision/coverage")
    model_summary = summary_df[summary_df["approach"].eq(approach)].copy() if "approach" in summary_df.columns else summary_df.copy()
    if not performance_df.empty and not model_summary.empty:
        mp = performance_df[performance_df["approach"].eq(approach) & ~performance_df["search_name"].eq(ALL) & performance_df["hours"].eq(window)]
        model_summary = model_summary.merge(mp[["search_name", "evaluated", "sold", "precision", "coverage"]], on="search_name", how="left")
    cols = [c for c in ["search_name", "threshold", "in_training_scope", "total_tracked", "total_scored", "passed", "pass_rate", "evaluated", "sold", "precision", "coverage", "mean_score", "p95_score", "max_score"] if c in model_summary.columns]
    sort_cols = [c for c in ["precision", "sold", "passed"] if c in model_summary.columns] or cols[:1]
    st.dataframe(
        model_summary[cols].sort_values(sort_cols, ascending=False, na_position="last"),
        width="stretch", hide_index=True,
        column_config={
            "threshold": st.column_config.NumberColumn(format="%.4f"),
            "pass_rate": st.column_config.NumberColumn(format="%.3f"),
            "precision": st.column_config.NumberColumn(format="%.3f"),
            "coverage": st.column_config.NumberColumn(format="%.3f"),
            "mean_score": st.column_config.NumberColumn(format="%.4f"),
            "p95_score": st.column_config.NumberColumn(format="%.4f"),
            "max_score": st.column_config.NumberColumn(format="%.4f"),
        },
    )


def render_fullscrape_giant_live(live_runs_dir, key_prefix):
    """Live per-search precision/coverage for full_scrape_giant_model (live_objective_per_search.csv)."""
    run = _latest_run_with(live_runs_dir, "live_objective_*", "live_objective_per_search.csv")
    if run is None:
        st.info(f"No full-scrape giant live objective runs found under `{live_runs_dir}`.")
        return
    per_search_path = run / "live_objective_per_search.csv"
    metrics_path = run / "live_objective_metrics.csv"

    st.caption(f"Run: `{run.name}`")
    c1, c2 = st.columns(2)
    c1.metric("Per-search rows", count_csv_rows(per_search_path))
    c2.metric("Updated", age_str(file_age(per_search_path)))

    per_search = pd.read_csv(per_search_path, low_memory=False)
    if per_search.empty:
        st.warning("No per-search rows yet.")
        return
    bps = per_search.copy()
    bps = bps[pd.to_numeric(bps.get("selected_count"), errors="coerce").fillna(0) > 0]
    if not bps.empty and "search_name" in bps.columns:
        bps = bps.sort_values(["precision", "selected_count", "roc_auc"], ascending=False, na_position="last")
        best = bps.drop_duplicates("search_name", keep="first")
        st.subheader("Best model per search (across mode × approach)")
        bcols = [c for c in ["search_name", "mode", "approach", "threshold", "selected_count", "precision", "recall", "coverage", "roc_auc"] if c in best.columns]
        st.dataframe(
            best[bcols].sort_values("search_name"),
            width="stretch", hide_index=True,
            column_config={
                "threshold": st.column_config.NumberColumn(format="%.4f"),
                "precision": st.column_config.NumberColumn(format="%.3f"),
                "recall": st.column_config.NumberColumn(format="%.3f"),
                "coverage": st.column_config.NumberColumn(format="%.3f"),
                "roc_auc": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    modes = sorted(per_search["mode"].dropna().astype(str).unique()) if "mode" in per_search.columns else []
    approaches = sorted(per_search["approach"].dropna().astype(str).unique()) if "approach" in per_search.columns else []
    fc1, fc2 = st.columns(2)
    mode = fc1.selectbox("Feature mode", options=modes or ["—"], index=0, key=f"{key_prefix}_mode")
    approach = fc2.selectbox("Model", options=approaches or ["—"], index=0, key=f"{key_prefix}_approach")

    scoped = per_search
    if "mode" in per_search.columns:
        scoped = scoped[scoped["mode"].astype(str).eq(mode)]
    if "approach" in per_search.columns:
        scoped = scoped[scoped["approach"].astype(str).eq(approach)]
    st.subheader("Per-search live precision/coverage")
    cols = [c for c in ["search_name", "rows", "base_rate", "threshold", "selected_count", "coverage", "precision", "recall", "lift", "precision_at_25", "roc_auc", "pr_auc"] if c in scoped.columns]
    sort_cols = [c for c in ["precision", "selected_count"] if c in scoped.columns] or cols[:1]
    st.dataframe(
        scoped[cols].sort_values(sort_cols, ascending=False, na_position="last"),
        width="stretch", hide_index=True,
        column_config={
            "base_rate": st.column_config.NumberColumn(format="%.3f"),
            "threshold": st.column_config.NumberColumn(format="%.4f"),
            "coverage": st.column_config.NumberColumn(format="%.3f"),
            "precision": st.column_config.NumberColumn(format="%.3f"),
            "recall": st.column_config.NumberColumn(format="%.3f"),
            "lift": st.column_config.NumberColumn(format="%.2f"),
            "precision_at_25": st.column_config.NumberColumn(format="%.3f"),
            "roc_auc": st.column_config.NumberColumn(format="%.3f"),
            "pr_auc": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    overall = pd.read_csv(metrics_path, low_memory=False) if metrics_path.exists() else pd.DataFrame()
    if not overall.empty:
        st.subheader("Overall (by mode / approach)")
        ocols = [c for c in ["mode", "approach", "threshold", "selected_count", "coverage", "precision", "recall", "precision_at_25", "roc_auc", "pr_auc"] if c in overall.columns]
        st.dataframe(overall[ocols], width="stretch", hide_index=True)


def _choose_live_threshold(scores, labels, min_precision, min_count):
    """Lowest score threshold (max coverage) with precision>=target and selected>=min_count,
    tuned on the given (in-sample) matured items. Fallback: threshold maximizing precision.
    Returns (threshold, precision, selected_count)."""
    import numpy as np
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if scores.size == 0:
        return (float("nan"), float("nan"), 0)
    order = np.argsort(-scores)
    s = scores[order]
    y = labels[order]
    n = np.arange(1, y.size + 1)
    prec = np.cumsum(y) / n
    ok = (n >= int(min_count)) & (prec >= float(min_precision))
    if ok.any():
        idx = int(np.max(np.where(ok)[0]))  # deepest qualifying depth = max coverage / lowest threshold
        return (float(s[idx]), float(prec[idx]), int(n[idx]))
    k = int(np.argmax(prec))
    return (float(s[k]), float(prec[k]), int(n[k]))


def render_threshold_compare_live(live_runs_dir, key_prefix):
    """Live data: for the deployed single-threshold model (main_image_scores), compare its
    GLOBAL threshold vs per-search thresholds tuned on live matured outcomes. In-sample."""
    run = _latest_run_with(live_runs_dir, "live_scoring_*", "live_scored_items.csv")
    if run is None:
        st.info(f"No live scoring run with scored items under `{live_runs_dir}`.")
        return
    d = pd.read_csv(run / "live_scored_items.csv", low_memory=False)
    score_cols = [c for c in d.columns if c.startswith("score__")]
    if d.empty or not score_cols or "SearchName" not in d.columns:
        st.warning("Live scored-items file missing score column or rows.")
        return
    sc = score_cols[0]
    key = sc[len("score__"):]
    thr_col = f"threshold__{key}"
    gthr_series = pd.to_numeric(d.get(thr_col), errors="coerce").dropna() if thr_col in d.columns else pd.Series(dtype=float)
    global_thr = float(gthr_series.iloc[0]) if not gthr_series.empty else 0.58
    st.caption(f"Run: `{run.name}` — model `{key}`, deployed global threshold `{global_thr:.4f}`")

    windows = sorted({int(m.group(1)) for c in d.columns for m in [re.fullmatch(r"sold_within_(\d+)h", c)] if m and f"evaluated_at_{m.group(1)}h" in d.columns})
    if not windows:
        st.warning("No matured outcome windows in live scored items.")
        return
    cc1, cc2, cc3 = st.columns(3)
    default_w = 48 if 48 in windows else windows[len(windows) // 2]
    window = cc1.selectbox("Maturation window (h)", windows, index=windows.index(default_w), key=f"{key_prefix}_w")
    min_prec = cc2.slider("Per-search precision target", 0.50, 0.95, 0.70, 0.05, key=f"{key_prefix}_mp")
    min_count = cc3.slider("Min selected count", 5, 100, 20, 5, key=f"{key_prefix}_mc")

    sold_col = f"sold_within_{window}h"
    eval_col = f"evaluated_at_{window}h"
    mat = d[d[eval_col].notna()].copy()
    ynum = pd.to_numeric(mat[sold_col], errors="coerce")
    if ynum.notna().any():
        mat["_y"] = (ynum.fillna(0) > 0).astype(float)
    else:
        mat["_y"] = mat[sold_col].astype(str).str.strip().str.lower().isin(["true", "1", "sold", "yes"]).astype(float)
    mat["_s"] = pd.to_numeric(mat[sc], errors="coerce")
    mat = mat[mat["_s"].notna()]
    if mat.empty:
        st.warning(f"No matured items at {window}h.")
        return

    rows = []
    for search, g in mat.groupby("SearchName"):
        y = g["_y"].to_numpy()
        s = g["_s"].to_numpy()
        N = int(len(g))
        gsel = s >= global_thr
        gN = int(gsel.sum())
        gp = float(y[gsel].mean()) if gN else float("nan")
        pthr, pp, pN = _choose_live_threshold(s, y, min_prec, min_count)
        rows.append({
            "search": str(search),
            "matured": N,
            "base_rate": float(y.mean()) if N else float("nan"),
            "global_thr": global_thr,
            "global_n": gN,
            "global_prec": gp,
            "ps_thr": pthr,
            "ps_n": pN,
            "ps_prec": pp,
            "Δprec": (pp - gp) if (pp == pp and gp == gp) else float("nan"),
            "Δn": pN - gN,
        })
    out = pd.DataFrame(rows).sort_values("search")
    st.subheader(f"main_image_scores — global vs per-search threshold @ {window}h")
    st.dataframe(
        out, width="stretch", hide_index=True,
        column_config={
            "base_rate": st.column_config.NumberColumn(format="%.3f"),
            "global_thr": st.column_config.NumberColumn("global thr", format="%.3f"),
            "global_prec": st.column_config.NumberColumn("global prec", format="%.3f"),
            "ps_thr": st.column_config.NumberColumn("ps thr", format="%.4f"),
            "ps_prec": st.column_config.NumberColumn("ps prec", format="%.3f"),
            "Δprec": st.column_config.NumberColumn("Δ prec", format="%.3f"),
        },
    )
    st.caption("Per-search thresholds tuned on the SAME live matured items (in-sample ⇒ optimistic). 'global' = deployed single threshold; 'ps' = lowest threshold per search hitting the precision target with ≥ min count. Δ = per-search − global.")


def render_threshold_compare(base, pattern, key_prefix):
    """Per search: best model under the run's GLOBAL threshold vs best under PER-SEARCH
    thresholds, from a model's offline per_search_threshold_comparison.csv (held-out test)."""
    run = _latest_run_with(base, pattern, "per_search_threshold_comparison.csv")
    if run is None:
        st.info(f"No offline threshold-comparison run under `{base}`.")
        return
    df = pd.read_csv(run / "per_search_threshold_comparison.csv", low_memory=False)
    st.caption(f"Offline run: `{run.name}` — held-out test split")
    if df.empty or "search_name" not in df.columns:
        st.warning("Comparison file empty.")
        return

    g = df.dropna(subset=["threshold_precision_global"]).copy()
    g = g[pd.to_numeric(g.get("threshold_count_global"), errors="coerce").fillna(0) > 0]
    g = g.sort_values(["threshold_precision_global", "threshold_count_global"], ascending=False).drop_duplicates("search_name", keep="first")
    g = g[["search_name", "approach", "threshold", "threshold_precision_global", "threshold_count_global", "threshold_recall_global"]].rename(
        columns={"approach": "global_model", "threshold": "global_thr", "threshold_precision_global": "global_prec", "threshold_count_global": "global_n", "threshold_recall_global": "global_recall"})

    p = df.dropna(subset=["threshold_precision_per_search"]).copy()
    p = p[pd.to_numeric(p.get("threshold_count_per_search"), errors="coerce").fillna(0) > 0]
    p = p.sort_values(["threshold_precision_per_search", "threshold_count_per_search"], ascending=False).drop_duplicates("search_name", keep="first")
    p = p[["search_name", "approach", "per_search_threshold", "threshold_precision_per_search", "threshold_count_per_search", "threshold_recall_per_search"]].rename(
        columns={"approach": "persearch_model", "per_search_threshold": "ps_thr", "threshold_precision_per_search": "ps_prec", "threshold_count_per_search": "ps_n", "threshold_recall_per_search": "ps_recall"})

    merged = g.merge(p, on="search_name", how="outer").sort_values("search_name")
    merged["prec_delta"] = merged["ps_prec"] - merged["global_prec"]
    merged["n_delta"] = merged["ps_n"] - merged["global_n"]

    st.subheader("Best model per search — global vs per-search threshold")
    st.dataframe(
        merged, width="stretch", hide_index=True,
        column_config={
            "global_thr": st.column_config.NumberColumn("global thr", format="%.4f"),
            "global_prec": st.column_config.NumberColumn("global prec", format="%.3f"),
            "global_recall": st.column_config.NumberColumn("global recall", format="%.3f"),
            "ps_thr": st.column_config.NumberColumn("ps thr", format="%.4f"),
            "ps_prec": st.column_config.NumberColumn("ps prec", format="%.3f"),
            "ps_recall": st.column_config.NumberColumn("ps recall", format="%.3f"),
            "prec_delta": st.column_config.NumberColumn("Δ prec (ps−global)", format="%.3f"),
            "n_delta": st.column_config.NumberColumn("Δ selected", format="%d"),
        },
    )
    st.caption("`global thr` = that approach's pooled-validation threshold (one per approach, NOT per search); `ps thr` = tuned per search. Best model can differ between the two strategies. Positive Δ prec ⇒ per-search wins for that search.")


def render_model_results_section(
    title: str,
    run_dir: Path,
    loader,
    metric_labels: list[tuple[str, str]],
) -> None:
    st.subheader(title)
    st.caption(f"Run: `{run_dir.relative_to(ROOT)}`")
    try:
        summary, overall, buckets, meta = loader(run_dir)
    except Exception as exc:
        st.error(f"Could not compute {title}: {exc}")
        return

    updated = meta.get("updated_at")
    if isinstance(updated, datetime):
        st.caption(f"Latest tracked update: {updated.strftime('%Y-%m-%d %H:%M:%S')} UTC ({age_str(file_age(run_dir / 'tracked_items.csv'))})")

    metric_cols = st.columns(len(metric_labels))
    for col, (label, key) in zip(metric_cols, metric_labels):
        col.metric(label, f"{meta.get(key, 0):,}")

    st.markdown("**Per-search Table**")
    st.dataframe(
        summary,
        width="stretch",
        hide_index=True,
        column_config={
            "S1 thr": st.column_config.NumberColumn("S1 thr", format="%.6f"),
            "S2 thr": st.column_config.NumberColumn("S2 thr", format="%.4f"),
            "Threshold": st.column_config.NumberColumn("Threshold", format="%.6f"),
        },
    )

    st.markdown("**Overall Matured Precision**")
    show_precision_table(overall)

    st.markdown("**Sold Hour Buckets**")
    if buckets.empty:
        st.info("No sold items detected yet.")
    else:
        st.dataframe(buckets, width="stretch", hide_index=True)


def value_counts_frame(df: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=[label, "Count"])
    return (
        df[column]
        .fillna("missing")
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis(label)
        .reset_index(name="Count")
    )


def log_issue_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        lines = path.read_text(errors="replace").splitlines()
    except Exception:
        return {}
    needles = ["Traceback", "Exception", "Error", "429", "failed", "FetchFailed", "killed"]
    return {needle: sum(needle.lower() in line.lower() for line in lines) for needle in needles}


def load_time_to_sell_collector(run_dir: Path) -> dict:
    tracked = read_csv_safe(run_dir / "tracked_items.csv")
    if tracked is None:
        tracked = pd.DataFrame()
    status = read_json_safe(run_dir / "latest_status.json") or {}
    daemon = read_json_safe(run_dir / "daemon.json") or {}
    manifest = read_json_safe(run_dir / "manifest.json") or {}
    plan = read_csv_safe(run_dir / "collector_plan.csv")
    balance = read_csv_safe(run_dir / "reports" / "bin_balance.csv")
    failures = read_csv_safe(run_dir / "full_items" / "full_scrape_failures.csv")
    events_path = run_dir / "events.jsonl"
    log_path = run_dir / "collector.log"

    if "sold_at" in tracked.columns:
        sold = pd.to_datetime(tracked["sold_at"], errors="coerce", utc=True).notna()
    else:
        sold = pd.Series(False, index=tracked.index)

    cohort_counts = (
        tracked.get("cohort", pd.Series(index=tracked.index, dtype=object))
        .fillna("missing")
        .astype(str)
        .value_counts()
        .to_dict()
    )
    high_rows = int(cohort_counts.get("high_score_candidate", 0))
    low_rows = int(cohort_counts.get("low_score_control", 0))

    full_counts = (
        tracked.get("FullScrapeStatus", pd.Series(index=tracked.index, dtype=object))
        .fillna("missing")
        .astype(str)
        .value_counts()
        .to_dict()
    )
    quality_counts = (
        tracked.get("QualityMethodStatus", pd.Series(index=tracked.index, dtype=object))
        .fillna("missing")
        .astype(str)
        .value_counts()
        .to_dict()
    )

    selected_at = pd.to_datetime(tracked.get("selected_at"), errors="coerce", utc=True) if "selected_at" in tracked.columns else pd.Series(pd.NaT, index=tracked.index)
    rechecked_at = pd.to_datetime(tracked.get("last_rechecked_at"), errors="coerce", utc=True) if "last_rechecked_at" in tracked.columns else pd.Series(pd.NaT, index=tracked.index)
    pid = daemon.get("pid") or status.get("pid")
    running = pid_number_alive(pid)
    if not running:
        detected_pid = find_process_by_cmdline("live_bin_collector.py", str(run_dir.resolve()))
        if detected_pid is not None:
            pid = detected_pid
            running = True

    return {
        "tracked": tracked,
        "status": status,
        "daemon": daemon,
        "manifest": manifest,
        "plan": plan if plan is not None else pd.DataFrame(),
        "balance": balance if balance is not None else pd.DataFrame(),
        "failures": failures if failures is not None else pd.DataFrame(),
        "running": running,
        "pid": pid,
        "log_path": log_path,
        "events_path": events_path,
        "log_counts": log_issue_counts(log_path),
        "meta": {
            "tracked_rows": int(len(tracked)),
            "high_rows": high_rows,
            "low_rows": low_rows,
            "sold_rows": int(sold.sum()),
            "full_success": int(full_counts.get("success", 0)),
            "full_pending": int(full_counts.get("pending", 0)),
            "full_failed": int(full_counts.get("failed", 0)),
            "quality_success": int(quality_counts.get("success", 0)),
            "quality_pending": int(quality_counts.get("pending", 0)),
            "latest_selected_at": selected_at.max() if len(selected_at) else pd.NaT,
            "latest_rechecked_at": rechecked_at.max() if len(rechecked_at) else pd.NaT,
            "events_rows": count_csv_rows(events_path) if events_path.exists() else "—",
        },
    }


def per_search_collector_summary(tracked: pd.DataFrame) -> pd.DataFrame:
    if tracked.empty or "SearchName" not in tracked.columns:
        return pd.DataFrame()
    work = tracked.copy()
    sold_source = work["sold_at"] if "sold_at" in work.columns else pd.Series(pd.NA, index=work.index)
    full_source = work["FullScrapeStatus"] if "FullScrapeStatus" in work.columns else pd.Series(pd.NA, index=work.index)
    work["_sold"] = pd.to_datetime(sold_source, errors="coerce", utc=True).notna()
    work["_full_success"] = full_source.fillna("").astype(str).eq("success")
    if "tracking_key" not in work.columns:
        work["tracking_key"] = tracking_key(work)
    pivot = work.pivot_table(index="SearchName", columns="cohort", values="tracking_key", aggfunc="count", fill_value=0)
    for col in ["high_score_candidate", "low_score_control"]:
        if col not in pivot.columns:
            pivot[col] = 0
    extra = work.groupby("SearchName").agg(
        Tracked=("tracking_key", "count"),
        Sold=("_sold", "sum"),
        FullScrapeSuccess=("_full_success", "sum"),
    )
    out = extra.join(pivot[["high_score_candidate", "low_score_control"]]).reset_index()
    out = out.rename(
        columns={
            "SearchName": "Search",
            "high_score_candidate": "High score",
            "low_score_control": "Low score",
            "FullScrapeSuccess": "Full scrape success",
        }
    )
    return out.sort_values("Search").reset_index(drop=True)


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Vinted Monitor", layout="wide")
st.title("Vinted — Live Monitor")

if st.sidebar.button("↻ Refresh", width="stretch"):
    st.rerun()

st.sidebar.caption(f"Updated: {_now_utc().strftime('%H:%M:%S')} UTC")
st.sidebar.divider()
log_lines = st.sidebar.slider("Log lines to show", 20, 200, 60, step=10)
show_sold_only = st.sidebar.toggle("Eventual sales: sold only", value=True)

# ── tabs ──────────────────────────────────────────────────────────────────────

PHOTO_DATASET = ROOT / "data" / "experiments" / "photo_arbitrage" / "features" / "sold_unsold_visuals_20260514_full" / "combined_scored.csv"
PHOTO_LABELS = ROOT / "data" / "experiments" / "photo_arbitrage" / "labels" / "photo_quality_labels.csv"

GIANT_MODEL_LIVE_RUNS = ROOT / "experiments" / "current" / "basic_5_giant_model" / "data" / "live_scoring"
VISUAL_GIANT_LIVE_RUNS = ROOT / "data" / "experiments" / "giant_basic_visual" / "live_scoring"
FULLSCRAPE_GIANT_LIVE_RUNS = ROOT / "data" / "experiments" / "full_scrape_giant_model" / "live_runs"
BASIC5_OFFLINE_RUNS = ROOT / "experiments" / "current" / "basic_5_giant_model" / "data" / "offline_runs"
VISUAL_OFFLINE_RUNS = ROOT / "data" / "experiments" / "giant_basic_visual" / "offline_runs"
FULLSCRAPE_OFFLINE_RUNS = ROOT / "data" / "experiments" / "full_scrape_giant_model" / "offline_runs"

tab_tts, tab_basic5, tab_visual, tab_fullscrape, tab_threshcompare, tab_eventual, tab_telegram, tab_photo = st.tabs([
    "Time-to-Sell Collector",
    "Giant Basic5",
    "Giant Basic5 Visual",
    "Full-Scrape Giant",
    "Threshold Compare",
    "Eventual Sales",
    "Telegram",
    "Photo Labeling",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Time-to-Sell Collector
# ════════════════════════════════════════════════════════════════════════════

with tab_tts:
    collector_runs = list_live_runs(TIME_TO_SELL_LIVE_RUNS, "bin_collector")

    if not collector_runs:
        st.warning("No time-to-sell collector runs found.")
    else:
        collector_run = st.selectbox(
            "Collector run",
            collector_runs,
            index=0,
            format_func=lambda p: p.name,
            key="time_to_sell_collector_run",
        )
        collector = load_time_to_sell_collector(collector_run)
        tracked = collector["tracked"]
        meta = collector["meta"]
        status = collector["status"]
        daemon = collector["daemon"]
        plan = collector["plan"]
        balance = collector["balance"]
        failures = collector["failures"]

        latest_activity_age = min(
            file_age(collector_run / "tracked_items.csv"),
            file_age(collector_run / "collector.log"),
            file_age(collector_run / "latest_status.json"),
        )
        status_row(
            "Time-to-Sell Collector",
            collector["running"],
            f"{age_str(latest_activity_age)} · PID {collector.get('pid', '—')}",
        )

        st.caption(f"Run: `{collector_run.relative_to(ROOT)}`")
        if daemon.get("started_at"):
            st.caption(f"Daemon started: `{daemon.get('started_at')}`")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Tracked", f"{meta['tracked_rows']:,}")
        m2.metric("High score", f"{meta['high_rows']:,}")
        m3.metric("Low score", f"{meta['low_rows']:,}")
        m4.metric("Sold detected", f"{meta['sold_rows']:,}")
        m5.metric("Full scrape OK", f"{meta['full_success']:,}")
        m6.metric("Pending / failed", f"{meta['full_pending']:,} / {meta['full_failed']:,}")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Last status", status.get("status", "—"))
        s2.metric("Due rows", status.get("due_rows", "—"))
        s3.metric("Checked rows", status.get("checked_rows", "—"))
        s4.metric("Max recheck age", status.get("max_recheck_age_hours", daemon.get("max_recheck_age_hours", "168")))

        if pd.notna(meta["latest_selected_at"]):
            st.caption(f"Latest selected item: `{meta['latest_selected_at']}`")
        if pd.notna(meta["latest_rechecked_at"]):
            st.caption(f"Latest recheck: `{meta['latest_rechecked_at']}`")

        with st.expander("Collector command and model plan", expanded=False):
            command = daemon.get("command")
            if command:
                st.code(" ".join(map(str, command)), language="bash")
            if not plan.empty:
                st.dataframe(plan, width="stretch", hide_index=True)

        st.subheader("Per-search collection")
        per_search = per_search_collector_summary(tracked)
        if per_search.empty:
            st.info("No tracked rows yet.")
        else:
            st.dataframe(per_search, width="stretch", hide_index=True)

        st.subheader("Bin balance")
        if balance.empty:
            st.info("No bin-balance report yet. It appears after the first report pass.")
        else:
            available_windows = sorted(pd.to_numeric(balance.get("hours"), errors="coerce").dropna().astype(int).unique())
            default_windows = [h for h in [2, 4, 6, 12, 24, 48, 72, 168] if h in available_windows] or available_windows[:5]
            selected_windows = st.multiselect(
                "Windows",
                options=available_windows,
                default=default_windows,
                key="tts_balance_windows",
            )
            balance_show = balance[pd.to_numeric(balance["hours"], errors="coerce").isin(selected_windows)].copy()
            readiness = (
                balance_show.groupby(["window", "hours"], dropna=False)[
                    ["high_sold", "low_unsold", "balanced_ready_pairs", "control_deficit"]
                ]
                .sum()
                .reset_index()
                .sort_values("hours")
            )
            st.markdown("**Dataset readiness by window**")
            st.dataframe(readiness, width="stretch", hide_index=True)
            st.markdown("**Per-search balance**")
            balance_cols = [
                "SearchName",
                "window",
                "high_tracked",
                "low_tracked",
                "high_evaluated",
                "high_sold",
                "low_evaluated",
                "low_unsold",
                "balanced_ready_pairs",
                "control_deficit",
            ]
            st.dataframe(
                balance_show[[c for c in balance_cols if c in balance_show.columns]],
                width="stretch",
                hide_index=True,
            )

        st.subheader("Scrape health")
        h1, h2, h3 = st.columns(3)
        h1.dataframe(value_counts_frame(tracked, "FullScrapeStatus", "Full scrape"), width="stretch", hide_index=True)
        h2.dataframe(value_counts_frame(tracked, "QualityMethodStatus", "Visual features"), width="stretch", hide_index=True)
        log_counts = pd.DataFrame(
            [{"Log signal": key, "Count": value} for key, value in collector["log_counts"].items()]
        )
        h3.dataframe(log_counts, width="stretch", hide_index=True)

        if not failures.empty:
            with st.expander(f"Full-scrape failures ({len(failures)})", expanded=False):
                fail_cols = [
                    "SearchName",
                    "cohort",
                    "Title",
                    "Brand",
                    "Price",
                    "FullScrapeError",
                    "FullScrapedAt",
                    "Link",
                ]
                fail_show = failures[[c for c in fail_cols if c in failures.columns]].tail(50)
                st.dataframe(
                    fail_show,
                    width="stretch",
                    hide_index=True,
                    column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open")} if "Link" in fail_show.columns else None,
                )

        st.subheader("Tracked items")
        if tracked.empty:
            st.info("No tracked rows yet.")
        else:
            tshow = tracked.copy()
            search_options = sorted(tshow["SearchName"].dropna().astype(str).unique()) if "SearchName" in tshow.columns else []
            cohort_options = sorted(tshow["cohort"].dropna().astype(str).unique()) if "cohort" in tshow.columns else []
            c1, c2, c3 = st.columns(3)
            chosen_searches = c1.multiselect("Search", search_options, default=[], key="tts_item_search")
            chosen_cohorts = c2.multiselect("Cohort", cohort_options, default=[], key="tts_item_cohort")
            status_filter = c3.selectbox("Full scrape", ["All", "success", "pending", "failed"], key="tts_item_full_status")
            if chosen_searches:
                tshow = tshow[tshow["SearchName"].astype(str).isin(chosen_searches)]
            if chosen_cohorts:
                tshow = tshow[tshow["cohort"].astype(str).isin(chosen_cohorts)]
            if status_filter != "All" and "FullScrapeStatus" in tshow.columns:
                tshow = tshow[tshow["FullScrapeStatus"].fillna("").astype(str) == status_filter]
            item_cols = [
                "SearchName",
                "cohort",
                "Title",
                "Brand",
                "Price",
                "Likes",
                "Stage1Model",
                "Stage1Score",
                "Stage1Rank",
                "FullScrapeStatus",
                "QualityMethodStatus",
                "last_recheck_status",
                "sold_at",
                "selected_at",
                "Link",
            ]
            item_show = tshow[[c for c in item_cols if c in tshow.columns]].tail(300)
            st.dataframe(
                item_show,
                width="stretch",
                hide_index=True,
                column_config={
                    "Link": st.column_config.LinkColumn("Link", display_text="Open"),
                    "Stage1Score": st.column_config.NumberColumn("Stage1Score", format="%.4f"),
                },
            )

        with st.expander("Collector log tail", expanded=False):
            st.code(tail_log(collector["log_path"], log_lines), language=None)


# ══════════════════════════════════════════════════════════════════════════════
# TAB — Giant Basic5
# ══════════════════════════════════════════════════════════════════════════════

with tab_basic5:
    st.subheader("Giant Basic5 — live matured precision/coverage")
    render_giant_live(GIANT_MODEL_LIVE_RUNS, "basic5")


# ══════════════════════════════════════════════════════════════════════════════
# TAB — Giant Basic5 Visual
# ══════════════════════════════════════════════════════════════════════════════

with tab_visual:
    st.subheader("Giant Basic5 Visual — live matured precision/coverage")
    render_giant_live(VISUAL_GIANT_LIVE_RUNS, "visual")


# ══════════════════════════════════════════════════════════════════════════════
# TAB — Full-Scrape Giant
# ══════════════════════════════════════════════════════════════════════════════

with tab_fullscrape:
    st.subheader("Full-Scrape Giant — live precision/coverage")
    render_fullscrape_giant_live(FULLSCRAPE_GIANT_LIVE_RUNS, "fullscrape")


# ══════════════════════════════════════════════════════════════════════════════
# TAB — Threshold Compare (global vs per-search)
# ══════════════════════════════════════════════════════════════════════════════

with tab_threshcompare:
    st.subheader("Global vs Per-Search Threshold — main_image_scores (live)")
    render_threshold_compare_live(VISUAL_GIANT_LIVE_RUNS, "tclive")


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Eventual Sales
# ════════════════════════════════════════════════════════════════════════════

with tab_eventual:
    eventual_age = file_age(EVENTUAL_LOG)
    status_row("Eventual Sale Checker", eventual_age < 60, age_str(eventual_age))

    # ── aggregate sold_eventually.csv from all per-search folders ──────────
    DISPLAY_COLS = ["SearchName", "Title", "Brand", "Price", "Size", "Likes",
                    "DealScore", "DealFinderScore", "DealFinderScoreBand", "Link"]

    frames = []
    for search_dir in sorted(SIMPLE_SCRAPE.iterdir()):
        if not search_dir.is_dir() or search_dir.name == "logs":
            continue
        f = search_dir / "eventual_sale_check" / "sold_eventually.csv"
        if f.exists():
            chunk = read_csv_safe(f)
            if chunk is not None and len(chunk) > 0:
                frames.append(chunk)

    if frames:
        df_ev = pd.concat(frames, ignore_index=True)
        # Cast any non-numeric columns that may have mixed types across search CSVs
        for col in df_ev.columns:
            if df_ev[col].dtype.kind == "O":
                df_ev[col] = df_ev[col].astype(str).replace("nan", "")

        # ── per-search summary ─────────────────────────────────────────────
        st.subheader("Sold items by search")
        summary = df_ev.groupby("SearchName").size().rename("Sold").reset_index()
        st.dataframe(summary, width="stretch", hide_index=True)

        # ── full table ─────────────────────────────────────────────────────
        st.subheader(f"All sold items ({len(df_ev):,} total)")

        search_filter_ev = st.multiselect(
            "Filter by search",
            options=sorted(df_ev["SearchName"].dropna().unique()),
            default=[],
            key="ev_search_filter",
        )
        df_show_ev = df_ev[df_ev["SearchName"].isin(search_filter_ev)] if search_filter_ev else df_ev

        avail_cols = [c for c in DISPLAY_COLS if c in df_show_ev.columns]
        col_cfg_ev = {}
        if "Link" in avail_cols:
            col_cfg_ev["Link"] = st.column_config.LinkColumn("Link", display_text="Open")
        if "DealScore" in avail_cols:
            col_cfg_ev["DealScore"] = st.column_config.NumberColumn("DealScore", format="%.2f")
        if "DealFinderScore" in avail_cols:
            col_cfg_ev["DealFinderScore"] = st.column_config.NumberColumn("DFScore", format="%.3f")

        st.dataframe(
            df_show_ev[avail_cols],
            width="stretch",
            hide_index=True,
            column_config=col_cfg_ev,
        )
    else:
        st.info("No sold_eventually.csv files found yet.")

    # ── eventual sale log ──────────────────────────────────────────────────
    with st.expander("Eventual sale log (tail)"):
        st.code(tail_log(EVENTUAL_LOG, log_lines), language=None)


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — Telegram
# ════════════════════════════════════════════════════════════════════════════

with tab_telegram:
    # ── load event log ─────────────────────────────────────────────────────
    def load_telegram_events() -> pd.DataFrame:
        if not TELEGRAM_EVENTS.exists():
            return pd.DataFrame()
        records = []
        for line in TELEGRAM_EVENTS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.sort_values("timestamp", ascending=False)
        return df

    df_tg = load_telegram_events()
    if "price" in df_tg.columns:
        # price arrives mixed str ('55.00') / float / NaN; coerce so grouping + Arrow render work
        df_tg["price"] = pd.to_numeric(df_tg["price"], errors="coerce")

    if df_tg.empty:
        st.info("No Telegram events recorded yet. Events will appear here once the bot sends items or receives button presses.")
    else:
        # ── summary metrics ────────────────────────────────────────────────
        counts = df_tg["event_type"].value_counts()
        cols = st.columns(5)
        cols[0].metric("Items sent", counts.get("item_sent", 0))
        cols[1].metric("Descriptions requested", counts.get("description_requested", 0))
        cols[2].metric("Descriptions generated", counts.get("description_generated", 0))
        cols[3].metric("Regenerations", counts.get("description_regenerated", 0))
        cols[4].metric("Errors", counts.get("bot_error", 0) + counts.get("description_error", 0) + counts.get("button_error", 0))

        st.divider()

        # ── filters ────────────────────────────────────────────────────────
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            event_types = ["All"] + sorted(df_tg["event_type"].dropna().unique().tolist())
            selected_type = st.selectbox("Event type", event_types)
        with col_f2:
            searches_tg = ["All"]
            if "search" in df_tg.columns:
                searches_tg += sorted(df_tg["search"].dropna().unique().tolist())
            selected_search = st.selectbox("Search", searches_tg, key="tg_search")

        df_filtered = df_tg.copy()
        if selected_type != "All":
            df_filtered = df_filtered[df_filtered["event_type"] == selected_type]
        if selected_search != "All" and "search" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["search"] == selected_search]

        # ── event table ────────────────────────────────────────────────────
        st.subheader(f"Events ({len(df_filtered):,})")
        SHOW_COLS = ["timestamp", "event_type", "search", "title", "brand", "price", "link", "cache_key"]
        avail = [c for c in SHOW_COLS if c in df_filtered.columns]

        col_cfg_tg: dict = {}
        if "link" in avail:
            col_cfg_tg["link"] = st.column_config.LinkColumn("Link", display_text="Open")
        if "timestamp" in avail:
            col_cfg_tg["timestamp"] = st.column_config.DatetimeColumn("Time", format="DD/MM HH:mm:ss")
        if "price" in avail:
            col_cfg_tg["price"] = st.column_config.NumberColumn("Price €", format="%.2f")

        st.dataframe(df_filtered[avail], width="stretch", hide_index=True, column_config=col_cfg_tg)

        # ── per-item history ───────────────────────────────────────────────
        if "title" in df_tg.columns:
            st.subheader("Per-item activity")
            item_summary = (
                df_tg.groupby(["title", "search", "price"], dropna=False)["event_type"]
                .value_counts()
                .unstack(fill_value=0)
                .reset_index()
            )
            st.dataframe(item_summary, width="stretch", hide_index=True)

        # ── cached items in pending_items ──────────────────────────────────
        with st.expander(f"Cached items in pending_items ({len(list(TELEGRAM_CACHE.glob('*.json'))) if TELEGRAM_CACHE.exists() else 0})"):
            if TELEGRAM_CACHE.exists():
                cache_rows = []
                for f in sorted(TELEGRAM_CACHE.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                    try:
                        data = json.loads(f.read_text(encoding="utf-8"))
                        item = data.get("item", {})
                        has_desc = data.get("description_payload") is not None
                        cache_rows.append({
                            "cache_key": f.stem,
                            "title": item.get("Title", ""),
                            "search": item.get("SearchName", ""),
                            "price": item.get("Price"),
                            "has_description": has_desc,
                        })
                    except Exception:
                        pass
                if cache_rows:
                    cache_df = pd.DataFrame(cache_rows)
                    cache_df["price"] = pd.to_numeric(cache_df["price"], errors="coerce")
                    st.dataframe(cache_df, width="stretch", hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 8 — Photo Labeling
# ════════════════════════════════════════════════════════════════════════════

with tab_photo:
    # ── load dataset ──────────────────────────────────────────────────────
    @st.cache_data(show_spinner="Loading photo dataset…")
    def load_photo_dataset() -> pd.DataFrame:
        df = pd.read_csv(PHOTO_DATASET, low_memory=False)
        df = df[df["LocalPrimaryImagePath"].notna() & (df["LocalPrimaryImagePath"] != "")]
        df = df[df["LocalPrimaryImagePath"].apply(lambda p: Path(str(p)).exists())]
        df = df.reset_index(drop=True)
        return df

    def load_labels() -> dict[str, str]:
        if not PHOTO_LABELS.exists():
            return {}
        try:
            ldf = pd.read_csv(PHOTO_LABELS, dtype=str)
            return dict(zip(ldf["image_path"], ldf["label"]))
        except Exception:
            return {}

    def save_label(image_path: str, label: str, item: "pd.Series") -> None:
        PHOTO_LABELS.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "item_id": str(item.get("Dataid", "")),
            "search": str(item.get("SearchName", "")),
            "image_path": image_path,
            "label": label,
            "labeled_at": datetime.now(tz=timezone.utc).isoformat(),
            "model_bad_score": item.get("CombinedBadPhotoScore", ""),
        }
        if PHOTO_LABELS.exists():
            pd.DataFrame([row]).to_csv(PHOTO_LABELS, mode="a", header=False, index=False)
        else:
            pd.DataFrame([row]).to_csv(PHOTO_LABELS, mode="w", header=True, index=False)

    # ── sidebar controls for this tab ─────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.caption("Photo Labeling")
        photo_search_filter = st.multiselect(
            "Filter by search",
            options=["griffati_donna_all", "griffati_uomo_all", "gucci", "nike", "prada", "ps4"],
            default=[],
            key="photo_search_filter",
        )
        photo_order = st.selectbox(
            "Order by",
            ["Uncertainty (score ≈ 0.5)", "Bad score (high first)", "Good score (high first)", "Random"],
            key="photo_order",
        )
        show_unlabeled_only = st.toggle("Skip already labeled", value=True, key="photo_skip_labeled")

    # ── load data ──────────────────────────────────────────────────────────
    try:
        df_photos = load_photo_dataset()
    except Exception as exc:
        st.error(f"Could not load photo dataset: {exc}")
        st.stop()

    labels_dict = load_labels()

    # ── apply filters ──────────────────────────────────────────────────────
    df_work = df_photos.copy()
    if photo_search_filter:
        df_work = df_work[df_work["SearchName"].isin(photo_search_filter)]
    if show_unlabeled_only:
        df_work = df_work[~df_work["LocalPrimaryImagePath"].isin(labels_dict)]

    # ── apply ordering ─────────────────────────────────────────────────────
    if "CombinedBadPhotoScore" in df_work.columns:
        score_col = "CombinedBadPhotoScore"
    elif "SimpleBadPhotoScore" in df_work.columns:
        score_col = "SimpleBadPhotoScore"
    else:
        score_col = None

    if photo_order == "Uncertainty (score ≈ 0.5)" and score_col:
        df_work = df_work.assign(_dist=(df_work[score_col] - 0.5).abs()).sort_values("_dist").drop(columns="_dist")
    elif photo_order == "Bad score (high first)" and score_col:
        df_work = df_work.sort_values(score_col, ascending=False)
    elif photo_order == "Good score (high first)" and score_col:
        df_work = df_work.sort_values(score_col, ascending=True)
    elif photo_order == "Random":
        df_work = df_work.sample(frac=1, random_state=42)

    df_work = df_work.reset_index(drop=True)

    # ── progress summary ───────────────────────────────────────────────────
    total_images = len(df_photos[~df_photos["LocalPrimaryImagePath"].isin(labels_dict) | df_photos["LocalPrimaryImagePath"].isin(labels_dict)])
    n_labeled = len(labels_dict)
    n_total = len(df_photos)
    star_counts = {str(s): sum(1 for v in labels_dict.values() if v == str(s)) for s in range(1, 6)}

    prog_cols = st.columns(7)
    prog_cols[0].metric("Total", f"{n_total:,}")
    prog_cols[1].metric("Labeled", f"{n_labeled:,}")
    for i, stars in enumerate(range(5, 0, -1)):
        prog_cols[2 + i].metric(f"{'★' * stars}", f"{star_counts[str(stars)]:,}")

    if n_total > 0:
        st.progress(n_labeled / n_total, text=f"{n_labeled}/{n_total} labeled ({100*n_labeled/n_total:.1f}%)")

    if len(df_work) == 0:
        st.success("All images in the current filter have been labeled!")
    else:
        st.divider()

        # ── session state for current index ────────────────────────────────
        if "photo_idx" not in st.session_state:
            st.session_state.photo_idx = 0
        if st.session_state.photo_idx >= len(df_work):
            st.session_state.photo_idx = 0

        idx = st.session_state.photo_idx
        row = df_work.iloc[idx]
        image_path = str(row["LocalPrimaryImagePath"])

        # ── navigation header ──────────────────────────────────────────────
        nav_col1, nav_col2, nav_col3 = st.columns([1, 4, 1])
        with nav_col1:
            if st.button("← Prev", disabled=(idx == 0), key="photo_prev"):
                st.session_state.photo_idx = max(0, idx - 1)
                st.rerun()
        with nav_col2:
            st.caption(f"Image {idx + 1} of {len(df_work)} (filtered queue)")
        with nav_col3:
            if st.button("Next →", disabled=(idx >= len(df_work) - 1), key="photo_next"):
                st.session_state.photo_idx = min(len(df_work) - 1, idx + 1)
                st.rerun()

        # ── image (full width, centered) ───────────────────────────────────
        _, img_center, _ = st.columns([1, 4, 1])
        with img_center:
            try:
                st.image(image_path, use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not load image: {exc}")

        # ── compact metadata row ───────────────────────────────────────────
        existing = labels_dict.get(image_path)
        meta_parts = [
            f"**{row.get('Title', '—')}**",
            f"`{row.get('Brand', '—')}`",
            f"`{row.get('Price', '—')}€`",
            f"`{row.get('SearchName', '—')}`",
            f"`{row.get('PhotoOutcomeLabel', '—')}`",
        ]
        score_parts = []
        for col_name, short in [
            ("CombinedBadPhotoScore", "bad"),
            ("AestheticGoodScore", "aes"),
            ("PyiqaBadPhotoScore", "pyiqa"),
        ]:
            if col_name in row.index and pd.notna(row[col_name]):
                score_parts.append(f"{short}={float(row[col_name]):.2f}")
        if score_parts:
            meta_parts.append("  ·  " + "  ".join(score_parts))
        if existing:
            meta_parts.append(f"  ·  already labeled: **{'★' * int(existing)}**")

        st.markdown("  ·  ".join(meta_parts))

        # ── labeling buttons ───────────────────────────────────────────────
        st.divider()
        STAR_LABELS = {
            "5": "★★★★★  Perfect",
            "4": "★★★★☆  Good",
            "3": "★★★☆☆  Average",
            "2": "★★☆☆☆  Bad",
            "1": "★☆☆☆☆  Terrible",
        }
        btn_cols = st.columns(6)
        for i, (star_val, star_label) in enumerate(STAR_LABELS.items()):
            with btn_cols[i]:
                is_primary = star_val in ("5", "4")
                if st.button(star_label, key=f"photo_star_{star_val}", use_container_width=True,
                             type="primary" if is_primary else "secondary"):
                    save_label(image_path, star_val, row)
                    st.session_state.photo_idx = min(len(df_work) - 1, idx + 1)
                    st.cache_data.clear()
                    st.rerun()
        with btn_cols[5]:
            if st.button("⏭ Skip", key="photo_skip", use_container_width=True):
                st.session_state.photo_idx = min(len(df_work) - 1, idx + 1)
                st.rerun()

        # ── jump to index ──────────────────────────────────────────────────
        with st.expander("Jump to image"):
            jump = st.number_input("Image index (1-based)", min_value=1, max_value=len(df_work), value=idx + 1, step=1, key="photo_jump")
            if st.button("Go", key="photo_jump_go"):
                st.session_state.photo_idx = int(jump) - 1
                st.rerun()
