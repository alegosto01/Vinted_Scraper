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
    checkpoints = [1, 6, 12, 24, 30, 36, 39, 42, 45, 48, 60, 72]
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
        [1, 2, 3, 6, 12, 18, 24, 27, 30, 33, 36, 39, 42, 45, 48, 60, 72],
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


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Vinted Monitor", layout="wide")
st.title("Vinted — Live Monitor")

if st.sidebar.button("↻ Refresh", width="stretch"):
    st.rerun()

st.sidebar.caption(f"Updated: {_now_utc().strftime('%H:%M:%S')} UTC")
st.sidebar.divider()
auto_refresh_results = st.sidebar.toggle("Auto-refresh page", value=True)
refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 15, 300, 60, step=15)
if auto_refresh_results:
    install_browser_autorefresh(refresh_seconds)
st.sidebar.divider()
log_lines = st.sidebar.slider("Log lines to show", 20, 200, 60, step=10)
show_sold_only = st.sidebar.toggle("Eventual sales: sold only", value=True)

# ── tabs ──────────────────────────────────────────────────────────────────────

PHOTO_DATASET = ROOT / "data" / "experiments" / "photo_arbitrage" / "features" / "sold_unsold_visuals_20260514_full" / "combined_scored.csv"
PHOTO_LABELS = ROOT / "data" / "experiments" / "photo_arbitrage" / "labels" / "photo_quality_labels.csv"

tab_scrape, tab_bench, tab_results, tab_eventual, tab_telegram, tab_photo = st.tabs([
    "Live Scraping",
    "Benchmark (Cascade)",
    "Model Results",
    "Eventual Sales",
    "Telegram",
    "Photo Labeling",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Scraping
# ════════════════════════════════════════════════════════════════════════════

with tab_scrape:
    scrape_age = file_age(SCRAPE_LOG)
    scrape_running = scrape_age < 10  # log written in last 10 min

    status_row("Live Scraper", scrape_running, age_str(scrape_age))

    # ── per-search summary ─────────────────────────────────────────────────
    st.subheader("Per-search snapshot")
    searches = sorted(d.name for d in SIMPLE_SCRAPE.iterdir() if d.is_dir() and d.name != "logs")
    rows = []
    for s in searches:
        base = SIMPLE_SCRAPE / s
        big = base / "big_raw.csv"
        sold = base / "sold_df.csv"
        rows.append({
            "Search": s,
            "big_raw rows": count_csv_rows(big),
            "sold rows": count_csv_rows(sold),
            "big_raw updated": age_str(file_age(big)),
            "sold updated": age_str(file_age(sold)),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ── log tail ───────────────────────────────────────────────────────────
    st.subheader("Scraper log (tail)")
    st.code(tail_log(SCRAPE_LOG, log_lines), language=None)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Benchmark (Cascade)
# ════════════════════════════════════════════════════════════════════════════

with tab_bench:
    bench_log_age = min(file_age(CASCADE_LOG), file_age(CASCADE_LOOP_LOG))
    # PID file may be stale after a manual restart; treat recent log activity as running too
    bench_running = pid_alive(CASCADE_PID) or bench_log_age < 90

    status_row("Cascade Benchmark", bench_running, age_str(bench_log_age))

    # ── manifest info ──────────────────────────────────────────────────────
    manifest = read_json_safe(CASCADE_MANIFEST)
    if manifest:
        col1, col2, col3 = st.columns(3)
        col1.metric("Searches", len(manifest.get("searches", [])))
        col2.metric("Stage 1", manifest.get("stage1_source", "—"))
        col3.metric("Recheck schedule", manifest.get("recheck_schedule", "—"))

    # ── latest iteration status ────────────────────────────────────────────
    status = read_json_safe(CASCADE_STATUS)
    if status:
        st.subheader("Last iteration")
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("Status", status.get("status", "—"))
        s_col2.metric("Due rows", status.get("due_rows", "—"))
        s_col3.metric("Checked rows", status.get("checked_rows", "—"))
        s_col4.metric("Dry run", str(status.get("dry_run", "—")))

    # ── tracked items ──────────────────────────────────────────────────────
    df = read_csv_safe(CASCADE_TRACKED)
    if df is not None:
        st.subheader(f"Tracked items ({len(df)} total)")

        DISPLAY_COLS = ["SearchName", "Title", "Brand", "Price", "Likes",
                        "Stage1Model", "Stage1Score", "Stage1Threshold",
                        "Stage2Model", "Stage2Score", "Stage2Threshold",
                        "Stage2Passed", "Stage2Status",
                        "first_stage1_pass_at", "last_rechecked_at",
                        "last_recheck_status", "sold_at", "Link"]
        available = [c for c in DISPLAY_COLS if c in df.columns]
        df_show = df[available].copy()

        # Colour-code by outcome
        # Normalise Stage2Passed to bool (mixed True/'True'/1.0/'False'/0.0 in CSV)
        if "Stage2Passed" in df_show.columns:
            df_show["Stage2Passed"] = df_show["Stage2Passed"].map(
                lambda x: str(x).strip().lower() in ("true", "1", "1.0")
            )

        search_filter = st.multiselect(
            "Filter by search",
            options=sorted(df["SearchName"].dropna().unique()),
            default=[],
        )
        if search_filter:
            df_show = df_show[df_show["SearchName"].isin(search_filter)]

        stage2_filter = st.selectbox(
            "Stage 2 status",
            ["All", "Passed", "Not passed", "Sold"],
            index=0,
        )
        if stage2_filter == "Passed" and "Stage2Passed" in df_show.columns:
            df_show = df_show[df_show["Stage2Passed"] == True]  # noqa: E712
        elif stage2_filter == "Not passed" and "Stage2Passed" in df_show.columns:
            df_show = df_show[df_show["Stage2Passed"] == False]  # noqa: E712
        elif stage2_filter == "Sold" and "sold_at" in df_show.columns:
            df_show = df_show[df_show["sold_at"].notna()]

        col_cfg = {}
        if "Link" in df_show.columns:
            col_cfg["Link"] = st.column_config.LinkColumn("Link", display_text="Open")
        if "Stage1Score" in df_show.columns:
            col_cfg["Stage1Score"] = st.column_config.NumberColumn("S1 Score", format="%.3f")
        if "Stage2Score" in df_show.columns:
            col_cfg["Stage2Score"] = st.column_config.NumberColumn("S2 Score", format="%.3f")

        st.dataframe(df_show, width="stretch", hide_index=True, column_config=col_cfg)

        # ── quick summary ──────────────────────────────────────────────────
        st.subheader("Per-search breakdown")
        grp_cols = {"Count": "size"}
        if "sold_at" in df.columns:
            sold_counts = df.groupby("SearchName")["sold_at"].apply(lambda x: x.notna().sum()).rename("Sold")
        else:
            sold_counts = None

        grp = df.groupby("SearchName").size().rename("Tracked").reset_index()
        if sold_counts is not None:
            grp = grp.merge(sold_counts.reset_index(), on="SearchName", how="left")
        if "Stage2Passed" in df.columns:
            s2_bool = df["Stage2Passed"].map(
                lambda x: str(x).strip().lower() in ("true", "1", "1.0")
            )
            passed = df.groupby("SearchName").apply(lambda g: s2_bool.loc[g.index].sum()).rename("Stage2 Passed").astype(int)
            grp = grp.merge(passed.reset_index(), on="SearchName", how="left")
        st.dataframe(grp, width="stretch", hide_index=True)

    # ── log tail ───────────────────────────────────────────────────────────
    with st.expander("Cascade loop log (tail)"):
        st.code(tail_log(CASCADE_LOOP_LOG, log_lines), language=None)
    with st.expander("Nohup log (tail)"):
        st.code(tail_log(CASCADE_LOG, log_lines), language=None)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Results
# ════════════════════════════════════════════════════════════════════════════

with tab_results:
    cascade_runs = list_live_runs(CASCADE_RESULTS_RUNS, "cascade")

    if not cascade_runs:
        st.warning("No cascade live runs found.")
    else:
        cascade_run = st.selectbox(
            "Cascade run",
            cascade_runs,
            index=0,
            format_func=lambda p: p.name,
            key="model_results_cascade_run",
        )

        st.divider()
        render_model_results_section(
            "Cascade",
            cascade_run,
            load_cascade_results,
            [
                ("S1 pass", "unique_stage1_pass"),
                ("S2 pass", "unique_stage2_pass"),
                ("S2 sold", "sold_stage2_pass"),
                ("Tracked rows", "tracked_rows"),
            ],
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Eventual Sales
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
# TAB 5 — Telegram
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
                    st.dataframe(pd.DataFrame(cache_rows), width="stretch", hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Photo Labeling
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
