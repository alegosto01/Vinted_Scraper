from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from config.logging_config import eventual_sales_log_context
from config.project_config import settings


LOGGER = logging.getLogger(__name__)
EVENTUAL_SALES_OUT_DIR_NAME = "eventual_sale_check"
EVENTUAL_SALES_STATE_DATE_KEY = "last_eventual_sale_refresh_date"
DEALS_RANKED_REFRESH_DATE_KEY = "last_deals_ranked_refresh_date"
EVENTUAL_SALES_MAX_WORKERS = 3
EVENTUAL_SALES_DELAY_SECONDS = 5.0
EVENTUAL_SALES_FETCH_SLEEP_SECONDS = 0.0
EVENTUAL_SALES_FETCH_MAX_ATTEMPTS = 1
EVENTUAL_SALES_BACKGROUND_INTERVAL_SECONDS = 60.0
EVENTUAL_SALES_BACKGROUND_WORKER_COUNT = 3
EVENTUAL_SALES_BACKGROUND_WORKER_STAGGER_SECONDS = 20.0
EVENTUAL_SALES_BACKGROUND_CURSOR_KEY = "eventual_sale_background_cursor"
EVENTUAL_SALES_BACKGROUND_LAST_CHECKED_AT_KEY = "eventual_sale_background_last_checked_at"
EVENTUAL_SALES_BACKGROUND_LAST_DATAID_KEY = "eventual_sale_background_last_dataid"
EVENTUAL_SALES_BACKGROUND_LAST_STATUS_KEY = "eventual_sale_background_last_status"
EVENTUAL_SALES_BACKGROUND_LAST_SOURCE_KEY = "eventual_sale_background_last_source"
PIPELINE_OUT_DIR_NAME = "pipeline_out"
DEALS_RANKED_FILENAME = "deals_ranked.csv"
DEALS_RANKED_DB_FILENAME = "index.sqlite"
PRIORITY_CHECK_QUEUE_FILENAME = "priority_check_queue.csv"
PRIORITY_QUEUE_SOURCE_COLUMN = "PriorityQueueSource"
PRIORITY_QUEUE_ENQUEUED_AT_COLUMN = "PriorityQueueEnqueuedAt"
PRIORITY_QUEUE_LAST_STATUS_COLUMN = "PriorityQueueLastStatus"
PRIORITY_QUEUE_LAST_CHECKED_AT_COLUMN = "PriorityQueueLastCheckedAt"
PRIORITY_QUEUE_ATTEMPTS_COLUMN = "PriorityQueueAttempts"
PRIORITY_QUEUE_MAX_AGE_DAYS = 30


def today_local_iso(now: datetime | None = None) -> str:
    now = now or datetime.now().astimezone()
    if now.tzinfo is None:
        now = now.astimezone()
    return now.date().isoformat()


def get_batch_input_path(output_folder: Path) -> Path | None:
    for filename in ("big_raw.csv", "old_df.csv"):
        candidate = output_folder / filename
        if candidate.exists():
            return candidate
    return None


def get_eventual_sales_input_path(output_folder: Path) -> Path | None:
    candidate = output_folder / PIPELINE_OUT_DIR_NAME / DEALS_RANKED_FILENAME
    return candidate if candidate.exists() else None


def _background_out_dir(output_folder: Path) -> Path:
    return output_folder / EVENTUAL_SALES_OUT_DIR_NAME


def _priority_queue_path(output_folder: Path) -> Path:
    return _background_out_dir(output_folder) / PRIORITY_CHECK_QUEUE_FILENAME


def _background_csv_paths(output_folder: Path) -> dict[str, Path]:
    out_dir = _background_out_dir(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "labeled": out_dir / "big_raw_eventual_sale_labeled.csv",
        "sold": out_dir / "sold_eventually.csv",
        "active": out_dir / "not_sold_yet.csv",
    }


def _read_csv_or_empty(path: Path, columns=None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path)
    except Exception as exc:
        LOGGER.warning("Failed reading %s: %s: %s", path, type(exc).__name__, exc)
        return pd.DataFrame(columns=columns)


def _normalize_priority_queue_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in (
        PRIORITY_QUEUE_SOURCE_COLUMN,
        PRIORITY_QUEUE_ENQUEUED_AT_COLUMN,
        PRIORITY_QUEUE_LAST_STATUS_COLUMN,
        PRIORITY_QUEUE_LAST_CHECKED_AT_COLUMN,
    ):
        if col not in out.columns:
            out[col] = pd.Series([pd.NA] * len(out), dtype="object")
        else:
            out[col] = out[col].astype("object")
    if PRIORITY_QUEUE_ATTEMPTS_COLUMN not in out.columns:
        out[PRIORITY_QUEUE_ATTEMPTS_COLUMN] = pd.Series([0] * len(out), dtype="Int64")
    else:
        out[PRIORITY_QUEUE_ATTEMPTS_COLUMN] = (
            pd.to_numeric(out[PRIORITY_QUEUE_ATTEMPTS_COLUMN], errors="coerce")
            .fillna(0)
            .astype("Int64")
        )
    return out


def _drop_stale_priority_rows(
    queue_df: pd.DataFrame,
    *,
    max_age_days: int = PRIORITY_QUEUE_MAX_AGE_DAYS,
    now_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if queue_df.empty or max_age_days <= 0:
        return queue_df.copy()

    out = queue_df.copy()
    now_ts = now_ts or pd.Timestamp.now(tz="UTC")
    cutoff = now_ts - pd.Timedelta(days=int(max_age_days))

    search_ts = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    if "SearchDate" in out.columns:
        parsed_search = pd.to_datetime(out["SearchDate"], errors="coerce", dayfirst=True, utc=True)
        search_ts = parsed_search.where(parsed_search.notna(), search_ts)
    if PRIORITY_QUEUE_ENQUEUED_AT_COLUMN in out.columns:
        parsed_enqueued = pd.to_datetime(out[PRIORITY_QUEUE_ENQUEUED_AT_COLUMN], errors="coerce", utc=True)
        search_ts = search_ts.where(search_ts.notna(), parsed_enqueued)

    stale_mask = search_ts.notna() & (search_ts < cutoff)
    if stale_mask.any():
        dropped = int(stale_mask.sum())
        LOGGER.info(
            "Dropping %s stale priority-queue rows older than %s days",
            dropped,
            max_age_days,
        )
        out = out.loc[~stale_mask].copy()
    return out


def _build_priority_queue_rows(rows_df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    queued = _normalize_priority_queue_df(rows_df)
    now_iso = datetime.now().astimezone().isoformat()
    queued[PRIORITY_QUEUE_SOURCE_COLUMN] = source
    if PRIORITY_QUEUE_ENQUEUED_AT_COLUMN in queued.columns:
        missing_enqueued = queued[PRIORITY_QUEUE_ENQUEUED_AT_COLUMN].isna() | (
            queued[PRIORITY_QUEUE_ENQUEUED_AT_COLUMN].astype(str).str.strip() == ""
        )
        queued.loc[missing_enqueued, PRIORITY_QUEUE_ENQUEUED_AT_COLUMN] = now_iso
    queued[PRIORITY_QUEUE_LAST_STATUS_COLUMN] = pd.NA
    queued[PRIORITY_QUEUE_LAST_CHECKED_AT_COLUMN] = pd.NA
    queued[PRIORITY_QUEUE_ATTEMPTS_COLUMN] = pd.Series([0] * len(queued), dtype="Int64")
    return queued


def enqueue_priority_background_checks(output_folder: str | Path, rows_df: pd.DataFrame, *, source: str = "compare_and_save") -> int:
    from scraping_options import dedupe_market_rows, write_csv_atomic

    if rows_df is None or rows_df.empty:
        return 0

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    queue_path = _priority_queue_path(output_path)
    queue_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _normalize_priority_queue_df(_read_csv_or_empty(queue_path))
    queued = _build_priority_queue_rows(rows_df, source=source)

    combined = pd.concat([existing, queued], ignore_index=True) if not existing.empty else queued.copy()
    combined = _normalize_priority_queue_df(combined)
    combined = dedupe_market_rows(combined, keep="last")
    write_csv_atomic(combined, str(queue_path))
    return int(len(queued))


def _extract_dataid_set(df: pd.DataFrame, *, sold_only: bool = False) -> set[int]:
    if df.empty or "Dataid" not in df.columns:
        return set()
    out = df.copy()
    if sold_only and "MarketStatus" in out.columns:
        out = out[out["MarketStatus"].astype(str) == "Sold"].copy()
    ids = pd.to_numeric(out["Dataid"], errors="coerce").dropna()
    return {int(v) for v in ids.tolist()}


def _known_sold_ids(output_folder: Path) -> set[int]:
    paths = _background_csv_paths(output_folder)
    sold_ids = _extract_dataid_set(_read_csv_or_empty(paths["sold"]))
    sold_ids |= _extract_dataid_set(_read_csv_or_empty(paths["labeled"]), sold_only=True)
    sold_ids |= _extract_dataid_set(_read_csv_or_empty(output_folder / "sold_df.csv"))
    return sold_ids


def _select_next_priority_candidate(search) -> tuple[Path | None, pd.DataFrame | None]:
    from scraping_options import dedupe_market_rows, write_csv_atomic

    output_folder = Path(str(settings.paths.simple_scrape_dir)) / search.folder
    output_folder.mkdir(parents=True, exist_ok=True)
    queue_path = _priority_queue_path(output_folder)
    queue_df = _normalize_priority_queue_df(_read_csv_or_empty(queue_path))
    if queue_df.empty:
        return queue_path, None

    queue_df = dedupe_market_rows(queue_df, keep="last")
    queue_df = _drop_stale_priority_rows(queue_df)
    sold_ids = _known_sold_ids(output_folder)
    if sold_ids and "Dataid" in queue_df.columns:
        queue_df = queue_df[~pd.to_numeric(queue_df["Dataid"], errors="coerce").isin(list(sold_ids))].copy()

    if queue_df.empty:
        write_csv_atomic(queue_df, str(queue_path))
        return queue_path, None

    queue_df["_AttemptsNum"] = pd.to_numeric(queue_df[PRIORITY_QUEUE_ATTEMPTS_COLUMN], errors="coerce").fillna(0)
    queue_df["_QueuedAtTs"] = pd.to_datetime(queue_df[PRIORITY_QUEUE_ENQUEUED_AT_COLUMN], errors="coerce")
    queue_df = queue_df.sort_values(
        by=["_AttemptsNum", "_QueuedAtTs"],
        ascending=[True, True],
        na_position="last",
        kind="stable",
    ).drop(columns=["_AttemptsNum", "_QueuedAtTs"], errors="ignore")
    write_csv_atomic(queue_df, str(queue_path))
    return queue_path, queue_df.iloc[[0]].copy()


def _finalize_priority_candidate(output_folder: Path, checked_row: pd.DataFrame) -> None:
    from scraping_options import dedupe_market_rows, write_csv_atomic

    queue_path = _priority_queue_path(output_folder)
    queue_df = _normalize_priority_queue_df(_read_csv_or_empty(queue_path))
    if queue_df.empty:
        return

    row = checked_row.iloc[0]
    dataid = row.get("Dataid")
    dataid_num = pd.to_numeric(pd.Series([dataid]), errors="coerce").iloc[0]
    link = row.get("Link")

    mask = pd.Series([False] * len(queue_df), index=queue_df.index)
    if pd.notna(dataid_num) and "Dataid" in queue_df.columns:
        mask = mask | (pd.to_numeric(queue_df["Dataid"], errors="coerce") == int(dataid_num))
    if link and "Link" in queue_df.columns:
        mask = mask | (queue_df["Link"].astype(str) == str(link))

    matching = queue_df.loc[mask].copy()
    remaining = queue_df.loc[~mask].copy()

    last_status = str(row.get("LastCheckStatus", "") or "")
    if last_status in {"FetchFailed", "MissingLink"}:
        retry_row = checked_row.copy()
        if not matching.empty:
            for col in (
                PRIORITY_QUEUE_SOURCE_COLUMN,
                PRIORITY_QUEUE_ENQUEUED_AT_COLUMN,
                PRIORITY_QUEUE_ATTEMPTS_COLUMN,
            ):
                if col in matching.columns and col in retry_row.columns:
                    retry_row.loc[:, col] = matching.iloc[0].get(col)
        retry_row = _normalize_priority_queue_df(retry_row)
        retry_row.loc[:, PRIORITY_QUEUE_LAST_STATUS_COLUMN] = last_status
        retry_row.loc[:, PRIORITY_QUEUE_LAST_CHECKED_AT_COLUMN] = datetime.now().astimezone().isoformat()
        retry_row.loc[:, PRIORITY_QUEUE_ATTEMPTS_COLUMN] = (
            pd.to_numeric(retry_row[PRIORITY_QUEUE_ATTEMPTS_COLUMN], errors="coerce")
            .fillna(0)
            .astype("Int64")
            + 1
        )
        remaining = pd.concat([remaining, retry_row], ignore_index=True)

    remaining = _normalize_priority_queue_df(remaining)
    remaining = dedupe_market_rows(remaining, keep="last")
    write_csv_atomic(remaining, str(queue_path))


def _sync_priority_result_to_tracking_files(output_folder: Path, checked_row: pd.DataFrame) -> None:
    from scraping_options import dedupe_market_rows, ensure_search_tracking_files, write_csv_atomic

    ensure_search_tracking_files(str(output_folder))
    row = checked_row.iloc[0].copy()
    last_status = str(row.get("LastCheckStatus", "") or "")
    if last_status in {"FetchFailed", "MissingLink"}:
        return

    old_df_path = output_folder / "old_df.csv"
    sold_df_path = output_folder / "sold_df.csv"
    non_really_path = output_folder / "non_really_sold_items_ids.csv"

    old_df = _read_csv_or_empty(old_df_path)
    sold_df = _read_csv_or_empty(sold_df_path)
    non_really_df = _read_csv_or_empty(non_really_path, columns=["Dataid"])
    dataid_num = pd.to_numeric(pd.Series([row.get("Dataid")]), errors="coerce").iloc[0]
    link = row.get("Link")

    def _row_mask(df: pd.DataFrame) -> pd.Series:
        mask = pd.Series([False] * len(df), index=df.index)
        if df.empty:
            return mask
        if pd.notna(dataid_num) and "Dataid" in df.columns:
            mask = mask | (pd.to_numeric(df["Dataid"], errors="coerce") == int(dataid_num))
        if link and "Link" in df.columns:
            mask = mask | (df["Link"].astype(str) == str(link))
        return mask

    sold_confirmed = False
    if last_status == "Sold" or str(row.get("MarketStatus", "")) == "Sold":
        sold_confirmed = True
        row["MarketStatus"] = "Sold"
        if not old_df.empty:
            old_df = old_df.loc[~_row_mask(old_df)].copy()
        sold_df = pd.concat([sold_df, pd.DataFrame([row.to_dict()])], ignore_index=True) if not sold_df.empty else pd.DataFrame([row.to_dict()])
        sold_df = dedupe_market_rows(sold_df, keep="last")
        if not non_really_df.empty and pd.notna(dataid_num):
            non_really_df = non_really_df.loc[
                pd.to_numeric(non_really_df["Dataid"], errors="coerce") != int(dataid_num)
            ].copy()
    elif last_status in {"OnSale", "On Sale"}:
        row["MarketStatus"] = "On Sale"
        row_df = pd.DataFrame([row.to_dict()])
        if old_df.empty:
            old_df = row_df.copy()
        else:
            old_df = old_df.loc[~_row_mask(old_df)].copy()
            old_df = pd.concat([old_df, row_df], ignore_index=True)
        old_df = dedupe_market_rows(old_df, keep="last")
        if pd.notna(dataid_num):
            non_really_add = pd.DataFrame([{"Dataid": int(dataid_num)}])
            non_really_df = pd.concat([non_really_df, non_really_add], ignore_index=True) if not non_really_df.empty else non_really_add
            non_really_df = non_really_df.drop_duplicates(subset=["Dataid"], keep="first")
    else:
        return

    write_csv_atomic(old_df, str(old_df_path))
    write_csv_atomic(sold_df, str(sold_df_path))
    write_csv_atomic(non_really_df, str(non_really_path))

    if sold_confirmed:
        try:
            from full_scraper import Full_Scraper

            scraper = Full_Scraper()
            scraper.collect_and_store_full_items(
                pd.DataFrame([row.to_dict()]),
                search_name=output_folder.name,
                reason="sold_confirmed_live",
                max_workers=1,
                image_mode="html",
                skip_existing=False,
            )
        except Exception as exc:
            LOGGER.warning(
                "Full scrape for confirmed sold item failed for search=%s item=%s: %s: %s",
                output_folder.name,
                row.get("Dataid", ""),
                type(exc).__name__,
                exc,
            )


def _build_daily_batch_args(search, output_folder: Path, input_path: Path) -> SimpleNamespace:
    pipeline_out_dir = output_folder / PIPELINE_OUT_DIR_NAME
    return SimpleNamespace(
        folder=search.folder,
        input=str(input_path),
        input_name=input_path.name,
        out_dir=str(pipeline_out_dir),
        pipeline_out_dir=PIPELINE_OUT_DIR_NAME,
        db=str(pipeline_out_dir / DEALS_RANKED_DB_FILENAME),
        model="paraphrase-multilingual-MiniLM-L12-v2",
        product_threshold=0.24,
        use_image_embeddings=False,
        image_embedding_model="facebook/dinov2-base",
        image_embedding_timeout=8.0,
        image_embedding_batch_size=8,
        product_image_weight=0.15,
        autotune_variants=False,
        variant_threshold=0.33,
        core_frac=0.70,
        variant_price_weight=0.35,
        variant_image_weight=0.20,
        min_product_size_for_variants=4,
        min_variant_size_for_deals=4,
        min_variant_size_for_confident_deals=8,
        min_variant_silhouette=0.35,
        max_variant_mad_ratio=0.25,
        hard_max_variant_mad_ratio=0.45,
        min_deal_confidence=0.55,
        min_centroid_similarity=0.55,
        min_informative_tokens=2,
        exclude_negflag_deals=False,
        resale_fee_rate=0.10,
        resale_fixed_cost=0.0,
        resale_safety_discount=0.05,
        min_expected_profit=0.0,
        min_expected_profit_margin=0.0,
        price_buffer_size=200,
        make_plots=False,
    )


def ensure_daily_deals_ranked(search, today_iso: str | None = None) -> Path | None:
    from scraping_options import read_schedule_state, write_schedule_state
    from workflow_runner import run_batch_command

    today_iso = today_iso or today_local_iso()
    output_folder = Path(str(settings.paths.simple_scrape_dir)) / search.folder
    output_folder.mkdir(parents=True, exist_ok=True)
    deals_path = output_folder / PIPELINE_OUT_DIR_NAME / DEALS_RANKED_FILENAME
    state = read_schedule_state(str(output_folder))

    if state.get(DEALS_RANKED_REFRESH_DATE_KEY) == today_iso and deals_path.exists():
        return deals_path

    input_path = get_batch_input_path(output_folder)
    if input_path is None:
        with eventual_sales_log_context():
            LOGGER.info(
                "Skipping daily deals-ranked refresh for %s; no big_raw.csv or old_df.csv found",
                search.folder,
            )
        return deals_path if deals_path.exists() else None

    with eventual_sales_log_context():
        LOGGER.info(
            "Refreshing daily deals-ranked for %s from %s",
            search.folder,
            input_path.name,
        )
    pipeline_out_dir = output_folder / PIPELINE_OUT_DIR_NAME
    pipeline_out_dir.mkdir(parents=True, exist_ok=True)
    args = _build_daily_batch_args(search, output_folder, input_path)
    run_batch_command(args)

    if not deals_path.exists():
        with eventual_sales_log_context():
            LOGGER.warning(
                "Daily deals-ranked refresh for %s finished without creating %s",
                search.folder,
                deals_path,
            )
        return None

    state.update(
        {
            DEALS_RANKED_REFRESH_DATE_KEY: today_iso,
            "last_deals_ranked_refresh_at": datetime.now().astimezone().isoformat(),
            "last_deals_ranked_refresh_input": input_path.name,
            "last_deals_ranked_refresh_output": deals_path.name,
        }
    )
    write_schedule_state(str(output_folder), state)
    return deals_path


def _merge_background_results(output_folder: Path, checked_row: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scraping_options import dedupe_market_rows, exclude_matching_market_rows

    paths = _background_csv_paths(output_folder)
    existing = _read_csv_or_empty(paths["labeled"], columns=list(checked_row.columns))
    merged = pd.concat([existing, checked_row], ignore_index=True) if not existing.empty else checked_row.copy()
    merged = dedupe_market_rows(merged, keep="last")
    merged = exclude_matching_market_rows(merged, _read_csv_or_empty(output_folder / "sold_df.csv"))
    sold = merged[merged["MarketStatus"] == "Sold"].copy()
    active = merged[merged["MarketStatus"] != "Sold"].copy()
    merged.to_csv(paths["labeled"], index=False)
    sold.to_csv(paths["sold"], index=False)
    active.to_csv(paths["active"], index=False)
    return merged, sold


def _select_next_background_candidate(search) -> tuple[str | None, Path | None, pd.DataFrame | None, dict]:
    from scraping_options import filter_eventual_sale_candidate_rows, read_schedule_state

    output_folder = Path(str(settings.paths.simple_scrape_dir)) / search.folder
    output_folder.mkdir(parents=True, exist_ok=True)
    priority_path, priority_row = _select_next_priority_candidate(search)
    if priority_row is not None and not priority_row.empty:
        return "priority", priority_path, priority_row, read_schedule_state(str(output_folder))

    input_path = get_eventual_sales_input_path(output_folder)
    if input_path is None:
        return None, None, None, {}

    raw_df = _read_csv_or_empty(input_path)
    if raw_df.empty:
        return "deals_ranked", input_path, None, read_schedule_state(str(output_folder))

    candidates = filter_eventual_sale_candidate_rows(
        raw_df,
        min_deal_score=2.0,
        min_deal_confidence=0.7,
        top_n=None,
        require_deal_eligible=True,
        sort_by="DealScore,DealConfidence,SearchCount",
    )
    if candidates.empty:
        return "deals_ranked", input_path, None, read_schedule_state(str(output_folder))

    state = read_schedule_state(str(output_folder))
    existing = _read_csv_or_empty(_background_csv_paths(output_folder)["labeled"])
    sold_ids = _known_sold_ids(output_folder)
    checked_ids = set()
    if not existing.empty and "Dataid" in existing.columns:
        checked_ids = set(pd.to_numeric(existing["Dataid"], errors="coerce").dropna().astype(int).tolist())

    if sold_ids and "Dataid" in candidates.columns:
        candidate_ids = pd.to_numeric(candidates["Dataid"], errors="coerce")
        candidates = candidates[~candidate_ids.isin(list(sold_ids))].reset_index(drop=True)
    if candidates.empty:
        return "deals_ranked", input_path, None, state

    if checked_ids and "Dataid" in candidates.columns:
        candidate_ids = pd.to_numeric(candidates["Dataid"], errors="coerce")
        unchecked = candidates[~candidate_ids.isin(list(checked_ids))].reset_index(drop=True)
    else:
        unchecked = candidates

    if not unchecked.empty:
        row = unchecked.iloc[[0]].copy()
    else:
        cursor = int(state.get(EVENTUAL_SALES_BACKGROUND_CURSOR_KEY, 0) or 0)
        row = candidates.iloc[[cursor % len(candidates)]].copy()
        state[EVENTUAL_SALES_BACKGROUND_CURSOR_KEY] = int((cursor + 1) % len(candidates))

    return "deals_ranked", input_path, row, state


def process_one_background_eventual_sale(search) -> bool:
    from scraping_options import _update_market_status_for_df, write_schedule_state

    with eventual_sales_log_context():
        output_folder = Path(str(settings.paths.simple_scrape_dir)) / search.folder
        source, input_path, candidate_df, state = _select_next_background_candidate(search)
        if candidate_df is None and source != "priority":
            ensure_daily_deals_ranked(search)
            source, input_path, candidate_df, state = _select_next_background_candidate(search)

        if input_path is None:
            LOGGER.info(
                "Background eventual-sale check skipped for %s; no priority queue or deals_ranked.csv found",
                search.folder,
            )
            return False
        if candidate_df is None or candidate_df.empty:
            LOGGER.info("Background eventual-sale check skipped for %s; no queued candidates", search.folder)
            return False

        dataid = candidate_df.iloc[0].get("Dataid", "unknown")
        LOGGER.info(
            "Background eventual-sale check for %s source=%s item=%s",
            search.folder,
            source,
            dataid,
        )

        checked_df, sold_df = _update_market_status_for_df(
            candidate_df,
            max_workers=EVENTUAL_SALES_MAX_WORKERS,
            delay=EVENTUAL_SALES_DELAY_SECONDS,
            initial_delay=0.0,
            fetch_sleep=EVENTUAL_SALES_FETCH_SLEEP_SECONDS,
            fetch_max_attempts=EVENTUAL_SALES_FETCH_MAX_ATTEMPTS,
        )
        checked_df = checked_df.reset_index(drop=True)
        if source == "priority":
            _finalize_priority_candidate(output_folder, checked_df)
            _sync_priority_result_to_tracking_files(
                output_folder,
                checked_df,
            )
        merged, merged_sold = _merge_background_results(output_folder, checked_df)

        status = str(checked_df.iloc[0].get("LastCheckStatus", "") or checked_df.iloc[0].get("MarketStatus", "Unknown"))
        state.update(
            {
                EVENTUAL_SALES_STATE_DATE_KEY: today_local_iso(),
                EVENTUAL_SALES_BACKGROUND_LAST_CHECKED_AT_KEY: datetime.now().astimezone().isoformat(),
                EVENTUAL_SALES_BACKGROUND_LAST_DATAID_KEY: str(dataid),
                EVENTUAL_SALES_BACKGROUND_LAST_STATUS_KEY: status,
                EVENTUAL_SALES_BACKGROUND_LAST_SOURCE_KEY: str(source or ""),
                "last_eventual_sale_refresh_input": input_path.name,
                "last_eventual_sale_refresh_checked": int(len(merged)),
                "last_eventual_sale_refresh_sold": int(len(merged_sold)),
            }
        )
        write_schedule_state(str(output_folder), state)
        LOGGER.info(
            "Background eventual-sale result for %s source=%s item=%s status=%s total_checked=%s total_sold=%s",
            search.folder,
            source,
            dataid,
            status,
            len(merged),
            len(merged_sold),
        )

        # ── Experiment tracker ──────────────────────────────────────────────
        try:
            from experiments.tracking.db import init_db, log_eventual_sale_check
            init_db()
            log_eventual_sale_check({
                "search_name": search.folder,
                "checked_at": datetime.now().astimezone().isoformat(),
                "source": source or "background",
                "items_checked": len(merged),
                "sold_found": len(merged_sold),
                "still_unsold": len(merged) - len(merged_sold),
                "errors": 0,
                "duration_seconds": None,
            })
        except Exception as exc:
            LOGGER.warning("[tracking] failed to log eventual sale check: %s", exc)
        # ─────────────────────────────────────────────────────────────────────

        return bool(not sold_df.empty)


def _background_worker_loop(
    programmed_searches,
    interval_seconds: float,
    initial_delay_seconds: float,
    stop_event: threading.Event,
    scheduler_state: dict[str, object],
) -> None:
    searches = list(programmed_searches)
    if not searches:
        with eventual_sales_log_context():
            LOGGER.info("Eventual-sale background worker started with no searches")
        return

    if initial_delay_seconds > 0:
        stop_event.wait(initial_delay_seconds)
        if stop_event.is_set():
            return

    with eventual_sales_log_context():
        LOGGER.info(
            "Eventual-sale background worker started searches=%s interval_seconds=%.1f initial_delay_seconds=%.1f",
            len(searches),
            interval_seconds,
            initial_delay_seconds,
        )
    while not stop_event.is_set():
        with scheduler_state["lock"]:
            index = int(scheduler_state["index"])
            search = searches[index % len(searches)]
            scheduler_state["index"] = index + 1
        try:
            process_one_background_eventual_sale(search)
        except Exception as exc:
            with eventual_sales_log_context():
                LOGGER.warning(
                    "Background eventual-sale check failed for %s: %s: %s",
                    getattr(search, "folder", "unknown"),
                    type(exc).__name__,
                    exc,
                )
        stop_event.wait(interval_seconds)


def start_eventual_sales_background_worker(programmed_searches, interval_seconds: float = EVENTUAL_SALES_BACKGROUND_INTERVAL_SECONDS):
    stop_event = threading.Event()
    scheduler_state = {"index": 0, "lock": threading.Lock()}
    workers = []
    for worker_index in range(EVENTUAL_SALES_BACKGROUND_WORKER_COUNT):
        initial_delay = float(worker_index) * float(EVENTUAL_SALES_BACKGROUND_WORKER_STAGGER_SECONDS)
        worker = threading.Thread(
            target=_background_worker_loop,
            args=(list(programmed_searches), float(interval_seconds), initial_delay, stop_event, scheduler_state),
            name=f"eventual-sales-background-{worker_index + 1}",
            daemon=True,
        )
        worker.start()
        workers.append(worker)
    return workers, stop_event


def refresh_daily_eventual_sales(programmed_searches, today_iso: str | None = None) -> None:
    from scraping_options import (
        read_schedule_state,
        update_eventual_sale_labels_for_csv,
        write_schedule_state,
    )

    today_iso = today_iso or today_local_iso()

    with eventual_sales_log_context():
        for search in programmed_searches:
            output_folder = Path(str(settings.paths.simple_scrape_dir)) / search.folder
            output_folder.mkdir(parents=True, exist_ok=True)

            state = read_schedule_state(str(output_folder))
            if state.get(EVENTUAL_SALES_STATE_DATE_KEY) == today_iso:
                LOGGER.info(
                    "Skipping eventual sale refresh for %s; already completed on %s",
                    search.folder,
                    today_iso,
                )
                continue

            input_path = ensure_daily_deals_ranked(search, today_iso=today_iso)
            if input_path is None:
                LOGGER.info(
                    "Skipping eventual sale refresh for %s; no deals_ranked.csv found",
                    search.folder,
                )
                continue

            LOGGER.info(
                "Running eventual sale refresh for %s from %s",
                search.folder,
                input_path.name,
            )
            summary = update_eventual_sale_labels_for_csv(
                str(input_path),
                out_dir=str(output_folder / EVENTUAL_SALES_OUT_DIR_NAME),
                max_workers=EVENTUAL_SALES_MAX_WORKERS,
                delay=EVENTUAL_SALES_DELAY_SECONDS,
                min_deal_score=2.0,
                min_deal_confidence=0.7,
                top_n=100,
                require_deal_eligible=True,
                sort_by="DealScore,DealConfidence,SearchCount",
                fetch_sleep=EVENTUAL_SALES_FETCH_SLEEP_SECONDS,
                fetch_max_attempts=EVENTUAL_SALES_FETCH_MAX_ATTEMPTS,
                exclude_known_sold_csv=str(output_folder / "sold_df.csv"),
            )

            state.update(
                {
                    EVENTUAL_SALES_STATE_DATE_KEY: today_iso,
                    "last_eventual_sale_refresh_at": datetime.now().astimezone().isoformat(),
                    "last_eventual_sale_refresh_input": input_path.name,
                    "last_eventual_sale_refresh_checked": int(summary.get("n_checked", 0) or 0),
                    "last_eventual_sale_refresh_sold": int(summary.get("n_sold", 0) or 0),
                }
            )
            write_schedule_state(str(output_folder), state)

            # ── Experiment tracker ──────────────────────────────────────────
            try:
                from experiments.tracking.db import init_db, log_eventual_sale_check
                init_db()
                log_eventual_sale_check({
                    "search_name": search.folder,
                    "checked_at": datetime.now().astimezone().isoformat(),
                    "source": "daily_refresh",
                    "items_checked": int(summary.get("n_checked", 0) or 0),
                    "sold_found": int(summary.get("n_sold", 0) or 0),
                    "still_unsold": int(summary.get("n_checked", 0) or 0) - int(summary.get("n_sold", 0) or 0),
                    "errors": 0,
                    "duration_seconds": None,
                })
            except Exception as exc:
                LOGGER.warning("[tracking] failed to log daily refresh: %s", exc)
            # ─────────────────────────────────────────────────────────────────
