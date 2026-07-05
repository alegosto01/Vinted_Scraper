import json
import logging
import math
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from config.logging_config import eventual_sales_log_context
from config.project_config import settings
from utils_lib.retry_utils import sleep_if_positive


columns_seller = ["SellerId", "SellerName", "Location", "ReviewsCount", "Stars"]
COLUMNS = ['Title','Price','Brand','Size','Link','Likes','Dataid',
           'MarketStatus','SearchDate','Upload_date','Images','LocalImagePaths','LocalPrimaryImagePath','SearchCount','Page']

bot_token = settings.telegram.bot_token
telegram_chat_id = settings.telegram.chat_id

pages_to_scrape = 1

data_folder = str(settings.paths.data_dir)
old_df_path = os.path.join(data_folder, "old_df.csv")
unsold_df_path = os.path.join(data_folder, "unsold_df.csv")
sold_df_path = os.path.join(data_folder, "sold_df.csv")
seller_df_path = os.path.join(data_folder, "sellers_df.csv")

data_folder_simple_scrape = str(settings.paths.simple_scrape_dir)

pathfile_simple_old_df = os.path.join(data_folder_simple_scrape, "old_df.csv")
unsold_df_simple_path = os.path.join(data_folder_simple_scrape, "unsold_df.csv")
sold_df_simple_path = os.path.join(data_folder_simple_scrape, "sold_df.csv")
seller_df_simple_path = os.path.join(data_folder_simple_scrape, "sellers_df.csv")

LOGGER = logging.getLogger(__name__)

VALID_SCRAPE_MODES = {'collect', 'online'}


def preflight_parallel_scrape(programmed_searches, mode='collect', app_settings=settings):
    result = app_settings.validate_for_simple_scrape(programmed_searches, require_proxy=False)
    if mode not in VALID_SCRAPE_MODES:
        result.errors.append(f"Invalid scrape mode: {mode}. Expected one of {sorted(VALID_SCRAPE_MODES)}")
    if not app_settings.proxy.has_datacenter_proxy:
        result.errors.append(
            'Missing Bright Data datacenter proxy configuration. '
            'Set BRIGHTDATA_DATACENTER_USERNAME and BRIGHTDATA_DATACENTER_PASSWORD in .env, '
            'or set BRIGHTDATA_DATACENTER_PROXY_URL.'
        )
    return result


def maybe_refresh_daily_eventual_sales_for_running_scheduler(programmed_searches, last_refresh_date: str | None):
    from daily_eventual_sales import refresh_daily_eventual_sales, today_local_iso

    today_iso = today_local_iso()
    if last_refresh_date == today_iso:
        return last_refresh_date

    LOGGER.info('Checking daily eventual-sale refresh for local_date=%s', today_iso)
    refresh_daily_eventual_sales(programmed_searches, today_iso=today_iso)
    return today_iso


def filter_eventual_sale_candidate_rows(
    df: pd.DataFrame,
    *,
    min_deal_score=None,
    min_deal_confidence=None,
    top_n=None,
    require_deal_eligible=False,
    sort_by=None,
) -> pd.DataFrame:
    out = dedupe_market_rows(df, keep="last")

    if require_deal_eligible and "DealEligible" in out.columns:
        eligible = out["DealEligible"]
        if str(eligible.dtype).lower() not in {"bool", "boolean"}:
            eligible = eligible.astype(str).str.lower().isin(["true", "1", "yes"])
        out = out[eligible].copy()

    if min_deal_score is not None and "DealScore" in out.columns:
        score_num = pd.to_numeric(out["DealScore"], errors="coerce")
        out = out[score_num >= float(min_deal_score)].copy()

    if min_deal_confidence is not None and "DealConfidence" in out.columns:
        conf_num = pd.to_numeric(out["DealConfidence"], errors="coerce")
        out = out[conf_num >= float(min_deal_confidence)].copy()

    if sort_by:
        sort_cols = []
        ascending = []
        for raw_col in sort_by.split(","):
            col = raw_col.strip()
            if not col or col not in out.columns:
                continue
            if col in {"DealScore", "DealConfidence", "SearchCount", "SnapshotCount"}:
                out[col] = pd.to_numeric(out[col], errors="coerce")
            sort_cols.append(col)
            ascending.append(False)
        if sort_cols:
            out = out.sort_values(sort_cols, ascending=ascending, kind="stable")

    if top_n is not None:
        out = out.head(int(top_n)).copy()

    return out.reset_index(drop=True)


def dedupe_market_rows(df: pd.DataFrame, keep: str = "last") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    dedupe_subset = ["Dataid"] if "Dataid" in df.columns else ["Link"] if "Link" in df.columns else None
    if not dedupe_subset:
        return df.copy()

    out = df.copy()
    out["_row_order"] = range(len(out))
    sort_cols = []
    temp_cols = ["_row_order"]

    if "SearchCount" in out.columns:
        out["_SearchCountNum"] = pd.to_numeric(out["SearchCount"], errors="coerce")
        sort_cols.append("_SearchCountNum")
        temp_cols.append("_SearchCountNum")
    if "SearchDate" in out.columns:
        out["_SearchDateTs"] = pd.to_datetime(out["SearchDate"], errors="coerce", dayfirst=True)
        sort_cols.append("_SearchDateTs")
        temp_cols.append("_SearchDateTs")
    if "Page" in out.columns:
        out["_PageNum"] = pd.to_numeric(out["Page"], errors="coerce")
        sort_cols.append("_PageNum")
        temp_cols.append("_PageNum")

    sort_cols.append("_row_order")
    out = out.sort_values(sort_cols, kind="stable")
    out = out.drop_duplicates(subset=dedupe_subset, keep=keep)
    out = out.drop(columns=temp_cols, errors="ignore")
    return out.reset_index(drop=True)


def exclude_matching_market_rows(df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or reference_df.empty:
        return df.copy()

    mask = pd.Series([False] * len(df), index=df.index)

    if "Dataid" in df.columns and "Dataid" in reference_df.columns:
        reference_ids = (
            pd.to_numeric(reference_df["Dataid"], errors="coerce")
            .dropna()
            .astype("Int64")
            .unique()
            .tolist()
        )
        if reference_ids:
            mask = mask | pd.to_numeric(df["Dataid"], errors="coerce").isin(reference_ids)

    if "Link" in df.columns and "Link" in reference_df.columns:
        reference_links = reference_df["Link"].dropna().astype(str).unique().tolist()
        if reference_links:
            mask = mask | df["Link"].astype(str).isin(reference_links)

    return df.loc[~mask].copy().reset_index(drop=True)


def append_csv_atomic(df_to_add: pd.DataFrame, path: str, dedupe: bool = False, keep: str = "last"):
    """Append without race/corruption (single-thread use)."""
    if df_to_add.empty:
        return
    if os.path.exists(path):
        prev = pd.read_csv(path)
        out = pd.concat([prev, df_to_add], ignore_index=True)
    else:
        out = df_to_add.copy()
    if dedupe:
        out = dedupe_market_rows(out, keep=keep)
    tmp = path + ".tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_csv_atomic(df: pd.DataFrame, path: str):
    """Write a CSV atomically."""
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def ensure_search_tracking_files(output_folder: str):
    """Create the per-search tracking CSVs if they do not exist yet."""
    for filename in ("old_df.csv", "unsold_df.csv", "sold_df.csv", "non_really_sold_items_ids.csv"):
        path = os.path.join(output_folder, filename)
        if os.path.exists(path):
            continue
        cols = ["Dataid"] if filename == "non_really_sold_items_ids.csv" else COLUMNS
        write_csv_atomic(pd.DataFrame(columns=cols), path)

SCHEDULE_STATE_FILENAME = "schedule_state.json"
MIN_SCHEDULER_SLEEP_SECONDS = 15
MAX_SCHEDULER_SLEEP_SECONDS = 900


def read_csv_or_empty(path: str, columns=None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path)
    except Exception as exc:
        LOGGER.warning('Failed reading CSV path=%s: %s: %s', path, type(exc).__name__, exc)
        return pd.DataFrame(columns=columns)


def read_schedule_state(output_folder: str) -> dict:
    path = os.path.join(output_folder, SCHEDULE_STATE_FILENAME)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        LOGGER.warning('Failed reading scheduler state for folder=%s: %s: %s', output_folder, type(exc).__name__, exc)
        return {}


def write_schedule_state(output_folder: str, state: dict):
    path = os.path.join(output_folder, SCHEDULE_STATE_FILENAME)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _parse_search_dates(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype='datetime64[ns, UTC]')
    return pd.to_datetime(series, errors='coerce', dayfirst=True, utc=True)


def _latest_search_date_from_df(df: pd.DataFrame):
    if df.empty or 'SearchDate' not in df.columns:
        return pd.NaT
    ts = _parse_search_dates(df['SearchDate'])
    return ts.max()


def get_latest_search_timestamp(output_folder: str):
    latest = pd.NaT
    for filename in ('old_df.csv', 'big_raw.csv'):
        df = read_csv_or_empty(os.path.join(output_folder, filename), columns=COLUMNS)
        candidate = _latest_search_date_from_df(df)
        if pd.notna(candidate) and (pd.isna(latest) or candidate > latest):
            latest = candidate
    return latest


def infer_next_search_count(output_folder: str) -> int:
    max_count = 0
    state = read_schedule_state(output_folder)
    stored_count = pd.to_numeric(pd.Series([state.get('last_search_count')]), errors='coerce').iloc[0]
    if pd.notna(stored_count):
        max_count = max(max_count, int(stored_count))

    for filename in ('old_df.csv', 'big_raw.csv', 'stream_assigned.csv'):
        path = os.path.join(output_folder, filename)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, usecols=lambda c: c == 'SearchCount')
        except ValueError:
            continue
        except Exception as exc:
            LOGGER.warning('Failed inferring SearchCount from %s: %s: %s', path, type(exc).__name__, exc)
            continue
        if 'SearchCount' not in df.columns or df.empty:
            continue
        counts = pd.to_numeric(df['SearchCount'], errors='coerce').dropna()
        if not counts.empty:
            max_count = max(max_count, int(counts.max()))

    return max_count + 1 if max_count > 0 else 1


def _recent_count_in_window(df: pd.DataFrame, cutoff, date_col: str = 'SearchDate') -> int:
    if df.empty or date_col not in df.columns:
        return 0
    ts = _parse_search_dates(df[date_col])
    return int((ts >= cutoff).fillna(False).sum())


def _load_deal_signal_df(output_folder: str) -> pd.DataFrame:
    for filename in ('stream_assigned.csv', os.path.join('pipeline_out', 'deals_ranked.csv')):
        path = os.path.join(output_folder, filename)
        if os.path.exists(path):
            return read_csv_or_empty(path)
    return pd.DataFrame()


def _count_recent_good_deals(df: pd.DataFrame, cutoff, min_score: float, min_confidence: float) -> int:
    if df.empty or 'SearchDate' not in df.columns:
        return 0

    mask = pd.Series(True, index=df.index)
    if 'DealEligible' in df.columns:
        eligible = df['DealEligible']
        if str(eligible.dtype).lower() not in {'bool', 'boolean'}:
            eligible = eligible.astype(str).str.lower().isin(['true', '1', 'yes'])
        mask &= eligible.fillna(False)
    if 'DealScore' in df.columns:
        mask &= pd.to_numeric(df['DealScore'], errors='coerce') >= float(min_score)
    if 'DealConfidence' in df.columns:
        mask &= pd.to_numeric(df['DealConfidence'], errors='coerce') >= float(min_confidence)

    ts = _parse_search_dates(df['SearchDate'])
    mask &= (ts >= cutoff).fillna(False)
    return int(mask.sum())


def _count_recent_fast_sales(old_df: pd.DataFrame, sold_df: pd.DataFrame, cutoff, fast_sale_hours: float) -> int:
    if old_df.empty or sold_df.empty:
        return 0
    key = 'Dataid' if 'Dataid' in old_df.columns and 'Dataid' in sold_df.columns else 'Link' if 'Link' in old_df.columns and 'Link' in sold_df.columns else None
    if key is None or 'SearchDate' not in old_df.columns or 'SearchDate' not in sold_df.columns:
        return 0

    old = old_df[[key, 'SearchDate']].copy()
    sold = sold_df[[key, 'SearchDate']].copy()
    old = old.rename(columns={'SearchDate': 'FirstSeenSearchDate'})
    sold = sold.rename(columns={'SearchDate': 'SoldSeenSearchDate'})
    old['FirstSeenSearchDate'] = _parse_search_dates(old['FirstSeenSearchDate'])
    sold['SoldSeenSearchDate'] = _parse_search_dates(sold['SoldSeenSearchDate'])
    old = old.dropna(subset=['FirstSeenSearchDate'])
    sold = sold.dropna(subset=['SoldSeenSearchDate'])
    if old.empty or sold.empty:
        return 0

    first_seen = old.sort_values('FirstSeenSearchDate').drop_duplicates(subset=[key], keep='first')
    sold_seen = sold.sort_values('SoldSeenSearchDate').drop_duplicates(subset=[key], keep='last')
    merged = sold_seen.merge(first_seen, on=key, how='inner')
    if merged.empty:
        return 0

    merged = merged[merged['SoldSeenSearchDate'] >= cutoff]
    if merged.empty:
        return 0

    sell_delay_hours = (merged['SoldSeenSearchDate'] - merged['FirstSeenSearchDate']).dt.total_seconds() / 3600.0
    return int(((sell_delay_hours >= 0.0) & (sell_delay_hours <= fast_sale_hours)).sum())


def compute_search_activity(ricerca, output_folder: str, now_ts=None) -> dict:
    now_ts = now_ts or pd.Timestamp.now(tz="UTC")
    window_hours = max(float(getattr(ricerca, 'activity_window_hours', 24) or 24), 1.0)
    min_delay_seconds = max(int(float(getattr(ricerca, 'min_delay_minutes', 15) or 15) * 60), 60)
    max_delay_seconds = max(min_delay_seconds, int(float(getattr(ricerca, 'max_delay_minutes', 240) or 240) * 60))
    default_delay_seconds = min(max_delay_seconds, max(min_delay_seconds, int(float(getattr(ricerca, 'default_delay_minutes', 60) or 60) * 60)))
    target_items_per_run = max(float(getattr(ricerca, 'target_items_per_run', 6.0) or 6.0), 1.0)
    target_value_per_run = max(float(getattr(ricerca, 'target_value_per_run', 4.0) or 4.0), 0.5)
    sold_weight = max(float(getattr(ricerca, 'sold_weight', 2.0) or 2.0), 0.0)
    good_deal_weight = max(float(getattr(ricerca, 'good_deal_weight', 1.0) or 1.0), 0.0)
    fast_sale_weight = max(float(getattr(ricerca, 'fast_sale_weight', 1.5) or 1.5), 0.0)
    fast_sale_hours = max(float(getattr(ricerca, 'fast_sale_hours', 12.0) or 12.0), 0.25)
    good_deal_min_score = float(getattr(ricerca, 'good_deal_min_score', 2.0) or 2.0)
    good_deal_min_confidence = float(getattr(ricerca, 'good_deal_min_confidence', 0.6) or 0.6)
    cutoff = now_ts - pd.Timedelta(hours=window_hours)

    raw_path = os.path.join(output_folder, 'big_raw.csv')
    old_path = os.path.join(output_folder, 'old_df.csv')
    sold_path = os.path.join(output_folder, 'sold_df.csv')

    new_df = read_csv_or_empty(raw_path if os.path.exists(raw_path) else old_path, columns=COLUMNS)
    old_df = read_csv_or_empty(old_path, columns=COLUMNS)
    sold_df = read_csv_or_empty(sold_path, columns=COLUMNS)
    deal_df = _load_deal_signal_df(output_folder)

    recent_new = _recent_count_in_window(new_df, cutoff)
    recent_sold = _recent_count_in_window(sold_df, cutoff)
    recent_good_deals = _count_recent_good_deals(deal_df, cutoff, good_deal_min_score, good_deal_min_confidence)
    recent_fast_sales = _count_recent_fast_sales(old_df, sold_df, cutoff, fast_sale_hours)

    value_units = (
        good_deal_weight * recent_good_deals
        + fast_sale_weight * recent_fast_sales
        + sold_weight * recent_sold
    )
    activity_per_hour = value_units / window_hours
    if activity_per_hour <= 0:
        recommended_delay_seconds = default_delay_seconds
    else:
        recommended_delay_seconds = int(round((target_value_per_run / activity_per_hour) * 3600.0))
        recommended_delay_seconds = max(min_delay_seconds, min(max_delay_seconds, recommended_delay_seconds))

    return {
        'window_hours': window_hours,
        'recent_new_items': recent_new,
        'recent_sold_items': recent_sold,
        'recent_good_deals': recent_good_deals,
        'recent_fast_sales': recent_fast_sales,
        'value_units': value_units,
        'activity_per_hour': activity_per_hour,
        'recommended_delay_seconds': recommended_delay_seconds,
        'min_delay_seconds': min_delay_seconds,
        'max_delay_seconds': max_delay_seconds,
        'default_delay_seconds': default_delay_seconds,
        'target_items_per_run': target_items_per_run,
        'target_value_per_run': target_value_per_run,
        'sold_weight': sold_weight,
        'good_deal_weight': good_deal_weight,
        'fast_sale_weight': fast_sale_weight,
        'fast_sale_hours': fast_sale_hours,
        'good_deal_min_score': good_deal_min_score,
        'good_deal_min_confidence': good_deal_min_confidence,
    }


def build_search_run_plan(ricerca, output_root: str, now_ts=None) -> dict:
    now_ts = now_ts or pd.Timestamp.now(tz="UTC")
    output_folder = os.path.join(output_root, ricerca.folder)
    os.makedirs(output_folder, exist_ok=True)
    ensure_search_tracking_files(output_folder)

    state = read_schedule_state(output_folder)
    activity = compute_search_activity(ricerca, output_folder, now_ts=now_ts)
    delay_seconds = activity['recommended_delay_seconds'] if getattr(ricerca, 'auto_schedule', True) else activity['default_delay_seconds']

    last_completed_at = pd.to_datetime(state.get('last_completed_at'), errors='coerce', utc=True)
    if pd.isna(last_completed_at):
        last_completed_at = get_latest_search_timestamp(output_folder)

    if pd.isna(last_completed_at):
        due_at = now_ts
    else:
        due_at = last_completed_at + pd.Timedelta(seconds=delay_seconds)

    due_in_seconds = max(0, int(math.ceil((due_at - now_ts).total_seconds()))) if due_at > now_ts else 0
    return {
        'ricerca': ricerca,
        'output_folder': output_folder,
        'state': state,
        'activity': activity,
        'delay_seconds': delay_seconds,
        'due_at': due_at,
        'due_in_seconds': due_in_seconds,
        'is_due': due_in_seconds <= 0,
        'next_search_count': infer_next_search_count(output_folder),
    }


def update_search_schedule_state(plan: dict, summary: dict, delay_seconds: int, activity: dict, completed_at=None):
    completed_at = completed_at or pd.Timestamp.now(tz="UTC")
    state = dict(plan.get('state') or {})
    state.update({
        'folder': summary['ricerca'].folder,
        'last_completed_at': completed_at.isoformat(),
        'last_due_delay_seconds': int(delay_seconds),
        'last_search_count': int(summary.get('search_count', 0) or 0),
        'last_scraped_items': int(summary.get('scraped', 0) or 0),
        'last_new_items': int(summary.get('new', 0) or 0),
        'recent_new_items': int(activity.get('recent_new_items', 0) or 0),
        'recent_sold_items': int(activity.get('recent_sold_items', 0) or 0),
        'recent_good_deals': int(activity.get('recent_good_deals', 0) or 0),
        'recent_fast_sales': int(activity.get('recent_fast_sales', 0) or 0),
        'value_units': float(activity.get('value_units', 0.0) or 0.0),
        'activity_per_hour': float(activity.get('activity_per_hour', 0.0) or 0.0),
        'activity_window_hours': float(activity.get('window_hours', 24.0) or 24.0),
        'next_run_at': (completed_at + pd.Timedelta(seconds=delay_seconds)).isoformat(),
    })
    write_schedule_state(plan['output_folder'], state)


def scrapeSpecificItems_InSequence(programmed_searches):
    """
    scrape and filter items that passed a series of manual filters
    """
    columns = ['Title', 'Price', 'Brand', 'Size', 'Link', 'Likes', 'Dataid',
                'MarketStatus', 'SearchDate', 'Upload_date', 'Images', 'LocalImagePaths', 'LocalPrimaryImagePath', "SearchCount", "Page"]

    for search_count in range(1, 500):
        for ricerca in programmed_searches:
            output_folder = os.path.join(data_folder_simple_scrape, ricerca['folder'])
            os.makedirs(output_folder, exist_ok=True)
            pathfile_old_df_item = os.path.join(output_folder, "old_df.csv")

            print(f"SEARCH: {ricerca['search']}")
            print("-" * 20)
            print(f"SEARCH COUNT: {search_count}")

            from simple_scraper import Simple_scraper

            simple_scraper = Simple_scraper()
            scraped_data = simple_scraper.scrape_products_serial(ricerca, search_count, pages_to_scrape, get_images=True)

            print("Scraped data first 5 items:")
            print(scraped_data[:5])

            scraped_df = pd.DataFrame(scraped_data, columns=columns)
            LOGGER.info('Search=%s scraped_items=%s', ricerca.folder, len(scraped_df))

            old_df = pd.read_csv(pathfile_old_df_item) if os.path.exists(pathfile_old_df_item) else pd.DataFrame(columns=columns)

            items_already_stored = []
            for index, row in scraped_df.iterrows():
                if int(row["Dataid"]) in old_df["Dataid"].values:
                    items_already_stored.append(index)

            new_df = scraped_df.drop(items_already_stored).reset_index(drop=True)
            LOGGER.info('Search=%s new_items=%s from_scraped=%s', ricerca.folder, len(new_df), len(scraped_df))

            old_df = pd.concat([old_df, new_df], ignore_index=True)
            old_df.to_csv(pathfile_old_df_item, index=False)

            if search_count > 0:
                import send_batch_items_to_telegram
                send_batch_items_to_telegram.send_new_items_to_telegram(new_df, bot_token, telegram_chat_id)

            sleep_if_positive(5)

        sleep_if_positive(300)

def _process_one_search(ricerca, search_count, pages_to_scrape):
    output_folder = os.path.join(data_folder_simple_scrape, ricerca.folder)
    os.makedirs(output_folder, exist_ok=True)
    ensure_search_tracking_files(output_folder)
    pathfile_old_df_item = os.path.join(output_folder, "old_df.csv")

    LOGGER.info('Starting search=%s search_count=%s', ricerca.folder, search_count)

    from simple_scraper import Simple_scraper

    simple_scraper = Simple_scraper()
    scraped_data = simple_scraper.scrape_products_serial(
        ricerca, search_count, pages_to_scrape, get_images=True
    )

    scraped_df = pd.DataFrame(scraped_data, columns=COLUMNS)
    LOGGER.info('Search=%s scraped_items=%s', ricerca.folder, len(scraped_df))

    if os.path.exists(pathfile_old_df_item):
        old_df = pd.read_csv(pathfile_old_df_item)
    else:
        old_df = pd.DataFrame(columns=COLUMNS)

    # faster than iterrows loop (vectorized)
    if not old_df.empty:
        new_df = scraped_df[~scraped_df["Dataid"].astype(int).isin(old_df["Dataid"].astype(int))].copy()
        new_df.reset_index(drop=True, inplace=True)
    else:
        new_df = scraped_df.copy()

    LOGGER.info('Search=%s new_items=%s from_scraped=%s', ricerca.folder, len(new_df), len(scraped_df))

    deal_finder_summary = {}
    if not new_df.empty:
        try:
            from full_scraper import Full_Scraper

            if "SearchName" not in new_df.columns:
                new_df = new_df.copy()
                new_df["SearchName"] = ricerca.folder
            full_scraper = Full_Scraper()
            new_df, deal_finder_summary = full_scraper.score_and_collect_extremes_for_live_rows(
                ricerca.folder,
                new_df,
                low_threshold=0.05,
                high_threshold=0.95,
                max_workers=2,
            )
            LOGGER.info(
                'Deal-finder scoring search=%s rows=%s extremes=%s summary=%s',
                ricerca.folder,
                len(new_df),
                deal_finder_summary.get("extreme_rows", 0),
                deal_finder_summary,
            )
        except Exception as exc:
            LOGGER.warning(
                'Deal-finder scoring/full-scrape hook failed for search=%s: %s: %s',
                ricerca.folder,
                type(exc).__name__,
                exc,
            )
            traceback.print_exc()



    LOGGER.info('Finished initial scrape for search=%s new_items=%s; checking sold-status reconciliation', ricerca.folder, len(new_df))
    persisted_old_df = False
    if not new_df.empty and not old_df.empty:
        last_old_date = pd.to_datetime(old_df.iloc[-1]["SearchDate"])
        last_new_date = pd.to_datetime(new_df.iloc[-1]["SearchDate"])
        time_diff = last_new_date - last_old_date
        LOGGER.info('Search=%s time difference from last run: %s', ricerca.folder, time_diff)
        if time_diff < pd.Timedelta(minutes=600):
            
            simple_scraper.compare_and_save_df_serial(
                new_df, old_df, unsold_df_path=output_folder + "/unsold_df.csv", sold_df_path=output_folder + "/sold_df.csv", non_really_sold_items_ids_df_path=output_folder + "/non_really_sold_items_ids.csv", output_folder=output_folder
            )
            persisted_old_df = True
        else:
            LOGGER.warning('Skipping compare_and_save for search=%s because last run is too old', ricerca.folder)


    # persist old_df.csv atomically (thread-safe per folder)
    if not persisted_old_df:
        # if we skipped compare_and_save, we just append new_df to old_df without marking items as sold (to avoid losing data)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = dedupe_market_rows(combined, keep="last")
        write_csv_atomic(combined, pathfile_old_df_item)

    return {
        "ricerca": ricerca,
        "search_count": search_count,
        "scraped": len(scraped_df),
        "new": len(new_df),
        "new_df": new_df,              # keep it for main thread
        "output_folder": output_folder, # optional
        "deal_finder_summary": deal_finder_summary,
    }


def scrape_specific_items_parallel(
    programmed_searches,
    pages_to_scrape=10,
    bot_token=bot_token,
    telegram_chat_id=telegram_chat_id,
    search_workers=6,
    max_search_counts=500,
    delay_between_jobs=5,
    delay_between_batch_of_searches=900,
    mode="collect",  # <-- NEW: "collect" or "online"
):
    output_folder = str(settings.paths.simple_scrape_dir)
    global_stream = os.path.join(output_folder, "stream_assigned_all.csv")
    os.makedirs(os.path.dirname(global_stream), exist_ok=True)

    preflight = preflight_parallel_scrape(programmed_searches, mode=mode, app_settings=settings)
    preflight.log(LOGGER)
    if not preflight.ok:
        return []

    scheduler_iterations = 0

    while scheduler_iterations < max_search_counts:
        now_ts = pd.Timestamp.now(tz="UTC")
        plans = [build_search_run_plan(ricerca, output_folder, now_ts=now_ts) for ricerca in programmed_searches]
        due_plans = [plan for plan in plans if plan['is_due']]

        if not due_plans:
            next_due_seconds = min(plan['due_in_seconds'] for plan in plans)
            sleep_seconds = max(MIN_SCHEDULER_SLEEP_SECONDS, min(MAX_SCHEDULER_SLEEP_SECONDS, next_due_seconds))
            LOGGER.info('No searches due right now. next_due_in=%ss sleeping=%ss', next_due_seconds, sleep_seconds)
            sleep_if_positive(sleep_seconds)
            continue

        scheduler_iterations += 1
        summaries = []
        plan_by_folder = {plan['ricerca'].folder: plan for plan in due_plans}

        with ThreadPoolExecutor(max_workers=min(search_workers, len(due_plans))) as ex:
            futures = []
            for i, plan in enumerate(due_plans):
                futures.append(ex.submit(_process_one_search, plan['ricerca'], plan['next_search_count'], pages_to_scrape))
                if i < len(due_plans) - 1:
                    sleep_if_positive(delay_between_jobs)

            for fut in as_completed(futures):
                try:
                    summaries.append(fut.result())
                except Exception as e:
                    LOGGER.warning('One search failed: %s: %r', type(e).__name__, e)
                    traceback.print_exc()

        LOGGER.info('Completed scheduler iteration=%s due_searches=%s', scheduler_iterations, len(summaries))

        for s in summaries:
            new_df = s.get("new_df")
            folder_name = getattr(s.get('ricerca'), 'folder', '')
            plan = plan_by_folder.get(folder_name)
            if plan is None:
                LOGGER.warning('Missing scheduler plan for search=%s', folder_name)
                continue

            if new_df is not None and not new_df.empty:
                if "Dataid" in new_df.columns:
                    new_df = new_df.drop_duplicates(subset=["Dataid"])
                elif "Link" in new_df.columns:
                    new_df = new_df.drop_duplicates(subset=["Link"])

                new_df = new_df.copy()
                new_df["SearchCount"] = s.get('search_count', plan['next_search_count'])
                new_df["SearchName"] = folder_name

                if mode == "collect":
                    try:
                        LOGGER.info('Appending %s new items to raw CSV for search=%s', len(new_df), folder_name)
                        raw_path = os.path.join(output_folder, folder_name, "big_raw.csv")
                        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
                        append_csv_atomic(new_df, raw_path, dedupe=True, keep="last")
                    except Exception as e:
                        LOGGER.warning('Raw append failed for search=%s: %s: %s', folder_name, type(e).__name__, e)
                        traceback.print_exc()

                elif mode == "online":
                    db_path_search = os.path.join(output_folder, folder_name, "index.sqlite")
                    try:
                        from experiments.old.clustering_approach.vinted_pipeline_incremental import process_new_df

                        assigned_df = process_new_df(
                            new_df,
                            db_path=db_path_search,
                            price_buffer_size=200
                        )
                        per_search_stream = os.path.join(output_folder, folder_name, "stream_assigned.csv")
                        os.makedirs(os.path.dirname(per_search_stream), exist_ok=True)
                        append_csv_atomic(assigned_df, per_search_stream)
                        append_csv_atomic(assigned_df, global_stream)
                    except Exception as e:
                        LOGGER.warning('Analysis failed for search=%s: %s: %s', folder_name, type(e).__name__, e)
                        traceback.print_exc()

                else:
                    raise ValueError("mode must be 'collect' or 'online'")

                LOGGER.info('Finished post-processing for search=%s new_items=%s', folder_name, len(new_df))
            else:
                LOGGER.info('No new items for search=%s in this run', folder_name)

            completed_at = pd.Timestamp.now(tz="UTC")
            refreshed_activity = compute_search_activity(s['ricerca'], plan['output_folder'], now_ts=completed_at)
            refreshed_delay_seconds = refreshed_activity['recommended_delay_seconds'] if getattr(s['ricerca'], 'auto_schedule', True) else refreshed_activity['default_delay_seconds']
            update_search_schedule_state(plan, s, refreshed_delay_seconds, refreshed_activity, completed_at=completed_at)
            LOGGER.info(
                'Updated schedule for search=%s delay=%ss good_deals=%s fast_sales=%s sold=%s value_units=%.2f next_run_at=%s',
                folder_name,
                refreshed_delay_seconds,
                refreshed_activity['recent_good_deals'],
                refreshed_activity['recent_fast_sales'],
                refreshed_activity['recent_sold_items'],
                refreshed_activity['value_units'],
                (completed_at + pd.Timedelta(seconds=refreshed_delay_seconds)).isoformat(),
            )

        total_scraped = sum(s.get('scraped', 0) for s in summaries)
        total_new = sum(s.get('new', 0) for s in summaries)
        LOGGER.info('Scheduler summary iteration=%s searches=%s scraped=%s new=%s', scheduler_iterations, len(summaries), total_scraped, total_new)

        # ── Experiment tracker ──────────────────────────────────────────────
        try:
            from experiments.old.tracking.db import init_db, log_scraper_iteration
            init_db()
            events = []
            for s in summaries:
                folder_name = getattr(s.get('ricerca'), 'folder', '')
                plan = plan_by_folder.get(folder_name)
                events.append({
                    "search_name": folder_name,
                    "scraped_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    "pages_scraped": pages_to_scrape,
                    "new_items": s.get('new', 0),
                    "sold_found": 0,  # sold detection happens in compare_and_save, not tracked here yet
                    "proxy_type": "datacenter",  # default; could be enriched later
                    "error": None,
                    "schedule_delay_seconds": plan.get('recommended_delay_seconds') if plan else None,
                })
            log_scraper_iteration({
                "iteration": scheduler_iterations,
                "started_at": now_ts.isoformat(),
                "finished_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "searches_due": len(due_plans),
                "searches_scraped": len(summaries),
                "total_new_items": total_new,
                "total_sold_found": 0,
                "proxy_datacenter_hits": len(summaries),
                "proxy_residential_hits": 0,
                "proxy_errors": sum(1 for s in summaries if s.get('error')),
                "events": events,
            })
        except Exception as exc:
            LOGGER.warning('[tracking] failed to log scraper iteration: %s', exc)
        # ─────────────────────────────────────────────────────────────────────

        if delay_between_batch_of_searches > 0:
            sleep_if_positive(min(delay_between_batch_of_searches, MIN_SCHEDULER_SLEEP_SECONDS))


def parse_relative_upload_date_to_days(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return pd.NA
    text = str(value).strip().lower()
    if not text or text in {"unknown", "none", "nan", "<na>"}:
        return pd.NA

    normalized = (
        text.replace("giorno", "giorni")
        .replace("day", "days")
        .replace("settimana", "settimane")
        .replace("week", "weeks")
        .replace("mese", "mesi")
        .replace("month", "months")
        .replace("anno", "anni")
        .replace("year", "years")
        .replace("ora", "ore")
        .replace("hour", "hours")
        .replace("minuto", "minuti")
        .replace("minute", "minutes")
    )

    if "oggi" in normalized or "today" in normalized:
        return 0.0
    if "ieri" in normalized or "yesterday" in normalized:
        return 1.0

    m = re.search(r"(\d+(?:[\.,]\d+)?)", normalized)
    if not m:
        return pd.NA
    number = float(m.group(1).replace(",", "."))

    if "minuti" in normalized or "minutes" in normalized:
        return round(number / (24.0 * 60.0), 4)
    if "ore" in normalized or "hours" in normalized:
        return round(number / 24.0, 4)
    if "settimane" in normalized or "weeks" in normalized:
        return round(number * 7.0, 4)
    if "mesi" in normalized or "months" in normalized:
        return round(number * 30.0, 4)
    if "anni" in normalized or "years" in normalized:
        return round(number * 365.0, 4)
    if "giorni" in normalized or "days" in normalized:
        return round(number, 4)
    return pd.NA


def _check_sold_with_own_driver(
    row,
    *,
    initial_delay=0.0,
    fetch_sleep=60.0,
    fetch_max_attempts=1,
):
    item_identifier = row.get("Dataid") or row.get("item_id") or row.get("Link") or row.name
    with eventual_sales_log_context():
        LOGGER.info("Checking if item %s is sold...", item_identifier)
        from simple_scraper import Simple_scraper

        scraper = Simple_scraper()
        try:
            item, sold, status = scraper.inspect_item_page(
                row.to_dict(),
                get_images=False,
                check_venduto=True,
                get_upload_date=True,
                initial_delay=initial_delay,
                fetch_sleep=fetch_sleep,
                fetch_max_attempts=fetch_max_attempts,
            )
            upload_date = item.get("Upload_date", "Unknown")
            page_price = item.get("ObservedPagePrice")
            previous_price = item.get("PreviousPrice")
            price_changed = bool(item.get("PriceChangedBeforeSold", False))
            LOGGER.info(
                "Checked item %s: status=%r upload=%r page_price=%r price_changed=%r",
                item_identifier,
                status,
                upload_date,
                page_price,
                price_changed,
            )
            return row.name, status, upload_date, page_price, previous_price, price_changed

        except Exception:
            LOGGER.warning("Error checking item %s: %s", item_identifier, traceback.format_exc())
            return row.name, "On Sale", "Unknown", None, None, False

def _update_market_status_for_df(
    df,
    max_workers=1,
    delay=0.0,
    initial_delay=0.0,
    fetch_sleep=60.0,
    fetch_max_attempts=1,
    recheck_sold_rows=False,
):
    out = df.copy()
    sold_rows = []
    if "MarketStatus" not in out.columns:
        out["MarketStatus"] = pd.NA
    else:
        out["MarketStatus"] = out["MarketStatus"].astype("object")
    if "Price" in out.columns:
        out["Price"] = out["Price"].astype("object")
    if "Upload_date" not in out.columns:
        out["Upload_date"] = pd.Series([pd.NA] * len(out), dtype="object")
    else:
        out["Upload_date"] = out["Upload_date"].astype("object")
    if "Upload_date_days" not in out.columns:
        out["Upload_date_days"] = pd.Series([pd.NA] * len(out), dtype="Float64")
    else:
        out["Upload_date_days"] = pd.to_numeric(out["Upload_date_days"], errors="coerce").astype("Float64")
    if "ObservedPagePrice" not in out.columns:
        out["ObservedPagePrice"] = pd.Series([pd.NA] * len(out), dtype="Float64")
    else:
        out["ObservedPagePrice"] = pd.to_numeric(out["ObservedPagePrice"], errors="coerce").astype("Float64")
    if "PreviousPrice" not in out.columns:
        out["PreviousPrice"] = pd.Series([pd.NA] * len(out), dtype="Float64")
    else:
        out["PreviousPrice"] = pd.to_numeric(out["PreviousPrice"], errors="coerce").astype("Float64")
    if "PriceChangedBeforeSold" not in out.columns:
        out["PriceChangedBeforeSold"] = pd.Series([False] * len(out), dtype="boolean")
    else:
        out["PriceChangedBeforeSold"] = out["PriceChangedBeforeSold"].astype("boolean")
    if "LastCheckStatus" not in out.columns:
        out["LastCheckStatus"] = pd.Series([pd.NA] * len(out), dtype="object")
    else:
        out["LastCheckStatus"] = out["LastCheckStatus"].astype("object")

    if recheck_sold_rows:
        to_check_idx = out.index
    else:
        to_check_idx = out.index[out["MarketStatus"] != "Sold"]
    with eventual_sales_log_context():
        LOGGER.info(
            "Checking %s items if they are sold%s",
            len(to_check_idx),
            " (including rows already marked Sold)" if recheck_sold_rows else "",
        )

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx in to_check_idx:
            futures.append(
                ex.submit(
                    _check_sold_with_own_driver,
                    out.loc[idx],
                    initial_delay=initial_delay,
                    fetch_sleep=fetch_sleep,
                    fetch_max_attempts=fetch_max_attempts,
                )
            )
            sleep_if_positive(delay)

        for fut in as_completed(futures):
            idx, status, upload_date, page_price, previous_price, price_changed = fut.result()
            out.at[idx, "LastCheckStatus"] = status
            upload_date_text = pd.NA if upload_date in {None, "Unknown", "None"} else str(upload_date)
            upload_date_days = parse_relative_upload_date_to_days(upload_date_text)
            if upload_date_text is not pd.NA:
                out.at[idx, "Upload_date"] = upload_date_text
            if pd.notna(upload_date_days):
                out.at[idx, "Upload_date_days"] = upload_date_days
            if page_price is not None:
                out.at[idx, "ObservedPagePrice"] = float(page_price)
            if previous_price is not None:
                out.at[idx, "PreviousPrice"] = float(previous_price)
            out.at[idx, "PriceChangedBeforeSold"] = bool(price_changed)
            if status == "Sold":
                out.at[idx, "MarketStatus"] = status
                if page_price is not None:
                    out.at[idx, "Price"] = float(page_price)
                sold_rows.append(out.loc[idx].copy())
                sold_identifier = out.at[idx, "Dataid"] if "Dataid" in out.columns else out.at[idx, "item_id"] if "item_id" in out.columns else out.at[idx, "Link"] if "Link" in out.columns else idx
                with eventual_sales_log_context():
                    LOGGER.info("Item %s is %s", sold_identifier, status.lower())
            elif recheck_sold_rows and status in {"OnSale", "On Sale"}:
                out.at[idx, "MarketStatus"] = "On Sale"

    sold_df = pd.DataFrame(sold_rows) if sold_rows else pd.DataFrame(columns=out.columns)
    return out, sold_df


def returnNewSoldItemsInCsv_parallel(
    csv,
    max_workers=1,
    delay=0.0,
    initial_delay=0.0,
    fetch_sleep=60.0,
    fetch_max_attempts=1,
):
    df = pd.read_csv(csv)
    _, sold_df = _update_market_status_for_df(
        df,
        max_workers=max_workers,
        delay=delay,
        initial_delay=initial_delay,
        fetch_sleep=fetch_sleep,
        fetch_max_attempts=fetch_max_attempts,
    )
    return sold_df.to_dict("records")


def update_eventual_sale_labels_for_csv(
    csv_path,
    out_dir=None,
    max_workers=1,
    delay=0.0,
    min_deal_score=None,
    min_deal_confidence=None,
    top_n=None,
    require_deal_eligible=False,
    sort_by=None,
    initial_delay=0.0,
    fetch_sleep=60.0,
    fetch_max_attempts=1,
    recheck_sold_rows=False,
    exclude_known_sold_csv=None,
):
    df = pd.read_csv(csv_path)
    df = dedupe_market_rows(df, keep="last")
    original_count = len(df)
    df = filter_eventual_sale_candidate_rows(
        df,
        min_deal_score=min_deal_score,
        min_deal_confidence=min_deal_confidence,
        top_n=top_n,
        require_deal_eligible=require_deal_eligible,
        sort_by=sort_by,
    )

    excluded_known_sold_count = 0
    known_sold_df = pd.DataFrame()
    if exclude_known_sold_csv:
        known_sold_df = read_csv_or_empty(exclude_known_sold_csv)
        before_exclusion = len(df)
        df = exclude_matching_market_rows(df, known_sold_df)
        excluded_known_sold_count = int(before_exclusion - len(df))

    labeled_df, sold_df = _update_market_status_for_df(
        df,
        max_workers=max_workers,
        delay=delay,
        initial_delay=initial_delay,
        fetch_sleep=fetch_sleep,
        fetch_max_attempts=fetch_max_attempts,
        recheck_sold_rows=recheck_sold_rows,
    )
    labeled_df = dedupe_market_rows(labeled_df, keep="last")
    sold_df = dedupe_market_rows(sold_df, keep="last")
    if not known_sold_df.empty:
        labeled_df = exclude_matching_market_rows(labeled_df, known_sold_df)
        sold_df = exclude_matching_market_rows(sold_df, known_sold_df)

    if out_dir is None:
        out_dir = os.path.dirname(csv_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    labeled_path = os.path.join(out_dir, "big_raw_eventual_sale_labeled.csv")
    sold_path = os.path.join(out_dir, "sold_eventually.csv")
    active_path = os.path.join(out_dir, "not_sold_yet.csv")

    labeled_df.to_csv(labeled_path, index=False)
    sold_df.to_csv(sold_path, index=False)
    labeled_df[labeled_df["MarketStatus"] != "Sold"].to_csv(active_path, index=False)

    return {
        "labeled_path": labeled_path,
        "sold_path": sold_path,
        "active_path": active_path,
        "n_input_rows": int(original_count),
        "n_checked": int(len(labeled_df)),
        "n_sold": int(len(sold_df)),
        "n_excluded_known_sold": int(excluded_known_sold_count),
        "filters": {
            "min_deal_score": min_deal_score,
            "min_deal_confidence": min_deal_confidence,
            "top_n": top_n,
            "require_deal_eligible": bool(require_deal_eligible),
            "sort_by": sort_by,
            "initial_delay": float(initial_delay),
            "fetch_sleep": float(fetch_sleep),
            "fetch_max_attempts": int(fetch_max_attempts),
            "recheck_sold_rows": bool(recheck_sold_rows),
            "exclude_known_sold_csv": exclude_known_sold_csv,
        },
    }

def scrapeToGetManuallyFilteredItems(programmed_searches):
    """
    scrape and filter items that passed a series of manual filters
    """
    columns = ['Title', 'Price', 'Brand', 'Size', 'Link', 'Likes', 'Dataid',
                'MarketStatus', 'SearchDate', 'Images', 'LocalImagePaths', 'LocalPrimaryImagePath', "SearchCount", "Page"]

    for search_count in range(1, 500):
        for ricerca in programmed_searches:
            output_folder = os.path.join(data_folder_simple_scrape, ricerca['folder'])
            os.makedirs(output_folder, exist_ok=True)
            pathfile_old_df_item = os.path.join(output_folder, "old_df.csv")

            print(f"SEARCH: {ricerca['search']}")
            print("-" * 20)
            print(f"SEARCH COUNT: {search_count}")

            from simple_scraper import Simple_scraper

            simple_scraper = Simple_scraper()
            scraped_data = simple_scraper.scrape_products_serial(ricerca, search_count, pages_to_scrape, get_images=True)
            
            print("Scraped data first 5 items:")
            print(scraped_data[:5])

            scraped_df = pd.DataFrame(scraped_data, columns=columns)

            LOGGER.info('Search=%s scraped_items=%s', ricerca.folder, len(scraped_df))

            old_df = pd.read_csv(pathfile_old_df_item) if os.path.exists(pathfile_old_df_item) else pd.DataFrame(columns=columns)

            items_already_stored = []

            for index, row in scraped_df.iterrows():
                if int(row["Dataid"]) in old_df["Dataid"].values:
                    items_already_stored.append(index)

            new_df = scraped_df.drop(items_already_stored).reset_index(drop=True)
            LOGGER.info('Search=%s new_items=%s from_scraped=%s', ricerca.folder, len(new_df), len(scraped_df))

            if search_count > 0:
                import send_batch_items_to_telegram

                import send_batch_items_to_telegram

                send_batch_items_to_telegram.send_new_items_to_telegram(new_df, bot_token, telegram_chat_id)
            
            sleep_if_positive(5)  # Sleep to avoid hitting the server too fast

        sleep_if_positive(300) #300  # Sleep to avoid hitting the server too fast


# sold = returnNewSoldItemsInCsv_parallel("/home/ale/Desktop/vinted/Vinted_New_Version/out/deals_ranked.csv")
# print(sold)

scrapeSpecificItems_parallel = scrape_specific_items_parallel
