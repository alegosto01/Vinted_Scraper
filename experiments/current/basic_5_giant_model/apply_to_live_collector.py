#!/usr/bin/env python3
"""Apply the trained Basic-5 Giant models to live collector tracked items."""
from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
for _path in (ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.current.basic_5_giant_model.paths import (  # noqa: E402
    EXPERIMENT_ROOT,
    MODELS_DIR,
    OFFLINE_RUNS_DIR,
    assert_experiment_path,
    ensure_experiment_dirs,
    run_id,
    write_json,
)
from config.project_config import settings  # noqa: E402
from experiments.deal_finder import model_sweep as base_sweep  # noqa: E402
from experiments.deal_finder.modeling import score_with_model  # noqa: E402
from telegram_implementation.accountability_service import (  # noqa: E402
    add_item,
    send_item_to_chat,
    update_tracking_message,
)
from telegram_implementation.event_log import log_event  # noqa: E402
from telegram_implementation.notify import build_recommended_keyboard  # noqa: E402


MODEL_SEARCHES = (
    "griffati_donna_all",
    "griffati_uomo_all",
    "gucci",
    "nike",
    "prada",
    "ps4",
    "donna_accessori_gioielli",
    "telefoni",
    "hobby_collezionismo",
)

BASIC5_APPROACH_NAMES = (
    "numeric_tree_v1",
    "random_forest_basic_v1",
    "hist_gradient_basic_numeric_v1",
    "xgboost_basic_v1",
)

OFFLINE_RUN_NAME = "basic_5_giant_20260525_185552"
NEW_SEARCHES_OFFLINE_RUN_NAME = "basic_5_giant_20260614_221642"
MODEL_GROUPS = (
    {
        "offline_run": OFFLINE_RUN_NAME,
        "searches": (
            "griffati_donna_all",
            "griffati_uomo_all",
            "gucci",
            "nike",
            "prada",
            "ps4",
        ),
    },
    {
        "offline_run": NEW_SEARCHES_OFFLINE_RUN_NAME,
        "searches": (
            "donna_accessori_gioielli",
            "telefoni",
            "hobby_collezionismo",
        ),
    },
)
SEED = 42
SCORE_PREFIX = "score__"
THRESHOLD_PREFIX = "threshold__"
PASS_PREFIX = "pass__"
ALL_SEARCHES = "__all__"
TELEGRAM_SENT_LOG = EXPERIMENT_ROOT / "live_scoring" / "telegram_sent_items.csv"
TELEGRAM_MIN_PRICE_EUR = 30.0
TELEGRAM_POLICY_DESCRIPTION = "any normal giant-model pass plus price > 30 EUR"


def sanitize_search_name(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return text.strip("_") or "unknown"


def search_onehot_column(search: str) -> str:
    return f"search__{sanitize_search_name(search)}"


def add_search_onehot(df: pd.DataFrame, searches: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    search_values = out["SearchName"].fillna("").astype(str)
    for search in searches:
        out[search_onehot_column(search)] = search_values.eq(str(search)).astype(int)
    return out


def model_basename(approach: str, offline_run: str = OFFLINE_RUN_NAME) -> str:
    return f"{offline_run}_global_basic_5_{approach}_seed{SEED}"


def model_path(approach: str, offline_run: str = OFFLINE_RUN_NAME) -> Path:
    return MODELS_DIR / f"{model_basename(approach, offline_run)}.pkl"


def metadata_path(approach: str, offline_run: str = OFFLINE_RUN_NAME) -> Path:
    return MODELS_DIR / f"{model_basename(approach, offline_run)}_metadata.json"


def load_model(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_live_tracked(live_run_dir: Path) -> pd.DataFrame:
    path = live_run_dir / "tracked_items.csv"
    if not path.exists():
        raise FileNotFoundError(f"No tracked_items.csv found in {live_run_dir}")
    return pd.read_csv(path, low_memory=False)


def offline_run_dir(offline_run: str = OFFLINE_RUN_NAME) -> Path:
    return OFFLINE_RUNS_DIR / offline_run


def load_threshold_table(offline_run: str = OFFLINE_RUN_NAME) -> pd.DataFrame:
    path = offline_run_dir(offline_run) / "per_search_threshold_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing per-search threshold table: {path}")
    return pd.read_csv(path, low_memory=False)


def thresholds_for_approach(
    approach: str,
    threshold_table: pd.DataFrame,
    metadata: dict[str, Any],
    searches: tuple[str, ...],
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    scoped = threshold_table[threshold_table["approach"].astype(str).eq(approach)]
    for _, row in scoped.iterrows():
        thresholds[str(row["search_name"])] = float(row["per_search_threshold"])
    fallback = float(metadata.get("threshold", 1.0))
    for search in searches:
        thresholds.setdefault(search, fallback)
    return thresholds


def selected_approaches(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return BASIC5_APPROACH_NAMES
    unknown = sorted(set(values) - set(BASIC5_APPROACH_NAMES))
    if unknown:
        raise ValueError(f"Unknown Basic5 approaches: {', '.join(unknown)}")
    return tuple(values)


def load_model_records(
    approaches: tuple[str, ...],
    *,
    offline_run: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for approach in approaches:
        groups: list[dict[str, Any]] = []
        model_groups = [group for group in MODEL_GROUPS if offline_run is None or group["offline_run"] == offline_run]
        for group in model_groups:
            group_run = str(group["offline_run"])
            searches = tuple(str(search) for search in group["searches"])
            threshold_table = load_threshold_table(group_run)
            artifact = model_path(approach, group_run)
            metadata_file = metadata_path(approach, group_run)
            if not artifact.exists():
                raise FileNotFoundError(f"Model not found for {approach}: {artifact}")
            if not metadata_file.exists():
                raise FileNotFoundError(f"Metadata not found for {approach}: {metadata_file}")
            metadata = load_json(metadata_file)
            groups.append(
                {
                    "offline_run": group_run,
                    "searches": searches,
                    "artifact_path": artifact,
                    "metadata_path": metadata_file,
                    "metadata": metadata,
                    "thresholds": thresholds_for_approach(approach, threshold_table, metadata, searches),
                }
            )
        records.append({"approach": approach, "groups": groups})
    return records


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the frame has the columns expected by all Basic5 model families."""
    out = df.copy()
    if "SearchName" not in out.columns:
        out["SearchName"] = ""
    out["SearchName"] = out["SearchName"].fillna("").astype(str)

    for col in ["Title", "Brand", "Size"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    for col in ["Price", "Likes"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    if "Page" not in out.columns:
        out["Page"] = np.nan
    out["Page"] = pd.to_numeric(out["Page"], errors="coerce")

    out = base_sweep.add_engineered_snapshot_features(out)
    return add_search_onehot(out, MODEL_SEARCHES)


def ensure_model_columns(frame: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    for col in metadata.get("numeric_features", []):
        if col not in out.columns:
            out[col] = np.nan
    for col in metadata.get("text_features", []):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    return out


def apply_models(
    tracked: pd.DataFrame,
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    scored = prepare_feature_frame(tracked)
    search_values = scored["SearchName"].astype(str)

    for record in records:
        approach = str(record["approach"])
        score_col = f"{SCORE_PREFIX}{approach}"
        threshold_col = f"{THRESHOLD_PREFIX}{approach}"
        pass_col = f"{PASS_PREFIX}{approach}"

        scored[score_col] = np.nan
        scored[threshold_col] = np.nan
        scored[pass_col] = False

        for group in record.get("groups", []):
            group_searches = set(str(search) for search in group["searches"])
            group_mask = search_values.isin(group_searches)
            if not group_mask.any():
                continue

            model = load_model(group["artifact_path"])
            feature_frame = ensure_model_columns(scored.loc[group_mask].copy(), group["metadata"])
            probs = np.clip(np.asarray(score_with_model(model, feature_frame), dtype=float), 0.0, 1.0)
            scored.loc[group_mask, score_col] = probs

            for search, threshold in group["thresholds"].items():
                search_mask = group_mask & search_values.eq(search)
                scored.loc[search_mask, threshold_col] = threshold
                scored.loc[search_mask, pass_col] = scored.loc[search_mask, score_col] >= threshold

    # Backwards-compatible columns for the first live XGBoost view/report.
    xgb = "xgboost_basic_v1"
    if f"{SCORE_PREFIX}{xgb}" in scored.columns:
        scored["xgboost_score"] = scored[f"{SCORE_PREFIX}{xgb}"]
        scored["xgboost_threshold"] = scored[f"{THRESHOLD_PREFIX}{xgb}"]
        scored["xgboost_pass"] = scored[f"{PASS_PREFIX}{xgb}"]
    return scored


def coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "yes", "sold"})


def item_identity(row: pd.Series | dict[str, Any]) -> str:
    for col in ("item_id", "Dataid", "dataid", "Link"):
        value = row.get(col) if isinstance(row, dict) else row.get(col)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    title = row.get("Title", "") if isinstance(row, dict) else row.get("Title", "")
    price = row.get("Price", "") if isinstance(row, dict) else row.get("Price", "")
    return f"{title}|{price}".strip()


def pass_columns_for_records(scored: pd.DataFrame, records: list[dict[str, Any]]) -> list[str]:
    return [f"{PASS_PREFIX}{record['approach']}" for record in records if f"{PASS_PREFIX}{record['approach']}" in scored.columns]


def telegram_price_series(scored: pd.DataFrame) -> pd.Series:
    if "price_num" in scored.columns:
        return pd.to_numeric(scored["price_num"], errors="coerce")
    if "Price" in scored.columns:
        return pd.to_numeric(scored["Price"], errors="coerce")
    return pd.Series(np.nan, index=scored.index)


def build_telegram_candidates(scored: pd.DataFrame, records: list[dict[str, Any]]) -> pd.DataFrame:
    """Return one row per item matching normal giant model thresholds and price >30 EUR."""
    pass_cols = pass_columns_for_records(scored, records)
    if not pass_cols:
        return pd.DataFrame()
    price = telegram_price_series(scored)
    pass_frame = pd.DataFrame({col: coerce_bool(scored[col]) for col in pass_cols}, index=scored.index)
    eligible = pass_frame.any(axis=1) & price.gt(TELEGRAM_MIN_PRICE_EUR)
    candidates = scored[eligible].copy()
    if candidates.empty:
        return candidates

    rows: list[pd.Series] = []
    approaches = [str(record["approach"]) for record in records]
    for _, row in candidates.iterrows():
        passed_models = [approach for approach in approaches if bool(row.get(f"{PASS_PREFIX}{approach}", False))]
        best_model = ""
        best_score = np.nan
        best_threshold = np.nan
        best_margin = -np.inf
        model_bits = []
        for approach in passed_models:
            score = pd.to_numeric(pd.Series([row.get(f"{SCORE_PREFIX}{approach}")]), errors="coerce").iloc[0]
            threshold = pd.to_numeric(pd.Series([row.get(f"{THRESHOLD_PREFIX}{approach}")]), errors="coerce").iloc[0]
            if pd.isna(score) or pd.isna(threshold):
                continue
            margin = float(score - threshold)
            model_bits.append(f"{approach}:{float(score):.3f}/{float(threshold):.3f}")
            if margin > best_margin:
                best_margin = margin
                best_model = approach
                best_score = float(score)
                best_threshold = float(threshold)

        out = row.copy()
        out["GiantPassedModels"] = ", ".join(passed_models)
        out["GiantPassedModelCount"] = int(len(passed_models))
        out["TelegramPassedModels"] = ", ".join(passed_models)
        out["TelegramModel"] = best_model
        out["TelegramModelScore"] = best_score
        out["TelegramOriginalThreshold"] = best_threshold
        out["TelegramAdjustedThreshold"] = best_threshold
        out["TelegramThresholdAdjustment"] = 0.0
        out["TelegramFilterReason"] = (
            f"price>{TELEGRAM_MIN_PRICE_EUR:.2f}; "
            f"normal giant pass; best={best_model} score={float(best_score):.3f} "
            f"threshold={float(best_threshold):.3f}"
        )
        out["GiantBestModel"] = best_model
        out["GiantBestScore"] = best_score
        out["GiantBestThreshold"] = best_threshold
        out["GiantBestMargin"] = (
            best_margin
            if np.isfinite(best_margin)
            else np.nan
        )
        out["GiantModelScores"] = "; ".join(model_bits)
        out["RecommendationReason"] = (
            f"Giant model pass: {len(passed_models)} model(s); "
            f"best={best_model} score={best_score:.3f} threshold={best_threshold:.3f}"
            if best_model
            else "Giant model pass"
        )
        out["TelegramItemKey"] = item_identity(out)
        rows.append(out)

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["GiantPassedModelCount", "GiantBestMargin", "GiantBestScore"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def load_telegram_sent_log(path: Path = TELEGRAM_SENT_LOG) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["item_key", "sent_at", "source_run", "models", "search_name", "title", "link"])
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def append_telegram_sent_rows(rows: list[dict[str, Any]], path: Path = TELEGRAM_SENT_LOG) -> None:
    if not rows:
        return
    path = assert_experiment_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = load_telegram_sent_log(path)
    updated = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True)
    updated.drop_duplicates(subset=["item_key"], keep="last").to_csv(path, index=False)


async def send_recommended_item(token: str, chat_id: int | str, item: dict[str, Any]) -> tuple[str, int | None]:
    cache_key = add_item(item, status="recommended")
    message_id = await send_item_to_chat(
        token,
        chat_id,
        item,
        build_recommended_keyboard(cache_key),
        cache_key=cache_key,
    )
    await update_tracking_message(token, cache_key)
    return cache_key, message_id


def send_candidates_to_telegram(
    candidates: pd.DataFrame,
    *,
    source_run: str,
    dry_run: bool = False,
    limit: int | None = None,
    sent_log_path: Path = TELEGRAM_SENT_LOG,
) -> dict[str, Any]:
    if candidates.empty:
        return {"status": "no_candidates", "candidates": 0, "already_sent": 0, "sent": 0, "dry_run": dry_run}

    sent_log = load_telegram_sent_log(sent_log_path)
    sent_keys = set(sent_log.get("item_key", pd.Series(dtype=str)).astype(str))
    candidates = candidates[~candidates["TelegramItemKey"].astype(str).isin(sent_keys)].copy()
    if limit is not None:
        candidates = candidates.head(max(0, int(limit))).copy()

    token = settings.telegram.bot_token
    chat_id = settings.telegram.resolved_recommended_deals_chat_id
    if chat_id is None and settings.telegram.chat_id:
        try:
            chat_id = int(str(settings.telegram.chat_id).strip())
        except ValueError:
            chat_id = None
    if not dry_run and (not token or chat_id is None):
        raise RuntimeError("Telegram bot token/chat is not configured.")

    sent_rows: list[dict[str, Any]] = []
    sent_count = 0
    errors: list[str] = []
    now = datetime.now(timezone.utc).isoformat()
    for _, row in candidates.iterrows():
        item = row.to_dict()
        item_key = str(row["TelegramItemKey"])
        if dry_run:
            sent_rows.append(
                {
                    "item_key": item_key,
                    "sent_at": now,
                    "source_run": source_run,
                    "models": str(row.get("TelegramPassedModels", row.get("GiantPassedModels", ""))),
                    "search_name": str(row.get("SearchName", "")),
                    "title": str(row.get("Title", "")),
                    "link": str(row.get("Link", "")),
                    "dry_run": "true",
                }
            )
            continue
        try:
            cache_key, message_id = asyncio.run(send_recommended_item(str(token), chat_id, item))
            sent_count += 1
            sent_rows.append(
                {
                    "item_key": item_key,
                    "sent_at": datetime.now(timezone.utc).isoformat(),
                    "source_run": source_run,
                    "models": str(row.get("TelegramPassedModels", row.get("GiantPassedModels", ""))),
                    "search_name": str(row.get("SearchName", "")),
                    "title": str(row.get("Title", "")),
                    "link": str(row.get("Link", "")),
                    "cache_key": cache_key,
                    "message_id": "" if message_id is None else str(message_id),
                    "dry_run": "false",
                }
            )
            log_event(
                "giant_model_recommendation_sent",
                item=item,
                cache_key=cache_key,
                details={
                    "source_run": source_run,
                    "models": row.get("TelegramPassedModels", row.get("GiantPassedModels", "")),
                    "best_model": row.get("GiantBestModel", ""),
                    "message_id": message_id,
                },
            )
        except Exception as exc:
            errors.append(f"{item_key}: {exc}")
            log_event(
                "giant_model_recommendation_error",
                item=item,
                details={"source_run": source_run, "error": str(exc)},
            )

    if not dry_run:
        append_telegram_sent_rows(sent_rows, sent_log_path)

    return {
        "status": "dry_run" if dry_run else ("sent_with_errors" if errors else "sent"),
        "candidates": int(len(candidates)),
        "already_sent": int(len(sent_keys)),
        "sent": int(len(sent_rows) if dry_run else sent_count),
        "dry_run": bool(dry_run),
        "chat_id": str(chat_id) if chat_id is not None else "",
        "sent_log": str(sent_log_path),
        "errors": errors,
    }


def outcome_windows(scored: pd.DataFrame) -> list[int]:
    return sorted(
        {
            int(match.group(1))
            for col in scored.columns
            for match in [re.fullmatch(r"sold_within_(\d+)h", col)]
            if match and f"evaluated_at_{match.group(1)}h" in scored.columns
        }
    )


def build_summary(scored: pd.DataFrame, records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        approach = str(record["approach"])
        score_col = f"{SCORE_PREFIX}{approach}"
        threshold_col = f"{THRESHOLD_PREFIX}{approach}"
        pass_col = f"{PASS_PREFIX}{approach}"
        for search in MODEL_SEARCHES:
            scope = scored[scored["SearchName"].astype(str).eq(search)]
            scores = pd.to_numeric(scope[score_col], errors="coerce")
            threshold = (
                float(scope[threshold_col].dropna().iloc[0])
                if threshold_col in scope.columns and scope[threshold_col].notna().any()
                else np.nan
            )
            passed = scope[scope[pass_col] == True]
            rows.append(
                {
                    "approach": approach,
                    "search_name": search,
                    "threshold": threshold,
                    "total_tracked": int(len(scope)),
                    "passed": int(len(passed)),
                    "pass_rate": float(len(passed) / len(scope)) if len(scope) else 0.0,
                    "mean_score": float(scores.mean()) if not scores.empty else np.nan,
                    "p50_score": float(scores.quantile(0.50)) if not scores.empty else np.nan,
                    "p90_score": float(scores.quantile(0.90)) if not scores.empty else np.nan,
                    "p95_score": float(scores.quantile(0.95)) if not scores.empty else np.nan,
                    "p99_score": float(scores.quantile(0.99)) if not scores.empty else np.nan,
                    "max_score": float(scores.max()) if not scores.empty else np.nan,
                    "max_margin": float(scores.max() - threshold) if pd.notna(threshold) and not scores.empty else np.nan,
                    "within_0_05_below": int(((scores < threshold) & (scores >= threshold - 0.05)).sum())
                    if pd.notna(threshold)
                    else 0,
                    "within_0_10_below": int(((scores < threshold) & (scores >= threshold - 0.10)).sum())
                    if pd.notna(threshold)
                    else 0,
                }
            )
    return pd.DataFrame(rows)


def passed_window_performance(scored: pd.DataFrame, records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    windows = outcome_windows(scored)
    scopes = (ALL_SEARCHES, *MODEL_SEARCHES)
    for record in records:
        approach = str(record["approach"])
        pass_col = f"{PASS_PREFIX}{approach}"
        for search in scopes:
            if search == ALL_SEARCHES:
                search_scope = scored
            else:
                search_scope = scored[scored["SearchName"].astype(str).eq(search)]
            passed = search_scope[search_scope[pass_col] == True]
            for hours in windows:
                eval_col = f"evaluated_at_{hours}h"
                sold_col = f"sold_within_{hours}h"
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


def write_report(
    path: Path,
    *,
    offline_run: str | None,
    live_run_dir: Path,
    summary: pd.DataFrame,
    performance: pd.DataFrame,
    telegram_candidates: pd.DataFrame,
    telegram_result: dict[str, Any] | None,
    reranker_status: dict[str, Any] | None,
) -> None:
    def fmt(value: object, decimals: int = 3) -> str:
        try:
            if value is None or pd.isna(value):
                return "-"
            return f"{float(value):.{decimals}f}"
        except Exception:
            return str(value)

    focus_hours = [h for h in (24, 48, 72) if h in set(performance["hours"].dropna().astype(int))]
    overall = performance[performance["search_name"].eq(ALL_SEARCHES)].copy()
    totals = summary.groupby("approach", as_index=False)["passed"].sum().rename(columns={"passed": "total_passed"})

    lines = [
        "# Basic 5 Giant Model - Live Collector Scoring",
        "",
        f"Offline run: `{offline_run}`",
        f"Live collector: `{live_run_dir.name}`",
        "",
        "## Model Overview",
        "",
        "| model | passed | 24h precision | 48h precision | 72h precision |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in totals.sort_values("total_passed", ascending=False).iterrows():
        approach = row["approach"]
        values = {}
        for hours in focus_hours:
            perf = overall[(overall["approach"].eq(approach)) & (overall["hours"].eq(hours))]
            values[hours] = fmt(perf["precision"].iloc[0]) if not perf.empty else "-"
        lines.append(
            f"| {approach} | {int(row['total_passed'])} | "
            f"{values.get(24, '-')} | {values.get(48, '-')} | {values.get(72, '-')} |"
        )

    lines.extend(["", "## Per-Search Pass Counts", ""])
    pivot = summary.pivot_table(index="approach", columns="search_name", values="passed", aggfunc="sum", fill_value=0)
    search_columns = [search for search in MODEL_SEARCHES if search in pivot.columns]
    lines.append("| model | " + " | ".join(search_columns) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in search_columns) + " |")
    for approach, row in pivot.loc[sorted(pivot.index)].iterrows():
        lines.append("| " + str(approach) + " | " + " | ".join(str(int(row.get(search, 0))) for search in search_columns) + " |")

    lines.extend(
        [
            "",
            "## Telegram Recommendations",
            "",
            f"- Candidate items passing Telegram filters: `{len(telegram_candidates)}`",
            f"- Candidate policy: `{TELEGRAM_POLICY_DESCRIPTION}`",
            f"- Full-scrape reranker status: `{json.dumps(reranker_status or {'status': 'not_applied'}, sort_keys=True)}`",
            f"- Send status: `{(telegram_result or {'status': 'not_requested'}).get('status')}`",
            f"- Sent/dry-run count: `{(telegram_result or {}).get('sent', 0)}`",
            "",
            "",
            "## Files",
            "",
            "- `live_scored_items.csv`: item-level scores, thresholds, and pass flags for all models.",
            "- `live_scored_summary.csv`: per-model, per-search score and threshold summary.",
            "- `live_scored_window_performance.csv`: sold precision by model, search, and outcome window.",
            "- `telegram_candidates.csv`: one row per item passing at least one normal giant-model threshold and the price filter.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    out_dir: Path,
    scored: pd.DataFrame,
    records: list[dict[str, Any]],
    *,
    offline_run: str | None,
    live_run_dir: Path,
    telegram_result: dict[str, Any] | None = None,
    reranker_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = assert_experiment_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored_path = out_dir / "live_scored_items.csv"
    scored.to_csv(scored_path, index=False)

    summary = build_summary(scored, records)
    summary_path = out_dir / "live_scored_summary.csv"
    summary.to_csv(summary_path, index=False)

    performance = passed_window_performance(scored, records)
    performance_path = out_dir / "live_scored_window_performance.csv"
    performance.to_csv(performance_path, index=False)

    telegram_candidates = build_telegram_candidates(scored, records)
    telegram_candidates_path = out_dir / "telegram_candidates.csv"
    telegram_candidates.to_csv(telegram_candidates_path, index=False)

    report_path = out_dir / "live_scored_report.md"
    write_report(
        report_path,
        offline_run=offline_run,
        live_run_dir=live_run_dir,
        summary=summary,
        performance=performance,
        telegram_candidates=telegram_candidates,
        telegram_result=telegram_result,
        reranker_status=reranker_status,
    )

    manifest = {
        "offline_run": offline_run or "configured_model_groups",
        "model_groups": [
            {"offline_run": str(group["offline_run"]), "searches": list(group["searches"])}
            for group in MODEL_GROUPS
        ],
        "live_run_dir": str(live_run_dir),
        "approaches": [str(record["approach"]) for record in records],
        "models": {
            str(record["approach"]): {
                "groups": [
                    {
                        "offline_run": str(group["offline_run"]),
                        "searches": list(group["searches"]),
                        "artifact_path": str(group["artifact_path"]),
                        "metadata_path": str(group["metadata_path"]),
                        "thresholds": group["thresholds"],
                    }
                    for group in record.get("groups", [])
                ],
            }
            for record in records
        },
        "outputs": {
            "scored": str(scored_path),
            "summary": str(summary_path),
            "performance": str(performance_path),
            "telegram_candidates": str(telegram_candidates_path),
            "report": str(report_path),
        },
        "telegram": telegram_result or {"status": "not_requested"},
        "full_scrape_reranker": reranker_status or {"status": "not_applied"},
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    live_run_dir = args.live_run_dir
    out_dir = args.out_dir or (EXPERIMENT_ROOT / "live_scoring" / run_id("live_scoring"))
    approaches = selected_approaches(args.approach)

    print(f"[apply_live] loading {len(approaches)} Basic5 giant models", flush=True)
    records = load_model_records(approaches, offline_run=args.offline_run)

    print(f"[apply_live] loading tracked items from {live_run_dir}", flush=True)
    tracked = load_live_tracked(live_run_dir)
    print(f"[apply_live] {len(tracked)} tracked rows", flush=True)

    print("[apply_live] scoring...", flush=True)
    scored = apply_models(tracked, records)
    reranker_status = {"status": "not_applied", "reason": TELEGRAM_POLICY_DESCRIPTION}

    for record in records:
        approach = str(record["approach"])
        total_passed = int((scored[f"{PASS_PREFIX}{approach}"] == True).sum())
        print(f"[apply_live] {approach}: {total_passed} passed per-search thresholds", flush=True)

    telegram_result: dict[str, Any] | None = None
    if args.send_telegram or args.telegram_dry_run:
        candidates = build_telegram_candidates(scored, records)
        print(
            f"[apply_live] {len(candidates)} unique items passed Telegram filters "
            f"(normal giant pass, price>{TELEGRAM_MIN_PRICE_EUR:.2f}€)",
            flush=True,
        )
        telegram_result = send_candidates_to_telegram(
            candidates,
            source_run=out_dir.name,
            dry_run=bool(args.telegram_dry_run),
            limit=args.telegram_limit,
        )
        print(f"[apply_live] telegram: {json.dumps(telegram_result, ensure_ascii=False)}", flush=True)

    manifest = write_outputs(
        out_dir,
        scored,
        records,
        offline_run=args.offline_run,
        live_run_dir=live_run_dir,
        telegram_result=telegram_result,
        reranker_status=reranker_status,
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--offline-run", default=None)
    parser.add_argument("--approach", action="append", choices=BASIC5_APPROACH_NAMES)
    parser.add_argument("--send-telegram", action="store_true", help="Send newly passing items to the recommended deals Telegram chat.")
    parser.add_argument("--telegram-dry-run", action="store_true", help="Count Telegram candidates without sending messages.")
    parser.add_argument("--telegram-limit", type=int, default=None, help="Optional max number of new Telegram recommendations to send.")
    parser.add_argument("--run-loop", action="store_true", help="Keep rescoring the live collector and sending newly passing items.")
    parser.add_argument("--sleep-seconds", type=float, default=7200.0, help="Sleep between run-loop scoring passes.")
    args = parser.parse_args()

    ensure_experiment_dirs()
    if args.run_loop and args.out_dir is not None:
        raise ValueError("--out-dir cannot be used with --run-loop because each loop pass writes a new live_scoring folder.")
    if args.run_loop:
        while True:
            run_once(args)
            print(f"[apply_live] sleeping {args.sleep_seconds:.1f}s before next scoring pass", flush=True)
            time.sleep(max(1.0, float(args.sleep_seconds)))
    run_once(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
