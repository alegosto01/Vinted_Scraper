"""SQLite experiment tracker.

Logs experiment runs, metrics, parameters, scraper iterations, and eventual-sale
checks to a single SQLite file at data/experiments/experiments.db.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DB_PATH = ROOT / "data" / "experiments" / "experiments.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Thread-local connection pool for concurrent use
_local = threading.local()


@contextmanager
def _get_conn():
    """Yield a SQLite connection with WAL mode and foreign keys."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
        _local.conn.row_factory = sqlite3.Row
    try:
        yield _local.conn
    except Exception:
        _local.conn.rollback()
        raise


def init_db() -> None:
    """Create tables if they do not exist. Safe to call repeatedly."""
    with _get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id            TEXT PRIMARY KEY,
                family            TEXT NOT NULL,
                command           TEXT NOT NULL,
                status            TEXT DEFAULT 'unknown',
                created_at        TIMESTAMP,
                finished_at       TIMESTAMP,
                run_dir           TEXT,
                git_branch        TEXT,
                git_head          TEXT,
                git_dirty         BOOLEAN,
                promotion_count   INTEGER
            );

            CREATE TABLE IF NOT EXISTS run_metrics (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id            TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
                search_name       TEXT,
                approach          TEXT,
                status            TEXT,
                seed              INTEGER,
                feature_policy    TEXT,
                live_ready        BOOLEAN,
                qualified         BOOLEAN,
                threshold         REAL,
                train_rows        INTEGER,
                validation_rows   INTEGER,
                test_rows         INTEGER,
                validation_precision REAL,
                test_precision    REAL,
                test_precision_at_10 REAL,
                test_precision_at_25 REAL,
                test_precision_at_50 REAL,
                test_roc_auc      REAL,
                test_pr_auc       REAL,
                test_lift_over_base REAL,
                fit_seconds       REAL,
                promotion_failures TEXT,
                reason            TEXT,
                task              TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_metrics_run ON run_metrics(run_id);
            CREATE INDEX IF NOT EXISTS idx_metrics_search ON run_metrics(search_name);
            CREATE INDEX IF NOT EXISTS idx_metrics_approach ON run_metrics(approach);

            CREATE TABLE IF NOT EXISTS run_parameters (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id            TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
                param_name        TEXT,
                param_value       TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_params_run ON run_parameters(run_id);

            CREATE TABLE IF NOT EXISTS live_run_status (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id            TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
                checked_at        TIMESTAMP,
                checked_rows      INTEGER,
                due_rows          INTEGER,
                sold_count        INTEGER,
                unsold_count      INTEGER
            );

            CREATE TABLE IF NOT EXISTS scraper_iterations (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration         INTEGER NOT NULL,
                started_at        TIMESTAMP,
                finished_at       TIMESTAMP,
                searches_due      INTEGER,
                searches_scraped  INTEGER,
                total_new_items   INTEGER,
                total_sold_found  INTEGER,
                proxy_datacenter_hits INTEGER,
                proxy_residential_hits INTEGER,
                proxy_errors      INTEGER,
                log_tail_hash     TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_scraper_iter ON scraper_iterations(iteration);

            CREATE TABLE IF NOT EXISTS scrape_events (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration_id      INTEGER REFERENCES scraper_iterations(id) ON DELETE CASCADE,
                search_name       TEXT,
                scraped_at        TIMESTAMP,
                pages_scraped     INTEGER,
                new_items         INTEGER,
                sold_found        INTEGER,
                proxy_type        TEXT,
                error             TEXT,
                schedule_delay_seconds INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_scrape_event_iter ON scrape_events(iteration_id);
            CREATE INDEX IF NOT EXISTS idx_scrape_event_search ON scrape_events(search_name);

            CREATE TABLE IF NOT EXISTS eventual_sale_checks (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                search_name       TEXT,
                checked_at        TIMESTAMP,
                source            TEXT,
                items_checked     INTEGER,
                sold_found        INTEGER,
                still_unsold      INTEGER,
                errors            INTEGER,
                duration_seconds  REAL
            );

            CREATE INDEX IF NOT EXISTS idx_esc_search ON eventual_sale_checks(search_name);
            CREATE INDEX IF NOT EXISTS idx_esc_source ON eventual_sale_checks(source);
            """
        )
        conn.commit()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_run(metadata: dict[str, Any]) -> str:
    """Insert or replace a run row. Returns run_id."""
    run_id = metadata["run_id"]
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs
            (run_id, family, command, status, created_at, finished_at, run_dir,
             git_branch, git_head, git_dirty, promotion_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                metadata.get("family", "unknown"),
                metadata.get("command", "unknown"),
                metadata.get("status", "unknown"),
                metadata.get("created_at") or _utc_now(),
                metadata.get("finished_at") or _utc_now(),
                metadata.get("run_dir"),
                metadata.get("git_branch"),
                metadata.get("git_head"),
                metadata.get("git_dirty"),
                metadata.get("promotion_count"),
            ),
        )
        conn.commit()
    return run_id


def log_metrics(run_id: str, metrics_df: pd.DataFrame) -> int:
    """Bulk-insert metrics from a DataFrame. Returns number of rows inserted."""
    if metrics_df is None or metrics_df.empty:
        return 0

    # Normalize column names
    col_map = {
        "search_name": "search_name",
        "approach": "approach",
        "status": "status",
        "seed": "seed",
        "feature_policy": "feature_policy",
        "live_ready": "live_ready",
        "qualified_for_paper_trading": "qualified",
        "threshold": "threshold",
        "train_rows": "train_rows",
        "validation_rows": "validation_rows",
        "test_rows": "test_rows",
        "validation_precision": "validation_precision",
        "test_precision": "test_precision",
        "test_precision_at_10": "test_precision_at_10",
        "test_precision_at_25": "test_precision_at_25",
        "test_precision_at_50": "test_precision_at_50",
        "test_roc_auc": "test_roc_auc",
        "test_pr_auc": "test_pr_auc",
        "test_lift_over_base": "test_lift_over_base",
        "fit_seconds": "fit_seconds",
        "promotion_failures": "promotion_failures",
        "reason": "reason",
        "task": "task",
    }

    records = []
    for _, row in metrics_df.iterrows():
        rec: dict[str, Any] = {"run_id": run_id}
        for csv_col, db_col in col_map.items():
            val = row.get(csv_col)
            if pd.isna(val):
                val = None
            elif db_col in ("live_ready", "qualified"):
                val = bool(val)
            rec[db_col] = val
        records.append(rec)

    if not records:
        return 0

    with _get_conn() as conn:
        conn.execute("DELETE FROM run_metrics WHERE run_id = ?", (run_id,))
        conn.executemany(
            """
            INSERT INTO run_metrics
            (run_id, search_name, approach, status, seed, feature_policy, live_ready,
             qualified, threshold, train_rows, validation_rows, test_rows,
             validation_precision, test_precision, test_precision_at_10,
             test_precision_at_25, test_precision_at_50, test_roc_auc, test_pr_auc,
             test_lift_over_base, fit_seconds, promotion_failures, reason, task)
            VALUES
            (:run_id, :search_name, :approach, :status, :seed, :feature_policy, :live_ready,
             :qualified, :threshold, :train_rows, :validation_rows, :test_rows,
             :validation_precision, :test_precision, :test_precision_at_10,
             :test_precision_at_25, :test_precision_at_50, :test_roc_auc, :test_pr_auc,
             :test_lift_over_base, :fit_seconds, :promotion_failures, :reason, :task)
            """,
            records,
        )
        conn.commit()
    return len(records)


def log_parameters(run_id: str, params: dict[str, Any]) -> int:
    """Bulk-insert parameters. Returns number of rows inserted."""
    if not params:
        return 0
    records = [
        {"run_id": run_id, "param_name": k, "param_value": json.dumps(v) if not isinstance(v, str) else v}
        for k, v in params.items()
    ]
    with _get_conn() as conn:
        conn.execute("DELETE FROM run_parameters WHERE run_id = ?", (run_id,))
        conn.executemany(
            "INSERT INTO run_parameters (run_id, param_name, param_value) VALUES (:run_id, :param_name, :param_value)",
            records,
        )
        conn.commit()
    return len(records)


def log_scraper_iteration(summary: dict[str, Any]) -> int:
    """Log a main.py scheduler iteration and its per-search events.
    Returns the iteration_db_id.
    """
    with _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO scraper_iterations
            (iteration, started_at, finished_at, searches_due, searches_scraped,
             total_new_items, total_sold_found, proxy_datacenter_hits,
             proxy_residential_hits, proxy_errors, log_tail_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary.get("iteration"),
                summary.get("started_at") or _utc_now(),
                summary.get("finished_at") or _utc_now(),
                summary.get("searches_due"),
                summary.get("searches_scraped"),
                summary.get("total_new_items"),
                summary.get("total_sold_found"),
                summary.get("proxy_datacenter_hits"),
                summary.get("proxy_residential_hits"),
                summary.get("proxy_errors"),
                summary.get("log_tail_hash"),
            ),
        )
        iteration_id = cur.lastrowid

        for event in summary.get("events", []):
            conn.execute(
                """
                INSERT INTO scrape_events
                (iteration_id, search_name, scraped_at, pages_scraped, new_items,
                 sold_found, proxy_type, error, schedule_delay_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    iteration_id,
                    event.get("search_name"),
                    event.get("scraped_at") or _utc_now(),
                    event.get("pages_scraped"),
                    event.get("new_items"),
                    event.get("sold_found"),
                    event.get("proxy_type"),
                    event.get("error"),
                    event.get("schedule_delay_seconds"),
                ),
            )
        conn.commit()
    return iteration_id


def log_eventual_sale_check(summary: dict[str, Any]) -> int:
    """Log a background eventual-sale check batch. Returns row id."""
    with _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO eventual_sale_checks
            (search_name, checked_at, source, items_checked, sold_found,
             still_unsold, errors, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary.get("search_name"),
                summary.get("checked_at") or _utc_now(),
                summary.get("source", "unknown"),
                summary.get("items_checked"),
                summary.get("sold_found"),
                summary.get("still_unsold"),
                summary.get("errors"),
                summary.get("duration_seconds"),
            ),
        )
        conn.commit()
        return cur.lastrowid


# ── Query utilities ──────────────────────────────────────────────────────────

def get_best_runs(
    search_name: str | None = None,
    metric: str = "test_precision",
    top_n: int = 5,
    days: int | None = None,
) -> list[dict[str, Any]]:
    """Return top-N runs by a given metric."""
    with _get_conn() as conn:
        sql = f"""
            SELECT r.run_id, r.family, r.command, r.created_at,
                   m.search_name, m.approach, m.{metric}
            FROM runs r
            JOIN run_metrics m ON r.run_id = m.run_id
            WHERE m.status = 'trained'
        """
        params: list[Any] = []
        if search_name:
            sql += " AND m.search_name = ?"
            params.append(search_name)
        if days:
            sql += " AND r.created_at > datetime('now', '-{} days')".format(days)
        sql += f" ORDER BY m.{metric} DESC LIMIT ?"
        params.append(top_n)
        cur = conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def compare_approaches(
    search_name: str,
    approach_a: str,
    approach_b: str,
    metric: str = "test_precision",
) -> list[dict[str, Any]]:
    """Compare two approaches on a single search across all runs."""
    with _get_conn() as conn:
        cur = conn.execute(
            f"""
            SELECT r.run_id, r.family, r.created_at, m.approach, m.{metric}
            FROM runs r
            JOIN run_metrics m ON r.run_id = m.run_id
            WHERE m.search_name = ? AND m.approach IN (?, ?) AND m.status = 'trained'
            ORDER BY r.created_at DESC
            """,
            (search_name, approach_a, approach_b),
        )
        return [dict(row) for row in cur.fetchall()]


def get_runs_since(days: int = 7, family: str | None = None) -> list[dict[str, Any]]:
    with _get_conn() as conn:
        sql = f"""
            SELECT run_id, family, command, status, created_at, promotion_count
            FROM runs
            WHERE created_at > datetime('now', '-{days} days')
        """
        params: list[Any] = []
        if family:
            sql += " AND family = ?"
            params.append(family)
        sql += " ORDER BY created_at DESC"
        cur = conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def get_promotion_candidates(search_name: str | None = None) -> list[dict[str, Any]]:
    with _get_conn() as conn:
        sql = """
            SELECT r.run_id, r.family, r.created_at, m.search_name, m.approach,
                   m.threshold, m.test_precision, m.test_precision_at_10, m.train_rows
            FROM runs r
            JOIN run_metrics m ON r.run_id = m.run_id
            WHERE m.qualified = 1
        """
        params: list[Any] = []
        if search_name:
            sql += " AND m.search_name = ?"
            params.append(search_name)
        sql += " ORDER BY r.created_at DESC"
        cur = conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def query_scraper_health(days: int = 7) -> list[dict[str, Any]]:
    """Daily scrape volume, sold detection, proxy error rate."""
    with _get_conn() as conn:
        cur = conn.execute(
            f"""
            SELECT
                date(started_at) as day,
                COUNT(*) as iterations,
                SUM(total_new_items) as new_items,
                SUM(total_sold_found) as sold_found,
                SUM(COALESCE(proxy_errors, 0)) as proxy_errors,
                SUM(COALESCE(proxy_datacenter_hits, 0)) as datacenter_hits,
                SUM(COALESCE(proxy_residential_hits, 0)) as residential_hits
            FROM scraper_iterations
            WHERE started_at > datetime('now', '-{days} days')
            GROUP BY date(started_at)
            ORDER BY day DESC
            """
        )
        return [dict(row) for row in cur.fetchall()]


def run_exists(run_id: str) -> bool:
    with _get_conn() as conn:
        cur = conn.execute("SELECT 1 FROM runs WHERE run_id = ? LIMIT 1", (run_id,))
        return cur.fetchone() is not None
