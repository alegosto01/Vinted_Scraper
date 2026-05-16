from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import settings

EVENT_LOG_PATH = settings.paths.data_dir / "telegram_implementation" / "events.jsonl"


def log_event(
    event_type: str,
    *,
    item: Mapping[str, Any] | None = None,
    cache_key: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "event_type": event_type,
    }
    if cache_key:
        record["cache_key"] = cache_key
    if item:
        record["item_id"] = str(item.get("Dataid") or item.get("item_id") or "")
        record["title"] = str(item.get("Title") or "")
        record["brand"] = str(item.get("Brand") or "")
        record["search"] = str(item.get("SearchName") or "")
        record["price"] = item.get("Price")
        record["link"] = str(item.get("Link") or "")
    if details:
        record["details"] = details
    with open(EVENT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
