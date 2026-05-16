from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import settings


DEFAULT_CACHE_DIR = settings.paths.data_dir / "telegram_implementation" / "pending_items"


def _normalize_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return value
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    return str(value)


def normalize_item_mapping(item: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _normalize_value(value) for key, value in item.items()}


def make_item_cache_key(item: Mapping[str, Any]) -> str:
    normalized = normalize_item_mapping(item)
    preferred = normalized.get("item_id") or normalized.get("Dataid") or normalized.get("Link") or normalized.get("Title") or "item"
    digest_input = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:12]
    return f"{preferred}_{digest}".replace("/", "_").replace(":", "_")[:60]


def cache_item(item: Mapping[str, Any], *, cache_dir: Path = DEFAULT_CACHE_DIR) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized = normalize_item_mapping(item)
    key = make_item_cache_key(normalized)
    payload = {
        "item": normalized,
        "description_payload": None,
    }
    (cache_dir / f"{key}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return key


def load_cached_entry(cache_key: str, *, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, Any]:
    path = cache_dir / f"{cache_key}.json"
    if not path.exists():
        raise FileNotFoundError(f"Cached Telegram item not found: {cache_key}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_cached_item(cache_key: str, *, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, Any]:
    entry = load_cached_entry(cache_key, cache_dir=cache_dir)
    item = entry.get("item")
    if not isinstance(item, dict):
        raise FileNotFoundError(f"Cached Telegram item not found: {cache_key}")
    return item


def save_description_payload(cache_key: str, description_payload: Mapping[str, Any], *, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    entry = load_cached_entry(cache_key, cache_dir=cache_dir)
    entry["description_payload"] = _normalize_value(dict(description_payload))
    path = cache_dir / f"{cache_key}.json"
    path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")


def load_description_payload(cache_key: str, *, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, Any] | None:
    entry = load_cached_entry(cache_key, cache_dir=cache_dir)
    payload = entry.get("description_payload")
    return payload if isinstance(payload, dict) else None
