from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.old.reselling_process.create_description.generate import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OLLAMA_MODEL,
    build_description_payload,
    clean_text,
    ensure_basic_columns,
    load_example_corpus,
    parse_search_name,
)

LOGGER = logging.getLogger(__name__)

# Corpus is large (50-100 MB per search). Cache it for the lifetime of the bot
# process so repeated button presses don't re-read disk on every call.
_CORPUS_CACHE: dict[str, tuple[pd.DataFrame, float]] = {}
_CORPUS_TTL_SECONDS = 3600  # refresh corpus once per hour


def _evict_expired_corpus_entries() -> None:
    """Drop expired entries so the cache doesn't grow unbounded over the bot's lifetime."""
    now = time.monotonic()
    expired = [key for key, (_, loaded_at) in _CORPUS_CACHE.items() if now - loaded_at >= _CORPUS_TTL_SECONDS]
    for key in expired:
        _CORPUS_CACHE.pop(key, None)
    if expired:
        LOGGER.info("Evicted %d expired corpus cache entries (%d remain)", len(expired), len(_CORPUS_CACHE))


def _get_cached_corpus(search_name: str | None, prefer_brand: str | None, all_searches: bool) -> pd.DataFrame:
    _evict_expired_corpus_entries()
    cache_key = f"{search_name or ''}|{prefer_brand or ''}|{all_searches}"
    cached = _CORPUS_CACHE.get(cache_key)
    if cached is not None:
        corpus, loaded_at = cached
        if time.monotonic() - loaded_at < _CORPUS_TTL_SECONDS:
            return corpus
    LOGGER.info("Loading corpus for search=%r brand=%r (cache miss or expired)", search_name, prefer_brand)
    corpus = load_example_corpus(
        search_names=[search_name] if search_name else None,
        prefer_brand=prefer_brand,
        all_searches=all_searches,
    )
    _CORPUS_CACHE[cache_key] = (corpus, time.monotonic())
    return corpus


def build_query_row(item: Mapping[str, Any]) -> pd.Series:
    frame = ensure_basic_columns(pd.DataFrame([dict(item)]))
    return frame.iloc[0]


def generate_description_for_item(
    item: Mapping[str, Any],
    *,
    top_k: int = 5,
    use_ollama: bool = False,
    use_gemini_api: bool | None = None,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    gemini_api_key: str | None = None,
    ollama_language: str = "english",
    ollama_pull_model: bool = False,
    all_searches: bool = False,
) -> dict[str, Any]:
    if use_gemini_api is None:
        use_gemini_api = bool(clean_text(gemini_api_key or os.getenv("GEMINI_API_KEY")))

    query_row = build_query_row(item)
    search_name = parse_search_name(query_row)
    corpus = _get_cached_corpus(
        search_name or None,
        clean_text(query_row.get("Brand")) or None,
        all_searches,
    )
    payload = build_description_payload(
        query_row,
        corpus=corpus,
        top_k=top_k,
        use_ollama=use_ollama and not use_gemini_api,
        use_gemini_api=use_gemini_api,
        ollama_model=ollama_model,
        gemini_model=gemini_model,
        gemini_api_key=gemini_api_key,
        ollama_language=ollama_language,
        ollama_pull_model=ollama_pull_model,
    )
    payload["source_search_name"] = search_name
    return payload
