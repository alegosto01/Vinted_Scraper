"""Fetch and parse a single Vinted item detail page, with a disk cache.

No proxy, no retries, no sleeps here — pacing and backoff are the caller's job.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from requests_html import HTML

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from full_scraper import Full_Scraper  # noqa: E402

LOG = logging.getLogger("full_item_fetch")

FIELDS = ("item_id", "title", "brand", "color", "category", "description",
          "price", "condition", "photo_urls", "picture_count", "upload_date", "fetched_at")


def _ld_json_product(html_obj: HTML) -> dict:
    for script in html_obj.find('script[type="application/ld+json"]'):
        raw = script.text or ""
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        candidates = payload if isinstance(payload, list) else [payload]
        for obj in candidates:
            if isinstance(obj, dict) and obj.get("@type") == "Product":
                return obj
    return {}


def _selector_text(html_obj: HTML, selector: str) -> str | None:
    element = html_obj.find(selector, first=True)
    return element.text if element else None


def _number(value: object) -> float | None:
    """JSON-LD and the picture counter both hand back strings."""
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _meta(html_obj: HTML, prop: str) -> str | None:
    element = html_obj.find(f'meta[property="{prop}"]', first=True)
    content = (element.attrs.get("content") if element else None) or None
    return content.removesuffix(" | Vinted").strip() if content else None


def parse_item_page(html: str, item_id: str) -> dict:
    html_obj = HTML(html=html)
    product = _ld_json_product(html_obj)

    title = product.get("name") if isinstance(product, dict) else None
    brand = None
    try:
        brand = product.get("brand", {}).get("name")
    except Exception:
        brand = None
    color = product.get("color") if isinstance(product, dict) else None
    category = product.get("category") if isinstance(product, dict) else None
    description = product.get("description") if isinstance(product, dict) else None
    price = None
    try:
        price = product.get("offers", {}).get("price")
    except Exception:
        price = None

    scraper = Full_Scraper.__new__(Full_Scraper)

    # Some item pages ship no ld+json block at all; fall back to the page's own markup.
    if not title:
        element = html_obj.find("h1", first=True)
        title = (element.text.strip() if element else None) or _meta(html_obj, "og:title")
    if not description:
        element = html_obj.find(
            "span[class='web_ui__Text__text web_ui__Text__body web_ui__Text__left web_ui__Text__format']",
            first=True,
        )
        description = (element.text.strip() if element else None) or _meta(html_obj, "og:description")
    if price is None:
        try:
            price = scraper.extract_item_page_price(html_obj)
        except Exception:
            LOG.warning("Item %s: price fallback failed", item_id)

    image_metadata = scraper.extract_item_page_image_metadata(html_obj)
    photo_urls = image_metadata.get("FullImageUrls", []) or []
    picture_count = image_metadata.get("PictureCount", 0) or 0

    condition = _selector_text(html_obj, 'div.details-list__item-value[itemprop="status"] span')
    upload_date = _selector_text(html_obj, 'div.details-list__item-value[itemprop="upload_date"] span')

    return {
        "item_id": item_id,
        "title": title,
        "brand": brand,
        "color": color,
        "category": category,
        "description": description,
        "price": _number(price),
        "condition": condition,
        "photo_urls": photo_urls,
        "picture_count": int(_number(picture_count) or 0),
        "upload_date": upload_date,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def cached_item(item_id: str, cache_dir: Path, max_age_days: float = 14.0) -> dict | None:
    """The cached parse if it is still fresh, so callers can skip pacing entirely."""
    cache_file = Path(cache_dir) / f"{item_id}.json"
    if not cache_file.exists():
        return None
    try:
        cached = json.loads(cache_file.read_text())
        age_days = (datetime.now(timezone.utc)
                    - datetime.fromisoformat(cached["fetched_at"])).total_seconds() / 86400
        return cached if age_days < max_age_days else None
    except Exception as exc:
        LOG.warning("Item %s: bad cache entry, refetching: %s", item_id, exc)
        return None


def fetch_item(
    session,
    item_id: str,
    cache_dir: Path,
    headers: dict,
    max_age_days: float = 14.0,
    on_status=None,
) -> dict | None:
    """`on_status` is called with the HTTP status whenever a live request happens,
    so the caller can back off on 403 without parsing logs."""
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"{item_id}.json"

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            fetched_at = datetime.fromisoformat(cached["fetched_at"])
            age_days = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 86400
            if age_days < max_age_days:
                return cached
        except Exception as exc:
            LOG.warning("Item %s: bad cache entry, refetching: %s", item_id, exc)

    url = f"https://www.vinted.it/items/{item_id}"
    try:
        response = session.get(url, headers=headers, timeout=25)
    except Exception as exc:
        LOG.warning("Item %s: fetch failed: %s", item_id, exc)
        return None

    if on_status is not None:
        on_status(response.status_code)
    if response.status_code != 200:
        LOG.warning("Item %s: HTTP %s", item_id, response.status_code)
        return None

    if not (response.text or "").strip():
        LOG.warning("Item %s: empty body", item_id)
        return None
    try:
        data = parse_item_page(response.text, item_id)
    except Exception:
        LOG.exception("Item %s: unparseable page", item_id)
        return None

    if not data.get("title"):
        # A titleless parse is useless downstream; leave it uncached so it retries later.
        LOG.warning("Item %s: parsed without a title, not caching", item_id)
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_dir / f"{item_id}.json.tmp"
    tmp_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    os.replace(tmp_file, cache_file)

    return data
