"""Scrape one safely paced Vinted title search without a proxy.

Usage:
  python scrape_title_candidates.py --query "Prada paradoxe 90 ml" --out results.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests_html
from curl_cffi import requests as cffi

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))

from simple_scraper import Simple_scraper  # noqa: E402

PRODUCT_SELECTOR = ".new-item-box__container"
STATUS_NAMES = {
    "6": "Nuovo con cartellino",
    "1": "Nuovo senza cartellino",
    "2": "Ottime condizioni",
    "3": "Buone condizioni",
    "4": "Condizioni discrete",
}
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.vinted.it/",
}


def parse_condition(product) -> str:
    overlay = product.find(".new-item-box__overlay", first=True)
    raw = overlay.attrs.get("title", "") if overlay else ""
    marker = "condizioni:"
    if marker not in raw.lower():
        return ""
    start = raw.lower().index(marker) + len(marker)
    return raw[start:].split(",", 1)[0].strip()


def catalog_url(query: str, status_id: str | None, page: int) -> str:
    condition = f"&status_ids[]={status_id}" if status_id else ""
    return (
        "https://www.vinted.it/catalog?currency=EUR&order=relevance"
        f"&search_text={quote_plus(query)}{condition}&page={page}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--page-gap", type=float, default=60.0)
    parser.add_argument("--status-id", choices=tuple(STATUS_NAMES))
    args = parser.parse_args()
    if args.pages < 1 or args.page_gap < 0:
        parser.error("--pages must be >= 1 and --page-gap must be >= 0")

    session = cffi.Session(impersonate="chrome")
    homepage = session.get("https://www.vinted.it/", headers=HEADERS, timeout=20)
    if homepage.status_code != 200:
        raise SystemExit(f"Vinted preflight failed: HTTP {homepage.status_code}")

    scraper = Simple_scraper()
    rows = []
    for page in range(1, args.pages + 1):
        response = session.get(
            catalog_url(args.query, args.status_id, page), headers=HEADERS, timeout=30
        )
        if response.status_code != 200:
            raise SystemExit(f"Catalog page {page} failed: HTTP {response.status_code}")
        products = requests_html.HTML(html=response.text).find(PRODUCT_SELECTOR)
        print(f"page={page} products={len(products)}", flush=True)
        for product in products:
            row = scraper.extract_catalog_item_meta(product, {}, page - 1, 0, get_images=True)
            if row:
                row["SearchName"] = args.query
                row["Condition"] = parse_condition(product)
                row["ConditionStatusId"] = args.status_id or ""
                if str(row["Link"]).startswith("/"):
                    row["Link"] = "https://www.vinted.it" + row["Link"]
                rows.append(row)
        if page < args.pages:
            time.sleep(args.page_gap)

    output = pd.DataFrame(rows).drop_duplicates("Dataid")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    conditions = output["Condition"].value_counts(dropna=False).to_dict()
    print(
        f"wrote={args.out} rows={len(output)} status_id={args.status_id or 'none'} "
        f"conditions={conditions}"
    )


if __name__ == "__main__":
    main()
