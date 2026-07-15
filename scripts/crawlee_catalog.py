"""Crawlee-based Vinted catalog fetcher (POC).

Replaces only the fetch + throttle + session + proxy layer of the scraper.
Reuses the existing Vinted parsing (Simple_scraper.extract_catalog_item_meta),
so output rows are identical to scrape_products_serial.

Anti-block strategy (see crawlee.dev):
  - ImpitHttpClient        -> TLS fingerprint impersonation (like curl_cffi)
  - ThrottlingRequestManager -> per-domain 429 backoff (exp, honors Retry-After)
  - ConcurrencySettings    -> max_tasks_per_minute paces under the IP ceiling
  - tiered_proxy_urls      -> tier 0 = no proxy; escalate to datacenter proxy
                              only when a tier keeps getting blocked

Run:
  python scripts/crawlee_catalog.py "nike" 15            # no proxy
  python scripts/crawlee_catalog.py "nike" 15 --proxy    # +datacenter fallback tier
"""
import asyncio
import re
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests_html  # noqa: E402

from crawlee import ConcurrencySettings  # noqa: E402
from crawlee.crawlers import HttpCrawler, HttpCrawlingContext  # noqa: E402
from crawlee.http_clients import ImpitHttpClient  # noqa: E402
from crawlee.proxy_configuration import ProxyConfiguration  # noqa: E402
from crawlee.request_loaders import ThrottlingRequestManager  # noqa: E402
from crawlee.storages import RequestQueue  # noqa: E402

from simple_scraper import Simple_scraper  # noqa: E402
from config.project_config import settings  # noqa: E402

PRODUCT_SEL = '.new-item-box__container'
LIKES_SEL = ('.u-background-white.u-flexbox.u-align-items-center'
             '.new-item-box__favourite-icon')
CATALOG = ('https://www.vinted.it/catalog?currency=EUR&search_text={q}'
           '&order=newest_first&page={p}')


def _page_of(url: str) -> int:
    m = re.search(r'[?&]page=(\d+)', url)
    return int(m.group(1)) if m else 0


async def crawl_catalog(search_text: str, pages: int, use_proxy: bool = False,
                        max_per_minute: int = 8):
    parser = Simple_scraper()
    rows: list[dict] = []
    stats = {'ok_pages': 0, 'soft_block': 0, 'products_seen': 0, 'kept': 0}

    tiers: list[list[str | None]] = [[None]]  # tier 0: no proxy
    if use_proxy and settings.proxy.datacenter_proxy_url:
        tiers.append([settings.proxy.datacenter_proxy_url])  # tier 1: escalate
    proxy_cfg = ProxyConfiguration(tiered_proxy_urls=tiers)

    queue = await RequestQueue.open()
    throttler = ThrottlingRequestManager(
        inner=queue,
        domains=['www.vinted.it'],
        request_manager_opener=RequestQueue.open,
        base_delay=timedelta(seconds=5),
        max_delay=timedelta(seconds=60),
    )

    crawler = HttpCrawler(
        http_client=ImpitHttpClient(),
        proxy_configuration=proxy_cfg,
        request_manager=throttler,
        concurrency_settings=ConcurrencySettings(
            min_concurrency=1, desired_concurrency=1, max_concurrency=1,
            max_tasks_per_minute=max_per_minute,
        ),
        max_request_retries=3,
        max_session_rotations=4,
        retry_on_blocked=True,
    )

    @crawler.router.default_handler
    async def handler(context: HttpCrawlingContext) -> None:
        page_no = _page_of(context.request.url)
        status = getattr(context.http_response, 'status_code', None)
        body = (await context.http_response.read()).decode('utf-8', 'ignore')
        html = requests_html.HTML(html=body)
        products = html.find(PRODUCT_SEL)

        # Soft-block: HTTP 200 but no listings (the per-IP quota page ~ after 10 pages).
        if status == 200 and not products:
            stats['soft_block'] += 1
            context.log.warning(
                f'SOFT-BLOCK page={page_no} status=200 items=0 '
                f'proxy={context.proxy_info}'
            )
            # With a proxy fallback tier: retire + raise so Crawlee escalates tier.
            # Without one: nothing to escalate to, so skip gracefully (no crash).
            if use_proxy:
                if context.session:
                    context.session.retire()
                raise RuntimeError('soft-block: 200 with 0 items')
            return

        likes = html.find(LIKES_SEL)
        stats['products_seen'] += len(products)
        for product in products:
            row = parser.extract_product_meta(product, likes, page_no, 0, get_images=False)
            if row and parser.validate_listing_row(row):
                rows.append(row)
                stats['kept'] += 1
        stats['ok_pages'] += 1
        context.log.info(
            f'OK page={page_no} status={status} items={len(products)} '
            f'kept_total={len(rows)} proxy={context.proxy_info}'
        )

    q = search_text.replace(' ', '%20')
    urls = [CATALOG.format(q=q, p=i + 1) for i in range(pages)]
    await throttler.add_requests(urls)
    await crawler.run()

    print(f'\n=== SUMMARY search={search_text!r} pages_requested={pages} '
          f'proxy_fallback={use_proxy} ===')
    print(stats)
    return rows, stats


def _main() -> None:
    args = sys.argv[1:]
    use_proxy = '--proxy' in args
    args = [a for a in args if a != '--proxy']
    search = args[0] if args else 'nike'
    pages = int(args[1]) if len(args) > 1 else 15
    rpm = int(args[2]) if len(args) > 2 else 8
    asyncio.run(crawl_catalog(search, pages, use_proxy=use_proxy, max_per_minute=rpm))


if __name__ == '__main__':
    _main()
