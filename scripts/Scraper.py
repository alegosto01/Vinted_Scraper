import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional

import pandas as pd
import requests
import config.searches as search
import urllib3
import utils_lib.utils as utils
from utils_lib.utils import parse_price_text as parse_price_value
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from requests_html import HTML, HTMLSession
from selenium.webdriver import ChromeOptions, Remote
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection
from selenium.webdriver.common.by import By

import filters as f
from config.project_config import settings
from utils_lib.download_tracker import estimate_response_bytes, record_download
from utils_lib.proxy_identity_tracker import maybe_track_proxy_identity
from utils_lib.retry_utils import sleep_if_positive, sleep_with_backoff

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SBR_WEBDRIVER = settings.proxy.scraping_browser_url
proxy = settings.proxy.web_unlocker_proxy

LOGGER = logging.getLogger(__name__)
DATACENTER_403_WAIT_SECONDS = 180.0


class Scraper:
    def __init__(self):
        self.logger = LOGGER
        self.driver = None

    def _get_api_token(self) -> Optional[str]:
        return os.getenv("API_TOKEN") or settings.proxy.api_token

    def _sleep_with_backoff(self, attempt: int, base_delay: float = 3.0) -> None:
        sleep_with_backoff(attempt, base_delay=base_delay)

    def _extract_listing_id_from_url(self, url: Optional[str]) -> Optional[str]:
        if not url:
            return None
        match = re.search(r'/items/(\d+)', str(url))
        return match.group(1) if match else None

    def _build_retry_session(self, total: int = 5, backoff_factor: float = 0.5) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=total,
            backoff_factor=backoff_factor,
            status_forcelist=[403, 429, 500, 502, 503, 504],
            allowed_methods=['GET'],
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def _response_body_bytes(self, response) -> bytes:
        body = getattr(response, 'content', None)
        if isinstance(body, bytes):
            return body
        if isinstance(body, str):
            return body.encode('utf-8', errors='ignore')

        text = getattr(response, 'text', None)
        if isinstance(text, str):
            return text.encode('utf-8', errors='ignore')
        return b''

    def _response_ok(self, response) -> bool:
        ok = getattr(response, 'ok', None)
        if ok is not None:
            return bool(ok)
        status_code = getattr(response, 'status_code', None)
        return bool(status_code is not None and int(status_code) < 400)

    def _default_page_headers(self) -> dict[str, str]:
        return {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': 'https://www.vinted.it/',
            'Connection': 'keep-alive',
        }

    def init_driver(self):
        chrome_options = Options()
        prefs = {
            'profile.managed_default_content_settings.images': 2,
            'profile.managed_default_content_settings.stylesheets': 2,
            'profile.managed_default_content_settings.fonts': 2,
            'profile.managed_default_content_settings.media_stream': 2,
            'profile.default_content_setting_values.notifications': 2,
            'profile.default_content_setting_values.popups': 2,
            'profile.default_content_setting_values.geolocation': 2,
            'profile.managed_default_content_settings.javascript': 1,
        }
        chrome_options.add_experimental_option('prefs', prefs)
        chrome_options.add_argument('--headless=new')
        chrome_options.add_argument('--window-size=375,667')

        for attempt in range(1, 4):
            try:
                if not SBR_WEBDRIVER:
                    raise RuntimeError('Missing Bright Data scraping browser configuration')
                self.logger.info('Connecting to scraping browser (attempt %s/3)', attempt)
                sbr_connection = ChromiumRemoteConnection(SBR_WEBDRIVER, 'goog', 'chrome')
                driver = Remote(sbr_connection, options=ChromeOptions())
                driver.set_page_load_timeout(120)
                self.logger.info('Connected to scraping browser')
                return driver
            except Exception as exc:
                self.logger.warning('Scraping browser connection failed on attempt %s/3: %s', attempt, exc)
                self._sleep_with_backoff(attempt, base_delay=5)

        self.logger.error('Failed to initialize scraping browser after 3 attempts')
        return None

    def ensure_driver(self):
        if self.driver is None:
            self.driver = self.init_driver()
        return self.driver

    def get_page_content_datacenter(self, url, timeout=40, sleep=10, max_attempts=3, request_kind='page'):
        proxy_url = settings.proxy.datacenter_proxy_url
        if not proxy_url:
            self.logger.error('Missing Bright Data datacenter proxy configuration')
            return None
        attempts_made = 0
        read_timeout = max(float(timeout), 1.0)
        connect_timeout = min(15.0, read_timeout)
        headers = self._default_page_headers()
        proxies = {'http': proxy_url, 'https': proxy_url}

        session = self._build_retry_session(total=5, backoff_factor=1.0)
        try:
            for attempt in range(1, max_attempts + 1):
                attempts_made = attempt
                try:
                    response = session.get(
                        url,
                        headers=headers,
                        proxies=proxies,
                        timeout=(connect_timeout, read_timeout),
                        verify=False,
                    )
                    body = self._response_body_bytes(response)
                    record_download(
                        kind=request_kind,
                        transport='datacenter_proxy',
                        url=url,
                        bytes_downloaded=estimate_response_bytes(response, body=body),
                        status_code=response.status_code,
                        ok=self._response_ok(response),
                        metadata={'attempt': attempt},
                    )
                    maybe_track_proxy_identity(
                        transport='datacenter_proxy',
                        proxy_url=proxy_url,
                        request_url=url,
                        session=session,
                        headers=headers,
                        response_headers=getattr(response, 'headers', None),
                        verify=False,
                    )
                    if response.status_code == 200 and body.strip():
                        sleep_if_positive(sleep)
                        self.logger.debug('Datacenter fetch succeeded for %s on attempt %s', url, attempt)
                        return response.text
                    if response.status_code == 404:
                        self.logger.warning(
                            'Datacenter fetch returned status=404 for %s on attempt %s; not retrying',
                            url,
                            attempt,
                        )
                        break
                    if response.status_code in {403, 429}:
                        wait_seconds = max(DATACENTER_403_WAIT_SECONDS, float(sleep) * max(attempt, 1), 20.0)
                        self.logger.warning(
                            'Datacenter fetch returned status=%s for %s on attempt %s; waiting %.1fs before retry',
                            response.status_code,
                            url,
                            attempt,
                            wait_seconds,
                        )
                        if attempt < max_attempts:
                            sleep_if_positive(wait_seconds)
                        continue
                    self.logger.warning(
                        'Datacenter fetch returned status=%s for %s on attempt %s',
                        response.status_code,
                        url,
                        attempt,
                    )
                except requests.exceptions.RequestException as exc:
                    self.logger.warning('Datacenter fetch error for %s on attempt %s: %s', url, attempt, exc)
                if attempt < max_attempts:
                    self._sleep_with_backoff(attempt)
        finally:
            session.close()

        self.logger.error('Datacenter fetch failed for %s after %s attempts', url, attempts_made)
        return None

    def _html_from_text(self, html_text, url: str):
        if not isinstance(html_text, str) or not html_text.strip():
            return None
        try:
            return HTML(html=html_text)
        except Exception as exc:
            self.logger.warning('HTML parsing failed for %s: %s', url, exc)
            return None

    def _get_page_content_datacenter_html(self, url, timeout=40, sleep=10, max_attempts=3, request_kind='page'):
        html_text = self.get_page_content_datacenter(
            url,
            timeout=timeout,
            sleep=sleep,
            max_attempts=max_attempts,
            request_kind=request_kind,
        )
        return self._html_from_text(html_text, url)

    def _get_page_content_rendered(self, url, api_token, timeout=100, sleep=10, max_attempts=3, request_kind='page'):
        last_status = None
        last_error = None
        attempts_made = 0

        headers = {
            'Authorization': api_token,
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/',
        }
        payload = {
            'zone': 'web_unlocker1',
            'url': url,
            'format': 'html',
            'method': 'GET',
            'country': 'IT',
        }

        session = HTMLSession()
        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        try:
            for attempt in range(1, max_attempts + 1):
                attempts_made = attempt
                response = None
                try:
                    sleep_if_positive(sleep)
                    response = session.get(url, json=payload, headers=headers, timeout=40)
                    body = self._response_body_bytes(response)
                    record_download(
                        kind=request_kind,
                        transport='rendered_fetch',
                        url=url,
                        bytes_downloaded=estimate_response_bytes(response, body=body),
                        status_code=response.status_code,
                        ok=self._response_ok(response),
                        metadata={'attempt': attempt, 'zone': payload['zone']},
                    )
                    last_status = response.status_code
                    if response.status_code == 404:
                        self.logger.warning(
                            'Rendered fetch returned status=404 for %s on attempt %s; not retrying',
                            url,
                            attempt,
                        )
                        break
                    if response.status_code == 403:
                        self.logger.warning(
                            'Rendered fetch returned status=403 for %s on attempt %s; switching to datacenter fallback immediately',
                            url,
                            attempt,
                        )
                        break
                    if response.status_code == 429:
                        cooldown = max(sleep * attempt, 20)
                        self.logger.warning(
                            'Rendered fetch hit 429 for %s on attempt %s; sleeping %.1fs before retry',
                            url,
                            attempt,
                            cooldown,
                        )
                        sleep_if_positive(cooldown)
                    elif response.status_code != 200:
                        self.logger.warning(
                            'Rendered fetch returned status=%s for %s on attempt %s',
                            response.status_code,
                            url,
                            attempt,
                        )
                    else:
                        html = response.html
                        if html is None:
                            self.logger.warning('Rendered fetch returned no HTML object for %s on attempt %s', url, attempt)
                        else:
                            self.logger.debug('Rendered fetch succeeded for %s on attempt %s', url, attempt)
                            return html, last_status, last_error
                except requests.exceptions.RequestException as exc:
                    last_error = exc
                    self.logger.warning('Rendered fetch request error for %s on attempt %s: %s', url, attempt, exc)
                finally:
                    if response is not None:
                        try:
                            response.close()
                        except Exception:
                            pass
                if attempt < max_attempts:
                    self._sleep_with_backoff(attempt)
        finally:
            session.close()

        self.logger.error('Rendered fetch failed for %s after %s attempts', url, attempts_made)
        return None, last_status, last_error

    def get_page_content(
        self,
        url,
        timeout=100,
        sleep=10,
        max_attempts=3,
        request_kind='item_page',
    ):
        datacenter_timeout = min(timeout, 40)
        datacenter_sleep = min(sleep, 10)

        api_token = self._get_api_token()
        if not api_token:
            self.logger.warning(
                'Missing API_TOKEN configuration for rendered page fetches; using datacenter for %s',
                url,
            )
            return self._get_page_content_datacenter_html(
                url,
                timeout=datacenter_timeout,
                sleep=datacenter_sleep,
                max_attempts=max_attempts,
                request_kind=request_kind,
            )

        html, last_status, last_error = self._get_page_content_rendered(
            url,
            api_token=api_token,
            timeout=timeout,
            sleep=sleep,
            max_attempts=max_attempts,
            request_kind=request_kind,
        )
        if html is not None:
            return html

        if last_status == 404:
            return None

        if last_status is not None:
            self.logger.warning(
                'Rendered fetch failed with status=%s for %s; falling back to datacenter',
                last_status,
                url,
            )
        elif last_error is not None:
            self.logger.warning(
                'Rendered fetch failed with request error for %s; falling back to datacenter: %s',
                url,
                last_error,
            )

        return self._get_page_content_datacenter_html(
            url,
            timeout=datacenter_timeout,
            sleep=datacenter_sleep,
            max_attempts=max_attempts,
            request_kind=request_kind,
        )

    def parse_price_text(self, raw_price: Any) -> Optional[float]:
        parsed_price = parse_price_value(raw_price)
        if parsed_price is None:
            self.logger.warning('Failed to parse price from value=%r', raw_price)
        return parsed_price

    def _iter_json_dicts(self, value):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from self._iter_json_dicts(child)
        elif isinstance(value, list):
            for child in value:
                yield from self._iter_json_dicts(child)

    def _extract_fee_included_page_price(self, html_content) -> Optional[float]:
        text_candidates = []
        for attr in ('text', 'html'):
            value = getattr(html_content, attr, None)
            if isinstance(value, str) and value.strip():
                text_candidates.append(value)

        # Only trust local patterns that show both the listing price and the
        # fee-included buyer total near the protection wording.
        pair_patterns = [
            re.compile(r'€\s*([\d.,]+)\D{0,40}€\s*([\d.,]+)\D{0,120}protezione acquisti', re.IGNORECASE),
            re.compile(r'([\d.,]+)\s*€\D{0,40}([\d.,]+)\s*€\D{0,120}protezione acquisti', re.IGNORECASE),
            re.compile(r'€\s*([\d.,]+)\D{0,40}€\s*([\d.,]+)\D{0,120}(buyer|purchase) protection', re.IGNORECASE),
            re.compile(r'([\d.,]+)\s*€\D{0,40}([\d.,]+)\s*€\D{0,120}(buyer|purchase) protection', re.IGNORECASE),
        ]

        for text in text_candidates:
            for pattern in pair_patterns:
                match = pattern.search(text)
                if not match:
                    continue
                first_raw = str(match.group(1)).rstrip(',. ')
                second_raw = str(match.group(2)).rstrip(',. ')
                first_price = self.parse_price_text(first_raw)
                second_price = self.parse_price_text(second_raw)
                valid = [p for p in (first_price, second_price) if p is not None and p > 0]
                if len(valid) == 2:
                    return max(valid)
        return None

    def extract_item_page_price(self, html_content) -> Optional[float]:
        fee_included_price = self._extract_fee_included_page_price(html_content)
        if fee_included_price is not None:
            return fee_included_price

        selector_candidates = [
            ('[data-testid="item-price-current"]', 'text'),
            ('[data-testid="item-price"]', 'text'),
            ('[itemprop="price"]', 'text'),
            ('meta[itemprop="price"]', 'content'),
            ('meta[property="product:price:amount"]', 'content'),
            ('meta[name="twitter:data1"]', 'content'),
        ]

        for selector, attr in selector_candidates:
            try:
                element = html_content.find(selector, first=True)
            except Exception:
                element = None
            if not element:
                continue
            raw_value = element.attrs.get(attr) if attr != 'text' else getattr(element, 'text', None)
            parsed_price = self.parse_price_text(raw_value)
            if parsed_price is not None and parsed_price > 0:
                return parsed_price

        try:
            scripts = html_content.find('script[type="application/ld+json"]')
        except Exception:
            scripts = []

        for script in scripts:
            raw_json = getattr(script, 'text', None) or ''
            if not raw_json.strip():
                continue
            try:
                payload = json.loads(raw_json)
            except json.JSONDecodeError:
                continue
            for obj in self._iter_json_dicts(payload):
                offers = obj.get('offers') if isinstance(obj, dict) else None
                if isinstance(offers, list):
                    offer_values = offers
                elif isinstance(offers, dict):
                    offer_values = [offers]
                else:
                    offer_values = []
                for offer in offer_values:
                    parsed_price = self.parse_price_text(offer.get('price'))
                    if parsed_price is not None and parsed_price > 0:
                        return parsed_price

        return None

    def update_item_with_latest_page_price(self, item: Dict[str, Any], html_content) -> Dict[str, Any]:
        page_price = self.extract_item_page_price(html_content)
        if page_price is None:
            return item

        item['ObservedPagePrice'] = page_price
        previous_price = self.parse_price_text(item.get('Price'))
        price_changed = previous_price is not None and abs(previous_price - page_price) >= 0.01

        if price_changed:
            item['PreviousPrice'] = previous_price
            item['PriceChangedBeforeSold'] = True
            item['Price'] = page_price
        else:
            item['PriceChangedBeforeSold'] = False
            if previous_price is None:
                item['Price'] = page_price

        return item

    def extract_likes_count(self, data_id: str, all_likes_counts) -> int:
        likes = 0
        for el in all_likes_counts:
            tid = el.attrs.get('data-testid', '')
            if data_id and data_id in tid:
                label = el.attrs.get('aria-label', '')
                if label != 'Aggiungi ai preferiti':
                    try:
                        likes = int(label.split('Aggiunto ai preferiti da ')[1].split()[0])
                    except (IndexError, ValueError):
                        self.logger.debug('Could not parse likes label=%r for data_id=%s', label, data_id)
                break
        return likes

    def validate_listing_row(self, row: Optional[Dict[str, Any]], required_fields=None) -> bool:
        if not row:
            return False
        required_fields = required_fields or ['Title', 'Link', 'Dataid']
        missing = [field for field in required_fields if not row.get(field)]
        if missing:
            self.logger.warning('Dropping scraped row with missing fields=%s row=%s', missing, row)
            return False
        return True

    def extract_catalog_item_meta(self, product, all_likes_counts, page, search_count, get_images=False):
        try:
            image_el = product.find('img', first=True)
            img = image_el.attrs.get('src') if image_el else None
            overlay = product.find('.new-item-box__overlay', first=True)
            if overlay is None:
                self.logger.warning('Skipping product without overlay on page=%s', page + 1)
                return None

            title_attr = overlay.attrs.get('title')
            link = overlay.attrs.get('href')
            if not title_attr or not link or 'referrer=catalog' not in link:
                self.logger.debug('Skipping product with missing title/link or non-catalog link=%r', link)
                return None

            title = utils.remove_illegal_characters(title_attr)
            components = utils.split_data(title)
            if len(components) < 4:
                self.logger.warning('Skipping product with unexpected title components=%r title=%r', components, title)
                return None

            parts = overlay.attrs.get('data-testid', '').split('-')
            data_id = parts[3] if len(parts) == 7 else parts[1] if len(parts) > 1 else ''
            likes = self.extract_likes_count(data_id, all_likes_counts)
            price_value = self.parse_price_text(components[1])

            row = {
                'Title': components[0],
                'Price': price_value if price_value is not None else 0.0,
                'Brand': components[2],
                'Size': components[3],
                'Link': link,
                'Likes': likes,
                'Dataid': data_id,
                'Images': img if get_images else img,
                'Page': page + 1,
                'SearchDate': time.strftime('%d/%m/%Y %H:%M:%S'),
                'SearchCount': search_count,
                'MarketStatus': 'On Sale',
                'Processed': False,
            }
            return row if self.validate_listing_row(row) else None
        except Exception as exc:
            self.logger.exception('Failed to extract catalog item metadata on page=%s: %s', page + 1, exc)
            return None

    def inspect_item_page(
        self,
        item,
        get_images=False,
        check_venduto=True,
        get_upload_date=False,
        initial_delay=0,
        fetch_sleep=50,
        fetch_max_attempts=3,
    ):
        if initial_delay:
            sleep_if_positive(initial_delay)

        item = dict(item)
        url = item.get('Link')
        if not url:
            self.logger.warning('Cannot inspect item without Link: %s', item)
            return item, False, 'MissingLink'

        html_content = self.get_page_content(
            url,
            timeout=60,
            sleep=fetch_sleep,
            max_attempts=fetch_max_attempts,
            request_kind='item_page',
        )
        if not html_content:
            return item, False, 'FetchFailed'

        item = self.update_item_with_latest_page_price(item, html_content)

        if get_images:
            try:
                images_links_element = html_content.find('div[class="item-photos"]', first=True)
                if images_links_element:
                    item['Images'] = [img.attrs['src'] for img in images_links_element.find('img') if 'src' in img.attrs]
            except Exception as exc:
                self.logger.debug('Image extraction failed for item=%s: %s', item.get('Dataid'), exc)

        if check_venduto:
            try:
                element = html_content.find('div[data-testid="item-status--content"]', first=True)
                if element and element.text == 'Venduto':
                    return item, True, 'Sold'
            except Exception as exc:
                self.logger.debug('Sold-status extraction failed for item=%s: %s', item.get('Dataid'), exc)

        if get_upload_date:
            try:
                upload_el = html_content.find('div.details-list__item-value[itemprop="upload_date"] span', first=True)
                item['Upload_date'] = upload_el.text if upload_el else 'Unknown'
            except Exception as exc:
                self.logger.debug('Upload-date extraction failed for item=%s: %s', item.get('Dataid'), exc)
                item['Upload_date'] = 'Unknown'

        return item, False, 'OnSale'

    def create_webpage(self, dictionary):
        input_search = '&search_text=' + str(dictionary.search).replace(' ', '%20')
        order = '&order=' + dictionary.sort
        price_from = '' if dictionary.prezzoDa == ' ' else '&price_from=' + dictionary.prezzoDa
        price_to = '' if dictionary.prezzoA == ' ' else '&price_to=' + dictionary.prezzoA

        color_search = ''
        if dictionary.colore != ' ':
            for color_id in f.find_color_ids(dictionary.colore.split('-')):
                color_search += '&color_ids[]=' + str(color_id)

        brands_search = ''
        if dictionary.brands != ' ':
            brands_list = dictionary.brands.split('-')
            brands_ids = []
            non_saved_brands = []
            brands_df = pd.read_csv(settings.paths.brand_ids_csv)

            for brand in brands_list:
                if brand not in brands_df['Brand'].values:
                    non_saved_brands.append(brand)
                else:
                    brands_ids.append(brands_df.loc[brands_df['Brand'] == brand, 'Brand_id'].iloc[0])

            if non_saved_brands:
                self.logger.info('Resolving missing brand ids for %s', non_saved_brands)
                self.driver = self.ensure_driver()
                if self.driver is not None:
                    self.driver.get(f'https://www.vinted.it/catalog?currency=EUR{input_search}')
                    try:
                        cookie = self.driver.find_element(By.ID, 'onetrust-accept-btn-handler')
                        cookie.click()
                    except Exception:
                        pass
                    sleep_if_positive(5)
                    brands_ids.extend(f.find_brand_ids(self.driver, non_saved_brands))

            for brand_id in brands_ids:
                brands_search += '&brand_ids[]=' + str(brand_id)

        condition = ''
        for elem in dictionary.condition.split('-'):
            if elem != ' ':
                condition += '&status_ids[]=' + elem

        category = '' if dictionary.category == ' ' else '&catalog[]=' + search.categories[dictionary.category]
        return f'https://www.vinted.it/catalog?currency=EUR{order}{input_search}{color_search}{price_from}{price_to}{condition}{brands_search}{category}'

    def get_all_product_images_urls(self, url):
        driver = self.ensure_driver()
        if driver is None:
            self.logger.error('Cannot fetch product images because driver initialization failed')
            return []

        utils.safe_get(driver, url)
        image_button = driver.find_element(By.XPATH, "//button[@class='item-thumbnail']")
        driver.execute_script('arguments[0].scrollIntoView(true);', image_button)
        driver.execute_script('arguments[0].click();', image_button)

        sleep_if_positive(2)
        image_carousel = driver.find_element(By.XPATH, "//div[contains(@class, 'image-carousel__image-wrapper')]")
        images_element = image_carousel.find_elements(By.TAG_NAME, 'img')
        image_urls = [img.get_attribute('src') for img in images_element]
        self.logger.info('Fetched %s images from %s', len(image_urls), url)
        return image_urls

    def get_all_product_images(self, url):
        return self.get_all_product_images_urls(url)
