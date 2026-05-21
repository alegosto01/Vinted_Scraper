import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

sys.modules.setdefault('searches', types.SimpleNamespace(categories={}))
sys.modules.setdefault('openpyxl', types.SimpleNamespace(load_workbook=lambda *args, **kwargs: None))
selenium_exc_mod = types.ModuleType('selenium.common.exceptions')
selenium_exc_mod.WebDriverException = Exception
sys.modules.setdefault('selenium.common.exceptions', selenium_exc_mod)
sys.modules.setdefault('filters', types.SimpleNamespace(find_color_ids=lambda values: [], find_brand_ids=lambda driver, brands: []))
sys.modules.setdefault(
    'project_config',
    types.SimpleNamespace(
        settings=types.SimpleNamespace(
            proxy=types.SimpleNamespace(
                scraping_browser_url='',
                web_unlocker_proxy='',
                api_token='',
            ),
            paths=types.SimpleNamespace(brand_ids_csv='data/brand_ids.csv'),
        )
    ),
)
sys.modules.setdefault(
    'retry_utils',
    types.SimpleNamespace(
        sleep_if_positive=lambda seconds: None,
        sleep_with_backoff=lambda attempt, base_delay=3.0: None,
    ),
)

requests_html_mod = types.ModuleType('requests_html')
requests_html_mod.HTMLSession = type('HTMLSession', (), {})
requests_html_mod.HTML = lambda *args, **kwargs: None
sys.modules.setdefault('requests_html', requests_html_mod)

selenium_mod = types.ModuleType('selenium')
webdriver_mod = types.ModuleType('selenium.webdriver')
webdriver_mod.ChromeOptions = type('ChromeOptions', (), {})
webdriver_mod.Remote = type('Remote', (), {})
chrome_options_mod = types.ModuleType('selenium.webdriver.chrome.options')
chrome_options_mod.Options = type('Options', (), {})
chromium_rc_mod = types.ModuleType('selenium.webdriver.chromium.remote_connection')
chromium_rc_mod.ChromiumRemoteConnection = type('ChromiumRemoteConnection', (), {})
common_by_mod = types.ModuleType('selenium.webdriver.common.by')
common_by_mod.By = type('By', (), {})
sys.modules.setdefault('selenium', selenium_mod)
sys.modules.setdefault('selenium.webdriver', webdriver_mod)
sys.modules.setdefault('selenium.webdriver.chrome.options', chrome_options_mod)
sys.modules.setdefault('selenium.webdriver.chromium.remote_connection', chromium_rc_mod)
sys.modules.setdefault('selenium.webdriver.common.by', common_by_mod)

from Scraper import Scraper


class FakeElement:
    def __init__(self, text=None, attrs=None):
        self.text = text
        self.attrs = attrs or {}


class FakeHTML:
    def __init__(self, mapping, text='', html=''):
        self.mapping = mapping
        self.text = text
        self.html = html

    def find(self, selector, first=False):
        values = self.mapping.get(selector, [])
        if first:
            return values[0] if values else None
        return values


class SoldPriceUpdateTests(unittest.TestCase):
    def test_extract_item_page_price_prefers_visible_price_over_meta(self):
        scraper = Scraper()
        html = FakeHTML({
            'meta[itemprop="price"]': [FakeElement(attrs={'content': '90'})],
            '[data-testid="item-price-current"]': [FakeElement(text='89.50 EUR')],
        })
        self.assertEqual(scraper.extract_item_page_price(html), 89.5)

    def test_extract_item_page_price_falls_back_to_meta_price(self):
        scraper = Scraper()
        html = FakeHTML({
            'meta[itemprop="price"]': [FakeElement(attrs={'content': '89.50'})],
        })
        self.assertEqual(scraper.extract_item_page_price(html), 89.5)

    def test_extract_item_page_price_prefers_fee_included_total_when_present(self):
        scraper = Scraper()
        html = FakeHTML(
            {},
            text='€110.00, €116.20 include la Protezione acquisti',
            html='€110.00, €116.20 include la Protezione acquisti',
        )
        self.assertEqual(scraper.extract_item_page_price(html), 116.2)

    def test_extract_item_page_price_ignores_single_fee_amount_without_two_price_pair(self):
        scraper = Scraper()
        html = FakeHTML(
            {'meta[itemprop="price"]': [FakeElement(attrs={'content': '89.50'})]},
            text='10,00 € Protezione acquisti',
            html='10,00 € Protezione acquisti',
        )
        self.assertEqual(scraper.extract_item_page_price(html), 89.5)

    def test_update_item_with_latest_page_price_overwrites_changed_price(self):
        scraper = Scraper()
        html = FakeHTML({
            'meta[itemprop="price"]': [FakeElement(attrs={'content': '79.90'})],
        })
        item = {'Price': 89.5, 'Link': 'x'}

        updated = scraper.update_item_with_latest_page_price(item, html)

        self.assertEqual(updated['Price'], 79.9)
        self.assertEqual(updated['PreviousPrice'], 89.5)
        self.assertEqual(updated['ObservedPagePrice'], 79.9)
        self.assertTrue(updated['PriceChangedBeforeSold'])

    def test_update_item_with_latest_page_price_uses_jsonld_fallback(self):
        scraper = Scraper()
        html = FakeHTML({
            'script[type="application/ld+json"]': [
                FakeElement(text='{"offers": {"price": "149.00"}}')
            ],
        })
        item = {'Price': 149.0, 'Link': 'x'}

        updated = scraper.update_item_with_latest_page_price(item, html)

        self.assertEqual(updated['ObservedPagePrice'], 149.0)
        self.assertFalse(updated['PriceChangedBeforeSold'])


if __name__ == '__main__':
    unittest.main()
