import sys
import types
import unittest
from pathlib import Path
import re
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

config_pkg = types.ModuleType('config')
config_pkg.__path__ = []
sys.modules['config'] = config_pkg

config_searches_mod = types.ModuleType('config.searches')
config_searches_mod.categories = {}
sys.modules['config.searches'] = config_searches_mod

config_project_mod = types.ModuleType('config.project_config')
config_project_mod.settings = types.SimpleNamespace(
    proxy=types.SimpleNamespace(
        scraping_browser_url='',
        web_unlocker_proxy='',
        api_token='',
        datacenter_proxy_url='http://dc-proxy.example:1234',
        residential_proxy_url='http://proxy.example:1234',
    ),
    paths=types.SimpleNamespace(brand_ids_csv='data/brand_ids.csv'),
)
sys.modules['config.project_config'] = config_project_mod

utils_pkg = types.ModuleType('utils_lib')
utils_pkg.__path__ = []
sys.modules['utils_lib'] = utils_pkg

utils_mod = types.ModuleType('utils_lib.utils')


def _parse_price_text(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace('EUR', '').replace('€', '').replace(',', '.')
    match = re.search(r'(\d+(?:\.\d+)?)', normalized)
    return float(match.group(1)) if match else None


utils_mod.parse_price_text = _parse_price_text
sys.modules['utils_lib.utils'] = utils_mod

retry_mod = types.ModuleType('utils_lib.retry_utils')
retry_mod.sleep_if_positive = lambda seconds: None
retry_mod.sleep_with_backoff = lambda attempt, base_delay=3.0: None
sys.modules['utils_lib.retry_utils'] = retry_mod

download_tracker_mod = types.ModuleType('utils_lib.download_tracker')
download_tracker_mod.estimate_response_bytes = lambda response, body=None: len(body or b'')
download_tracker_mod.record_download = lambda **kwargs: None
sys.modules['utils_lib.download_tracker'] = download_tracker_mod

proxy_identity_tracker_mod = types.ModuleType('utils_lib.proxy_identity_tracker')
proxy_identity_tracker_mod.maybe_track_proxy_identity = lambda **kwargs: None
sys.modules['utils_lib.proxy_identity_tracker'] = proxy_identity_tracker_mod

filters_mod = types.ModuleType('filters')
filters_mod.find_color_ids = lambda values: []
filters_mod.find_brand_ids = lambda driver, brands: []
sys.modules['filters'] = filters_mod

requests_html_mod = types.ModuleType('requests_html')
requests_html_mod.HTMLSession = type('HTMLSession', (), {})
requests_html_mod.HTML = lambda *args, **kwargs: object()
sys.modules['requests_html'] = requests_html_mod

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
sys.modules['selenium'] = selenium_mod
sys.modules['selenium.webdriver'] = webdriver_mod
sys.modules['selenium.webdriver.chrome.options'] = chrome_options_mod
sys.modules['selenium.webdriver.chromium.remote_connection'] = chromium_rc_mod
sys.modules['selenium.webdriver.common.by'] = common_by_mod

from Scraper import Scraper
import Scraper as scraper_module


class ScraperFetchFallbackTests(unittest.TestCase):
    def setUp(self):
        Scraper._reset_fetch_mode_state_for_tests()

    def tearDown(self):
        Scraper._reset_fetch_mode_state_for_tests()

    def test_rendered_failure_switches_to_residential_and_sets_cooldown(self):
        scraper = Scraper()
        sentinel = object()

        with patch.object(scraper, '_get_api_token', return_value='token'), \
             patch.object(scraper, '_get_page_content_rendered', return_value=(None, 403, None)) as rendered_mock, \
             patch.object(scraper, '_get_page_content_residential_html', return_value=sentinel) as residential_mock:
            result = scraper.get_page_content('https://example.com/item', sleep=0, max_attempts=1)

        self.assertIs(result, sentinel)
        rendered_mock.assert_called_once()
        residential_mock.assert_called_once()
        self.assertGreater(Scraper._residential_cooldown_remaining(), 50.0)

    def test_active_cooldown_uses_residential_first(self):
        Scraper._activate_residential_cooldown('test', seconds=300)
        scraper = Scraper()
        sentinel = object()

        with patch.object(scraper.logger, 'warning') as warning_mock, \
             patch.object(scraper, '_get_api_token', return_value='token'), \
             patch.object(scraper, '_get_page_content_residential_html', return_value=sentinel) as residential_mock, \
             patch.object(scraper, '_get_page_content_rendered') as rendered_mock:
            result = scraper.get_page_content('https://example.com/items/1234567890-example', sleep=0, max_attempts=1)

        self.assertIs(result, sentinel)
        residential_mock.assert_called_once()
        rendered_mock.assert_not_called()
        warning_mock.assert_any_call('RESIDENTIAL-MODE checking item %s', '1234567890')

    def test_active_cooldown_falls_back_to_rendered_when_residential_fails(self):
        Scraper._activate_residential_cooldown('test', seconds=300)
        scraper = Scraper()
        sentinel = object()

        with patch.object(scraper, '_get_api_token', return_value='token'), \
             patch.object(scraper, '_get_page_content_residential_html', return_value=None) as residential_mock, \
             patch.object(scraper, '_get_page_content_rendered', return_value=(sentinel, 200, None)) as rendered_mock:
            result = scraper.get_page_content('https://example.com/item', sleep=0, max_attempts=1)

        self.assertIs(result, sentinel)
        residential_mock.assert_called_once()
        rendered_mock.assert_called_once()

    def test_rendered_only_mode_ignores_residential_cooldown(self):
        Scraper._activate_residential_cooldown('test', seconds=300)
        scraper = Scraper()
        sentinel = object()

        with patch.object(scraper, '_get_api_token', return_value='token'), \
             patch.object(scraper, '_get_page_content_residential_html') as residential_mock, \
             patch.object(scraper, '_get_page_content_rendered', return_value=(sentinel, 200, None)) as rendered_mock:
            result = scraper.get_page_content(
                'https://example.com/items/1234567890-example',
                sleep=0,
                max_attempts=1,
                allow_residential_fallback=False,
            )

        self.assertIs(result, sentinel)
        residential_mock.assert_not_called()
        rendered_mock.assert_called_once()

    def test_no_residential_mode_uses_datacenter_only(self):
        scraper = Scraper()
        sentinel = object()

        with patch.object(scraper, '_get_page_content_datacenter_html', return_value=sentinel) as datacenter_mock, \
             patch.object(scraper, '_get_page_content_residential_html') as residential_mock, \
             patch.object(scraper, '_get_page_content_rendered') as rendered_mock:
            result = scraper.get_page_content(
                'https://example.com/items/1234567890-example',
                sleep=0,
                max_attempts=1,
                no_residential=True,
            )

        self.assertIs(result, sentinel)
        datacenter_mock.assert_called_once()
        residential_mock.assert_not_called()
        rendered_mock.assert_not_called()

    def test_rendered_helper_stops_after_first_403(self):
        class FakeResponse:
            def __init__(self, status_code):
                self.status_code = status_code
                self.html = None

            def close(self):
                return None

        class FakeSession:
            def __init__(self):
                self.calls = 0

            def mount(self, *args, **kwargs):
                return None

            def get(self, *args, **kwargs):
                self.calls += 1
                return FakeResponse(403)

            def close(self):
                return None

        fake_session = FakeSession()
        scraper = Scraper()

        with patch.object(scraper_module, 'HTMLSession', return_value=fake_session):
            html, last_status, last_error = scraper._get_page_content_rendered(
                'https://example.com/item',
                api_token='token',
                sleep=0,
                max_attempts=3,
            )

        self.assertIsNone(html)
        self.assertEqual(last_status, 403)
        self.assertIsNone(last_error)
        self.assertEqual(fake_session.calls, 1)

    def test_residential_fetch_stops_after_first_404(self):
        class FakeResponse:
            def __init__(self, status_code):
                self.status_code = status_code
                self.text = ''

        class FakeSession:
            def __init__(self):
                self.calls = 0

            def get(self, *args, **kwargs):
                self.calls += 1
                return FakeResponse(404)

            def close(self):
                return None

        fake_session = FakeSession()
        scraper = Scraper()

        with patch.object(scraper, '_build_retry_session', return_value=fake_session):
            html = scraper.get_page_content_residential('https://example.com/item', sleep=0, max_attempts=3)

        self.assertIsNone(html)
        self.assertEqual(fake_session.calls, 1)

    def test_datacenter_fetch_waits_and_retries_after_403(self):
        class FakeResponse:
            def __init__(self, status_code, text=''):
                self.status_code = status_code
                self.text = text
                self.content = text.encode('utf-8')

        class FakeSession:
            def __init__(self):
                self.calls = 0

            def get(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return FakeResponse(403, '')
                return FakeResponse(200, '<html>ok</html>')

            def close(self):
                return None

        fake_session = FakeSession()
        scraper = Scraper()

        with patch.object(scraper, '_build_retry_session', return_value=fake_session):
            html = scraper.get_page_content_datacenter('https://example.com/item', sleep=0, max_attempts=3)

        self.assertEqual(html, '<html>ok</html>')
        self.assertEqual(fake_session.calls, 2)


if __name__ == '__main__':
    unittest.main()
