import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

stub_project_config = sys.modules.get('config.project_config')
if stub_project_config is not None and not hasattr(stub_project_config, 'LoggingConfig'):
    sys.modules.pop('config.project_config', None)
    sys.modules.pop('config', None)

stub_scraping_options = sys.modules.get('scraping_options')
if stub_scraping_options is not None and not hasattr(stub_scraping_options, 'maybe_refresh_daily_eventual_sales_for_running_scheduler'):
    sys.modules.pop('scraping_options', None)


from config.project_config import LoggingConfig, PathsConfig, ProxyConfig, Settings, TelegramConfig
from scraping_options import maybe_refresh_daily_eventual_sales_for_running_scheduler, preflight_parallel_scrape


class ScrapingPreflightTests(unittest.TestCase):
    def make_settings(self, with_residential_proxy: bool, with_datacenter_proxy: bool = False) -> Settings:
        temp_dir = Path(__file__).resolve().parent
        proxy = ProxyConfig(
            residential_username='user' if with_residential_proxy else None,
            residential_password='pass' if with_residential_proxy else None,
            datacenter_username='dc-user' if with_datacenter_proxy else None,
            datacenter_password='dc-pass' if with_datacenter_proxy else None,
        )
        return Settings(
            paths=PathsConfig(project_root=temp_dir, data_dir=temp_dir, simple_scrape_dir=temp_dir, full_scrape_dir=temp_dir, models_dir=temp_dir, searches_yaml=Path(__file__).resolve(), brand_ids_csv=Path(__file__).resolve()),
            telegram=TelegramConfig(bot_token='token', chat_id='chat'),
            proxy=proxy,
            logging=LoggingConfig(),
        )

    def test_preflight_rejects_invalid_mode(self):
        result = preflight_parallel_scrape([object()], mode='bad-mode', app_settings=self.make_settings(with_residential_proxy=True))
        self.assertFalse(result.ok)
        self.assertTrue(any('Invalid scrape mode' in message for message in result.errors))

    def test_preflight_rejects_missing_proxy(self):
        result = preflight_parallel_scrape([object()], mode='collect', app_settings=self.make_settings(with_residential_proxy=False))
        self.assertFalse(result.ok)
        self.assertTrue(any('Missing Bright Data residential proxy configuration' in message for message in result.errors))

    def test_preflight_accepts_valid_collect_mode(self):
        result = preflight_parallel_scrape([object()], mode='collect', app_settings=self.make_settings(with_residential_proxy=True))
        self.assertTrue(result.ok)

    def test_preflight_accepts_datacenter_only_searches(self):
        search = type('Search', (), {'no_residential': True})()
        result = preflight_parallel_scrape(
            [search],
            mode='collect',
            app_settings=self.make_settings(with_residential_proxy=False, with_datacenter_proxy=True),
        )
        self.assertTrue(result.ok)

    def test_preflight_rejects_missing_datacenter_proxy_for_no_residential_searches(self):
        search = type('Search', (), {'no_residential': True})()
        result = preflight_parallel_scrape(
            [search],
            mode='collect',
            app_settings=self.make_settings(with_residential_proxy=True, with_datacenter_proxy=False),
        )
        self.assertFalse(result.ok)
        self.assertTrue(any('Missing Bright Data datacenter proxy configuration' in message for message in result.errors))

    def test_maybe_refresh_daily_eventual_sales_runs_once_per_process_day(self):
        searches = [object()]

        with patch('daily_eventual_sales.today_local_iso', return_value='2026-04-06'), \
             patch('daily_eventual_sales.refresh_daily_eventual_sales') as refresh_mock:
            current = maybe_refresh_daily_eventual_sales_for_running_scheduler(searches, None)
            same_day = maybe_refresh_daily_eventual_sales_for_running_scheduler(searches, current)

        self.assertEqual(current, '2026-04-06')
        self.assertEqual(same_day, '2026-04-06')
        refresh_mock.assert_called_once_with(searches, today_iso='2026-04-06')
