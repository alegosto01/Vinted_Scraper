import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from config.project_config import LoggingConfig, PathsConfig, ProxyConfig, Settings, TelegramConfig


class RuntimeValidationTests(unittest.TestCase):
    def test_validate_for_simple_scrape_requires_searches(self):
        settings = Settings(
            paths=PathsConfig(project_root=Path('.'), data_dir=Path('.'), simple_scrape_dir=Path('.'), full_scrape_dir=Path('.'), models_dir=Path('.'), searches_yaml=Path('missing.yaml'), brand_ids_csv=Path('missing_brand_ids.csv')),
            telegram=TelegramConfig(bot_token=None, chat_id=None),
            proxy=ProxyConfig(datacenter_username=None, datacenter_password=None, datacenter_proxy_url_override=None),
            logging=LoggingConfig(),
        )

        result = settings.validate_for_simple_scrape([], require_proxy=True)

        self.assertFalse(result.ok)
        self.assertTrue(any('No enabled searches' in message for message in result.errors))
        self.assertTrue(any('Missing Bright Data datacenter proxy configuration' in message for message in result.errors))

    def test_validate_for_simple_scrape_can_skip_proxy_requirement(self):
        temp_dir = Path(__file__).resolve().parent
        settings = Settings(
            paths=PathsConfig(project_root=temp_dir, data_dir=temp_dir, simple_scrape_dir=temp_dir, full_scrape_dir=temp_dir, models_dir=temp_dir, searches_yaml=Path(__file__).resolve(), brand_ids_csv=Path(__file__).resolve()),
            telegram=TelegramConfig(bot_token='token', chat_id='chat'),
            proxy=ProxyConfig(datacenter_username=None, datacenter_password=None, datacenter_proxy_url_override=None),
            logging=LoggingConfig(),
        )

        result = settings.validate_for_simple_scrape([object()], require_proxy=False)

        self.assertTrue(result.ok)

    def test_runtime_paths_default_under_project_root(self):
        root = Path('/tmp/vinted-test-root')
        paths = PathsConfig(project_root=root)

        self.assertEqual(paths.runtime_root, root / 'runtime')
        self.assertEqual(paths.runtime_logs_root, root / 'runtime' / 'logs')
        self.assertEqual(paths.runtime_pids_root, root / 'runtime' / 'pids')
        self.assertEqual(paths.simple_scrape_runtime_logs_dir, root / 'runtime' / 'logs' / 'simple_scrape')
        self.assertEqual(paths.scraper_log_path, root / 'runtime' / 'logs' / 'simple_scrape' / 'usual_live_scrape.log')
        self.assertEqual(paths.eventual_sales_log_path, root / 'runtime' / 'logs' / 'simple_scrape' / 'eventual_sale.log')
        self.assertEqual(paths.download_stats_path, root / 'runtime' / 'logs' / 'simple_scrape' / 'download_stats.jsonl')
        self.assertEqual(paths.proxy_identity_stats_path, root / 'runtime' / 'logs' / 'simple_scrape' / 'proxy_identity_stats.jsonl')
