import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))

import config.logging_config as logging_config


class EventualSalesLoggingTests(unittest.TestCase):
    def test_eventual_sales_context_writes_to_separate_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_settings = SimpleNamespace(
                paths=SimpleNamespace(simple_scrape_dir=Path(tmpdir)),
                logging=SimpleNamespace(app_level="INFO", third_party_level="WARNING"),
            )
            with patch.object(logging_config, "settings", fake_settings):
                logging_config.configure_logging(force=True)
                with logging_config.eventual_sales_log_context():
                    logging.getLogger("urllib3.connectionpool").warning("proxy hiccup")

            log_path = Path(tmpdir) / logging_config.EVENTUAL_SALES_LOG_FILENAME
            self.assertTrue(log_path.exists())
            content = log_path.read_text(encoding="utf-8")
            self.assertIn("proxy hiccup", content)
            self.assertIn("urllib3.connectionpool", content)


if __name__ == "__main__":
    unittest.main()
