import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils_lib.download_tracker import record_download, summarize_downloads


class DownloadTrackerTests(unittest.TestCase):
    def test_record_and_summarize_downloads(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            stats_file = Path(tmp_dir) / "download_stats.jsonl"
            with patch.dict(os.environ, {"VINTED_DOWNLOAD_STATS_PATH": str(stats_file)}, clear=False):
                record_download(
                    kind="catalog_page",
                    transport="residential_proxy",
                    url="https://www.vinted.it/catalog?search_text=ps4",
                    bytes_downloaded=1024,
                    status_code=200,
                    ok=True,
                )
                record_download(
                    kind="listing_image",
                    transport="direct",
                    url="https://images1.vinted.net/example.webp",
                    bytes_downloaded=2048,
                    status_code=200,
                    ok=True,
                )

                summary = summarize_downloads()

        self.assertEqual(summary["events"], 2)
        self.assertEqual(summary["total_bytes"], 3072)
        self.assertEqual(summary["by_transport"]["direct"]["bytes"], 2048)
        self.assertEqual(summary["by_transport"]["residential_proxy"]["bytes"], 1024)
        self.assertEqual(summary["by_kind"]["catalog_page"]["events"], 1)
        self.assertEqual(summary["by_host"]["www.vinted.it"]["bytes"], 1024)

    def test_missing_stats_file_returns_empty_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            stats_file = Path(tmp_dir) / "missing.jsonl"
            with patch.dict(os.environ, {"VINTED_DOWNLOAD_STATS_PATH": str(stats_file)}, clear=False):
                summary = summarize_downloads()

        self.assertEqual(summary["events"], 0)
        self.assertEqual(summary["total_bytes"], 0)
