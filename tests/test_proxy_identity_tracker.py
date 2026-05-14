import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils_lib.proxy_identity_tracker import (
    iter_proxy_identity_events,
    maybe_track_proxy_identity,
    reset_proxy_identity_state_for_tests,
    summarize_proxy_identities,
)


class ProxyIdentityTrackerTests(unittest.TestCase):
    def setUp(self):
        reset_proxy_identity_state_for_tests()

    def tearDown(self):
        reset_proxy_identity_state_for_tests()

    def test_sampling_tracks_unique_ips_and_changes(self):
        class FakeResponse:
            def __init__(self, ip):
                self.status_code = 200
                self.ok = True
                self.headers = {}
                self._ip = ip

            def json(self):
                return {"ip": self._ip, "country": "IT"}

        class FakeSession:
            def __init__(self):
                self.calls = 0

            def get(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return FakeResponse("1.1.1.1")
                return FakeResponse("2.2.2.2")

        with tempfile.TemporaryDirectory() as tmp_dir:
            stats_file = Path(tmp_dir) / "proxy_identity_stats.jsonl"
            env = {
                "VINTED_PROXY_IDENTITY_STATS_PATH": str(stats_file),
                "VINTED_PROXY_IDENTITY_SAMPLE_EVERY": "1",
                "VINTED_PROXY_IDENTITY_MIN_INTERVAL_SECONDS": "0",
            }
            with patch.dict(os.environ, env, clear=False):
                session = FakeSession()
                maybe_track_proxy_identity(
                    transport="datacenter_proxy",
                    proxy_url="http://proxy.example:1234",
                    request_url="https://www.vinted.it/catalog?page=1",
                    session=session,
                    headers={"User-Agent": "test"},
                )
                maybe_track_proxy_identity(
                    transport="datacenter_proxy",
                    proxy_url="http://proxy.example:1234",
                    request_url="https://www.vinted.it/catalog?page=2",
                    session=session,
                    headers={"User-Agent": "test"},
                )
                summary = summarize_proxy_identities()
                events = list(iter_proxy_identity_events())

        self.assertEqual(summary["events"], 2)
        self.assertEqual(summary["change_events"], 1)
        self.assertEqual(summary["unique_ips"], 2)
        self.assertEqual(summary["by_transport"]["datacenter_proxy"]["unique_ips"], 2)
        self.assertEqual(events[0]["proxy_ip"], "1.1.1.1")
        self.assertFalse(events[0]["changed"])
        self.assertEqual(events[1]["previous_ip"], "1.1.1.1")
        self.assertTrue(events[1]["changed"])

    def test_response_headers_can_record_identity_without_sample_request(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            stats_file = Path(tmp_dir) / "proxy_identity_stats.jsonl"
            with patch.dict(
                os.environ,
                {"VINTED_PROXY_IDENTITY_STATS_PATH": str(stats_file)},
                clear=False,
            ):
                session = Mock()
                maybe_track_proxy_identity(
                    transport="residential_proxy",
                    proxy_url="http://proxy.example:1234",
                    request_url="https://www.vinted.it/items/123",
                    session=session,
                    response_headers={"X-BRD-IP": "9.9.9.9", "X-BRD-Country": "IT"},
                )
                events = list(iter_proxy_identity_events())

        session.get.assert_not_called()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["source"], "response_header")
        self.assertEqual(events[0]["proxy_ip"], "9.9.9.9")
        self.assertEqual(events[0]["country"], "IT")


if __name__ == "__main__":
    unittest.main()
