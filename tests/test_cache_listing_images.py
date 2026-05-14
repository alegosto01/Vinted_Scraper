import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
ANALYSIS_DIR = SCRIPTS_DIR / "analysis_pipeline"
for path in (SCRIPTS_DIR, ANALYSIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis_pipeline.evaluation.cache_listing_images import build_cache_paths, safe_token


class CacheListingImagesTests(unittest.TestCase):
    def test_safe_token_normalizes_folder_names(self):
        self.assertEqual(safe_token("ps4 / sold_df", "fallback"), "ps4_sold_df")
        self.assertEqual(safe_token("", "fallback"), "fallback")

    def test_build_cache_paths_is_tidy_and_stable(self):
        root = Path("/tmp/cache-root")
        path = build_cache_paths(
            dataset_root=root / "sold_df",
            data_id="8292590185",
            image_index=1,
        )
        self.assertEqual(path, root / "sold_df" / "8292590185" / "image_01")

    def test_cached_csv_shape_supports_local_images_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame([{"Dataid": "1", "Images": "https://example.com/a.webp"}])
            df["LocalImagePaths"] = [json.dumps([str(Path(tmp) / "a.webp")])]
            df["LocalPrimaryImagePath"] = [str(Path(tmp) / "a.webp")]
            self.assertIn("LocalImagePaths", df.columns)
            self.assertIn("LocalPrimaryImagePath", df.columns)


if __name__ == "__main__":
    unittest.main()
