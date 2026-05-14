from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPTS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis_pipeline._legacy_redirect import load_legacy_module, run_legacy_module

load_legacy_module(globals(), "analysis_pipeline.evaluation.cache_listing_images")

if __name__ == "__main__":
    run_legacy_module("analysis_pipeline.evaluation.cache_listing_images")

