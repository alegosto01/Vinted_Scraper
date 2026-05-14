from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.clustering_approach.vinted_pipeline_incremental import *  # noqa: F401,F403
from experiments.clustering_approach.vinted_pipeline_incremental import parse_args, run


if __name__ == "__main__":
    run(parse_args())
