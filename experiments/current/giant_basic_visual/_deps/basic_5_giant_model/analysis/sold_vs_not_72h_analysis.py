#!/usr/bin/env python3
"""Run the sold-within-72h vs not-sold analysis pipeline end to end."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import common  # noqa: E402


STAGES = (
    ("data_prep", "01_data_prep.py"),
    ("text_features", "02_text_features.py"),
    ("stats", "03_stats.py"),
    ("report", "04_visualize_report.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["all", *[name for name, _ in STAGES]],
        default="all",
        help="Run the full pipeline or a single stage.",
    )
    return parser.parse_args()


def run_stage(name: str, script_name: str) -> None:
    script_path = SCRIPT_DIR / script_name
    print(f"\n=== {name}: {script_path.name} ===")
    runpy.run_path(str(script_path), run_name="__main__")


def main() -> int:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vinted-analysis")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    args = parse_args()
    stages = STAGES if args.stage == "all" else tuple(s for s in STAGES if s[0] == args.stage)
    for name, script_name in stages:
        run_stage(name, script_name)

    print(f"\nOutputs written to: {common.OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
