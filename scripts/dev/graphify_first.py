#!/usr/bin/env python3
"""Start a low-token code investigation with Graphify.

Usage:
    python scripts/dev/graphify_first.py "where is Telegram sending decided?"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUDGET = 1500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a scoped Graphify query before reading source files.")
    parser.add_argument("question", help="Natural-language codebase question to ask Graphify.")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="Token budget for graphify query output.")
    parser.add_argument("--dfs", action="store_true", help="Use DFS traversal instead of Graphify's default BFS.")
    parser.add_argument(
        "--update-if-missing",
        action="store_true",
        help="Run 'graphify update . --force' if graphify-out/graph.json is missing.",
    )
    return parser.parse_args()


def ensure_graph(update_if_missing: bool) -> None:
    graph_path = ROOT / "graphify-out" / "graph.json"
    if graph_path.exists():
        return
    if not update_if_missing:
        raise SystemExit(
            "Missing graphify-out/graph.json. Run 'graphify update . --force' "
            "or pass --update-if-missing."
        )
    subprocess.run(["graphify", "update", ".", "--force"], cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    ensure_graph(args.update_if_missing)

    cmd = ["graphify", "query", args.question, "--budget", str(args.budget)]
    if args.dfs:
        cmd.append("--dfs")
    subprocess.run(cmd, cwd=ROOT, check=True)

    print(
        "\nNext: read only the files/nodes Graphify surfaced. "
        "Use 'graphify explain \"<node>\"' for a focused concept or "
        "'graphify path \"<A>\" \"<B>\"' for relationships.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

