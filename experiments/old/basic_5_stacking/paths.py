from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parent / "data"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def ensure_experiment_dirs() -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)


def assert_experiment_path(path: Path) -> Path:
    resolved = path.resolve()
    root = EXPERIMENT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Refusing to write outside basic_5_stacking: {resolved}")
    return resolved


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = assert_experiment_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def git_text(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        return f"<git unavailable: {type(exc).__name__}: {exc}>"
    return result.stdout.strip()


def write_manifest(path: Path, *, command: str, extra: dict[str, Any]) -> Path:
    return write_json(
        path,
        {
            "created_at": utc_now_iso(),
            "command": command,
            "git": {
                "branch": git_text(["branch", "--show-current"]),
                "status_short": git_text(["status", "--short"]),
                "head": git_text(["rev-parse", "--short", "HEAD"]),
            },
            **extra,
        },
    )
