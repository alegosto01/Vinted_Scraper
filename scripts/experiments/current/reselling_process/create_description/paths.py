from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = ROOT / "scripts"
SIMPLE_SCRAPE_DIR = ROOT / "data" / "simple_scrape"
EXPERIMENT_ROOT = ROOT / "data" / "experiments" / "reselling_process" / "create_description"
RUNS_DIR = EXPERIMENT_ROOT / "runs"
REPORTS_DIR = EXPERIMENT_ROOT / "reports"


def ensure_project_imports() -> None:
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def ensure_experiment_dirs() -> None:
    for path in (RUNS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def assert_experiment_path(path: Path) -> Path:
    resolved = path.resolve()
    root = EXPERIMENT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Refusing to write outside experiment root: {resolved}")
    return resolved


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    def _default(value: Any):
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path = assert_experiment_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_default), encoding="utf-8")
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


def git_snapshot() -> dict[str, Any]:
    return {
        "branch": git_text(["branch", "--show-current"]),
        "status_short": git_text(["status", "--short"]),
        "head": git_text(["rev-parse", "--short", "HEAD"]),
    }


def write_manifest(path: Path, *, command: str, extra: dict[str, Any] | None = None) -> Path:
    payload: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "command": command,
        "git": git_snapshot(),
    }
    if extra:
        payload.update(extra)
    return write_json(path, payload)