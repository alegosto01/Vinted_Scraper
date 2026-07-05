"""Paths and helpers for the basic_plus_visual experiment."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
EXPERIMENT_ROOT = Path(__file__).resolve().parent / "data"
OFFLINE_RUNS_DIR = EXPERIMENT_ROOT / "offline_runs"
LIVE_RUNS_DIR = EXPERIMENT_ROOT / "live_runs"
MODELS_DIR = EXPERIMENT_ROOT / "models"
REPORTS_DIR = EXPERIMENT_ROOT / "reports"


def run_id(prefix: str) -> str:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def git_snapshot() -> dict[str, str]:
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()
        return {"branch": branch, "head": head, "status_short": status}
    except Exception:
        return {"branch": "", "head": "", "status_short": ""}


def write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def write_manifest(path: Path, *, command: str = "", extra: dict[str, object] | None = None) -> None:
    manifest = {
        "created_at": utc_now_iso(),
        "command": command,
        "git": git_snapshot(),
    }
    if extra:
        manifest.update(extra)
    write_json(path, manifest)


def assert_experiment_path(path: Path) -> Path:
    path = Path(path).resolve()
    root = EXPERIMENT_ROOT.resolve()
    if not str(path).startswith(str(root)):
        raise ValueError(f"Path {path} is outside experiment root {root}")
    return path


def ensure_experiment_dirs() -> None:
    for d in (OFFLINE_RUNS_DIR, LIVE_RUNS_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
