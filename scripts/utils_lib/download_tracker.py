from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


_WRITE_LOCK = threading.Lock()


def stats_path(path: str | Path | None = None) -> Path:
    if path is not None:
        resolved = Path(path)
    else:
        override = os.getenv("VINTED_DOWNLOAD_STATS_PATH")
        if override:
            resolved = Path(override)
        else:
            from config.project_config import settings

            resolved = settings.paths.download_stats_path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def estimate_response_bytes_with_source(response: Any, *, body: bytes | None = None):
    """Bytes plus where they came from: the wire size or the decompressed body."""
    size = estimate_response_bytes(response, body=body)
    if size is None:
        return None, "unknown"
    header = None
    try:
        header = (getattr(response, "headers", {}) or {}).get("content-length")
    except Exception:
        header = None
    return size, "wire" if header not in (None, "") else "decompressed"


def estimate_response_bytes(response: Any, *, body: bytes | None = None) -> int | None:
    if response is None:
        return None

    header_value = None
    try:
        header_value = (getattr(response, "headers", {}) or {}).get("content-length")
    except Exception:
        header_value = None

    if header_value not in (None, ""):
        try:
            parsed = int(str(header_value))
            if parsed >= 0:
                return parsed
        except (TypeError, ValueError):
            pass

    if body is not None:
        return len(body)

    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return len(content)
    if isinstance(content, str):
        return len(content.encode("utf-8", errors="ignore"))

    text = getattr(response, "text", None)
    if isinstance(text, str):
        return len(text.encode("utf-8", errors="ignore"))

    return None


def record_download(
    *,
    kind: str,
    transport: str,
    url: str,
    bytes_downloaded: int | None,
    status_code: int | None = None,
    ok: bool | None = None,
    metadata: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": str(kind),
        "transport": str(transport),
        "url": str(url),
        "host": urlparse(str(url)).netloc,
        "bytes_downloaded": int(bytes_downloaded or 0),
        "status_code": None if status_code is None else int(status_code),
        "ok": ok,
        "metadata": metadata or {},
    }

    output_path = stats_path(path)
    with _WRITE_LOCK:
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")


def iter_download_events(path: str | Path | None = None) -> Iterable[dict[str, Any]]:
    input_path = stats_path(path)
    if not input_path.exists():
        return []

    events: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return events


def summarize_downloads(path: str | Path | None = None) -> dict[str, Any]:
    input_path = stats_path(path)
    summary = {
        "path": str(input_path.resolve()),
        "events": 0,
        "total_bytes": 0,
        "by_transport": {},
        "by_kind": {},
        "by_host": {},
    }
    if not input_path.exists():
        return summary

    by_transport: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"events": 0, "bytes": 0})
    by_kind: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"events": 0, "bytes": 0})
    by_host: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"events": 0, "bytes": 0})

    total_events = 0
    total_bytes = 0
    for event in iter_download_events(input_path):
        total_events += 1
        size = int(event.get("bytes_downloaded", 0) or 0)
        total_bytes += size
        transport = str(event.get("transport", "unknown") or "unknown")
        kind = str(event.get("kind", "unknown") or "unknown")
        host = str(event.get("host", "") or "")

        by_transport[transport]["events"] += 1
        by_transport[transport]["bytes"] += size
        by_kind[kind]["events"] += 1
        by_kind[kind]["bytes"] += size
        by_host[host]["events"] += 1
        by_host[host]["bytes"] += size

    summary["events"] = total_events
    summary["total_bytes"] = total_bytes
    summary["by_transport"] = dict(sorted(by_transport.items(), key=lambda item: (-item[1]["bytes"], item[0])))
    summary["by_kind"] = dict(sorted(by_kind.items(), key=lambda item: (-item[1]["bytes"], item[0])))
    summary["by_host"] = dict(sorted(by_host.items(), key=lambda item: (-item[1]["bytes"], item[0])))
    return summary
