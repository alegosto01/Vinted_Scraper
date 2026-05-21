from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests


_WRITE_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_STATE: dict[str, dict[str, Any]] = {}

_DEFAULT_CHECK_URL = "https://lumtest.com/myip.json"
_HEADER_IP_KEYS = (
    "x-brd-ip",
    "x-luminati-ip",
    "x-proxy-ip",
    "x-superproxy-ip",
)
_HEADER_COUNTRY_KEYS = (
    "x-brd-country",
    "x-luminati-country",
    "x-proxy-country",
)
_HEADER_CITY_KEYS = (
    "x-brd-city",
    "x-luminati-city",
    "x-proxy-city",
)


def proxy_identity_stats_path(path: str | Path | None = None) -> Path:
    if path is not None:
        resolved = Path(path)
    else:
        override = os.getenv("VINTED_PROXY_IDENTITY_STATS_PATH")
        if override:
            resolved = Path(override)
        else:
            from config.project_config import settings

            resolved = Path(str(settings.paths.simple_scrape_dir)) / "proxy_identity_stats.jsonl"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    try:
        return max(int(raw_value), minimum)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    try:
        return max(float(raw_value), minimum)
    except (TypeError, ValueError):
        return default


def _normalize_headers(headers: Any) -> dict[str, str]:
    if not headers:
        return {}
    normalized: dict[str, str] = {}
    try:
        items = dict(headers).items()
    except Exception:
        return {}
    for key, value in items:
        if key is None:
            continue
        normalized[str(key).lower()] = "" if value is None else str(value)
    return normalized


def _first_present(headers: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = headers.get(key)
        if value:
            return value
    return None


def _parse_identity_payload(response: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        candidate = response.json()
        if isinstance(candidate, dict):
            payload = candidate
    except Exception:
        try:
            text = getattr(response, "text", "")
            candidate = json.loads(text) if isinstance(text, str) and text.strip() else {}
            if isinstance(candidate, dict):
                payload = candidate
        except Exception:
            payload = {}

    headers = _normalize_headers(getattr(response, "headers", None))
    proxy_ip = payload.get("ip") or payload.get("address") or _first_present(headers, _HEADER_IP_KEYS)
    country = payload.get("country") or payload.get("country_code") or _first_present(headers, _HEADER_COUNTRY_KEYS)
    city = payload.get("city") or _first_present(headers, _HEADER_CITY_KEYS)
    return {
        "proxy_ip": None if proxy_ip in (None, "") else str(proxy_ip),
        "country": None if country in (None, "") else str(country),
        "city": None if city in (None, "") else str(city),
    }


def _next_request_state(transport: str, *, sample_every: int, min_interval_seconds: float) -> tuple[int, bool]:
    now = time.monotonic()
    with _STATE_LOCK:
        state = _STATE.setdefault(
            str(transport),
            {
                "request_count": 0,
                "last_sample_request_count": 0,
                "last_sample_monotonic": 0.0,
                "last_ip": None,
            },
        )
        state["request_count"] += 1
        request_count = int(state["request_count"])
        requests_since_sample = request_count - int(state["last_sample_request_count"])
        should_sample = (
            request_count == 1
            or requests_since_sample >= sample_every
            or (state["last_sample_monotonic"] and (now - float(state["last_sample_monotonic"])) >= min_interval_seconds)
        )
        if should_sample:
            state["last_sample_request_count"] = request_count
            state["last_sample_monotonic"] = now
        return request_count, should_sample


def _mark_request_as_sampled(transport: str, request_count: int) -> None:
    with _STATE_LOCK:
        state = _STATE.setdefault(str(transport), {})
        state["last_sample_request_count"] = int(request_count)
        state["last_sample_monotonic"] = time.monotonic()


def reset_proxy_identity_state_for_tests() -> None:
    with _STATE_LOCK:
        _STATE.clear()


def record_proxy_identity_event(
    *,
    transport: str,
    request_url: str,
    request_count: int,
    source: str,
    proxy_ip: str | None = None,
    country: str | None = None,
    city: str | None = None,
    changed: bool | None = None,
    previous_ip: str | None = None,
    ok: bool | None = None,
    status_code: int | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transport": str(transport),
        "request_url": str(request_url),
        "target_host": urlparse(str(request_url)).netloc,
        "request_count": int(request_count),
        "source": str(source),
        "proxy_ip": None if proxy_ip in (None, "") else str(proxy_ip),
        "country": None if country in (None, "") else str(country),
        "city": None if city in (None, "") else str(city),
        "changed": bool(changed) if changed is not None else False,
        "previous_ip": None if previous_ip in (None, "") else str(previous_ip),
        "ok": ok,
        "status_code": None if status_code is None else int(status_code),
        "error": None if error in (None, "") else str(error),
        "metadata": metadata or {},
    }
    output_path = proxy_identity_stats_path(path)
    with _WRITE_LOCK:
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")


def _record_observation(
    *,
    transport: str,
    request_url: str,
    request_count: int,
    source: str,
    identity: dict[str, Any],
    ok: bool,
    status_code: int | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> None:
    proxy_ip = identity.get("proxy_ip")
    previous_ip = None
    changed = False
    if proxy_ip:
        with _STATE_LOCK:
            state = _STATE.setdefault(str(transport), {})
            previous_ip = state.get("last_ip")
            changed = previous_ip is not None and previous_ip != proxy_ip
            state["last_ip"] = proxy_ip

    record_proxy_identity_event(
        transport=transport,
        request_url=request_url,
        request_count=request_count,
        source=source,
        proxy_ip=proxy_ip,
        country=identity.get("country"),
        city=identity.get("city"),
        changed=changed,
        previous_ip=previous_ip,
        ok=ok,
        status_code=status_code,
        error=error,
        metadata=metadata,
        path=path,
    )


def maybe_track_proxy_identity(
    *,
    transport: str,
    proxy_url: str | None,
    request_url: str,
    session: requests.Session | Any | None,
    headers: dict[str, str] | None = None,
    response_headers: Any = None,
    verify: bool = False,
    timeout: tuple[int, int] = (10, 25),
    path: str | Path | None = None,
) -> None:
    if not proxy_url or transport != "datacenter_proxy":
        return

    sample_every = _env_int("VINTED_PROXY_IDENTITY_SAMPLE_EVERY", 20)
    min_interval_seconds = _env_float("VINTED_PROXY_IDENTITY_MIN_INTERVAL_SECONDS", 300.0)
    request_count, should_sample = _next_request_state(
        transport,
        sample_every=sample_every,
        min_interval_seconds=min_interval_seconds,
    )

    normalized_headers = _normalize_headers(response_headers)
    header_identity = {
        "proxy_ip": _first_present(normalized_headers, _HEADER_IP_KEYS),
        "country": _first_present(normalized_headers, _HEADER_COUNTRY_KEYS),
        "city": _first_present(normalized_headers, _HEADER_CITY_KEYS),
    }
    if header_identity["proxy_ip"]:
        _mark_request_as_sampled(transport, request_count)
        _record_observation(
            transport=transport,
            request_url=request_url,
            request_count=request_count,
            source="response_header",
            identity=header_identity,
            ok=True,
            metadata={"sampled": False},
            path=path,
        )
        return

    if not should_sample or session is None:
        return

    proxies = {"http": proxy_url, "https": proxy_url}
    check_url = os.getenv("VINTED_PROXY_IDENTITY_CHECK_URL", _DEFAULT_CHECK_URL)
    try:
        response = session.get(
            check_url,
            headers=headers or {},
            proxies=proxies,
            timeout=timeout,
            verify=verify,
        )
        identity = _parse_identity_payload(response)
        ok = bool(getattr(response, "ok", None))
        if getattr(response, "ok", None) is None:
            ok = int(getattr(response, "status_code", 0) or 0) < 400
        _record_observation(
            transport=transport,
            request_url=request_url,
            request_count=request_count,
            source="sample_request",
            identity=identity,
            ok=ok and bool(identity.get("proxy_ip")),
            status_code=getattr(response, "status_code", None),
            metadata={"sampled": True, "check_url": check_url},
            path=path,
        )
    except requests.RequestException as exc:
        record_proxy_identity_event(
            transport=transport,
            request_url=request_url,
            request_count=request_count,
            source="sample_request",
            ok=False,
            error=str(exc),
            metadata={"sampled": True, "check_url": check_url},
            path=path,
        )


def iter_proxy_identity_events(path: str | Path | None = None) -> Iterable[dict[str, Any]]:
    input_path = proxy_identity_stats_path(path)
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


def summarize_proxy_identities(path: str | Path | None = None) -> dict[str, Any]:
    input_path = proxy_identity_stats_path(path)
    summary = {
        "path": str(input_path.resolve()),
        "events": 0,
        "successful_events": 0,
        "change_events": 0,
        "unique_ips": 0,
        "by_transport": {},
    }
    if not input_path.exists():
        return summary

    all_ips: set[str] = set()
    transport_stats: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "events": 0,
            "successful_events": 0,
            "change_events": 0,
            "unique_ips": set(),
            "ips": Counter(),
            "first_timestamp": None,
            "last_timestamp": None,
        }
    )

    for event in iter_proxy_identity_events(input_path):
        transport = str(event.get("transport", "unknown") or "unknown")
        stats = transport_stats[transport]
        stats["events"] += 1
        summary["events"] += 1

        timestamp = event.get("timestamp")
        if stats["first_timestamp"] is None:
            stats["first_timestamp"] = timestamp
        stats["last_timestamp"] = timestamp

        ok = bool(event.get("ok"))
        if ok:
            stats["successful_events"] += 1
            summary["successful_events"] += 1

        if bool(event.get("changed")):
            stats["change_events"] += 1
            summary["change_events"] += 1

        proxy_ip = event.get("proxy_ip")
        if proxy_ip:
            proxy_ip_str = str(proxy_ip)
            stats["unique_ips"].add(proxy_ip_str)
            stats["ips"][proxy_ip_str] += 1
            all_ips.add(proxy_ip_str)

    formatted_transport: dict[str, dict[str, Any]] = {}
    for transport, stats in transport_stats.items():
        formatted_transport[transport] = {
            "events": stats["events"],
            "successful_events": stats["successful_events"],
            "change_events": stats["change_events"],
            "unique_ips": len(stats["unique_ips"]),
            "ips": dict(stats["ips"].most_common()),
            "first_timestamp": stats["first_timestamp"],
            "last_timestamp": stats["last_timestamp"],
        }

    summary["unique_ips"] = len(all_ips)
    summary["by_transport"] = dict(
        sorted(formatted_transport.items(), key=lambda item: (-item[1]["events"], item[0]))
    )
    return summary
