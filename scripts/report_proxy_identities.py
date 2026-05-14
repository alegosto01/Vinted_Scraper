#!/usr/bin/env python3
from __future__ import annotations

import argparse

from utils_lib.proxy_identity_tracker import iter_proxy_identity_events, summarize_proxy_identities


def print_transport_summary(transport: str, stats: dict[str, object], top_ips: int) -> None:
    print(f"{transport}:")
    print(f"  events={stats['events']}")
    print(f"  successful_events={stats['successful_events']}")
    print(f"  unique_ips={stats['unique_ips']}")
    print(f"  change_events={stats['change_events']}")
    print(f"  first_seen={stats['first_timestamp']}")
    print(f"  last_seen={stats['last_timestamp']}")
    ips = stats.get("ips", {}) or {}
    if not ips:
        print("  top_ips=none")
        return
    print("  top_ips:")
    for index, (ip, count) in enumerate(ips.items(), start=1):
        if index > top_ips:
            break
        print(f"    {ip}: samples={count}")


def print_recent_changes(path: str | None, limit: int) -> None:
    changed_events = [event for event in iter_proxy_identity_events(path) if event.get("changed")]
    if not changed_events:
        print("Recent changes:")
        print("  none")
        return
    print("Recent changes:")
    for event in changed_events[-limit:]:
        print(
            "  "
            f"{event.get('timestamp')} transport={event.get('transport')} "
            f"{event.get('previous_ip')} -> {event.get('proxy_ip')} "
            f"request_count={event.get('request_count')} host={event.get('target_host')}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize proxy exit IP samples and change events.")
    ap.add_argument("--path", default=None, help="Optional path to a proxy_identity_stats.jsonl file")
    ap.add_argument("--top-ips", type=int, default=5, help="How many sampled IPs to show per transport")
    ap.add_argument("--recent-changes", type=int, default=10, help="How many IP change events to show")
    args = ap.parse_args()

    summary = summarize_proxy_identities(args.path)
    print(f"path: {summary['path']}")
    print(f"events: {summary['events']}")
    print(f"successful_events: {summary['successful_events']}")
    print(f"unique_ips: {summary['unique_ips']}")
    print(f"change_events: {summary['change_events']}")
    print("By transport:")
    if not summary["by_transport"]:
        print("  none")
    else:
        for transport, stats in summary["by_transport"].items():
            print_transport_summary(transport, stats, args.top_ips)
    print_recent_changes(args.path, args.recent_changes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
