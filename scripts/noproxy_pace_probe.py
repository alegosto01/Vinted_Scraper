"""No-proxy scrape-pace probe for the laptop (hobby / gioielli / telefoni).

Answers: how many catalog pages can a single home IP pull before Vinted/DataDome
soft-blocks, how fast does the quota recover, and does the discovery regime
(3 searches x P pages every 15 min) sustain with NO proxy.

Reuses the real search config + URL builder + parsing. Fetches with curl_cffi
(Chrome TLS impersonation) and NO proxy. Writes logs to --out (scratchpad).

Modes:
  probe    : cold burst until soft-block -> N pages; then poll to measure recovery
  cadence  : run 3 searches x P pages every I minutes for C cycles; report blocks
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests_html  # noqa: E402
from curl_cffi import requests as cffi  # noqa: E402

from simple_scraper import Simple_scraper  # noqa: E402
from config.project_config import settings  # noqa: E402
from config.search_loader import load_searches  # noqa: E402

SEARCHES = ["hobby_collezionismo", "donna_accessori_gioielli", "telefoni"]
PRODUCT_SEL = ".new-item-box__container"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.vinted.it/",
    "Connection": "keep-alive",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_base_urls(scraper) -> dict[str, str]:
    searches = load_searches(str(settings.paths.searches_yaml))
    urls = {}
    for name in SEARCHES:
        s = searches[name]
        s.sort = "newest_first"
        urls[name] = scraper.create_webpage(s)  # exact URL the loop hits
    return urls


def fetch(session, url: str) -> tuple[int | None, int, int]:
    """Return (status, item_count, body_len). item_count 0 on soft-block."""
    try:
        r = session.get(url, headers=HEADERS, timeout=30)
    except Exception as e:  # noqa: BLE001
        print(f"    ERR {type(e).__name__}: {e}")
        return None, 0, 0
    items = len(requests_html.HTML(html=r.text).find(PRODUCT_SEL)) if r.status_code == 200 else 0
    return r.status_code, items, len(r.text)


def classify(status, items) -> str:
    if status in (403, 429):
        return f"HARD_BLOCK({status})"
    if status == 200 and items == 0:
        return "SOFT_BLOCK"
    if status == 200:
        return "OK"
    return f"HTTP_{status}"


def write_row(log: Path, row: dict) -> None:
    new = not log.exists()
    with log.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def probe(urls: dict[str, str], out: Path, max_pages: int, gap: float, recov_cap: int) -> None:
    """Cold burst until soft-block, then poll every 60s to measure recovery."""
    log = out / "probe.csv"
    session = cffi.Session(impersonate="chrome")
    names = list(urls)
    print(f"\n=== MODE A: cold burst (gap={gap}s, up to {max_pages} pages) ===")
    blocked_at = None
    for n in range(1, max_pages + 1):
        name = names[(n - 1) % len(names)]
        page = (n - 1) // len(names) + 1
        url = f"{urls[name]}&page={page}"
        status, items, blen = fetch(session, url)
        state = classify(status, items)
        print(f"  fetch {n:2d}: {state:14s} {name} page={page} items={items} len={blen}")
        write_row(log, {"ts": now(), "phase": "burst", "n": n, "search": name,
                        "page": page, "status": status, "items": items, "state": state})
        if state != "OK":
            blocked_at = n
            break
        time.sleep(gap)

    if blocked_at is None:
        print(f"  NO BLOCK within {max_pages} pages -> ceiling is HIGHER than tested.")
        return
    print(f"\n  >>> BLOCKED at page #{blocked_at} (so ~{blocked_at - 1} clean pages from cold)")

    print(f"\n=== recovery poll (1 page/60s, cap {recov_cap} min) ===")
    probe_url = f"{urls[names[0]]}&page=1"
    for minute in range(1, recov_cap + 1):
        time.sleep(60)
        status, items, _ = fetch(session, probe_url)
        state = classify(status, items)
        print(f"  +{minute:2d}min: {state:14s} items={items}")
        write_row(log, {"ts": now(), "phase": "recovery", "n": minute, "search": names[0],
                        "page": 1, "status": status, "items": items, "state": state})
        if state == "OK":
            print(f"  >>> RECOVERED after ~{minute} min")
            return
    print(f"  >>> still blocked after {recov_cap} min")


def cadence(urls: dict[str, str], out: Path, pages: int, interval: float, cycles: int) -> None:
    """Run 3 searches x `pages` pages every `interval` min for `cycles` cycles."""
    log = out / "cadence.csv"
    session = cffi.Session(impersonate="chrome")
    print(f"\n=== MODE B: {len(urls)} searches x {pages} pages every {interval}min "
          f"x {cycles} cycles ({len(urls) * pages} pages/scan) ===")
    for c in range(1, cycles + 1):
        scan_ok = scan_soft = scan_hard = 0
        for name, base in urls.items():
            for page in range(1, pages + 1):
                status, items, _ = fetch(session, f"{base}&page={page}")
                state = classify(status, items)
                scan_ok += state == "OK"
                scan_soft += state == "SOFT_BLOCK"
                scan_hard += state.startswith("HARD")
                write_row(log, {"ts": now(), "cycle": c, "search": name, "page": page,
                                "status": status, "items": items, "state": state})
                time.sleep(7)  # matches the prod loop's per-page pacing
        verdict = "SUSTAINED" if scan_soft == 0 and scan_hard == 0 else "BLOCKED"
        print(f"  cycle {c}: OK={scan_ok} soft={scan_soft} hard={scan_hard} -> {verdict}")
        if c < cycles:
            time.sleep(max(0.0, interval * 60.0 - len(urls) * pages * 7))
    print("=== cadence done ===")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["probe", "cadence"])
    ap.add_argument("--out", default="/tmp/claude-1000/-home-ale-Desktop-vinted-Vinted-"
                    "New-Version/1ea4fd98-9614-499c-82af-7b26e55dc216/scratchpad/noproxy_probe")
    ap.add_argument("--max-pages", type=int, default=20)
    ap.add_argument("--gap", type=float, default=2.0)
    ap.add_argument("--recov-cap", type=int, default=15)
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--cycles", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scraper = Simple_scraper()
    urls = build_base_urls(scraper)
    print(f"searches: {list(urls)}")
    for n, u in urls.items():
        print(f"  {n}: {u}")

    if args.mode == "probe":
        probe(urls, out, args.max_pages, args.gap, args.recov_cap)
    else:
        cadence(urls, out, args.pages, args.interval, args.cycles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
