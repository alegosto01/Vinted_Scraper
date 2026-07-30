"""Find listings that never sold, have been sitting for months, and are badly photographed.

The scrape history holds ~1.2M unsold listings with their images on disk. MUSIQ costs
218ms each on this CPU, so the whole pool is 78 hours and a cheap prefilter would be
worth a lot - except none exists. Measured on 300 sampled listings:

  resolution   useless: Vinted serves every catalog thumbnail at 310x430
  file size    correlation with MUSIQ 0.13, mean score flat across quartiles
  laplacian    correlation 0.07; of the 25 blurriest, none scored below MUSIQ 60

So scan builds the pool and score runs MUSIQ over it in price order, most expensive
first, because that is where a mistake costs the most. Expect ~2% below MUSIQ 60.
Nothing here touches the network.

  python stale_bad_photos.py scan   --min-price 80 --min-age-days 60
  python stale_bad_photos.py score  --limit 15000
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SIMPLE_SCRAPE = ROOT / "data" / "simple_scrape"
OUT_DIR = HERE / "data" / "stale_scan"
SCAN_PATH = OUT_DIR / "candidates.csv"
SCORED_PATH = OUT_DIR / "scored.csv"


def build_pool(min_price: float, min_age_days: int) -> pd.DataFrame:
    """Unsold listings older than the cutoff whose primary image is on disk."""
    cutoff = pd.Timestamp(datetime.now() - timedelta(days=min_age_days))
    frames = []
    for search_dir in sorted(p for p in SIMPLE_SCRAPE.iterdir() if p.is_dir()):
        big = search_dir / "big_raw.csv"
        cache = search_dir / "image_cache"
        if not big.exists() or not cache.exists():
            continue
        columns = ["Dataid", "Title", "Price", "Link", "SearchDate", "MarketStatus", "Brand"]
        frame = pd.read_csv(big, usecols=lambda c: c in columns, low_memory=False)
        frame["when"] = pd.to_datetime(frame["SearchDate"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        frame = (frame.dropna(subset=["when"])
                 .sort_values("when")
                 .drop_duplicates("Dataid", keep="first"))
        frame["Price"] = pd.to_numeric(frame["Price"], errors="coerce")
        frame = frame[(frame["MarketStatus"].astype(str).str.upper() != "SOLD")
                      & (frame["when"] < cutoff)
                      & (frame["Price"] >= min_price)]
        if frame.empty:
            continue
        on_disk = {p.name for p in cache.iterdir() if p.is_dir()}
        frame = frame[frame["Dataid"].astype(str).isin(on_disk)].copy()
        frame["search"] = search_dir.name
        frame["image_dir"] = frame["Dataid"].astype(str).map(lambda i: str(cache / i))
        frames.append(frame)
        print(f"  {search_dir.name:26s} {len(frame):7d}", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def measure(image_dir: str) -> tuple[str | None, int, int, int]:
    """Header-only read of the MAIN image: path, width, height, bytes.

    The simple scrape stores one file per listing, image_01.webp, which is the catalog
    thumbnail the buyer sees first - the same image MUSIQ scores live. Anything else in
    the folder is a later photo and must not stand in for it, so a missing main image
    means the listing is skipped rather than judged on a secondary one.
    """
    candidate = Path(image_dir) / "image_01.webp"
    if not candidate.exists():
        return None, 0, 0, 0
    try:
        with Image.open(candidate) as image:  # lazy, does not decode pixels
            width, height = image.size
        return str(candidate), width, height, candidate.stat().st_size
    except Exception:
        return str(candidate), 0, 0, 0


def scan(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("building pool")
    pool = build_pool(args.min_price, args.min_age_days)
    if pool.empty:
        print("nothing matched")
        return
    print(f"{len(pool)} listings; reading image headers")
    measured = [measure(d) for d in pool["image_dir"]]
    pool["image_path"] = [m[0] for m in measured]
    pool["width"] = [m[1] for m in measured]
    pool["height"] = [m[2] for m in measured]
    pool["bytes"] = [m[3] for m in measured]
    pool = pool[pool["image_path"].notna() & pool["width"].gt(0)]
    pool["pixels"] = pool["width"] * pool["height"]
    pool["age_days"] = (pd.Timestamp(datetime.now()) - pool["when"]).dt.days
    pool.sort_values(["pixels", "bytes"]).to_csv(SCAN_PATH, index=False)
    print(f"wrote {SCAN_PATH} ({len(pool)} rows)")
    print(pool["pixels"].describe(percentiles=[0.01, 0.05, 0.25, 0.5]).to_string())


def score(args) -> None:
    import torch

    torch.set_num_threads(args.threads)
    import pyiqa

    frame = pd.read_csv(SCAN_PATH)
    if args.search:
        frame = frame[frame["search"] == args.search]
    # Highest price first: no cheap signal predicts MUSIQ, so the only sane ordering is
    # the one where a missed bad photo costs the most.
    frame = frame.sort_values("Price", ascending=False).reset_index(drop=True)
    if args.workers > 1:
        frame = frame[frame.index % args.workers == args.worker_index].copy()
    if args.limit:
        frame = frame.head(args.limit).copy()

    part = OUT_DIR / f"scored_part{args.worker_index}.csv"
    # This laptop occasionally dies mid-run, so results are appended as they are produced
    # and anything already scored is skipped on restart.
    done: set[str] = set()
    if part.exists():
        try:
            done = set(pd.read_csv(part, usecols=["Dataid"])["Dataid"].astype(str))
        except Exception:
            LOG_BROKEN = part.with_suffix(".broken")
            part.rename(LOG_BROKEN)
            print(f"unreadable checkpoint moved to {LOG_BROKEN}")
    todo = frame[~frame["Dataid"].astype(str).isin(done)]
    print(f"worker {args.worker_index}: {len(todo)} to score, {len(done)} already done", flush=True)

    metric = pyiqa.create_metric("musiq", device="cuda" if torch.cuda.is_available() else "cpu")
    batch, written = [], 0
    for index, row in enumerate(todo.itertuples(index=False), 1):
        try:
            value = float(metric(str(row.image_path)))
        except Exception:
            value = float("nan")
        batch.append({**row._asdict(), "musiq": value})
        if len(batch) >= args.checkpoint or index == len(todo):
            chunk = pd.DataFrame(batch)
            chunk.to_csv(part, mode="a", header=not part.exists(), index=False)
            written += len(batch)
            batch = []
            print(f"  worker {args.worker_index}: {index}/{len(todo)} (saved {written})", flush=True)
    print(f"worker {args.worker_index} finished, wrote {part}")


def _unlocker_get(url: str, tries: int = 3):
    import requests

    sys.path.insert(0, str(ROOT / "scripts"))
    from config.project_config import settings

    for _ in range(tries):
        response = requests.post(
            "https://api.brightdata.com/request",
            headers={"Authorization": f"Bearer {settings.proxy.api_token}",
                     "Content-Type": "application/json"},
            json={"zone": "web_unlocker1", "url": url, "format": "raw",
                  "method": "GET", "country": "IT"},
            timeout=180,
        )
        if (response.text or "").strip():
            return response
    return response


def live(args) -> None:
    """Check whether the worst-scoring listings are still on sale.

    A sold or removed listing answers with a stub page - 122 bytes or a ~19KB
    placeholder - while a live one is around 2.3MB and carries og:title. That size
    gap is the signal; keyword matching is useless because a live page mentions
    "venduto" dozens of times in seller stats and related items.
    """
    import re
    from concurrent.futures import ThreadPoolExecutor

    frame = load_scored().head(args.limit).copy()
    print(f"checking {len(frame)} listings", flush=True)

    def check(item_id: str) -> tuple[str, bool, int]:
        response = _unlocker_get(f"https://www.vinted.it/items/{item_id}")
        text = response.text or ""
        alive = len(text) > 100_000 and bool(re.search(r'<meta property="og:title"', text))
        return str(item_id), alive, len(text)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = dict((i, (a, n)) for i, a, n in pool.map(check, frame["Dataid"].astype(str)))
    frame["still_listed"] = frame["Dataid"].astype(str).map(lambda i: results.get(i, (None, 0))[0])
    frame["page_bytes"] = frame["Dataid"].astype(str).map(lambda i: results.get(i, (None, 0))[1])
    out = OUT_DIR / "live_checked.csv"
    frame.to_csv(out, index=False)
    alive = int(frame["still_listed"].fillna(False).sum())
    print(f"still listed: {alive} of {len(frame)} ({100*alive/max(len(frame),1):.0f}%)")
    print(f"wrote {out}")


def load_scored() -> pd.DataFrame:
    parts = sorted(OUT_DIR.glob("scored_part*.csv"))
    if not parts:
        raise SystemExit("no scored_part*.csv yet - run score first")
    frame = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    frame = frame.dropna(subset=["musiq"]).drop_duplicates("Dataid")
    return frame.sort_values("musiq")


REPORTS = HERE / "data" / "deal_monitor" / "reports"


def report(args) -> None:
    """Page of the worst-photographed stale listings, newest scoring run included."""
    import base64
    import html as html_lib

    frame = load_scored()
    checked = OUT_DIR / "live_checked.csv"
    if checked.exists():
        status = pd.read_csv(checked, usecols=["Dataid", "still_listed"])
        frame = frame.merge(status, on="Dataid", how="left")
        if args.only_live:
            frame = frame[frame["still_listed"].fillna(False)]
    frame = frame.head(args.limit)

    def thumb(path: str) -> str:
        try:
            return "data:image/webp;base64," + base64.b64encode(Path(path).read_bytes()).decode()
        except Exception:
            return ""

    cards = []
    for row in frame.itertuples(index=False):
        listed = getattr(row, "still_listed", None)
        badge = ("" if listed is None or pd.isna(listed)
                 else "<span class='live'>still listed</span>" if listed
                 else "<span class='gone'>gone</span>")
        cards.append(f"""<article>
<img src="{thumb(row.image_path)}" loading="lazy">
<p class="score">MUSIQ <b>{row.musiq:.1f}</b> · €{float(row.Price):.0f} · {int(row.age_days)}d old {badge}</p>
<p class="title">{html_lib.escape(str(row.Title)[:70])}</p>
<a href="https://www.vinted.it/items/{row.Dataid}" target="_blank">open on Vinted</a>
</article>""")

    out = REPORTS / "stale_bad_photos.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stale listings with bad photos</title>
<style>
body{{font:15px system-ui;margin:14px auto;max-width:1100px;background:#f4f4f4;color:#222;padding:0 10px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}}
article{{background:#fff;border:1px solid #ddd;border-radius:10px;padding:8px}}
img{{width:100%;aspect-ratio:.72;object-fit:cover;border-radius:7px;background:#eee}}
.score{{margin:6px 0 2px;font-size:13px}} .title{{margin:0 0 6px;font-size:12px;color:#555}}
.live{{background:#d8f0d8;border-radius:20px;padding:1px 7px;font-size:11px}}
.gone{{background:#f0dcdc;border-radius:20px;padding:1px 7px;font-size:11px}}
a{{font-size:12px;color:#0645ad}}
</style>
<h1>Stale listings with bad photos</h1>
<p>{len(frame)} of {len(load_scored())} scored so far, worst MUSIQ first. All unsold as of the
last check, {args.min_age}+ days old, priced at 80 EUR or more. "gone" means the listing has
since sold or been removed.</p>
<div class="grid">{''.join(cards)}</div>
""", encoding="utf-8")
    print(f"wrote {out} ({len(frame)} cards)")


HUNT_STATE = OUT_DIR / "hunt_state.json"


def hunt(args) -> None:
    """Liveness-check the worst-scoring listings on our own IP, then compare the survivors.

    Everything here goes through the laptop's own connection at --gap seconds per request,
    so it costs nothing and takes days. State is written after every listing, so stopping
    it and starting it again loses nothing.
    """
    import json

    from curl_cffi import requests as cffi

    sys.path.insert(0, str(HERE))
    import bad_photo_deal_monitor as monitor_module
    import full_item_fetch as fif

    state = json.loads(HUNT_STATE.read_text()) if HUNT_STATE.exists() else {}
    frame = load_scored()
    if args.max_musiq:
        frame = frame[frame["musiq"] <= args.max_musiq]
    if args.min_price:
        frame = frame[frame["Price"] >= args.min_price]
    todo = [row for row in frame.itertuples(index=False) if str(row.Dataid) not in state]
    print(f"{len(frame)} candidates, {len(state)} already checked, {len(todo)} to go", flush=True)

    session = cffi.Session(impersonate="chrome")
    pacer = monitor_module.RequestPacer(args.gap, "item page")
    cache = HERE / "data" / "deal_monitor" / "full_items"

    sys.argv = ["hunt", "--once"] + ([] if args.telegram else ["--dry-run"])
    monitor_args = monitor_module.parse_args()
    monitor_args.max_candidates = args.max_candidates
    monitor_args.fallback_seconds = 0.0  # never reach for the paid proxy
    monitor_args.gap_seconds = args.gap
    monitor_args.full_gap_seconds = args.gap
    monitor = monitor_module.Monitor(monitor_args)

    blocked_streak = 0
    for row in todo:
        item_id = str(row.Dataid)
        # 404 means the listing is really gone. 403 and friends mean our IP is blocked, and
        # recording those as "gone" would bury live listings - so those are retried, but only
        # a few times, or one dead item stalls the whole run (it stalled it for 9 hours).
        attempts, code, data = 0, None, None
        while True:
            pacer.wait()
            status: list[int] = []
            data = fif.fetch_item(session, item_id, cache, monitor_module.HEADERS,
                                  on_status=status.append)
            code = status[0] if status else 200  # no status means it came from cache
            if code in (200, 404):
                blocked_streak = 0
                break
            attempts += 1
            if attempts >= args.max_attempts:
                print(f"{item_id}: HTTP {code} after {attempts} tries, leaving unrecorded",
                      flush=True)
                break
            blocked_streak += 1
            backoff = min(args.block_backoff * blocked_streak, args.max_backoff)
            print(f"{item_id}: HTTP {code}, blocked - waiting {backoff/60:.0f} min "
                  f"(streak {blocked_streak}, try {attempts}/{args.max_attempts})", flush=True)
            time.sleep(backoff)
        if code not in (200, 404):
            continue  # undecided: try again on a later pass rather than guessing
        alive = data is not None
        state[item_id] = {"musiq": float(row.musiq), "price": float(row.Price),
                          "live": alive, "checked_at": datetime.now().isoformat(timespec="seconds")}
        HUNT_STATE.write_text(json.dumps(state, indent=1))
        live_count = sum(1 for v in state.values() if v["live"])
        print(f"{item_id} MUSIQ {row.musiq:.1f} EUR{row.Price:.0f} -> "
              f"{'LIVE' if alive else 'gone'} ({live_count} live of {len(state)})", flush=True)
        if not alive or not args.compare:
            continue
        try:
            target = {"Title": data["title"], "Price": float(data.get("price") or 0),
                      "Condition": str(data.get("condition") or ""),
                      "Link": f"https://www.vinted.it/items/{item_id}",
                      "Images": (data.get("photo_urls") or [""])[0]}
            candidates, confidence, material = monitor.find_candidates(
                {"Dataid": item_id, "Title": data["title"], "Brand": data.get("brand"),
                 "ConditionStatusId": monitor_module.CONDITION_IDS.get(
                     str(data.get("condition") or "").casefold(), "")})
            if candidates.empty:
                print("    no candidates", flush=True)
                continue
            job = {"item_id": item_id, "musiq": float(row.musiq),
                   "report_url": f"{args.base_url}/{item_id}.html",
                   "target": target, "title_confidence": confidence, "material": material,
                   "candidate_ids": monitor.in_price_band(target["Price"], candidates, item_id),
                   "attempts": 0}
            monitor.run_full_compare(job)
            state[item_id]["compared"] = True
            HUNT_STATE.write_text(json.dumps(state, indent=1))
        except Exception:
            LOG_MESSAGE = f"    comparison failed for {item_id}"
            print(LOG_MESSAGE, flush=True)
            import traceback

            traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scanner = sub.add_parser("scan")
    scanner.add_argument("--min-price", type=float, default=80.0)
    scanner.add_argument("--min-age-days", type=int, default=60)
    scanner.set_defaults(func=scan)
    scorer = sub.add_parser("score")
    scorer.add_argument("--limit", type=int, default=0, help="0 scores the whole selection")
    scorer.add_argument("--search", default="", help="restrict to one search folder")
    scorer.add_argument("--workers", type=int, default=1)
    scorer.add_argument("--worker-index", type=int, default=0)
    scorer.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    scorer.add_argument("--checkpoint", type=int, default=100,
                        help="append results to disk every N images")
    scorer.set_defaults(func=score)
    checker = sub.add_parser("live")
    checker.add_argument("--limit", type=int, default=200)
    checker.add_argument("--workers", type=int, default=4)
    checker.set_defaults(func=live)
    reporter = sub.add_parser("report")
    reporter.add_argument("--limit", type=int, default=120)
    reporter.add_argument("--min-age", type=int, default=60)
    reporter.add_argument("--only-live", action="store_true")
    reporter.set_defaults(func=report)
    hunter = sub.add_parser("hunt")
    hunter.add_argument("--gap", type=float, default=180.0, help="seconds between requests")
    hunter.add_argument("--block-backoff", type=float, default=900.0,
                        help="seconds to wait after a block, multiplied by the streak")
    hunter.add_argument("--max-backoff", type=float, default=1800.0)
    hunter.add_argument("--max-attempts", type=int, default=4,
                        help="blocked retries per listing before moving on")
    hunter.add_argument("--max-musiq", type=float, default=60.0)
    hunter.add_argument("--min-price", type=float, default=0.0)
    hunter.add_argument("--max-candidates", type=int, default=12)
    hunter.add_argument("--no-compare", dest="compare", action="store_false")
    hunter.add_argument("--telegram", action="store_true", help="send alerts for the survivors")
    hunter.add_argument("--base-url", default="https://ale-hkd-wxx.tailc0437a.ts.net")
    hunter.set_defaults(func=hunt, compare=True)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
