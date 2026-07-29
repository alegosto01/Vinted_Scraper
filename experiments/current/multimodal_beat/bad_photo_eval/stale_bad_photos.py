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
import sys
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
    import pyiqa
    import torch

    frame = pd.read_csv(SCAN_PATH)
    # Highest price first: no cheap signal predicts MUSIQ, so the only sane ordering is
    # the one where a missed bad photo costs the most.
    frame = frame.sort_values("Price", ascending=False)
    if args.skip:
        frame = frame.iloc[args.skip:]
    frame = frame.head(args.limit).copy()
    metric = pyiqa.create_metric("musiq", device="cuda" if torch.cuda.is_available() else "cpu")
    scores = []
    for index, path in enumerate(frame["image_path"], 1):
        try:
            scores.append(float(metric(str(path))))
        except Exception:
            scores.append(float("nan"))
        if index % 250 == 0:
            print(f"  {index}/{len(frame)}", flush=True)
    frame["musiq"] = scores
    out = SCORED_PATH if not args.skip else SCORED_PATH.with_name(f"scored_{args.skip}.csv")
    frame.sort_values("musiq").to_csv(out, index=False)
    usable = frame["musiq"].dropna()
    print(f"wrote {out}")
    print(usable.describe(percentiles=[0.01, 0.05, 0.25, 0.5]).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scanner = sub.add_parser("scan")
    scanner.add_argument("--min-price", type=float, default=80.0)
    scanner.add_argument("--min-age-days", type=int, default=60)
    scanner.set_defaults(func=scan)
    scorer = sub.add_parser("score")
    scorer.add_argument("--limit", type=int, default=15000)
    scorer.add_argument("--skip", type=int, default=0,
                        help="rows to skip, so several processes can split the pool")
    scorer.set_defaults(func=score)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
