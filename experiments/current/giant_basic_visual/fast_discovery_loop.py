#!/usr/bin/env python3
"""Fast newest-first discovery + score-once + observation log (v1).

Goal: catch a good deal as soon as it is posted, and record how score/likes
evolve so a quicker (less likes-dependent) model can be trained. See the
`likes-timing-finding` note: half of eventual sellers have 0 likes when first
captured, and the existing snapshots freeze per-item state, so the velocity
signal must be logged going forward.

Cost-neutral config (chosen 2026-07-11): 15-min cadence, page-1 `newest_first`
with overlap early-stop. NEW items are scored once with the live-trained visual
model; passing items are sent to Telegram once (shared dedup log) and never
re-scraped. Every observed item is appended to an append-only observation log
(likes, score, dt-since-first-seen).

Modes:
  --replay CSV   score+log an existing CSV, NO scraping (free; for wiring tests)
  --once         one real scan of all model searches (costs ~N proxy requests)
  --loop         repeat every --interval minutes
Dry-run by default (no Telegram); pass --send to actually send.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import experiments.current.giant_basic_visual.apply_live_trained_to_live_collector as scorer  # noqa: E402

MODEL_SEARCHES = set(scorer.MODEL_SEARCHES)
DEFAULT_OUT = scorer.EXPERIMENT_ROOT / "fast_discovery"
OBS_COLUMNS = ["obs_ts", "search", "item_id", "first_seen_ts", "dt_min", "likes",
               "price", "score", "threshold", "passed", "is_new"]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class RequestCounter:
    """Counts real datacenter proxy requests (catalog page fetches, incl. retries).

    Persists a lifetime total so cost can be computed exactly: cost = total * EUR/request.
    """

    def __init__(self, out_dir: Path):
        self.total_path = out_dir / "requests_total.txt"
        self.log_path = out_dir / "requests.csv"
        self.total = int(self.total_path.read_text()) if self.total_path.exists() else 0
        self.scan = 0

    def wrap(self, scraper) -> None:
        orig = scraper.get_page_content_datacenter

        def counted(*a, **k):
            self.scan += 1
            self.total += 1
            return orig(*a, **k)

        scraper.get_page_content_datacenter = counted

    def commit(self, extra: dict) -> None:
        self.total_path.write_text(str(self.total))
        row = {"scan_ts": now_utc().isoformat(), "requests_this_scan": self.scan,
               "requests_total": self.total, **extra}
        pd.DataFrame([row]).to_csv(self.log_path, mode="a",
                                   header=not self.log_path.exists(), index=False)


def load_seen(out_dir: Path) -> dict[tuple[str, str], str]:
    path = out_dir / "seen.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, low_memory=False)
    return {(str(r["search"]), str(r["item_id"])): str(r["first_seen_ts"]) for _, r in df.iterrows()}


def save_seen(out_dir: Path, seen: dict[tuple[str, str], str]) -> None:
    rows = [{"search": s, "item_id": i, "first_seen_ts": ts} for (s, i), ts in seen.items()]
    pd.DataFrame(rows, columns=["search", "item_id", "first_seen_ts"]).to_csv(out_dir / "seen.csv", index=False)


def append_obs(out_dir: Path, rows: pd.DataFrame) -> None:
    path = out_dir / "observations.csv"
    rows = rows.reindex(columns=OBS_COLUMNS)
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)


def scrape_search(scraper, search, pages: int, get_images: bool = True) -> pd.DataFrame:
    """Scrape newest_first page(s) for one search. Returns a catalog DataFrame."""
    search.sort = "newest_first"
    data = scraper.scrape_products_serial(dictionary=search, search_count=0,
                                          pages_to_scrape=pages, get_images=get_images)
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["SearchName"] = search.folder
    if "item_id" not in df.columns:
        df["item_id"] = df.get("Dataid")
    df["item_id"] = df["item_id"].astype(str)
    return df


def add_visual_features(rows: pd.DataFrame, out_dir: Path, methods: str, device: str) -> pd.DataFrame:
    """Live only: PIL + aesthetic/dino quality scores on the images the scraper already
    downloaded (LocalPrimaryImagePath is set by scrape with get_images=True).

    Uses giant_basic_visual's OWN _deps -- not time_to_sell -- so it resolves on the VPS
    (the sender imports the same deps). Same computation as the collector's
    add_live_visual_features, minus the redundant image-download step. Heavy imports lazy.
    """
    from experiments.current.giant_basic_visual._deps.photo_arbitrage.features import add_photo_features
    from experiments.current.giant_basic_visual._deps.photo_arbitrage.quality_methods import (
        add_quality_method_scores, MethodConfig,
        DEFAULT_PYIQA_MODEL, DEFAULT_AESTHETIC_MODEL, DEFAULT_DINO_MODEL,
    )
    from experiments.current.giant_basic_visual._deps.full_scrape_model.compare_feature_modalities import (
        add_dino_embedding_columns,
    )
    featured = add_photo_features(rows)
    config = MethodConfig(
        methods=tuple(p.strip() for p in methods.split(",") if p.strip()) if methods else ("simple",),
        pyiqa_model=DEFAULT_PYIQA_MODEL, aesthetic_model=DEFAULT_AESTHETIC_MODEL,
        dino_model=DEFAULT_DINO_MODEL, max_images_per_item=1, device=device,
    )
    scored = add_quality_method_scores(featured, config=config)
    scored, _ = add_dino_embedding_columns(scored)
    return scored


def score_frame(df: pd.DataFrame, model, threshold: float, per_search: dict[str, float]) -> pd.DataFrame:
    prepared = scorer.prepare_feature_frame(df)
    return scorer.apply_model(prepared, model, threshold, per_search)


def run_scan(df: pd.DataFrame, *, out_dir: Path, seen: dict[tuple[str, str], str],
             model, threshold: float, per_search: dict[str, float],
             live: bool, methods: str, device: str, send: bool, score: bool = True) -> dict:
    if df.empty:
        return {"seen": 0, "new": 0, "passed": 0, "sent": 0}
    df = df[df["SearchName"].isin(MODEL_SEARCHES)].copy()
    obs_ts = now_utc()
    keys = list(zip(df["SearchName"].astype(str), df["item_id"].astype(str)))
    is_new = np.array([k not in seen for k in keys])
    df["is_new"] = is_new

    # compute visual features for NEW items only (score-once); reappearances are log-only.
    # --no-score (score=False) => observation-log only (likes/price/dt), no visual/model/send.
    new_df = df[is_new].copy()
    if score and not new_df.empty:
        if live:
            new_df = add_visual_features(new_df, out_dir, methods, device)
        scored_new = score_frame(new_df, model, threshold, per_search)
    else:
        scored_new = pd.DataFrame()

    # observation log: every item seen this scan
    first_seen = {k: seen.get(k, obs_ts.isoformat()) for k in keys}
    score_by_key = {}
    if not scored_new.empty:
        for _, r in scored_new.iterrows():
            score_by_key[(str(r["SearchName"]), str(r["item_id"]))] = (
                r.get(scorer.SCORE_COL), r.get(scorer.THRESHOLD_COL), r.get(scorer.PASS_COL))
    obs_rows = []
    for k, row in zip(keys, df.to_dict("records")):
        fs = first_seen[k]
        dt_min = (obs_ts - datetime.fromisoformat(fs)).total_seconds() / 60.0
        sc, th, ps = score_by_key.get(k, (np.nan, np.nan, np.nan))
        obs_rows.append({
            "obs_ts": obs_ts.isoformat(), "search": k[0], "item_id": k[1],
            "first_seen_ts": fs, "dt_min": round(dt_min, 2),
            "likes": pd.to_numeric(row.get("Likes"), errors="coerce"),
            "price": pd.to_numeric(row.get("Price"), errors="coerce"),
            "score": sc, "threshold": th, "passed": ps, "is_new": k not in seen,
        })
    append_obs(out_dir, pd.DataFrame(obs_rows))

    # send passers once (the sender dedups against the shared sent-log internally),
    # then mark seen forever. dry_run=not send => dry runs simulate without sending.
    sent = 0
    passed = 0
    if not scored_new.empty:
        candidates = scorer.build_telegram_candidates(scored_new)
        passed = int(len(candidates))
        if not candidates.empty:
            result = scorer.send_candidates_to_telegram(
                candidates,
                source_run="fast_discovery",
                dry_run=not send,
                sent_log_path=scorer.TELEGRAM_SENT_LOG,
            )
            sent = int(result.get("sent", 0))
    for k in keys:
        seen.setdefault(k, first_seen[k])
    save_seen(out_dir, seen)
    return {"seen": len(keys), "new": int(is_new.sum()), "passed": passed, "sent": sent}


def live_searches(subset: set[str] | None = None):
    from config.project_config import settings
    from config.search_loader import load_searches
    searches = load_searches(str(settings.paths.searches_yaml))
    # Cover every search the model knows, regardless of the yaml enabled/cascade_only
    # flags (the main 6 are enabled=False there but are still scored live).
    wanted = (MODEL_SEARCHES & subset) if subset else MODEL_SEARCHES
    return [s for s in searches.values() if s.folder in wanted]


def make_noproxy_fetch(scraper):
    """No-proxy catalog fetch (curl_cffi Chrome TLS impersonation), drop-in for
    scraper.get_page_content_datacenter. Viable per the per-search ~10-page/window
    quota measured 2026-07-13 (see the vinted-per-ip-quota memory)."""
    from curl_cffi import requests as cffi
    session = cffi.Session(impersonate="chrome")
    headers = scraper._default_page_headers()

    def fetch(url, timeout=40, sleep=10, max_attempts=3, request_kind='page'):
        for attempt in range(1, max_attempts + 1):
            try:
                r = session.get(url, headers=headers, timeout=timeout)
                if r.status_code == 200 and r.text.strip():
                    if sleep and sleep > 0:
                        time.sleep(sleep)
                    return r.text
                if r.status_code in (403, 429):
                    time.sleep(min(60.0, float(sleep) * attempt))
            except Exception as exc:  # noqa: BLE001
                print(f"  noproxy fetch err attempt={attempt}: {exc}")
            if attempt < max_attempts:
                time.sleep(2.0 * attempt)
        return None

    return fetch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", help="Score+log an existing CSV, no scraping (free).")
    ap.add_argument("--once", action="store_true", help="One real scan of all model searches.")
    ap.add_argument("--loop", action="store_true", help="Repeat every --interval minutes.")
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--pages", type=int, default=1)
    ap.add_argument("--send", action="store_true", help="Actually send to Telegram (default off).")
    ap.add_argument("--no-proxy", action="store_true",
                    help="Fetch catalog pages with no proxy (curl_cffi impersonation).")
    ap.add_argument("--no-score", action="store_true",
                    help="Observation-log only: skip visual features, model scoring, and sends.")
    ap.add_argument("--searches", default=None,
                    help="Comma-separated subset of MODEL_SEARCHES to run (default: all).")
    ap.add_argument("--pages-map", default=None,
                    help="Per-search page depth override, e.g. "
                         "'hobby_collezionismo=5,telefoni=2'. Falls back to --pages.")
    ap.add_argument("--methods", default="simple,aesthetic,dino")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--model-dir", default=None, help="Override model dir (default: production; VPS-only path).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = load_seen(out_dir)
    if args.no_score:
        # Observation-log only: no model needed (avoids the VPS-only model dir).
        model, threshold, per_search = None, 0.0, {}
        print(f"NO-SCORE mode: observation log only. seen={len(seen)} out={out_dir}")
        _no_score_early = True
    else:
        _no_score_early = False
        model_dir = Path(args.model_dir) if args.model_dir else scorer.MODEL_DIR
        model_path = model_dir / scorer.MODEL_PATH.name
        model = scorer._load_cached_model(str(model_path))
        import json as _json
        threshold = float(_json.loads((model_dir / "results.json").read_text())["results"]["main_image_scores"]["threshold"])
        per_search = scorer.load_per_search_thresholds(model_dir)
    if not _no_score_early:
        print(f"model_dir={model_dir.name} model={model_path.name} seen={len(seen)} out={out_dir}")
        if per_search:
            print(f"PER-SEARCH THRESHOLDS ACTIVE: {len(per_search)} searches -> "
                  + ", ".join(f"{s}={t:.3f}" for s, t in sorted(per_search.items())))
        else:
            print(f"WARNING: no per_search_thresholds.json in {model_dir} -> using GLOBAL threshold "
                  f"{threshold:.4f} for every search. Per the decisions doc, the newer searches "
                  f"(donna_accessori_gioielli, hobby_collezionismo, telefoni) get ~0 passes at the "
                  f"global threshold. Calibrate before trusting live sends.")

    if args.replay:
        df = pd.read_csv(args.replay, low_memory=False)
        df["item_id"] = df.get("item_id", df.get("Dataid")).astype(str)
        stats = run_scan(df, out_dir=out_dir, seen=seen, model=model, threshold=threshold,
                         per_search=per_search, live=False, methods=args.methods,
                         device=args.device, send=False)
        print("replay:", stats)
        return 0

    if not (args.once or args.loop):
        ap.error("choose --replay, --once, or --loop")
    from simple_scraper import Simple_scraper
    scraper = Simple_scraper()
    if args.no_proxy:
        scraper.get_page_content_datacenter = make_noproxy_fetch(scraper)
        print("NO-PROXY mode: catalog pages fetched via curl_cffi (no datacenter proxy).")
    counter = RequestCounter(out_dir)
    counter.wrap(scraper)
    subset = {s.strip() for s in args.searches.split(",") if s.strip()} if args.searches else None
    searches = live_searches(subset)
    pages_map = {}
    if args.pages_map:
        for kv in args.pages_map.split(","):
            k, _, v = kv.partition("=")
            if k.strip() and v.strip():
                pages_map[k.strip()] = int(v)
    if pages_map:
        print(f"per-search pages: {pages_map} (default {args.pages})")
    print(f"searches: {[s.folder for s in searches]}  requests_total_so_far={counter.total}")

    while True:
        counter.scan = 0
        frames = []
        for s in searches:
            try:
                fr = scrape_search(scraper, s, pages_map.get(s.folder, args.pages),
                                   get_images=not args.no_score)
                if not fr.empty:
                    overlap = sum((s.folder, str(i)) in seen for i in fr["item_id"])
                    # Full page with zero overlap => newest_first didn't reach items we
                    # already have => we likely missed some since the last scan. Signal to
                    # deepen --pages for this (trending) search. This is the data-driven
                    # input for per-search adaptive depth (v2).
                    if len(fr) >= 90 and overlap == 0 and len(seen) > 0:
                        print(f"  GAP WARNING search={s.folder}: full page ({len(fr)}) but 0 overlap "
                              f"with seen -> items may have been missed; consider --pages {args.pages + 1}")
                    frames.append(fr)
            except Exception as exc:
                print(f"  scrape failed search={s.folder}: {exc}")
        df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
        stats = run_scan(df, out_dir=out_dir, seen=seen, model=model, threshold=threshold,
                         per_search=per_search, live=True, methods=args.methods,
                         device=args.device, send=args.send, score=not args.no_score)
        counter.commit(stats)
        print(f"{now_utc().isoformat()} scan: {stats} requests_this_scan={counter.scan} "
              f"requests_total={counter.total}")
        if not args.loop:
            break
        time.sleep(max(60.0, args.interval * 60.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
