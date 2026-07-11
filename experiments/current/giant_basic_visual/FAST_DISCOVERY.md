# Fast Discovery Loop

Catch a good deal as soon as it is posted, spend less proxy money, and log the
likes/score-vs-time trajectory to train a quicker (less likes-dependent) model.

## Why
- Half of eventual 24h-sellers have **0 likes** when first captured, and the old
  scoring snapshots **freeze** per-item state, so the velocity signal must be
  logged going forward (see the `likes-timing-finding` memory).
- The old collector re-scrapes the **whole tracked set** every cycle, so cost
  **grows** with the backlog (July: ~€8-11/day and rising). This loop only
  discovers new items + scores once + never re-scrapes, so cost is **flat**
  (~€1-2/day, estimated; measured exactly via the request counter).

## What it does (every 15 min)
1. Scrape each model search `order=newest_first`, page 1 (`--pages`), and stop
   there. ~13 catalog requests/scan.
2. Score each **new** item once with the live-trained visual model
   (`main_image_scores`), using the same per-search thresholds as the current
   sender. Passing items (price > €30) are sent to Telegram once via the shared
   dedup log, then never scraped again.
3. Append every observed item to `observations.csv`:
   `obs_ts, search, item_id, first_seen_ts, dt_min, likes, price, score,
   threshold, passed, is_new`. Re-appearances are log-only (build the trajectory).

## Cost tracking (exact)
- `requests.csv` — per scan: `requests_this_scan, requests_total, ...`
- `requests_total.txt` — lifetime proxy-request count (survives restarts).
- Cost = `requests_total * (your BrightData €/request)`. Counter wraps the real
  datacenter call, so it includes retries.

## Run
Dry-run one scan (no Telegram), offline replay for wiring tests (no proxy):
```
python fast_discovery_loop.py --replay <catalog.csv> --model-dir <local_model_dir>
```
One real scan, no send (measures real requests/scan):
```
python fast_discovery_loop.py --once
```
Production loop (sends):
```
python fast_discovery_loop.py --loop --interval 15 --pages 1 --send
```
Defaults: model dir = production `live_trained_20260629_202123` (with its
per-search thresholds); output = `data/fast_discovery/`. Startup prints whether
per-search thresholds are active or it is falling back to the global threshold.

## Deploy (VPS) — replaces vinted-visual-scoring
```
sudo systemctl stop vinted-visual-scoring        # stop the old re-scrape loop (required for the savings)
cp deploy/systemd/vinted-fast-discovery.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start vinted-fast-discovery
journalctl --user -u vinted-fast-discovery -f    # watch scans + request count
```
Do a dry run first (drop `--send` from the unit's ExecStart) and confirm
`requests.csv` + scores look right before enabling sends.

## Not in v1 (next)
- **Per-search adaptive page depth.** v1 is fixed `--pages 1` and prints a
  `GAP WARNING` when a full page has zero overlap with seen (items likely missed
  => deepen that search). Use the accrued `observations.csv` dwell data to auto-tune.
- **Sold-check labels** stay in the existing eventual-sale worker (needed to
  train the faster model: features from here + sold/not-sold outcomes from there).
