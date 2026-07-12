# Fast Discovery — Project Status & Next Steps

_Last updated: 2026-07-12_

Goal: catch a good deal **as soon as it is posted** (buy before others), **spend no more proxy than before**, and **log likes/score-vs-time** to train a faster, less likes-dependent model. Branch: `fast-discovery-loop`.

---

## 1. What we found (analysis)

- **Likes ablation** (`main_image_scores`, live data): dropping `Likes` hurts on matured data (AUC 0.756→0.695, precision 0.857→0.286). But that's the wrong test for "catch at t=0".
- **At t=0 likes are ~useless:** 85% of items have ≤1 like when captured; **49% of eventual 24h-sellers had 0 likes, 68% ≤1**. Half the winners are invisible to the likes signal → the instant-buy call must lean on **price + visual**, with likes/velocity as *later* confirmation.
- **The 329 `live_scoring` snapshots FREEZE per-item state** (0% of items ever change likes/score across snapshots) → **no velocity data exists**; it must be logged going forward. This is why we built the observation log.
- **hobby_collezionismo:** not data-starved (2002 matured@72h, sold-rate 0.25) and not signal-dead (AUC 0.608) — it's **moderate signal capping ~48% precision**. More rows won't help; needs better features (velocity) or a hobby-specific model. **Currently disabled from sending** (threshold 1.01), still logged.
- **Production ran on GLOBAL threshold 0.58, not per-search** — the 0629 model dir had no `per_search_thresholds.json` (calibration step was skipped). Now fixed (see §3).

Memory: `likes-timing-finding` in the auto-memory.

---

## 2. What we built

- **`fast_discovery_loop.py`** — every 15 min: scrape 9 model searches `newest_first` page-1 → score each **new** item once with the live-trained visual model (dino/aesthetic from `giant_basic_visual/_deps`) → **send passers once** (shared Telegram dedup log) → never re-scrape. Appends every observation to `data/fast_discovery/observations.csv`.
- **Exact request counter:** `data/fast_discovery/requests.csv` (per scan) + `requests_total.txt` (lifetime). **Measured: 9 proxy requests/scan** = 864/day at 15-min. Cost = `requests_total × your €/request`.
- **`deploy/systemd/vinted-fast-discovery.service`** — the unit (replaces `vinted-visual-scoring`).
- **`FAST_DISCOVERY.md`** — usage/deploy doc.
- **Per-search thresholds calibrated** and written to the production model dir (`live_trained_20260629_202123/per_search_thresholds.json`).

### Calibrated per-search thresholds (72h labels)
| search | threshold | prec | n | note |
|---|--:|--:|--:|---|
| telefoni | 0.402 | 0.67 | 113 | big volume unlocked |
| nike | 0.524 | 0.85 | 13 | lowered from 0.58 |
| ps4 | 0.584 | 0.86 | 7 | |
| donna_accessori_gioielli | 0.587 | 0.67 | 12 | |
| griffati_donna_all | 0.582 | 1.00 | 3 | thin n |
| gucci | 0.595 | 1.00 | 4 | |
| griffati_uomo_all | 0.603 | 1.00 | 3 | thin n |
| prada | 0.636 | 1.00 | 3 | thin n |
| **hobby_collezionismo** | **1.010** | — | — | **disabled (send off), still logged** |

Recalibrate as more data matures (several searches at n=3).

---

## 3. Current state (2026-07-12)

- Loop **deployed on VPS** as `vinted-fast-discovery` (user systemd). Old sender `vinted-visual-scoring` **disabled**.
- **Per-search thresholds ACTIVE**; hobby off. Memory peak ~1.1 G (4 G cap) — no OOM.
- **Bug found & fixed:** the send call used the wrong signature → crashed every scan at the send step (nothing sent, `seen` never saved → permanent 0-overlap GAP warnings). Fixed in commit `b1100f2` (`dry_run=not send`, keyword args). **Pending: re-pull + restart on VPS, then verify.**

### Verify after restart
```bash
cd ~/Vinted_New_Version && git fetch origin
git checkout origin/fast-discovery-loop -- experiments/current/giant_basic_visual/fast_discovery_loop.py
systemctl --user restart vinted-fast-discovery
# after the 2nd scan:
cat experiments/current/giant_basic_visual/data/fast_discovery/requests.csv
journalctl --user -u vinted-fast-discovery --no-pager | grep -E "scan:|GAP|Traceback" | tail
```
- New `requests.csv` rows + Telegram burst = working.
- **Watch scan-2 `new` count + GAP warnings** (see §4).

---

## 4. Open risks / watch

- **Throughput (unconfirmed):** each scan dino's every new item on a 4 GB CPU VPS (~1.4 s/item). If a search churns >96 new items faster than a scan completes, page-1 misses items (persistent GAP warnings) and it never stabilizes. **If scan-2 still shows `new` ~850 + GAP warnings → go two-stage:** score cheap on all (price+likes+PIL, dino=NaN-imputed), run dino/aesthetic only on the few pre-passers. Keeps it light + fast.
- **Two dino stacks in RAM** (this loop + `vinted-collector` both run `--quality-methods ...dino`). OK so far (~1.1 G each) but watch on the 4 G box.
- **`observations.csv` has duplicate rows** from the crashed scans — dedup before analysis.

---

## 5. Next steps (priority order)

1. **Finish Phase 1 verify** (above): confirm scan-2 is light + GAP-free + sends flowing. If throughput bad → build two-stage scoring.
2. **Phase 2 — the real €-saving (collector trim).** ALL proxy spend is `vinted-collector` (the scoring services don't scrape). It re-scrapes tracked items across windows out to **1008h (42 days)** — that's the ~€8–11/day sink and also the label source. Trim carefully:
   - Turn its discovery down (`--collect-every-hours` up; the loop discovers now).
   - Cut recheck windows from 1008h → ≤72h (keep only the labels the fast-sell model needs).
   Do with measurement; it's the label pipeline.
3. **Track cost:** `requests_total × €/request` vs the old €8–11/day. Should end well below once Phase 2 lands.
4. **Velocity model (original goal):** once `observations.csv` accrues (~1–2 weeks), engineer likes-velocity (Δlikes/Δt from first-seen), test whether it calls deals earlier than absolute likes, and train the faster model. Also **retest hobby** with velocity features — its only real improvement path.
5. **Recalibrate per-search thresholds** periodically (the n=3 searches are noisy).

---

## 6. Key paths
- Loop: `experiments/current/giant_basic_visual/fast_discovery_loop.py`
- Unit: `deploy/systemd/vinted-fast-discovery.service`
- Model dir (prod, VPS): `.../giant_basic_visual/data/live_trained/live_trained_20260629_202123/` (+ `per_search_thresholds.json`)
- Loop output (VPS): `.../giant_basic_visual/data/fast_discovery/` (`observations.csv`, `requests.csv`, `requests_total.txt`, `seen.csv`)
- Collector unit: `vinted-collector` (`scripts/experiments/current/time_to_sell/live_bin_collector.py run-loop ...`)
