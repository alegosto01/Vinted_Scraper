# Handoff — bad-photo deal monitor (2026-07-31)

Resume doc for the live monitor and the archive hunt. The find-bad-images research record
(how MUSIQ was chosen, the failed CLIP/blur attempts, the label sets) is in `FINDINGS.md`
and `MUSIQ_RESULTS.md` and has not changed since 2026-07-26. MUSIQ is still the detector;
what changed since then is everything downstream of it.

---

## TL;DR

- The monitor finds Vinted listings with a bad first photo, searches for the same product,
  and judges what is genuinely comparable before quoting a price.
- **The comparison is no longer embeddings.** gpt-5-nano rewrites the search query and
  decides which candidates are comparable; gpt-4.1-mini judges the photos; DINOv2 only
  ranks. Text embeddings are gone - description cosine sat inside a 0.04 band across every
  candidate of a target and separated nothing.
- **Per-search MUSIQ thresholds**: default 55, telefoni 65 (one global 67.7 sent ~550
  alerts/day, most of them phones).
- **Two long jobs run as systemd user services** and survive reboots: `vinted-stale-score`
  (MUSIQ over the archive) and `vinted-stale-hunt` (liveness check).
- **The live monitor is stopped and disabled on purpose** so testing can proceed without new
  alerts: `vinted-bad-photo-deal-monitor`.

---

## Pipeline as it stands

1. Catalog page per search, MUSIQ on the first photo, per-search threshold
   (`SEARCH_MUSIQ_THRESHOLDS`).
2. Fetch the target's own item page **before searching** - lazy titles name the product only
   in the description ("Accessories" + "18 Kt Rose gold ring" -> `Sparkle Allure ring`).
3. `title_query.py` (gpt-5-nano) returns brand / model / material / product type, an English
   query and an Italian one. Material is deliberately **not** in the query.
4. Search runs three ways and merges on item id, interleaved so a later cap draws from all
   three: English, Italian, English + material. Under `--widen-below` candidates, a brand +
   product-type search is added.
5. Candidates filtered to the price band (0.1x - 10x of target) and capped by
   `--max-candidates` (40).
6. Full item pages fetched, then `spec_compare.py` (gpt-5-nano, batches of 12) returns per
   candidate: same_product, size_or_capacity, full_article, spec_relation, disqualifier, note.
7. Photos compared **only for text-accepted candidates**: DINOv2 cosine over all photo pairs
   for ranking, `image_compare.py` (gpt-4.1-mini, every photo of both listings) for the
   verdict. The photo verdict never overrides the text verdict.
8. Price median over usable comparables -> one Telegram alert, sent only once the comparison
   is ready, carrying a line naming what was excluded and why.

Reports are served over Tailscale Funnel at `https://ale-hkd-wxx.tailc0437a.ts.net/<item>.html`
by a plain `python -m http.server 8765` in `data/deal_monitor/reports`.

---

## Numbers measured this session

| thing | value |
|---|---|
| archive scored | 433,290 listings, 11,300 below MUSIQ 60 |
| archive left to score | ~700k (the sub-EUR80 tier) |
| liveness of archived bad-photo listings | **45 live of 116 checked (39%)** |
| MUSIQ throughput | 218 ms/image, no GPU |
| text vs photo agreement | 75% on phones/perfume, 98% on clothing (mutual rejection dominates) |
| acceptance rates | text ~5% of candidates, photos ~29% |
| unlocker cost | ~EUR0.001 per item page, ~EUR0.20 per full comparison |
| datacenter proxy | works on catalog and item pages, but flaky: 3/3 403 one hour, 200 the next |

**No cheap prefilter for MUSIQ exists.** Resolution is constant (Vinted serves every catalog
thumbnail at 310x430), file size correlates 0.13 with MUSIQ, Laplacian blur correlates 0.07,
and of the 25 blurriest images in a 300-listing sample not one scored below MUSIQ 60. Do not
spend an afternoon rediscovering this.

---

## What the archive hunt has shown so far

Two live archived listings were compared end to end. Both were **overpriced**, not
underpriced: a FENDI t-shirt asking EUR250 against a EUR122 comparable, and Gucci sunglasses
at EUR247 where every Gucci sunglass in the price band sat between EUR55 and EUR200.

That is the thesis under test. A listing that sat for months with a bad photo is usually
sitting because of its price. **Stale + bad photo + below comparables** is the signal; any
two of the three is noise. The liveness map decides whether this route deserves more work.

---

## Services and long jobs

```bash
systemctl --user status vinted-stale-score   # MUSIQ over the archive
systemctl --user status vinted-stale-hunt    # liveness, own IP, 180s per request
systemctl --user status vinted-bad-photo-deal-monitor   # live monitor: STOPPED + DISABLED
journalctl --user -u vinted-stale-hunt -f
```

Both long jobs checkpoint constantly and resume, so they can be killed freely:
`data/stale_scan/scored_part*.csv`, `data/stale_scan/hunt_state.json`.

The laptop runs hot (90C observed) and has crashed before, so the scoring service is capped
at `CPUQuota=100%`, 2 threads, `Nice=15`, `MemoryMax=5G`, and rests 60s every 200 images. An
uncapped run once had the kernel kill three jobs simultaneously.

```bash
PY=/home/ale/miniconda3/envs/vinted_scraper/bin/python
cd experiments/current/multimodal_beat/bad_photo_eval

$PY stale_bad_photos.py scan --min-price 0 --min-age-days 60   # rebuild the candidate list
$PY stale_bad_photos.py score --workers 2 --worker-index 0     # score (or use the service)
$PY stale_bad_photos.py live  --limit 200                      # liveness via unlocker (costs)
$PY stale_bad_photos.py hunt  --gap 180 --no-compare           # liveness on our own IP (free)
$PY stale_bad_photos.py report --limit 120 --only-live         # build the review page
```

---

## Traps worth knowing

- **Vinted changed the catalog card format on 2026-07-29.** The euro sign moved after the
  amount and the labels were capitalised, so every row silently became `Price 0.0`,
  `Brand "No brand"`, `Size 0`. Fixed in `utils.split_data`; both formats parse. Nothing
  detected it - a canary on the share of rows with price 0 is still not built.
- **A sold or removed listing answers 200 with a stub page** titled "Page not found", which
  the og:title fallback happily cached as a real item with no price. Now rejected.
- **404 means gone, 403 means our IP is blocked.** Conflating them had the hunter spend 9.5
  hours retrying one dead listing and checking nothing.
- **The photo model reads the fit label** off collar close-ups ("SLIM FIT" vs "CUSTOM FIT")
  and rejected valid comparables until the prompt named that exact failure.
- **A model reference lifted from the description can be unsearchable**: "Gucci GG0563SKN
  sunglasses" returns 0 rows where "Gucci sunglasses" returns 96.
- **gpt-5 models reject `temperature=0`**, so rewrites are not deterministic run to run;
  small per-target counts move by one or two for that reason alone.
- **`setsid` forks**, so `$!` is the wrapper pid, not the python process. Two workers assumed
  dead ran for another 7 hours and cooked the laptop. Prefer the systemd units.
- Prices stored by the scraper are the **buyer-protection inclusive** figure, not the asking
  price (EUR263.20 on a EUR250 listing).

---

## Open questions

1. **Labels.** `judge_clothing.html` holds 67 candidates with radio buttons and a copy
   button. Until those labels exist, precision and recall for the text and photo verdicts are
   unknown and no threshold can be tuned honestly.
2. **What "comparable" should mean for model-specific items.** Strict same-model is correct
   and usually returns zero comparables. A category fallback ("Gucci sunglasses, new,
   EUR55-200, median EUR140, weak signal") would have flagged both live archive items as
   overpriced instead of returning nothing.
3. **Free vs paid transport.** ~12 item pages per block window on our own IP, versus
   unlimited through the unlocker at ~EUR0.001 a page. This decides whether the monitor does
   20 or 200 comparisons a day.
4. **English vs Italian queries.** Both are generated and searched; which retrieves better has
   never been measured.
5. **Seller-profile expansion.** Bad photography is a seller trait, so a confirmed stale
   bad-photo listing is a better seed than a new listing for crawling a seller's other items.
