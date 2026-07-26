# Handoff — Find-Bad-Images session (2026-07-26)

Practical resume doc. For the full experimental record + numbers see `FINDINGS.md`;
for model detail see `MUSIQ_RESULTS.md` and `data/blur_batch/blur_results.md`.

---

## TL;DR

- Goal: rank Vinted listings by first-photo technical quality (bad = blur/dark/low-res/
  crop/clutter/tilt/glare). Ranking, not probability. First image only.
- **Outcome: a general "any bad photo" detector does NOT work with zero-shot CLIP.**
  The tractable, validated target is **blur + low-resolution**.
- **Chosen production model: MUSIQ** (pyiqa pretrained no-reference IQA). Fresh-holdout
  AUROC **0.847**. Operating threshold **MUSIQ < 67.7 = bad** (Youden on ~134 human labels).
- All code committed & pushed: branch `find_bad_items`, commit `3f994bf`
  (`github.com/alegosto01/Vinted_Scraper`). `data/` (253M) is gitignored.

---

## What happened this session (arc)

1. Built full frozen-embedding baselines (v1/v2/v3 CLIP margins, simple pixel features)
   + alignment recovery + 3-target evaluation vs 244 ChatGPT pseudo-labels + a fresh
   63-image human holdout.
2. On the fresh holdout **all zero-shot methods were ~random** (AUROC ≤ 0.57). v3's
   `crop` margin was anti-predictive (fires on normal tight framing). Built a rules-only
   hybrid v4 (gate routing); marginal.
3. Pivoted to **blur only** → texture-normalized `blur_score` (AUROC 0.80). Tried to beat
   it with NR features + supervised model: overfit, no gain (only ~32 positives).
4. Tried **pretrained IQA (pyiqa)**: **MUSIQ wins** (fresh AUROC 0.847), and revives the
   general target too. Bake-off: MANIQA best by +0.02 (0.867/0.853) but ~44min/210imgs on
   CPU → hours for 19k; DBCNN 0.860/0.827; CLIP-IQA+ 0.832/0.814. **Voting/ensemble: no gain.**
5. Scored all 19,018 with MUSIQ, built a review queue, ran a **live no-proxy scrape**
   (griffati uomo+donna, newest 10 pages each) + MUSIQ flagging.

---

## Current state

### Data (all under `data/`, gitignored)
- `scored_bad_photo_candidates.csv` — 19,018 items, v1/v2/v3/simple + blur_score.
- `scored_with_v4.csv`, `scored_with_blur.csv`, `scored_with_musiq.csv` — progressive
  augmentations; **`scored_with_musiq.csv` is the current master** (adds `musiq`,`musiq_bad`).
- `musiq_ranking.csv` / `blur_ranking.csv` — top-N worst per search (review queues).
- `live_musiq_test.csv` — live-scrape result: 1,654 listings, **329 flagged bad (20%)**.
- Labels: `evaluation/` (244 pseudo + 46 human audit), `holdout/` (63 human),
  `blur_batch/` + `blur_batch2/` (120, blur y/n), `musiq_batch/` (60 human general).

### Models / thresholds
- Production: **MUSIQ**, threshold **< 67.7**. `blur_score` = dependency-free blur fallback.
- MANIQA = accuracy ceiling; only worth it to re-rank the top-N MUSIQ shortlist.

### Tests: 39 unittest, all passing.
`/home/ale/miniconda3/envs/vinted_scraper/bin/python -m unittest discover -s tests -p 'test_*.py'`

---

## Key commands

```bash
PY=/home/ale/miniconda3/envs/vinted_scraper/bin/python
cd experiments/current/multimodal_beat/bad_photo_eval

# score all items with MUSIQ (one image pass; resumable)
$PY build_musiq.py

# review queue: top-N worst-MUSIQ per search
$PY - <<'P'
import pandas as pd
t=pd.read_csv("data/scored_with_musiq.csv",low_memory=False)
t["sl"]=t.search_names.fillna("").str.split("|")
# ...nsmallest(N,"musiq") per search...
P

# live no-proxy scrape + MUSIQ flag (run from scripts/ dir on path; has preflight guard)
cd ../../../../scripts && $PY ../experiments/current/multimodal_beat/bad_photo_eval/live_musiq_noproxy.py
```

---

## Gotchas / constraints

- **No-proxy scraping:** use `curl_cffi` chrome impersonation, **1 page/min** dodges
  DataDome (recovery ~1 min/search). Fast bursts / datacenter-proxy get blocked (403).
  `live_musiq_noproxy.py` has a preflight that aborts cleanly if the IP is blocked.
  User directive: **do NOT use brightdata/datacenter proxies.**
- **MUSIQ scope:** detects blur + low-resolution/soft; MISSES crop/clutter/tilt/glare.
  So "bad" = technically low-quality photo, mostly soft/blurry.
- **Label bottleneck:** only ~30–40 clear human "bad". Any supervised head needs ~50+/defect.
- **Background jobs die at session teardown** here — long scrapes/scoring may need re-launch.
- MUSIQ P@K on tiny fresh sets is noisy; trust AUROC. 20%-flag @67.7 is high-recall — for a
  tighter queue use top-N ranking instead of the threshold.

---

## Open / next steps (none blocking)

1. Wire the MUSIQ review queue into `app.py` / dashboard ("worst photos to review").
2. Optional MUSIQ→MANIQA re-rank on the top-N shortlist for the marginal accuracy gain.
3. To beat ~0.85 or catch semantic defects: collect ~50+ clear bad/defect and train a
   small supervised head — the only remaining lever.
4. Live-scrape display bug (cosmetic): earlier console print doubled the URL prefix; the
   `live_musiq_test.csv` `Link` column is correct.
