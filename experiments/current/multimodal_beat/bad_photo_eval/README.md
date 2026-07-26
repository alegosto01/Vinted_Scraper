# Bad-Photo Detector

Rank Vinted listings by the **technical quality of their first image** only.

## Goal

Answer one question per listing: *how likely is the first image to contain a
meaningful technical or presentation defect?* (blur, bad exposure, glare, bad crop,
low resolution/noise, clutter, extreme tilt.) Output is a **ranking score for manual
review, not a calibrated probability**.

## Non-goals (explicitly NOT evaluated)

Product value, price, resale profitability, brand desirability, seller quality, time
to sell, whether to buy. Product-type classification is **not required** — defects are
scored independently of what the item is. Only the **first image** is judged. No
business-value decision is performed here.

## Data alignment requirement

The frozen CLIP arrays are aligned to the **row order** of one exact CSV. That CSV was
recovered and locked (see `alignment.py`):

- source CSV: `experiments/current/time_to_sell/data/live_runs/bin_collector_20260602_214104/tracked_items.csv`
- rows: 20,450 | valid images: 19,173 | embedding: `(20450, 512)`, model `openai/clip-vit-base-patch32`, L2-normalized
- unique listings after dedup: **19,018**

Alignment proof (prompt-independent): the image mask was produced by walking the CSV
`item_id` column in order, so `mask[i]` must equal *"an image exists for
csv.item_id[i]"* for every row. The recovered CSV matches on **20,450/20,450 rows (0
disagreements)** and contains all 40 item_ids from the retained v1 review. The sidecar
`cache/img_clip_live19k_item_ids.npy` + `cache/img_clip_live19k_manifest.json` persist
this. Every scoring run asserts row-count, item-id order, and dim==512 via
`alignment.load_aligned()`.

## Methods compared

| key | name | definition |
|-----|------|------------|
| `v1_generic_clip` | frozen generic zero-shot | softmax over 5 good + 5 bad prompts, sum P(bad). From `zero_shot_badimg.py`. |
| `v2_typed_clip` | frozen type-aware | product type from search+title (image-CLIP fallback), then within-type `sigmoid(bad-good)`. From `zero_shot_typed_badimg.py`. Typing is heuristic/buggy — kept only as a baseline. |
| `v3_generic_defect_margin` | **new** | per defect: `mean(img·bad_prompts) - mean(img·good_prompts)`; overall = `max` over 7 defects. No softmax/sigmoid/calibration/type-selection. `prompts.py` + `clip_margins.py`. |
| `simple_technical` | **new** | cheap first-image features (resolution, exposure, blur, glare proxy, framing) → within-dataset "worse-quality" percentiles; component scores; overall = max. `simple_features.py`. |

Optional diagnostics (PyIQA/MUSIQ, FashionCLIP, aesthetic) may be added later; their
loading failures must never stop the core CLIP-margin + simple-feature run.

## Commands

```bash
PY=/home/ale/miniconda3/envs/vinted_scraper/bin/python
cd experiments/current/multimodal_beat/bad_photo_eval

# Phase 1 — lock alignment (writes item_id sidecar + manifest)
$PY alignment.py --input-csv <aligned_csv>

# Phase 2-6 — one scored table (add --max-items 100 for a smoke run)
$PY score_bad_photos.py --out data/scored_bad_photo_candidates.csv

# Phase 7-8 — blind evaluation set + gallery
$PY build_bad_photo_eval.py --scores data/scored_bad_photo_candidates.csv \
    --per-method-search 8 --random-per-search 8 --repeat-fraction 0.10 \
    --seed 24072026 --out-dir data/evaluation

# ...label by hand in data/evaluation/blind_gallery.html, export blind_label_sheet.csv...

# Phase 9 — metrics
$PY evaluate_bad_photo_eval.py \
    --private-candidates data/evaluation/eval_candidates_private.csv \
    --labels data/evaluation/blind_label_sheet.csv --out-dir data/evaluation

# Tests
$PY -m unittest discover -s tests -p 'test_*.py'
```

## Output files

- `data/scored_bad_photo_candidates.csv` — one row per unique item_id: all method
  scores, every CLIP defect margin, all simple features + component scores, source CSV
  hash, embedding-cache id, scoring version. **No combined final score.**
- `data/evaluation/eval_candidates_private.csv` — private key (scores + selection buckets).
- `data/evaluation/blind_label_sheet.csv` — blind: **no scores/method/type/rank/item_id**.
- `data/evaluation/blind_gallery.html` — self-contained labeling gallery.
- `data/evaluation/eval_manifest.json`, `metrics_*.{json,csv}`, `false_*.csv`, `results.md`.

## Label instructions

Judge only the first image's technical photo quality — not product value.

- `technical_quality`: good | bad | uncertain | not_item_photo
- `hurts_listing_presentation`: yes | no | uncertain
- `fixable_by_retake`: yes | no | uncertain
- `defect_tags` (multi): blur, dark, overexposed, glare, bad_crop, extreme_tilt,
  low_resolution, noise, clutter, item_not_clear, other

Keep `uncertain` explicit — it is never forced into good/bad. Labels must stay **blind
to model scores**.

## Metric definitions

- `precision@K` (K = 3,5,8): among a method's top-K ranked candidates in a pool,
  fraction labeled positive. Uncertain rows and repeat rows are excluded before ranking.
- Main positive: `technical_quality == bad`. Strict positive: `bad` AND
  `hurts_listing_presentation == yes` AND `fixable_by_retake == yes`.
- Reported: per search, macro-averaged across searches, pooled, by defect tag, strict
  target, and lift over the random bucket. **Primary metric: macro precision@8 across
  searches.** No dataset-wide recall is reported from this enriched sample.

## Decision rule

The new generic CLIP-margin method (`v3`) is worth retaining if:

```
macro precision@8 >= 0.50  AND  lift over random >= 2.0
AND  precision@8 >= 0.375 in at least 4 searches
```

and it should beat the better legacy CLIP method by >= 0.10 absolute macro p@8 (or
match it with much clearer defect explanations). If `simple_technical` beats CLIP,
prioritize the simple baseline. If **no** method reaches 0.50 macro p@8: stop prompt
tuning, do not add product-type rules, and collect labels to train a small model later
(needs ~50 clear bad + ~50 clear good before training).

## v4 hybrid (rules-only, gate-routed)

Frozen evaluation on 244 ChatGPT pseudo-labels showed **simple > v3 > v1/v2** on macro
P@8, and that v3's `crop` margin is anti-predictive (fires on normal tightly-framed
photos). Conclusion: route each defect to the signal that can see it.

`v4 = max over KEPT defect gates`, each gate a within-dataset percentile (higher =
worse), **no thresholds fit for ranking**:

- pixel gates reuse `simple_*` percentile scores: `gate_blur`, `gate_exposure`, `gate_resolution`
- CLIP gates use paired antonym question margins (`hybrid.QUESTION_PAIRS`) → percentile:
  `gate_crop`, `gate_clutter`, `gate_item_visibility`, `gate_tilt`
- `gate_glare` is hybrid (max of pixel glare proxy and CLIP glare margin)

**Gate selection is a development step, not validation.** `develop_gates.py` screens
gates on the **46 human audit labels** (bad-vs-good) with pooled AUROC + leave-one-
search-out AUROC + provisional Youden thresholds, and writes `surviving_gates.json`.
On this tiny enriched dev set the kept gates were **blur, item_visibility, clutter**
(crop dropped, AUROC 0.41 — anti-predictive). These dev numbers must **never** be cited
as validation.

Final comparison uses a **fresh blind holdout** (`build_holdout.py`) that excludes all
244 already-used items, selects new items by top simple/v3/v4 per search + a random
control, and emits a fillable blind gallery + ChatGPT bundle. `compare_final.py` then
ranks simple vs v3 vs v4 on the freshly-labeled holdout — the only place a validation
claim may be made.

```bash
# 1. attach question margins to the scored table, freeze v4
python develop_gates.py            # -> data/hybrid_dev/surviving_gates.json (dev only)
# 2. build the fresh blind holdout (excludes the 244)
python build_holdout.py --seed 20260725 --out-dir data/holdout
# 3. label data/holdout/holdout_blind_gallery.html (human) or the chatgpt_bundle
# 4. final validation
python compare_final.py --private data/holdout/holdout_candidates_private.csv \
    --labels data/holdout/holdout_label_sheet.csv --out-dir data/holdout/results
```

## Known limitations

- Scores are **ranking scores, not calibrated probabilities**.
- CLIP is semantic-first; it may penalize unusual content rather than true optics.
- The glare proxy is uncalibrated and stored independently.
- v2 typing is heuristic and has documented misclassifications; it is a baseline only.
- Labels are a small enriched sample — good for top-K precision, not for recall.
- **No method is claimed to work until blind labels are completed and evaluated.**
