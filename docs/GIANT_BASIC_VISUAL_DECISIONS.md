# Giant Basic Visual — Ground Rules and Key Decisions

This document records the architectural and operational decisions made for the `giant_basic_visual` experiment so that future sessions start with full context rather than re-deriving it from scratch.

---

## Why This Model Exists

The `basic_5_giant_model` (9-model union, numeric/text features) was the original Telegram sender. It was ruled by `rules_price_v1` (56% of candidates, 0.40 precision) making the feed low-quality. The visual model adds image quality features (MainImage* simple stats + dino/aesthetic scores) and was live-trained on the live collector's own data to fix the base-rate mismatch of the offline models.

---

## Model Architecture Decisions

### Single model, global → per-search thresholds
- **Original design**: single global threshold across all searches (chosen by `choose_threshold` on the pooled val set).
- **Decision (2026-07-01)**: switched to **per-search thresholds**, calibrated on live matured 72h labels from the current tracked population using `scripts/experiments/current/giant_basic_visual/calibrate_thresholds.py` (or the inline equivalent). Saved as `per_search_thresholds.json` alongside the model pkl.
- **Fallback**: if a search has no entry, the global threshold applies.
- **Why**: 3 new searches (donna_accessori_gioielli, hobby_collezionismo, telefoni) had 0 passes at the global threshold (0.558) because the model assigns them uniformly low scores. Per-search thresholds were calibrated at lower values for those searches while tightening others.

### Feature set
- `VISUAL_FEATURES = BASIC_FEATURES + MAIN_IMAGE_FEATURES + SCORE_COLS`
- `BASIC_FEATURES`: Price, Likes, search one-hot columns
- `MAIN_IMAGE_FEATURES`: 10 simple PIL stats (width, height, brightness, contrast, saturation, sharpness, edge density, quality score, screenshot risk, aspect ratio) — computed from the raw jpg on first encounter, cached indefinitely in `main_image_feature_cache.csv`
- `SCORE_COLS` (MAIN_IMAGE_SCORE_FEATURES): SimpleBadPhotoScore, AestheticGoodScore, AestheticBadPhotoScore, DinoEmbeddingNorm, CombinedBadPhotoScore — precomputed by the collector's `--quality-methods simple,aesthetic,dino` pass

### DinoEmbeddingNorm is a gated HuggingFace model
- Model: `facebook/dinov3-vits16-pretrain-lvd1689m` (gated)
- Requires HuggingFace token at `~/.cache/huggingface/token` on every machine running the collector.
- **Failure mode is silent**: if the token is missing, `DinoEmbeddingNorm` is written as NaN for all items and `DinoStatus` = `load_failed:LocalTokenNotFoundError...`. The collector does NOT crash — it continues with NaN dino values.
- The scoring/sender script uses `SimpleImputer(strategy="median")` which handles NaN natively.
- **Important**: dino was silently broken on the VPS from 2026-06-25 (migration) to 2026-06-29 (token fixed). All items collected during that window have DinoEmbeddingNorm=NaN permanently.

### Image cache fallback
- `apply_live_trained_to_live_collector.py` caches MainImage* features in `main_image_feature_cache.csv` keyed by absolute image path.
- **Decision**: code was updated to look up this cache even when the source jpg is gone from disk (e.g., after a disk cleanup). Derived features do not need the source file to persist. Rows with a missing jpg but a cache hit are scored normally.

---

## Training Decisions

### Train on live data, not offline curated data
- Offline giant_basic_visual models (in `offline_runs/`) were trained on the photo_arbitrage curated dataset (~47% positive rate). This causes severe threshold miscalibration when applied to the live collector population (~12-15% sold@24h).
- `train_on_live.py` retrains on the live collector's own tracked items with real outcome labels.

### Label: sold_within_24h (not 72h)
- `LABEL_COL = "sold_within_24h"` — original design choice.
- **Effect**: the model is optimized to predict fast-selling items (within 24h). Items that sell in 24-72h but not within 24h are counted as negatives during training, making the training label harder than the live 72h precision metric.
- **Threshold calibration** (the separate step) uses 72h labels, which is the metric users care about.
- Changing to 72h training label is possible but not yet explored.

### Don't hard-require DinoEmbeddingNorm during training
- `has_visual` filter does NOT require `DinoEmbeddingNorm` to be non-null.
- `HistGradientBoostingClassifier` tolerates NaN natively (no imputer needed for it, but the pipeline wraps a `SimpleImputer` anyway for the full feature set).
- Reason: the dino token was broken during the initial VPS deployment period, so historical rows from that window have permanently-NaN dino values. Requiring dino would have made the training set size = 0.

### Stratification at small N
- Original script stratified on `SearchName × label` jointly. With <200 rows per search and 2-3% base rate, this produces strata with 1 member and `train_test_split` fails.
- **Decision**: stratify on label only when per-search joint stratification fails due to small N.

### When to retrain
- Model drifts: measured 0.857 → 0.778 @ 72h precision in 5 days post original training (2026-06-13 model).
- Rule of thumb: retrain when live 72h precision drops below 0.70 for more than 2-3 consecutive scoring cycles.
- Recalibrate per-search thresholds whenever retraining (run calibration script against the new live eval run's scored items).
- Future retrain will have real dino signal for VPS-collected items (token is now fixed).

---

## Metric Guide — Don't Confuse These

| Metric | What it is | Typical value |
|---|---|---|
| **Offline test precision @24h** | Held-out test-split precision at selected threshold, sold_within_24h label | ~0.50–0.68 |
| **Live 72h precision** | % of items passing the live threshold that actually sold within 72h, measured on the full tracked population | ~0.78–0.85 |

**These are NOT comparable.** Always compare live-to-live or offline-to-offline. The live 72h metric is what matters for Telegram quality.

---

## Sender Architecture

### Telegram sender
- `apply_live_trained_to_live_collector.py --run-loop --send-telegram` (VPS systemd unit: `vinted-visual-scoring`)
- Uses per-search thresholds from `per_search_thresholds.json` in the model dir; falls back to global threshold.
- Price filter: price > 30 EUR (hard floor, independent of threshold).
- Dedup: shared sent-log at `experiments/current/basic_5_giant_model/data/live_scoring/telegram_sent_items.csv` — also checked/written by basic5 dry-run loop, ensuring no duplicate sends.

### basic5 comparison loop
- `apply_to_live_collector.py --run-loop --telegram-dry-run` (VPS unit: `vinted-scoring`)
- Scores and produces reports, does NOT send to Telegram.
- Kept alive to monitor basic5 model drift and as a fallback.

### Items skipped by visual model
- Items with no readable main image are never sent (not even via basic5 fallback).
- Items in donna_accessori_gioielli, hobby_collezionismo, telefoni with low scores don't pass even the relaxed per-search thresholds yet — expected, will improve as VPS collects more items for those searches with working dino.

---

## VPS Deployment

### Scraper location
- The collector runs on the VPS (Hetzner CX23, IP 167.233.132.113).
- Scraping goes through BrightData's remote scraping browser (`zproxy.lum-superproxy.io`) — the host machine's IP is irrelevant.
- Collector systemd unit: `vinted-collector` (user: `vinted`, linger enabled).

### Disk management
- `rechecks/` snapshots are write-only audit logs (never read back). Capped at 20 files by `vinted-prune-rechecks.timer` (daily).
- `image_cache/` (raw jpgs) for INFERENCE only needs the VPS's own freshly-collected items — historical backfill is only needed for training MainImage* features. After training: safe to delete pre-migration images (identified by mtime < 2026-06-25), keep VPS-native ones.
- `visual_features/` (precomputed dino/aesthetic CSVs) should be kept — small (230M) and useful for future retrains.
- Disk target: stay below 85% on the 38G root volume.

### Path portability
- Tracked items' `LocalPrimaryImagePath` stores absolute laptop paths (`/home/ale/Desktop/vinted/...`).
- A symlink on VPS resolves this: `/home/ale/Desktop/vinted/Vinted_New_Version -> /home/vinted/Vinted_New_Version`
- The cascade collector's model `artifact_path` metadata is also absolute — the same symlink covers it.

---

## Calibration Procedure (after each retrain)

1. Run `apply_live_trained_to_live_collector.py --model-dir <new_dir>` (no --send-telegram) to produce a live_scoring run with the new model.
2. Run `calibrate_thresholds.py` pointed at the live_scored_items.csv from step 1 and the new model dir.
3. Review per-search thresholds: check that precision >= 0.60 and n >= 3 for each calibrated search.
4. Run step 1 again with the newly-saved `per_search_thresholds.json` to verify per-search pass counts and precision.
5. Only then swap `MODEL_DIR` in the apply script and restart `vinted-visual-scoring`.
