# Giant Basic Visual

This experiment tests whether the Basic5 giant model improves when the model gets
features from the listing main image.

The current live model is one global `HistGradientBoostingClassifier` trained on
live collector data to predict `sold_within_24h`. It combines `Price`, `Likes`,
search one-hot columns, simple main-image statistics, and cached image-quality
scores. Inference uses per-search thresholds calibrated against matured 72-hour
outcomes, with a global threshold as fallback. See
[`GIANT_BASIC_VISUAL_DECISIONS.md`](GIANT_BASIC_VISUAL_DECISIONS.md) for training,
calibration, sender, and deployment decisions.

Hard rules:

- Use only `LocalPrimaryImagePath`.
- Do not use `LocalImagePaths` as fallback.
- Skip rows with missing or unreadable main images.
- Do not use secondary image counts, duplicate-image features, averages across images, or raw DINO embedding dimensions.
- Do not change Telegram policy.
- Live scoring is shadow-only.

Package:

```text
experiments/current/giant_basic_visual/
```

Outputs:

```text
experiments/current/giant_basic_visual/data/
```

Offline smoke run:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python experiments/current/giant_basic_visual/run.py --limit-rows 50
```

Live shadow run:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python experiments/current/giant_basic_visual/apply_to_live_collector.py \
  --live-run-dir experiments/current/time_to_sell/data/live_runs/<run> \
  --shadow-only
```

Modes:

- `basic_5_control`: Basic5 fields plus search one-hot, but only on rows with readable main images.
- `main_image_simple`: control plus main-image size, brightness, contrast, saturation, sharpness, edge density, quality score, and screenshot risk.
- `main_image_scores`: simple plus cached main-image score columns when available.
- `main_image_dino_outlier_diagnostic`: adds `DinoOutlierScore`; diagnostic only, not live-ready.

Telegram note: this package never sends messages. It only writes shadow candidates
and comparison reports.

## SHAP feature analysis

The live-model SHAP analysis was reconstructed on 2026-07-11 because the original
generated report was not present. It used:

- Model: `data/live_trained/live_trained_20260613_200922/main_image_scores_hist_gradient_seed42.pkl`
- Scored input: `data/live_scoring/live_scoring_20260619_125732/live_scored_items.csv`
- Scope: 28,083 rows loaded; deterministic sample of 1,000 rows explained; no price or pass filter
- Explainer: TreeSHAP with `check_additivity=False`
- Expected value: `-1.966238` on the raw-margin/log-odds scale (about 12.28% base probability)
- Full scored input contained 127 Telegram-eligible rows; only 7 of the sampled rows passed

The configured production-model pointer referenced a missing 2026-06-29 artifact,
so this reconstruction explicitly used the available 2026-06-13 model. Results
therefore describe that model and scored population, not a newer missing artifact.

### Likes result

`Likes` ranked second of 24 features, behind only `Price`:

| Metric | Value |
|---|---:|
| Mean absolute SHAP | 0.443182 |
| Mean SHAP | +0.001073 |
| Positive SHAP rate | 31.1% |
| Share of summed mean absolute SHAP | 28.53% |
| Importance relative to Price | 80.38% |
| Pearson correlation, Likes vs Likes SHAP | 0.788 |
| Spearman correlation, Likes vs Likes SHAP | 0.817 |
| Correlation, Likes SHAP vs final score | 0.660 |

Mean signed SHAP is near zero because zero-like penalties and liked-item boosts
cancel; it does **not** mean Likes is unimportant. All 689 zero-like rows had a
negative Likes contribution, while all 311 rows with at least one like had a
positive contribution.

| Likes | Rows | Mean Likes SHAP | Approx. odds multiplier |
|---:|---:|---:|---:|
| 0 | 689 | -0.3208 | 0.73x |
| 1 | 161 | +0.4337 | 1.54x |
| 2 | 67 | +0.6064 | 1.83x |
| 3-4 | 40 | +0.9699 | 2.64x |
| 5-9 | 23 | +1.6056 | 4.98x |
| 10-19 | 16 | +1.7423 | 5.71x |
| 20-49 | 4 | +2.0192 | 7.53x |

Likes SHAP ranged from -0.4085 to +2.4187. Quartiles were -0.3293,
-0.3021, and +0.4124; the 90th, 95th, and 99th percentiles were +0.6470,
+1.1356, and +1.7667.

### All feature results

| Rank | Feature | Mean abs SHAP | Mean SHAP | Positive rate |
|---:|---|---:|---:|---:|
| 1 | Price | 0.551330 | +0.083664 | 54.9% |
| 2 | Likes | 0.443182 | +0.001073 | 31.1% |
| 3 | search__gucci | 0.090743 | -0.011592 | 84.9% |
| 4 | search__ps4 | 0.086633 | -0.018463 | 87.7% |
| 5 | MainImageContrast | 0.050302 | +0.006854 | 70.4% |
| 6 | MainImageWidth | 0.034455 | +0.007688 | 84.5% |
| 7 | MainImageBrightness | 0.033006 | -0.003932 | 51.9% |
| 8 | MainImageSaturation | 0.032530 | -0.000573 | 56.6% |
| 9 | DinoEmbeddingNorm | 0.032391 | +0.000918 | 60.8% |
| 10 | search__telefoni | 0.030263 | -0.001296 | 3.3% |
| 11 | CombinedBadPhotoScore | 0.029479 | -0.003938 | 58.7% |
| 12 | AestheticGoodScore | 0.021288 | -0.002471 | 59.1% |
| 13 | search__griffati_donna_all | 0.019841 | -0.003359 | 17.5% |
| 14 | MainImageEdgeDensity | 0.019070 | +0.000781 | 45.4% |
| 15 | MainImageQualityScore | 0.018736 | +0.000828 | 80.1% |
| 16 | search__prada | 0.016467 | -0.002530 | 86.9% |
| 17 | search__griffati_uomo_all | 0.015109 | -0.001532 | 18.4% |
| 18 | MainImageSharpness | 0.008965 | +0.000921 | 47.3% |
| 19 | MainImageHeight | 0.008801 | -0.001175 | 86.7% |
| 20 | MainImageAspectRatio | 0.007425 | -0.001823 | 72.3% |
| 21 | AestheticBadPhotoScore | 0.003386 | -0.001704 | 37.0% |
| 22 | search__nike | 0.000000 | 0.000000 | 0.0% |
| 23 | MainImageScreenshotRisk | 0.000000 | 0.000000 | 0.0% |
| 24 | SimpleBadPhotoScore | 0.000000 | 0.000000 | 0.0% |

Summed mean absolute SHAP share: `Price + Likes` 64.02%, search identity
16.68%, and visual features 19.30%. Likes is therefore a major engagement signal,
not a small auxiliary feature. SHAP describes model behavior, not causality; a
time-aware held-out ablation with and without Likes is needed to measure whether
the feature improves generalization or leaks listing age/popularity.

Generated artifacts:

- `data/shap_likes/live_shap_likes_20260711_232319/shap_report.md`
- `data/shap_likes/live_shap_likes_20260711_232319/feature_importance.csv`
- `data/shap_likes/live_shap_likes_20260711_232319/item_explanations.csv`
