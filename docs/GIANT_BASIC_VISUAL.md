# Giant Basic Visual

This experiment tests whether the Basic5 giant model improves when the model gets
features from the listing main image.

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

