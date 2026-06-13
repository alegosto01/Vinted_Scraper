> ⚠️ **Caveman-compressed** — terse/fragment style to save tokens. Technical substance, code, commands, URLs kept verbatim. Original backed up under `~/.local/share/caveman-compress/backups/`.

# Photo-Improvement Arbitrage

Second project track. Separate from fast-sale deal finder.

Goal:

Find listings where item maybe commercially interesting but photos bad enough that better presentation lift resale.

Local analysis only. No buy, no contact seller, no message, no account action.

## Output Folders

All outputs go under:

```text
data/experiments/photo_arbitrage/
```

Main folders:

- `candidates/`: candidate rows from local search data.
- `features/`: scored rows w/ visual features + opportunity scores.
- `labels/`: manual label sheets for photo review.
- `models/`: trained photo-quality model artifacts.
- `reports/`: review queue + compact reports.

## Commands

Build local candidates:

```bash
python scripts/experiments/current/photo_arbitrage/build_candidates.py --all-searches
```

Export manual label sheet:

```bash
python scripts/experiments/current/photo_arbitrage/export_label_sheet.py --limit 500
```

Train first photo-quality model after labels filled:

```bash
python scripts/experiments/current/photo_arbitrage/train_photo_quality.py
```

Train DINOv3-enriched classifier once machine got gated model access:

```bash
python scripts/experiments/current/photo_arbitrage/train_photo_quality.py --methods simple,dino --device auto
```

Score current candidates + make review queue:

```bash
python scripts/experiments/current/photo_arbitrage/score_candidates.py --all-searches
```

Compare all photo-quality methods in one table:

```bash
python scripts/experiments/current/photo_arbitrage/compare_quality_methods.py --all-searches --methods all
```

Image-only review queue, skip rows w/o local cached images:

```bash
python scripts/experiments/current/photo_arbitrage/compare_quality_methods.py --all-searches --methods all --require-local-image
```

Generate compact report:

```bash
python scripts/experiments/current/photo_arbitrage/report.py
```

## Labels

Manual labels:

- `photo_quality_bad`
- `photo_quality_good`
- `unclear`
- `not_item_photo`

Optional review tags:

- `dark`
- `blurry`
- `bad_crop`
- `cluttered_background`
- `low_resolution`
- `mirror_or_glare`
- `item_not_clear`
- `only_one_photo`

Only `photo_quality_bad` + `photo_quality_good` used for training.

## Version 1 Features

V1 use simple local visual features:

- brightness
- contrast
- saturation
- sharpness
- edge density
- width and height
- aspect ratio
- picture count
- duplicate image count
- low-quality image fraction
- missing-image flag

First score:

```text
PhotoOpportunityScore =
  bad-photo probability
  + simple item/value signal
  - risk penalties
```

No trained model yet → scoring fall back to `heuristic_v0`.

## Multi-Method Quality Comparison

Comparison table for manual review. Writes:

```text
data/experiments/photo_arbitrage/features/latest_photo_quality_comparison.csv
data/experiments/photo_arbitrage/reports/photo_quality_comparison_review_queue.csv
```

Compare methods side by side:

- `simple`: local features — blur, darkness, contrast, saturation, resolution, duplicates, picture count.
- `pyiqa`: pretrained no-reference image-quality score via PyIQA when pkg + weights installed.
- `aesthetic`: pretrained aesthetic classifier via HF Transformers when weights available.
- `dino`: DINOv3 embeddings only. Outlier/visual-style signal until enough labels exist for DINO bad-photo classifier.

Key output columns:

- `SimpleBadPhotoScore`
- `PyiqaQualityScore`
- `PyiqaBadPhotoScore`
- `AestheticGoodScore`
- `AestheticBadPhotoScore`
- `DinoOutlierScore`
- `CombinedBadPhotoScore`
- `manual_label`
- `manual_notes`

Pretrained methods = optional adapters. Pkg/model missing → row get status like `pyiqa_not_installed` or `load_failed`, rest of table still made.

Model weights cached locally under:

```text
data/experiments/photo_arbitrage/model_cache/
```

Default DINO use `facebook/dinov3-vits16-pretrain-lvd1689m` only. HF model gated → local machine must auth w/ account that accepted access. No access → comparison table records `load_failed_gated_repo:facebook/dinov3-vits16-pretrain-lvd1689m` in `DinoStatus`. No DINOv2 fallback.

## Future Upgrade

Once label sheet got enough examples, train local classifier w/ `--methods simple,dino` to mix interpretable photo features + DINOv3 signals. Embeddings stay extra feature source, not replacement for simple interpretable metrics.