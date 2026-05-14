# Photo-Improvement Arbitrage

This is a second project track, separate from the fast-sale deal finder.

Goal:

Find listings where the item may be commercially interesting, but the photos are poor enough that better presentation could improve resale potential.

This track is local analysis only. It does not make purchases, contact sellers, send messages, or perform account actions.

## Output Folders

All outputs are stored under:

```text
data/experiments/photo_arbitrage/
```

Main folders:

- `candidates/`: candidate item rows built from local search data.
- `features/`: scored rows with visual features and photo opportunity scores.
- `labels/`: manual label sheets for photo quality review.
- `models/`: trained photo-quality model artifacts.
- `reports/`: review queue and compact reports.

## Commands

Build local candidates:

```bash
python scripts/experiments/photo_arbitrage/build_candidates.py --all-searches
```

Export a manual label sheet:

```bash
python scripts/experiments/photo_arbitrage/export_label_sheet.py --limit 500
```

Train the first photo-quality model after labels are filled:

```bash
python scripts/experiments/photo_arbitrage/train_photo_quality.py
```

Train a DINOv3-enriched classifier once the machine has access to the gated model:

```bash
python scripts/experiments/photo_arbitrage/train_photo_quality.py --methods simple,dino --device auto
```

Score current candidates and create the review queue:

```bash
python scripts/experiments/photo_arbitrage/score_candidates.py --all-searches
```

Compare all photo-quality methods in one table:

```bash
python scripts/experiments/photo_arbitrage/compare_quality_methods.py --all-searches --methods all
```

For an image-only review queue, skip rows without local cached images:

```bash
python scripts/experiments/photo_arbitrage/compare_quality_methods.py --all-searches --methods all --require-local-image
```

Generate a compact report:

```bash
python scripts/experiments/photo_arbitrage/report.py
```

## Labels

Supported manual labels:

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

Only `photo_quality_bad` and `photo_quality_good` are used for model training.

## Version 1 Features

The first version uses simple local visual features:

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

The first score is:

```text
PhotoOpportunityScore =
  bad-photo probability
  + simple item/value signal
  - risk penalties
```

If no trained model is available yet, scoring falls back to `heuristic_v0`.

## Multi-Method Quality Comparison

The comparison table is designed for manual review. It writes:

```text
data/experiments/photo_arbitrage/features/latest_photo_quality_comparison.csv
data/experiments/photo_arbitrage/reports/photo_quality_comparison_review_queue.csv
```

It can compare these methods side by side:

- `simple`: local features such as blur, darkness, contrast, saturation, resolution, duplicate images, and picture count.
- `pyiqa`: pretrained no-reference image-quality scoring through PyIQA when the package and model weights are installed.
- `aesthetic`: a pretrained aesthetic classifier through Hugging Face Transformers when the model weights are available.
- `dino`: DINOv3 embeddings only. This is an outlier/visual-style signal until enough manual labels exist for a DINO-based bad-photo classifier.

The key output columns are:

- `SimpleBadPhotoScore`
- `PyiqaQualityScore`
- `PyiqaBadPhotoScore`
- `AestheticGoodScore`
- `AestheticBadPhotoScore`
- `DinoOutlierScore`
- `CombinedBadPhotoScore`
- `manual_label`
- `manual_notes`

The pretrained methods are optional adapters. If a package/model is missing, the row receives a status such as `pyiqa_not_installed` or `load_failed`, and the rest of the table is still produced.

Model weights are cached locally under:

```text
data/experiments/photo_arbitrage/model_cache/
```

Current default DINO behavior uses `facebook/dinov3-vits16-pretrain-lvd1689m` only. This Hugging Face model is gated, so the local machine must be authenticated with an account that has accepted access to that model. If access is missing, the comparison table records `load_failed_gated_repo:facebook/dinov3-vits16-pretrain-lvd1689m` in `DinoStatus` and does not use DINOv2 as a fallback.

## Future Upgrade

After the label sheet has enough examples, train a local classifier with `--methods simple,dino` to combine interpretable photo features with DINOv3-derived signals. Embeddings remain an additional feature source, not a replacement for simple interpretable image metrics.
