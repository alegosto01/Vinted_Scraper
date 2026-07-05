> ⚠️ **Caveman-compressed** — terse/fragment style to save tokens. Technical substance, code, commands, URLs kept verbatim. Original backed up under `~/.local/share/caveman-compress/backups/`.

# Photo-Improvement Arbitrage

Second project track. Separate from fast-sale deal finder.

Goal:

Find listings where item maybe commercially interesting but photos bad enough that better presentation lift resale.

Local analysis only. No buy, no contact seller, no message, no account action.

## Output Folders

All outputs go under:

```text
experiments/old/photo_arbitrage/data/
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
python experiments/old/photo_arbitrage/build_candidates.py --all-searches
```

Export manual label sheet:

```bash
python experiments/old/photo_arbitrage/export_label_sheet.py --limit 500
```

Train first photo-quality model after labels filled:

```bash
python experiments/old/photo_arbitrage/train_photo_quality.py
```

Train DINOv3-enriched classifier once machine got gated model access:

```bash
python experiments/old/photo_arbitrage/train_photo_quality.py --methods simple,dino --device auto
```

Score current candidates + make review queue:

```bash
python experiments/old/photo_arbitrage/score_candidates.py --all-searches
```

Compare all photo-quality methods in one table:

```bash
python experiments/old/photo_arbitrage/compare_quality_methods.py --all-searches --methods all
```

Image-only review queue, skip rows w/o local cached images:

```bash
python experiments/old/photo_arbitrage/compare_quality_methods.py --all-searches --methods all --require-local-image
```

Generate compact report:

```bash
python experiments/old/photo_arbitrage/report.py
```

After FashionCLIP manual review, generate a usefulness/method-comparison report:

```bash
python experiments/old/photo_arbitrage/report.py --fashionclip-review experiments/old/photo_arbitrage/data/reports/fashionclip_pseudo_label_review_queue.csv
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
experiments/old/photo_arbitrage/data/features/latest_photo_quality_comparison.csv
experiments/old/photo_arbitrage/data/reports/photo_quality_comparison_review_queue.csv
```

Compare methods side by side:

- `simple`: local features — blur, darkness, contrast, saturation, resolution, duplicates, picture count.
- `pyiqa`: pretrained no-reference image-quality score via PyIQA when pkg + weights installed.
- `aesthetic`: pretrained aesthetic classifier via HF Transformers when weights available.
- `fashionclip`: FashionCLIP/CLIP prompt similarity for good-vs-bad marketplace photo pseudo-labels.
- `dino`: DINOv3 embeddings only. Outlier/visual-style signal until enough labels exist for DINO bad-photo classifier.

Key output columns:

- `SimpleBadPhotoScore`
- `PyiqaQualityScore`
- `PyiqaBadPhotoScore`
- `AestheticGoodScore`
- `AestheticBadPhotoScore`
- `FashionClipGoodScore`
- `FashionClipBadScore`
- `FashionClipBadPhotoScore`
- `FashionClipScoreMargin`
- `DinoOutlierScore`
- `CombinedBadPhotoScore`
- `manual_label`
- `manual_notes`

Pretrained methods = optional adapters. Pkg/model missing → row get status like `pyiqa_not_installed` or `load_failed`, rest of table still made.

FashionCLIP default model:

```text
patrickjohncyh/fashion-clip
```

FashionCLIP loading is local-cache-first. `compare_quality_methods.py` and `train_photo_quality.py` pass `local_files_only=True` unless explicitly run with:

```bash
--allow-fashionclip-downloads
```

FashionCLIP pseudo-label review outputs:

```text
experiments/old/photo_arbitrage/data/reports/fashionclip_pseudo_label_review_queue.csv
experiments/old/photo_arbitrage/data/reports/fashionclip_pseudo_label_review_queue.html
experiments/old/photo_arbitrage/data/reports/fashionclip_pseudo_label_review_queue_<run_id>.csv
experiments/old/photo_arbitrage/data/reports/fashionclip_pseudo_label_review_queue_<run_id>.html
```

Use these for manual review before training. The HTML file is a thumbnail contact sheet; the CSV is the editable label sheet. Suggested labels live in `FashionClipPseudoLabel`; fill or approve `manual_label` before using the sheet as training labels.

If the FashionCLIP comparison command is rerun, existing non-empty `manual_label` and `manual_notes` values in the latest CSV are restored onto matching regenerated rows by item identity/link/image fallback. This protects manual review work while still refreshing scores and timestamps.

Each comparison manifest includes `quality_status_counts`, so a run with zero FashionCLIP review rows can quickly distinguish missing dependencies, uncached model weights, missing local images, and successful scoring.

Training defaults to reviewed labels only:

```bash
python experiments/old/photo_arbitrage/train_photo_quality.py --labels experiments/old/photo_arbitrage/data/reports/fashionclip_pseudo_label_review_queue.csv --methods simple,fashionclip
```

Training requires at least two reviewed `photo_quality_bad` rows and two reviewed `photo_quality_good` rows. When trained, model metadata includes train-set metrics plus a deterministic stratified cross-validation summary (`evaluation`) so the first classifier has a small held-out sanity check instead of only fit-on-train feedback.

Check whether the edited queue is ready without writing model artifacts:

```bash
python experiments/old/photo_arbitrage/train_photo_quality.py --labels experiments/old/photo_arbitrage/data/reports/fashionclip_pseudo_label_review_queue.csv --methods simple,fashionclip --check-only
```

For a weak-label baseline before manual review, opt in explicitly:

```bash
python experiments/old/photo_arbitrage/train_photo_quality.py --labels experiments/old/photo_arbitrage/data/reports/fashionclip_pseudo_label_review_queue.csv --methods simple,fashionclip --label-source fashionclip_pseudo
```

Do not treat the weak-label baseline as validated photo quality; it measures whether a small local model can imitate FashionCLIP prompt labels.

The compact report reads the FashionCLIP review queue and, once `manual_label` has good/bad decisions, reports pseudo-label agreement, bad-photo precision/recall, method score metrics, high-confidence failure examples, and the remaining non-fashion holdout-data gap. Before manual labels exist, it records the review as pending.

Model weights cached locally under:

```text
experiments/old/photo_arbitrage/data/model_cache/
```

Default DINO use `facebook/dinov3-vits16-pretrain-lvd1689m` only. HF model gated → local machine must auth w/ account that accepted access. No access → comparison table records `load_failed_gated_repo:facebook/dinov3-vits16-pretrain-lvd1689m` in `DinoStatus`. No DINOv2 fallback.

## Future Upgrade

Once label sheet got enough examples, train local classifier w/ `--methods simple,fashionclip,dino` to mix interpretable photo features + prompt/embedding signals. Embeddings stay extra feature source, not replacement for simple interpretable metrics.
