# Handoff: improving the CLIP bad-photo detector

## What I want help with

I am trying to identify Vinted listings whose first photo is technically poor, across
different searches and product types. Examples of defects:

- blur or missed focus
- darkness / underexposure
- glare or strong reflections
- bad crop or framing
- extreme tilt
- low-resolution / noisy image
- visually confusing clutter

The desired output is a useful ranking for manual review, not necessarily a globally
calibrated probability. The important metric should probably be precision among the
top-ranked listings, split by search and product type.

Please review the approach below, identify its main failure modes, and propose the
smallest high-signal next experiment. Do not jump directly to a large model or a new
pipeline. Prefer reusing the existing frozen CLIP embeddings and creating a small,
reliable evaluation set.

## Repository location

`experiments/current/multimodal_beat/`

Relevant files:

- `zero_shot_badimg.py` — original generic zero-shot detector
- `zero_shot_typed_badimg.py` — newest type-aware detector (v2)
- `embed_blocks.py` — creates and caches CLIP image embeddings
- `cache/badimg_live19k/worst.csv` — retained v1 top-40 ranking
- `cache/badimg_live19k/gallery.html` — retained visual review gallery
- `RESULTS.md` — broader multimodal sales-prediction experiment; it does not evaluate
  bad-photo detection accuracy

## Available data and embeddings

- Model: `openai/clip-vit-base-patch32`
- One normalized 512-dimensional embedding per listing's first image
- Cache: `cache/img_clip_live19k.npy`
- Shape: `(20450, 512)`
- Image mask: `cache/img_clip_live19k_mask.npy`
- Images found: `19173 / 20450` (`93.76%`)
- Embeddings are CPU-generated and frozen.
- Original images are resolved by `item_id` from several collector image caches.
- Only the first image is used.

Critical reproducibility issue: these arrays are aligned to the exact row order of the
original 20,450-row input CSV, but that CSV path was not recorded in the experiment.
Before rerunning either detector, recover or reconstruct the exact aligned CSV. Add a
row-count assertion and ideally save an `item_id` sidecar beside every embedding cache.

Observed searches in the retained v1 top results include:

- `prada`
- `gucci`
- `nike`
- `ps4`
- `griffati_donna_all`
- `griffati_uomo_all`

The searches are heterogeneous. Brand searches can contain clothing, bags, perfume,
shoes, eyewear, and other products. Search names can also contain irrelevant listings.

## Last actual test: type-aware v2 review

The latest test output was found outside the repository at:

`/home/ale/Desktop/badimg_review`

It was generated on 2026-07-21 by `zero_shot_typed_badimg.py`, apparently with:

```text
--per-type 8 --out-dir /home/ale/Desktop/badimg_review
```

Contents:

- `worst.csv`
- 10 product-type directories
- 8 selected images per type
- 80 rows total, representing 75 unique `item_id` values
- 5 duplicate rows caused by the same listing appearing under multiple searches

Score ranges are highly saturated:

```text
bag             0.9820–0.9987
clothing        0.9925–0.9974
collectible     0.9186–0.9813
console_device  0.9832–0.9976
eyewear         0.9623–0.9976
jewelry         0.9987–0.9999
perfume         0.9978–0.9997
phone_tablet    0.9898–0.9994
shoes           0.9919–0.9994
video_game      0.9933–0.9992
```

Qualitative review of all 80 thumbnails suggests poor precision. Many selected photos
are clear and usable. The detector often appears to penalize ordinary backgrounds,
non-catalog presentation, or semantic mismatch rather than true technical defects.
This is not a formal human-label evaluation, but it is strong evidence that prompt
tuning alone should not be trusted.

The test also exposes deterministic type-classification errors:

- `Kingdom Hearts ... Plush` and `Lots Couches lavables` become `video_game` because
  `ps4` is the default search type.
- French `casquette` (cap) becomes `console_device` because the regex contains
  `casque`, intended to mean headset.
- `Kit crampon ...` becomes `console_device` because the generic keyword `kit` is in
  that rule.
- `Final fantasy pixel remastered` becomes `phone_tablet` because `pixel` is treated
  as the Google phone brand without contextual boundaries.
- `Elden Ring` becomes `jewelry` because the rule matches the standalone word `ring`.
- PSP listings become `bag` because titles mention `pochette`.
- Shoe listings become `bag` because titles mention `dust bag`.
- Perfume atomizers become `bag` because French titles contain `sac`.
- Singular French `lunette` can miss the eyewear rule, which expects `lunettes`.

Duplicate IDs in this test:

```text
9089916210  griffati_uomo_all + griffati_donna_all
9090394550  griffati_uomo_all + gucci
9097773202  griffati_uomo_all + griffati_donna_all
9098597454  griffati_donna_all + gucci
9134560353  griffati_uomo_all + griffati_donna_all
```

Before another scoring experiment, deduplicate by `item_id` and separate product-type
classification quality from photo-quality ranking quality.

## Version 1: generic CLIP prompts

`zero_shot_badimg.py` compares every image to five generic good prompts and five bad
prompts.

Good prompts:

```text
a clear sharp in-focus photo
a bright evenly well-lit photo
a person wearing the clothing item
a straight well-framed photo
a normal quality product photo
```

Bad prompts:

```text
a blurry out-of-focus photo
a dark underexposed dim photo
a crooked tilted misaligned photo
a grainy low quality noisy photo
a motion blurred photo
```

Current scoring:

```text
CLIP logits for all 10 prompts
-> one softmax across all prompts
-> sum probability assigned to the five bad prompts
```

It filters missing images, optionally filters searches, sorts globally by `bad_score`,
copies the top images, and writes `worst.csv`.

Retained output:

- `cache/badimg_live19k/` contains 40 copied images
- `cache/badimg_live19k/worst.csv`
- `cache/badimg_live19k/gallery.html`
- top scores are extremely saturated (`~0.99`)

Likely v1 problems:

- Generic prompts confuse product identity/content with image quality.
- One softmax across many prompts makes the score depend on prompt count and wording.
- CLIP is semantic-first and may not reliably measure blur, noise, or exposure.
- A globally ranked list lets some searches/product types dominate.
- No labeled evaluation exists, so visual anecdotes can drive prompt overfitting.
- The first image may be atypical; no multi-image aggregation is tested.

## Version 2: type-aware zero-shot scoring

`zero_shot_typed_badimg.py` was the latest work, modified after v1.

### Stage 1: assign product type

It first uses multilingual title regexes, then a default type for known searches.
If still unresolved, it chooses the product type whose CLIP text prompt has the highest
image similarity.

Supported types:

- `video_game`
- `console_device`
- `phone_tablet`
- `shoes`
- `clothing`
- `bag`
- `perfume`
- `jewelry`
- `eyewear`
- `collectible`

Examples:

- `ps4` defaults to `video_game`
- `nike`, `gucci`, `prada`, and Griffati searches default to `clothing`
- title keywords override those defaults, so a Prada perfume or sunglasses listing can
  receive the corresponding type

### Stage 2: type-specific quality comparison

Each type has one good prompt and one bad prompt. Example for perfume:

```text
good: a sharp, bright, well-framed photo of a perfume bottle centered on a clean surface
bad:  a blurry, out-of-focus, dark, or badly framed photo of a perfume bottle
```

For each row, the detector compares only the assigned type's good and bad logits:

```text
bad_score = sigmoid(bad_logit - good_logit)
```

It then takes the worst N items within every product type and creates one output
subdirectory per type.

Likely v2 problems:

- Product typing is heuristic and has not been measured.
- Search defaults can silently assign wrong types.
- Only one good and one bad prompt represent each type.
- Several defects are packed into one sentence, making attribution impossible.
- Scores are selected per type, but there is still no explicit per-search balancing.
- `--searches` filters rows only after scoring.
- There is no `len(df) == len(embeddings)` assertion in v2.
- The CLIP model can be loaded twice when image-based type fallback is needed.
- Absolute scores are not comparable across types; only within-type rank is intended.
- There are no human labels, baselines, precision metrics, or error analysis.

## What has and has not been established

Established:

- Cached CLIP embeddings cover most of the 20,450 listings.
- The generic detector can produce a ranked review gallery.
- The type-aware implementation exists.
- CLIP image embeddings were useful in the separate sales-prediction experiment, but
  that does not prove they detect bad photos.

Not established:

- Exact labeled top-K precision for v1 or v2. Qualitative v2 precision looks poor.
- Whether v2 improves over v1.
- Which searches/types work or fail.
- Whether scores detect actual technical defects instead of unusual content.
- Whether prompt changes generalize beyond the reviewed examples.
- Whether cheap non-semantic image-quality features would beat CLIP.

## Suggested evaluation direction

Please improve this proposal if needed:

1. Recover the aligned 20,450-row CSV and persist its ordered `item_id` list.
2. Build one small, fixed evaluation set before changing prompts:
   - stratify by search and product type;
   - deduplicate by `item_id`;
   - include v1 top-ranked, v2 top-ranked, middle-ranked, and random items;
   - hide model scores during labeling;
   - label `good` / `bad` plus defect reason;
   - keep uncertain cases explicit instead of forcing a label.
3. Compare:
   - random baseline;
   - v1 generic CLIP;
   - v2 type-aware CLIP;
   - simple prompt ensembles scored as mean bad similarity minus mean good similarity.
4. Report precision at K overall, per search, per type, and per defect.
5. Inspect false positives and false negatives before adding more machinery.
6. Only if zero-shot CLIP is weak, try the smallest next option:
   - cheap blur/exposure/resolution features;
   - or a linear/logistic head on frozen CLIP embeddings using the manual labels.

Avoid choosing thresholds from the same examples used to write prompts.

## Questions for you

Please answer these concretely:

1. Is type-aware prompting the right decomposition, or should quality defects be scored
   independently of product type?
2. What CLIP score formulation is least sensitive to prompt count and logit saturation?
3. What is the smallest labeling design that can compare v1 and v2 credibly?
4. How should results be balanced across searches and product types?
5. Which cheap image-quality signals complement CLIP for blur, exposure, glare, crop,
   and low resolution?
6. What exact next experiment would you run, including sample construction, metrics,
   decision rule, and stopping criterion?
7. Which bugs or reproducibility fixes should be made before any new experiment?

End with one recommended next step and a minimal implementation plan.
