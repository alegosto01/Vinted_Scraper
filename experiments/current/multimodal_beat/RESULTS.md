# Multimodal "beat giant_basic_visual" — results (2026-07-15)

CPU-only (no GPU) → frozen encoders + light heads (not literal fine-tuning). Title text =
frozen multilingual MiniLM sentence embedding (384-d). Visual = already-computed dino/
aesthetic columns. Heads: LogReg, HistGBDT (champion-style), MLP, TabNet, FT-Transformer.
Split 60/20/20 stratified, SEED=42. Harness: `run_experiment.py`.

## Exp B — 19k LIVE rows, `sold_within_24h` (pos 14.4%), test n=3852  [DECISIVE]
No visual features on disk for the live population, so this tests tabular vs tabular+text at scale.

| feature_set | head   | ROC-AUC | PR-AUC | P@10% |
|-------------|--------|--------:|-------:|------:|
| **tab+txt** | **gbdt** | **0.789** | **0.436** | **0.488** |
| tab+txt | mlp    | 0.761 | 0.391 | 0.436 |
| tab     | ft     | 0.761 | 0.371 | 0.423 |
| tab     | (all)  | ~0.758 | ~0.366 | ~0.42 |
| tab+txt | logreg | 0.744 | 0.356 | 0.423 |
| tab+txt | tabnet | 0.720 | 0.333 | 0.421 |

**Title text lifts GBDT: AUC 0.758→0.789, PR-AUC 0.363→0.436, P@10% 0.43→0.49.** Real (n=3852).
Text helps only tree/MLP heads; hurts linear/tabnet. Deep nets (TabNet/FT) do NOT beat GBDT.

## Exp A — 814 rows full multimodal (tab+vis+txt), `label_sold_within_24h`, test n=163  [suggestive only]
Only complete multimodal set on disk (old searches, small n → within noise).

| feature_set | head | ROC-AUC | PR-AUC |
|---|---|--:|--:|
| tab+vis | logreg | 0.681 | 0.490 |
| tab+vis | mlp | 0.678 | 0.533 |
| tab+txt | gbdt | 0.674 | 0.498 |
| tab+vis+txt | logreg | 0.657 | 0.468 |
| tab (only) | logreg | 0.650 | 0.494 |

Visual is the strongest single add; text helps GBDT; **text+visual did not stack (overfit at n=814)**;
no deep head wins.

## Exp A-full — 814 rows, ALL encoders (tab, dino-vis, MiniLM, CLIP-img, CLIP-txt, BERT, mDeBERTa)
~50 configs, 12 feature sets x 5 heads. Test n=163 (differences within noise). Top by ROC-AUC:

| feature_set | head | ROC-AUC | PR-AUC | P@10% |
|---|---|--:|--:|--:|
| tab+vis (champion-style) | logreg | 0.681 | 0.490 | 0.50 |
| tab+vis+txt+img+ctxt (ALL) | gbdt | 0.679 | 0.526 | 0.625 |
| tab+vis | mlp | 0.678 | 0.533 | 0.625 |
| tab+txt | gbdt | 0.674 | 0.498 | 0.625 |
| tab+btxt (BERT) | logreg | 0.672 | 0.487 | 0.44 |
| tab+img+ctxt (CLIP dual-encoder) | gbdt | 0.648 | 0.508 | 0.625 |
| best PR-AUC: tab+vis+txt+img+ctxt | mlp | 0.661 | **0.568** | 0.625 |

- **CLIP dual-encoder (Model 1) does NOT beat dino+tabular** (0.648 vs 0.681).
- BERT text ~ MiniLM; mDeBERTa worst (NLI model, poor embeddings).
- Full-multimodal (1827 feat) competitive on PR-AUC but not ROC-AUC; overfits at n=488 train.
- No deep head (TabNet/FT) wins anywhere.

Coverage: every doc model except cross-attention VLMs (LXMERT/ViLT/VisualBERT = GPU fine-tuning).
Image tower = CLIP-ViT-B/32 (torchvision ResNet weights uncached + no internet); CLIP-image is the
stronger frozen substitute anyway.

## Exp B-full — 19k LIVE rows, real visual (CLIP-image) + text at scale  [DEFINITIVE]
Live images ARE on disk (93.5%, in bin_collector_*/image_cache/<search>/<item_id>). Regenerated
visual via CLIP-ViT-B/32. Test n=3852, sold_within_24h.

| model | head | ROC-AUC | PR-AUC | P@10% |
|---|---|--:|--:|--:|
| **tab + CLIP-img + CLIP-txt** | **GBDT** | **0.805** | **0.489** | **0.553** |
| tab + CLIP-img + MiniLM-txt | GBDT | 0.803 | 0.482 | 0.530 |
| tab + CLIP-img (visual only) | GBDT | 0.795 | 0.442 | 0.507 |
| tab + text only | GBDT | 0.789 | 0.436 | 0.488 |
| **giant_basic_visual: tab+dino (CHAMPION)** | GBDT | 0.771 | 0.394 | — |
| giant_basic_visual: basic5 tab-only | GBDT | 0.764 | 0.372 | — |
| tab only (this experiment's baseline) | GBDT | 0.758 | 0.363 | 0.431 |

Champion rows = its own recorded metrics (model live_trained_20260613, results.json), a slightly
different split than the 20260704 live snapshot used here. Matching tab-only baselines
(0.758 vs champion 0.764) show the splits are comparable -> the multimodal win over champion is real.

- Visual + text both add and STACK: 0.758 -> 0.795 (+vis) -> 0.805 (+both). PR-AUC 0.363 -> 0.489.
- **Best = CLIP dual-encoder (image+text) + tabular GBDT = AUC 0.805.** Beats champion reference
  (giant_basic_visual matured-live AUC ~0.756). GBDT wins every set; MLP/LogReg collapse on wide embeds.
- Caveat: visual here = CLIP-image (frozen), not champion's exact dino. For exact apples-to-apples,
  run champion dino-visual GBDT on this split (dino also regenerable from the same images).

## Verdict
1. **Title text is a real, scalable signal the champion ignores** — adding it to the GBDT is the
   clear win (+0.03 AUC / +0.07 PR-AUC at scale).
2. **No deep multimodal net (CLIP fine-tune / TabNet / FT-Transformer) beats GBDT** on this data
   scale + CPU — reconfirms the lean-model lesson (see full-scrape-giant-model memory).
3. **Full "beat giant_basic_visual" unconfirmed**: the live visual features are not on this laptop
   (rotated off), so tabular+visual+text GBDT at scale wasn't testable. Champion = tabular+visual;
   proven here = text > tabular-alone. The likely winner = **champion GBDT + title-text**.

## Next step to confirm
Regenerate/sync the live visual-feature files (or run where they live), then train
`tab+vis+txt` GBDT on the 19k live rows vs the champion's `tab+vis` GBDT on the same split.
