# Find-Bad-Images — Findings & Model Decisions

Full record of what was tried, what the numbers said, and what was chosen. Companion to
`README.md` (how-to), `MUSIQ_RESULTS.md`, and `data/blur_batch/blur_results.md`.

Last updated: 2026-07-26.

---

## 1. Goal & scope

Rank Vinted listings by the **technical quality of their FIRST photo** (blur, dark,
overexposed, glare, bad crop, extreme tilt, low resolution/noise, clutter, item-not-clear).
NOT about product value, price, brand, seller, or resale. Ranking score, not a calibrated
probability. Only the first image.

**Hard-won scope conclusion:** a *general* "any bad photo" detector does not work with the
signals available. The tractable target is **blur + low-resolution** (sharpness/detail),
which is what the chosen model actually detects.

---

## 2. Data & alignment (frozen)

- Source CSV recovered & proven: `time_to_sell/.../bin_collector_20260602_214104/tracked_items.csv`
  (20,450 rows). Alignment proof: mask == image-exists for **20,450/20,450 rows, 0 disagreements**.
- Unique listings after dedup: **19,018**. Valid images: 19,173. MUSIQ scored: 17,810
  (rest = missing/undecodable images).
- Frozen CLIP image embeddings (openai/clip-vit-base-patch32, 512-d) used by v1/v2/v3.

---

## 3. Methods tried, in order

### Zero-shot / hand-crafted (all FAILED on fresh human labels)
| method | what | verdict |
|---|---|---|
| v1 generic CLIP | softmax over good/bad prompts | weak |
| v2 typed CLIP | product-type-aware prompts | weak, buggy typing |
| v3 CLIP defect margins | mean(bad)-mean(good) per defect, max | weak; **crop margin anti-predictive** |
| simple | pixel features → percentiles | best of the four, still weak |
| v4 hybrid | rules-only gate routing (pixel+CLIP), gates screened on 46 human labels | marginal |

**Evaluation evidence:**
- On 244 ChatGPT pseudo-labels (T1 bad-vs-good), macro P@8: simple 0.417 > v3 0.354 >
  v1 0.292 > v2 0.271. Looked like "simple wins."
- On a **fresh 63-image human holdout** (the honest test), everything collapsed to
  ~random: AUROC simple 0.387, v3 0.416, v4 0.569; macro P@8 ≤ 0.28. The earlier ranking
  was an artifact of enriched, model-selected, ChatGPT-labeled data.
- v3's false positives were all `crop` on normal tightly-framed photos (caps, shoes,
  sunglasses) — confirmed by human review. CLIP over-flags ordinary framing.

### Single-defect pivot: blur
- Raw Laplacian variance conflates blur with *plain/low-texture* photos → poor precision.
- **Fix = texture-normalized `blur_score`** (`blur_score.py`):
  `z(laplacian/edge) + z(laplacian/dynamic_range²)`. From existing columns, no image reload.
- Validated on **175 human labels (32 blurry)**: **AUROC 0.796**, P@5 1.00, P@8 0.75,
  P@10 0.70. Two independent labeling batches agreed. First thing that clearly worked.
- Tried to beat it with no-reference image features (re-blur, FFT) + a supervised model:
  **no gain — overfit on 32 positives** (grouped-CV dropped). Bottleneck = labels, not model.

### Pretrained no-reference IQA (the real improvement)
- `pyiqa` MUSIQ, zero training → tiny labels only validate, no overfit.
- **MUSIQ beats blur_score and revives the general target:**
  - grouped-CV: MUSIQ blur **0.843** vs blur_score 0.777; general **0.837** vs 0.728
    (vs ~0.5 for the CLIP methods).
  - **fresh holdout AUROC 0.847** vs blur_score 0.707 — holds on unseen data.
- Confirmation batch (60, stratified by worst-MUSIQ, 14 bad): AUROC 0.736, **worst-3% of
  MUSIQ = 41% bad** (~2× base rate), top-8 = 50% bad. So: a useful **~2× review queue**,
  not a precise classifier. Defects it flags: mostly **blur + low_resolution**; it misses
  crop/clutter/tilt/glare.

### IQA model bake-off (all pretrained, offline weights)
| model | blur AUROC | general AUROC |
|---|---|---|
| **MANIQA** | **0.867** | **0.853** |
| DBCNN | 0.860 | 0.827 |
| MUSIQ | 0.848 | 0.846 |
| CLIP-IQA+ | 0.832 | 0.814 |
MANIQA is marginally best (+0.02 over MUSIQ, within noise on 32 positives) but ~44 min
for 210 images on CPU → hours for 19k.

### Ensemble / voting (tested, does NOT help)
On 134 general-labeled (grouped-CV): MUSIQ alone **0.804** beats every combination —
musiq+blur 0.782, musiq+semantic 0.751, all-7 logistic 0.791. Voting fails because the
only strong member is MUSIQ; the others are correlated (other IQA) or near-random
(CLIP semantic). No free lunch.

---

## 4. CHOSEN MODEL

**MUSIQ (pyiqa) as the production quality/blur detector.**
- Best accuracy/compute trade-off: fresh-holdout AUROC 0.85, already scored on all 19,018,
  fast enough to rescore.
- **MANIQA is the accuracy ceiling** (+0.02) — reserve for re-ranking only the top-N MUSIQ
  candidates if the marginal gain is ever wanted (cheap on a shortlist, hours on the full set).
- Operating threshold (Youden on 134 human labels): **MUSIQ < 67.7 = "bad"**. For a
  cleaner queue, prefer top-N ranking over the hard threshold.
- **What it detects:** blur + low-resolution/soft/low-detail. NOT crop/clutter/tilt/glare.

`blur_score` is kept as a lightweight, dependency-free blur-only fallback (AUROC 0.80).

---

## 5. Key files

| file | purpose |
|---|---|
| `alignment.py` | recover/lock embedding↔CSV alignment |
| `score_bad_photos.py` | main table: v1/v2/v3/simple + blur_score (19,018 rows) |
| `blur_score.py` | texture-normalized blur ranker |
| `build_musiq.py` | MUSIQ over all 19k → `data/scored_with_musiq.csv` (`musiq`, `musiq_bad`) |
| `rank_blur.py` / `data/blur_ranking.csv` | top-N blurriest per search |
| `data/musiq_ranking.csv` | top-N worst-MUSIQ per search (the review queue) |
| `analyze_chatgpt_labels.py` | frozen 3-target evaluation vs pseudo-labels |
| `build_holdout.py` / `compare_final.py` | fresh blind holdout + final comparison |
| `hybrid.py` / `develop_gates.py` | v4 gate routing + screening (superseded by MUSIQ) |
| `build_blur_batch.py` / `build_musiq_batch.py` | stratified labeling batches |
| `live_musiq_noproxy.py` | live scrape (no proxy, paced) + MUSIQ flagging |
| `MUSIQ_RESULTS.md`, `data/blur_batch/blur_results.md` | detailed result docs |
| `tests/` | 39 unittest tests (alignment, margins, dedup, metrics, blur, hybrid) |

Human labels collected this project: 244 ChatGPT pseudo + ~230 human (holdout 63,
audit 46, blur batches 120, musiq batch 60). Clear human "bad": still only ~30–40 — the
main constraint on training anything supervised.

---

## 6. Live test (in progress at time of writing)

`live_musiq_noproxy.py`: scrape newest 10 pages of griffati_uomo_all + griffati_donna_all
with **no proxy** (`curl_cffi` chrome impersonation, 1 page/min to dodge DataDome), MUSIQ-
score first images, flag MUSIQ < 67.7. Output → `data/live_musiq_test.csv`.
Note: the slow pace cleared DataDome where a fast/proxy burst had been blocked.

---

## 7. Honest limitations

- Validation labels are modest (~30–40 clear "bad"); AUROCs are solid but precision@K has
  wide error bars.
- MUSIQ = sharpness/detail sensor; crop/clutter/tilt/glare remain undetected by any tried method.
- Scores are rankings, not probabilities. Bad photos are rare (~5–20% depending on region).
- No dataset-wide recall (enriched samples).

## 8. Next steps (not done)

1. **Ship MUSIQ queue** into `app.py`/dashboard as "worst photos to review".
2. Optional **MUSIQ→MANIQA re-rank** on the top-N shortlist for the marginal accuracy gain.
3. To ever beat ~0.85 or catch semantic defects: collect **~50+ clear bad per defect** and
   train a small supervised head — the only lever left; hand-crafted/zero-shot is exhausted.
