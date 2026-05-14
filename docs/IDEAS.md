# Ideas

This file is a loose notebook for project ideas, experiments, questions, and future improvements.

It does not need to be perfectly clean. The point is to avoid losing thoughts.

When an idea becomes a real project rule or workflow, move it into the correct structured doc.

## Deal Ranking

Possible future ranking signals:

- Seller reliability.
- Seller review count.
- Seller star rating.
- Time since upload.
- View count.
- Interested count.
- View-to-interest ratio.
- Image quality.
- Brand/category-specific pricing behavior.

## Second Project Track: Photo-Improvement Arbitrage

Idea:

- Look for listings where the item may be valuable or desirable, but the listing photos are very poor.
- The opportunity is not only "this will sell fast now", but "this could sell better after buying it, improving presentation, and relisting with better photos".
- Build a visual model that scores photo quality and detects listings with bad presentation.
- Combine photo-quality score with item/value signals so the project finds items that are both under-presented and commercially interesting.

Possible model direction:

- Use a vision foundation model such as DINOv3 or DINOv2 to create image embeddings.
- Train a lightweight classifier or regressor on top of those embeddings to predict "bad listing photos" vs "good listing photos".
- Add simple image-quality metrics too: brightness, blur, contrast, saturation, resolution, aspect ratio, clutter, number of photos, and whether the item is clearly visible.
- Consider a separate aesthetic-quality model such as CLIP/LAION-style aesthetic scoring as a baseline, but adapt it to marketplace photos because "beautiful photo" and "useful selling photo" are not exactly the same thing.

Labels to create:

- `photo_quality_bad`: dark, blurry, low-resolution, cluttered, cropped badly, item not clearly visible, screenshots, mirror glare, bad background.
- `photo_quality_good`: clear lighting, item centered, multiple angles, clean background, details visible.
- `under_presented_candidate`: item has poor photos but seems potentially valuable based on brand, category, price, condition, and comparable listings.
- `not_worth_relisting`: poor photos but also weak item/value signal.

Evaluation idea:

- First evaluate visually: can the model separate obviously bad photos from good photos?
- Then evaluate commercially: do bad-photo candidates have higher resale upside after filtering by brand/category/price?
- Track real outcomes separately from the fast-sale model, because this strategy is about relisting improvement, not just immediate sell-speed.

Important boundaries:

- This track should not automate purchases, messages, or seller contact.
- Any buying decision should stay manual.
- Relisted photos should accurately represent the item and should not hide defects.
- Profit estimates must include shipping, fees, return risk, time cost, and the chance the item does not resell.

Status:

Second project track idea.

## Two-Stage Full-Data Deal Model

Idea:

- Keep the first-stage model based only on public listing snapshot fields, so every first-page candidate can be scored cheaply and immediately.
- Add a second-stage model for shortlisted candidates after full item/seller enrichment is available.
- The second-stage model can use richer fields such as description, condition, upload age, seller reviews, seller stars, location, interested count, view count, picture count, image count, and image-quality features.
- Use this model to rerank only candidates that pass an initial threshold or land in the top-K, instead of fully enriching every item.

Why it might help:

- False positives may be caused by missing context in the first-page snapshot.
- Seller quality, condition, description wording, and picture count may separate genuinely attractive deals from listings that only look good from price/title.
- This creates a practical funnel: broad cheap scoring first, richer precise scoring second.

Status:

Idea only.

## Model Ensemble And Voting

Idea:

- Combine multiple live-ready models instead of relying on one model per search.
- Test simple voting rules such as "recommend only if at least 2 of 3 models pass threshold".
- Test score averaging or weighted averaging across the strongest models.
- Test conservative intersections for high precision and unions for broader recall.
- Keep thresholds search-specific and model-specific, based on validation or live paper-trading results.

Why it might help:

- Different models catch different patterns.
- Agreement between independent approaches may reduce false positives.
- Disagreement between models can become a useful uncertainty signal.

Status:

Idea only.

## Clean Live-Labeled Deal Dataset

Idea:

- Build a clean dataset from paper-trading/live benchmark results.
- Each row should represent one observed candidate at one ranking time.
- Store the exact search name, timestamp, model version, threshold version, features available at ranking time, model score, rank, and whether it was selected.
- Add outcome labels only after enough time has passed: `sold_within_2h`, `sold_within_12h`, `sold_within_2d`, and `sold_within_7d`.
- Keep unevaluated rows separate from true negatives, because an item that has not reached the 2-day window yet is not a confirmed 2-day negative.

Why it might help:

- The historical CSVs are useful, but live labels are cleaner because the observation time is exact.
- This dataset becomes the main source for improving models against the real goal: fast sale after being observed.
- It allows fair comparison between models, thresholds, and searches.

Status:

Idea only.

## False-Positive Analysis

Idea:

- Study items that were ranked highly or selected by threshold but did not sell quickly.
- Group false positives by search, model, threshold, brand, price bucket, likes, upload age, seller review count, seller stars, condition, picture count, and item type.
- Compare false positives against true positives in the same search to find what the model is overvaluing.
- Create recurring failure tags such as "too expensive", "weak seller", "bad condition", "too niche", "low demand", "misleading brand/title", or "already saturated item type".

Why it might help:

- High precision usually improves by removing repeated false-positive patterns.
- False positives can become hard-negative examples for the next training run.
- It can reveal where simple rules should override model enthusiasm.

Status:

Idea only.

## False-Negative Analysis

Idea:

- Study items that sold quickly but were ranked low or not selected by the model.
- Use all saved first-page candidates, not only selected items, so missed opportunities can be found.
- Compare false negatives against selected true positives to find signals the model ignored.
- Check whether false negatives share patterns such as low likes, unusual title wording, rare sizes, specific brands, low starting price, or very fresh upload age.

Why it might help:

- False negatives show what the current model does not understand yet.
- They are useful for adding features, adjusting thresholds, and deciding whether a separate search-specific model is needed.
- They help improve recall without blindly lowering thresholds.

Status:

Idea only.

## Search-Specific Thresholds

Idea:

- Choose thresholds separately for each search instead of using one global threshold.
- Optimize thresholds against live outcomes, especially precision among selected items for 2h, 12h, and 2d sale windows.
- Use stricter thresholds for noisy searches and looser thresholds for searches where the model is consistently precise.
- Track threshold versions so old results remain interpretable.

Why it might help:

- Searches have different base sale rates and different item mixes.
- A probability score of `0.80` may mean different things in `nike`, `gucci`, `ps4`, or `griffati_donna_all`.
- Search-specific thresholds can improve precision without retraining the model.

Status:

Idea only.

## Hard-Negative Mining From False Positives

Idea:

- Take high-scoring false positives from live tests and add them as especially important negative examples in later training.
- Train future models to separate "looks like a good deal but did not sell quickly" from true fast-sale items.
- Keep hard negatives search-specific when the failure pattern is search-specific.
- Avoid using rows whose evaluation window is not mature yet.

Why it might help:

- The model may currently learn broad sold/not-sold patterns but miss subtle reasons a listing does not sell quickly.
- Hard negatives teach the model about difficult mistakes, not just easy bad listings.
- This can improve precision at the top of the ranking.

Status:

Idea only.

## Model Calibration

Idea:

- Check whether predicted probabilities match real live sale rates.
- For example, among items scored around `0.80`, verify whether about 80% actually sell within the target window.
- Calibrate models per search if needed using validation/live data.
- Report calibration curves or probability buckets alongside precision metrics.

Why it might help:

- Good ranking is not the same as good probability estimation.
- Threshold decisions are easier when scores mean something consistent.
- Calibration can make probability thresholds more trustworthy and easier to compare across models.

Status:

Idea only.

## Time-Of-Day And Upload-Age Features

Idea:

- Add features for observation hour, day of week, freshness/upload age, and possibly time since first seen.
- Track whether listings observed at certain times sell faster.
- Separate "newly uploaded and cheap" from "old listing still visible on page 1".
- Use upload-age features carefully so only ranking-time information is used.

Why it might help:

- Fast-sale behavior is strongly tied to freshness.
- Some searches may be more active at specific hours or days.
- Upload age can distinguish genuinely hot listings from stale listings that remain visible.

Status:

Idea only.

## Separate Models Per Search

Idea:

- Train and evaluate separate models per search when behavior is clearly different.
- Keep a global model as a baseline, but do not force one model to fit every search.
- Decide search-specific modeling based on live precision, false-positive patterns, false-negative patterns, and enough sample size.
- Use simpler models for small searches and richer models for searches with enough data.

Why it might help:

- `ps4`, `nike`, `gucci`, and branded-clothing searches likely have different demand patterns.
- Features that help one search can hurt another.
- Separate models can learn category-specific price, brand, title, seller, and image patterns.

Status:

Idea only.

## Image Quality

Current listing images are mostly cached as `.webp`.

Possible future experiment:

- Keep normal `.webp` cache for broad ranking.
- Fetch higher-resolution images only for shortlisted buy candidates.
- Compare machine-vision performance using current images vs higher-resolution images.

Status:

Idea only.

## Bright Data MCP

Could be useful for:

- Debugging live Vinted pages.
- Testing selectors.
- Investigating blocked or dynamic pages.
- Prototyping scraping logic interactively.

Probably not ideal as the main production pipeline.

Better likely split:

- Production scraper uses Python plus Bright Data proxy/Web Unlocker/Browser API.
- Codex/debugging uses Bright Data MCP when live page inspection is useful.

Status:

Maybe later.

## Eventual-Sale Evaluation

Possible improvements:

- Track how long it takes for an item to sell after first scrape.
- Separate quick sales from slow sales.
- Evaluate deal score by sale speed, not only sold/not sold.
- Use eventual-sale labels to tune category-specific thresholds.

Status:

Idea only.

## Final Buy Filter

Possible improvements:

- Save a clearer explanation for why each candidate was recommended or rejected.
- Add seller-level risk flags.
- Add image-based condition checks.
- Add category-specific final-buy rules.

Status:

Idea only.

## Remote Workflow

Possible improvements:

- Use `tmux` session names per task, such as `vinted-main`, `vinted-eval`, and `vinted-debug`.
- Add a small script to show active scraper jobs.
- Add a small script to tail important logs.

Status:

Idea only.
