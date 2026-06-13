> ⚠️ **Caveman-compressed** — terse/fragment style to save tokens. Technical substance, code, commands, URLs kept verbatim. Original backed up under `~/.local/share/caveman-compress/backups/`.

# Ideas

Loose notebook for ideas, experiments, questions, future improvements.

No need clean. Point: no lose thought.

Idea become real rule/workflow → move to structured doc.

## Deal Ranking

Future ranking signals:

- Seller reliability.
- Seller review count.
- Seller star rating.
- Time since upload.
- View count.
- Interested count.
- View-to-interest ratio.
- Image quality.
- Brand/category pricing behavior.

## Second Project Track: Photo-Improvement Arbitrage

Idea:

- Find listings: item maybe valuable but photos bad.
- Opportunity: not just "sell fast now" but "buy, fix photos, relist better".
- Build vision model: score photo quality, detect bad presentation.
- Combine photo score + value signals → find under-presented but commercially good items.

Model direction:

- Use vision foundation model (DINOv3/DINOv2) for image embeddings.
- Train light classifier/regressor on embeddings → predict bad vs good listing photos.
- Add simple metrics: brightness, blur, contrast, saturation, resolution, aspect ratio, clutter, photo count, item visible.
- Maybe separate aesthetic model (CLIP/LAION) as baseline, but adapt to marketplace — "beautiful" ≠ "good selling photo".

Labels:

- `photo_quality_bad`: dark, blurry, low-res, cluttered, bad crop, item hidden, screenshots, mirror glare, bad background.
- `photo_quality_good`: clear light, item centered, multiple angles, clean background, details visible.
- `under_presented_candidate`: bad photos but maybe valuable by brand/category/price/condition/comparables.
- `not_worth_relisting`: bad photos + weak value signal.

Evaluation:

- First visual: model separate bad from good?
- Then commercial: bad-photo candidates higher resale upside after brand/category/price filter?
- Track outcomes separate from fast-sale model — this strategy = relist improvement, not immediate speed.

Boundaries:

- No automated buy, message, seller contact.
- Buying stays manual.
- Relisted photos must show item truth, no hide defects.
- Profit estimates include shipping, fees, return risk, time cost, no-resell chance.

Status:

Second project track idea.

## Two-Stage Full-Data Deal Model

Idea:

- Stage 1 model: only public snapshot fields → every first-page candidate scored cheap, fast.
- Stage 2 model: shortlisted candidates after full item/seller enrichment.
- Stage 2 use richer fields: description, condition, upload age, seller reviews/stars, location, interested, views, picture/image count, image-quality features.
- Rerank only candidates past threshold or top-K, no full enrich every item.

Why help:

- False positives maybe from missing snapshot context.
- Seller quality, condition, description wording, picture count → separate real deals from price/title-only good.
- Funnel: broad cheap → richer precise.

Status:

Idea only.

## Model Ensemble And Voting

Idea:

- Combine many live-ready models, no rely on one per search.
- Test vote rules: "recommend if ≥2 of 3 pass threshold".
- Test score averaging / weighted averaging across strongest models.
- Test conservative intersections (precision) + unions (recall).
- Thresholds search-specific + model-specific, from validation/paper-trade.

Why help:

- Different models catch different patterns.
- Agreement = fewer false positives.
- Disagreement = useful uncertainty signal.

Status:

Idea only.

## Clean Live-Labeled Deal Dataset

Idea:

- Build clean dataset from paper-trade/live benchmark.
- Each row = one candidate at one ranking time.
- Store search name, timestamp, model version, threshold version, ranking-time features, score, rank, selected flag.
- Add outcomes after time: `sold_within_2h`, `sold_within_12h`, `sold_within_2d`, `sold_within_7d`.
- Keep unevaluated rows separate from true negatives — item not past 2d window ≠ confirmed 2d negative.

Why help:

- Historical CSVs useful, but live labels cleaner — exact observation time.
- Main source for improving models vs real goal: fast sale after observed.
- Fair compare across models, thresholds, searches.

Status:

Idea only.

## False-Positive Analysis

Idea:

- Study items ranked high / selected but no sell fast.
- Group by search, model, threshold, brand, price bucket, likes, upload age, seller reviews/stars, condition, picture count, item type.
- Compare false positives vs true positives same search → find what model overvalues.
- Create failure tags: "too expensive", "weak seller", "bad condition", "too niche", "low demand", "misleading brand/title", "saturated item type".

Why help:

- Precision improve by removing repeated false-positive patterns.
- False positives = hard negatives for next training.
- Reveal where simple rules override model enthusiasm.

Status:

Idea only.

## False-Negative Analysis

Idea:

- Study items sold fast but ranked low / not selected.
- Use all saved first-page candidates, not just selected, find missed.
- Compare false negatives vs selected true positives → find ignored signals.
- Check shared patterns: low likes, weird title wording, rare sizes, specific brands, low start price, very fresh upload.

Why help:

- False negatives show model gaps.
- Useful for new features, threshold tuning, search-specific model decision.
- Improve recall without blindly lowering thresholds.

Status:

Idea only.

## Search-Specific Thresholds

Idea:

- Choose thresholds per search, no one global.
- Optimize vs live outcomes — precision among selected for 2h/12h/2d windows.
- Stricter for noisy searches, looser for consistently precise searches.
- Track threshold versions → old results stay interpretable.

Why help:

- Searches have different base sale rates + item mixes.
- Score `0.80` mean different in `nike`, `gucci`, `ps4`, or `griffati_donna_all`.
- Search-specific thresholds = better precision, no retrain.

Status:

Idea only.

## Hard-Negative Mining From False Positives

Idea:

- Take high-scoring false positives from live → add as important negatives in next training.
- Train models to separate "looks like deal but no sell fast" from real fast-sale.
- Keep hard negatives search-specific when failure pattern is search-specific.
- Skip rows with immature evaluation window.

Why help:

- Model now learn broad sold/unsold, miss subtle reasons for slow sale.
- Hard negatives teach difficult mistakes, not just easy bad listings.
- Boost precision at ranking top.

Status:

Idea only.

## Model Calibration

Idea:

- Check predicted probabilities match real live sale rates.
- Example: items scored ~`0.80` → ~80% actually sell in window?
- Calibrate per search if needed from validation/live data.
- Report calibration curves / probability buckets alongside precision.

Why help:

- Good ranking ≠ good probability estimation.
- Threshold decisions easier when scores mean consistent.
- Calibration → probability thresholds more trust, easier cross-model compare.

Status:

Idea only.

## Time-Of-Day And Upload-Age Features

Idea:

- Add features: observation hour, day of week, freshness/upload age, time since first seen.
- Track whether listings observed at certain times sell faster.
- Separate "fresh + cheap" from "old still on page 1".
- Use upload age careful — only ranking-time info.

Why help:

- Fast-sale strongly tied to freshness.
- Some searches more active at specific hours/days.
- Upload age distinguish hot from stale-visible.

Status:

Idea only.

## Separate Models Per Search

Idea:

- Train + evaluate separate models per search when behavior clearly different.
- Keep global model baseline, no force one model fit all.
- Decide by live precision, false-positive/negative patterns, sample size.
- Simpler models small searches, richer models big data searches.

Why help:

- `ps4`, `nike`, `gucci`, branded-clothing → different demand.
- Features help one search can hurt other.
- Separate models learn category-specific price/brand/title/seller/image patterns.

Status:

Idea only.

## Image Quality

Listing images mostly cached as `.webp`.

Future experiment:

- Keep `.webp` cache for broad ranking.
- Fetch higher-res only for shortlisted buy candidates.
- Compare vision performance: current vs higher-res.

Status:

Idea only.

## Bright Data MCP

Useful for:

- Debug live Vinted pages.
- Test selectors.
- Investigate blocked/dynamic pages.
- Prototype scraping interactively.

Probably not ideal as main production pipeline.

Better split:

- Production scraper = Python + Bright Data proxy/Web Unlocker/Browser API.
- Codex/debug = Bright Data MCP when live page inspection useful.

Status:

Maybe later.

## Eventual-Sale Evaluation

Improvements:

- Track time from first scrape to sale.
- Separate quick from slow sales.
- Score deals by sale speed, not just sold/unsold.
- Use eventual-sale labels → tune category thresholds.

Status:

Idea only.

## Final Buy Filter

Improvements:

- Save clearer reason why candidate recommended/rejected.
- Add seller risk flags.
- Add image-based condition checks.
- Add category-specific final-buy rules.

Status:

Idea only.

## Remote Workflow

Improvements:

- Use `tmux` session names per task: `vinted-main`, `vinted-eval`, `vinted-debug`.
- Small script show active scraper jobs.
- Small script tail important logs.

Status:

Idea only.