---
name: giant-model-results
description: Summarize Basic5 giant-model live/offline performance for this repo. Use for model performance tables, best model overall, best per search, AUC/accuracy/precision/recall, price buckets, sold item links, hourly performance, and "results so far".
---

# Giant Model Results

Use this skill to answer result-analysis questions for the Basic5 giant-model experiment.

## Rules

- Use Graphify before reading source if you need to rediscover metric code.
- Prefer existing generated outputs under `data/experiments/basic_5_giant_model/` before writing new analysis code.
- Do not retrain models unless the user explicitly asks.
- Do not send Telegram messages.
- Keep result tables scoped to the requested slice: live run, offline run, search, model, price bucket, threshold, or time window.

## Common Data Locations

- Offline runs: `data/experiments/basic_5_giant_model/offline_runs/`
- Live scoring runs: `data/experiments/basic_5_giant_model/live_scoring/live_scoring_*/`
- Telegram candidate outputs: `data/experiments/basic_5_giant_model/live_scoring/live_scoring_*/telegram_candidates.csv`
- Telegram sent log: `data/experiments/basic_5_giant_model/live_scoring/telegram_sent_items.csv`
- Current docs: `docs/BASIC_5_GIANT_MODEL.md`

## Workflow

1. Identify the relevant latest run or the explicit run the user named:

   ```bash
   find data/experiments/basic_5_giant_model/live_scoring -maxdepth 1 -type d -name 'live_scoring_*' -printf '%T@ %p\n' | sort -nr | head
   find data/experiments/basic_5_giant_model/offline_runs -maxdepth 1 -type d -name 'basic_5_giant_*' -printf '%T@ %p\n' | sort -nr | head
   ```

2. Inspect only the relevant outputs, usually:
   - `live_scored_report.md`
   - `performance_by_model.csv`
   - `performance_by_search_model.csv`
   - `telegram_candidates.csv`
   - `live_scored_items.csv`
   - `summary.json`
   - `manifest.json`

3. For custom summaries, prefer a short one-off Python read of CSV/JSON outputs over opening large data files.

4. Always state:
   - run folder
   - number of evaluated/tracked rows
   - whether metrics are live matured labels or offline test labels
   - whether filters such as price `>30 EUR` or `30-100 EUR` were applied

## Useful Existing Commands

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/basic_5_giant_model/report_per_search_thresholds.py --run-dir <offline-run-dir>
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/basic_5_giant_model/report_weighted_voting.py --run-dir <offline-run-dir>
```

## Finish With Verification

End with one focused verification command, usually:

```bash
python3 -m py_compile scripts/experiments/current/basic_5_giant_model/apply_to_live_collector.py
```

If only reading current live status, use:

```bash
ps -ef | rg 'apply_to_live_collector|basic_5_giant'
```
