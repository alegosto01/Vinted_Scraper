# Basic 5 Giant Model Results

This page tracks the giant Basic5 follow-ups so the results do not get lost across chats.

## Main Run

Run folder:

```text
data/experiments/basic_5_giant_model/offline_runs/basic_5_giant_20260525_185552/
```

Training command:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/basic_5_giant_model/run.py
```

Dataset:

- Searches: `griffati_donna_all`, `griffati_uomo_all`, `gucci`, `nike`, `prada`, `ps4`
- Rows: 34,347 total; 20,608 train; 6,869 validation; 6,870 test
- Features: `Price`, `Likes`, `Title`, `Brand`, `Size`, plus one-hot `SearchName`

## Results Tracker

| Follow-up | Best row | Selected | Positives | Precision | Takeaway |
|---|---|---:|---:|---:|---|
| Single global threshold | `xgboost_basic_v1` | 140 | 139 | 0.993 | Very precise, but under-selects several searches. |
| Per-search thresholds, XGBoost only | `xgboost_basic_v1` | 290 | 276 | 0.952 | Best current coverage/precision tradeoff. |
| Best model per search | HistGradient for Griffati, XGBoost for Gucci/Nike/PS4, SGD for Prada | 254 | 245 | 0.965 | Higher precision than XGBoost-only, but less coverage. |
| Weighted voting | `auc_hard_weighted_vote` | 258 | 244 | 0.946 | Did not beat XGBoost per-search thresholds. |

## Weighted Voting

Command:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/basic_5_giant_model/report_weighted_voting.py --run-dir data/experiments/basic_5_giant_model/offline_runs/basic_5_giant_20260525_185552
```

Output:

```text
data/experiments/basic_5_giant_model/offline_runs/basic_5_giant_20260525_185552/weighted_voting_report.md
```

Weighted voting tested soft score averaging and hard thresholded voting with uniform, AUC, PR-AUC, P@25, and qualified-precision weights. The best weighted row was `auc_hard_weighted_vote`, which selected 258 rows at precision 0.946. This is slightly worse than XGBoost per-search thresholds, which selected 290 rows at precision 0.952.

## Live Telegram Sending

The live scorer can notify the recommended deals Telegram chat when an item passes at least one giant-model threshold:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/basic_5_giant_model/apply_to_live_collector.py \
  --live-run-dir data/experiments/time_to_sell/live_runs/bin_collector_20260525_071846 \
  --send-telegram
```

Useful safety options:

- `--telegram-dry-run`: count candidates without sending messages.
- `--telegram-limit N`: cap a single send pass.
- `--run-loop --send-telegram`: keep rescoring the live collector and send only newly passing items.

Sends are deduped by item in:

```text
data/experiments/basic_5_giant_model/live_scoring/telegram_sent_items.csv
```

## Telefoni Run (2026-06-11)

Run folder:

```text
data/experiments/basic_5_giant_model/offline_runs/basic_5_giant_20260611_204911/
```

Added `telefoni` as the 7th search (2,536 labeled rows: 1,617 sold / 919 not_sold, built via `scripts/stage_balanced_full_scrape.py --search telefoni --output-subdir full_scrape_stage_resume_20260611` plus `scripts/experiments/current/basic_5_giant_model/merge_balanced_full_scrape.py`; ~700 unsold candidates were dead links and could not be labeled).

Telefoni test metrics (508 rows, base rate 0.638): `xgboost_basic_v1` selects 90 at precision 0.922 (recall 0.256); `numeric_tree_v1`/`hist_gradient` reach ~0.97 precision at ~35 selected. Old-6 aggregate with xgboost per-search thresholds moved from 290 selected @ 0.952 to 317 @ 0.924 (prada/nike/gucci improved or held; griffati_uomo_all dipped to 0.741 while selecting 58 vs 25 — split was re-randomized, so part of the movement is noise).

## Current Decision

Use `xgboost_basic_v1` with per-search thresholds as the current best single giant-model candidate. Keep the best-per-search mix as a precision-oriented alternative if lower coverage is acceptable.

## Main-Image Visual Shadow

`giant_basic_visual` tests whether main-image-only features improve the Basic5
giant model on the same filtered rows and split:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/giant_basic_visual/run.py --limit-rows 50
```

This is shadow-only. It does not change Telegram sending. See
`docs/GIANT_BASIC_VISUAL.md`.
