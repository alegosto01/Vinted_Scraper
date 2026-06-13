---
name: telegram-policy
description: Change, explain, or verify the Basic5 giant-model Telegram sending policy. Use for price filters, model-pass rules, thresholds, reranking removal/addition, dry-runs, dedupe behavior, sent-log behavior, and tests around Telegram candidates.
---

# Telegram Policy

Use this skill when the user asks which giant-model items are sent to Telegram
or asks to change that behavior.

## Current Policy

The canonical policy lives in:

```text
scripts/experiments/current/basic_5_giant_model/apply_to_live_collector.py
```

At the time this skill was written, Telegram candidates are:

- one row per unique item
- item passes at least one normal Basic5 giant-model threshold
- item price is strictly greater than `30 EUR`
- already-sent items are deduped through `data/experiments/basic_5_giant_model/live_scoring/telegram_sent_items.csv`

The key functions/constants are:

- `TELEGRAM_MIN_PRICE_EUR`
- `TELEGRAM_POLICY_DESCRIPTION`
- `telegram_price_series()`
- `build_telegram_candidates()`
- `send_candidates_to_telegram()`
- `run_once()`

## Rules

- Use Graphify first if source locations are uncertain.
- Keep behavior changes narrow and documented.
- Default to `--telegram-dry-run` when validating candidate counts.
- Do not send live Telegram messages unless the user explicitly asks for a real send.
- Do not edit credentials or private Telegram configuration.
- Preserve the sent-log dedupe behavior unless the user explicitly asks to reset or change it.

## Workflow

1. Read the current policy code and tests:

   ```bash
   sed -n '300,470p' scripts/experiments/current/basic_5_giant_model/apply_to_live_collector.py
   sed -n '150,230p' tests/test_basic_5_giant_model.py
   ```

2. If changing policy, update all of:
   - candidate-building code
   - `TELEGRAM_POLICY_DESCRIPTION`
   - tests in `tests/test_basic_5_giant_model.py`
   - docs in `docs/BASIC_5_GIANT_MODEL.md` if user-visible behavior changed

3. Validate with dry-run or tests before suggesting live send:

   ```bash
   /home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/basic_5_giant_model/apply_to_live_collector.py --live-run-dir <live-run-dir> --telegram-dry-run
   ```

## Finish With Verification

End with the focused Telegram policy tests:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python -m unittest tests.test_basic_5_giant_model
```

If code changed, also refresh the graph:

```bash
graphify update . --force
```
