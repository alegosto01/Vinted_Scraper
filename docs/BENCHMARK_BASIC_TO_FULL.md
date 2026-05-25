# Benchmark Basic To Full

## Goal

This experiment tests a two-stage deal ranking cascade:

1. Run a cheap teacher-student basic model first.
2. Fully collect item/seller details only for items that pass the basic model.
3. Add photo-arbitrage visual features.
4. Run the full-scrape-plus-visual model as the final precision filter.

The experiment is isolated under:

```text
data/experiments/benchmark_basic_to_full/
```

It does not write to production `old_df.csv`, `sold_df.csv`, `unsold_df.csv`, or normal full-scrape files.

## Models

The default model source is:

```text
data/experiments/full_scrape_model/offline_runs/sold_status_feature_modalities_20260515_full_visual/
```

The default first-stage student source is now the precision-trained run:

```text
data/experiments/teacher_student_basic_filter/offline_runs/student_fullvisual_score_precision_20260516_222723/
```

The previous recall-trained student is preserved for comparison at:

```text
data/experiments/teacher_student_basic_filter/offline_runs/student_fullvisual_score_20260515_154011/
```

The cascade uses:

- Stage 1: best teacher-student basic model per search, trained to imitate the full+visual teacher score. Threshold is chosen on validation for **teacher precision** (P(teacher pass | student pass) ≥ target), not recall.
- Stage 2: best `full_scrape_plus_visual` model per search.

The old `basic_5` models can still be used for comparison with:

```bash
--stage1-source basic_5
```

The recall-trained student can be re-selected explicitly with:

```bash
--student-run student_fullvisual_score_20260515_154011 --student-objective recall --student-recall-target 0.95
```

Current default stage plan uses the 95% teacher-precision target:

| Search | Stage 1 | Stage 1 Threshold | Stage 2 | Stage 2 Threshold |
| --- | --- | ---: | --- | ---: |
| `griffati_donna_all` | `extra_trees_basic_student_v1` | `0.9569` | `numeric_tree_v1` | `0.9728` |
| `griffati_uomo_all` | `sgd_huber_basic_student_v1` | `0.5677` | `numeric_tree_v1` | `0.9680` |
| `gucci` | `sgd_huber_basic_student_v1` | `0.8759` | `logistic_v1_baseline` | `0.9643` |
| `nike` | `sgd_huber_basic_student_v1` | `0.3399` | `numeric_tree_v1` | `0.9917` |
| `prada` | `sgd_huber_basic_student_v1` | `0.9561` | `linear_svm_calibrated_v1` | `0.9955` |
| `ps4` | `extra_trees_basic_student_v1` | `0.9651` | `linear_svm_calibrated_v1` | `0.9542` |

Rationale for the switch: the recall-trained first stage kept ~94% of teacher-approved test items but sent many items into the expensive full-scrape stage that the teacher would later reject. Switching the first stage to a precision objective makes it a tight filter that only forwards items the full+visual teacher is very likely to also approve. `gucci`, `ps4`, and `prada` hit the 95% teacher-precision floor on validation cleanly. `griffati_uomo_all` and `nike` did not — no student threshold on those searches reached 95% teacher precision on validation, so the threshold selector fell back to the best-achievable precision row.

## Recheck Schedule

Tracked items are rechecked with this schedule:

- First 24 hours: every 1 hour.
- From 24 to 48 hours: every 3 hours.
- After 48 hours: every 12 hours until 7 days.

Precision is reported for checkpoint windows:

```text
1h, 2h, 3h, ..., 24h,
27h, 30h, 33h, ..., 48h,
60h, 72h, ..., 168h
```

## Non-Skippable Rules

These rules are part of the cascade contract and must not be skipped by any
runner, restart, repair script, or manual collection command:

- Item identity is `(SearchName, item_id)`, with `item_id` normalized from
  `Dataid` or the Vinted item URL. This key decides whether an item has already
  been seen.
- Each item is scored by stage 1 at most once per run. After a stage-1 decision
  exists, later hourly catalog snapshots must not score that item again.
- Stage-1 rejects must be remembered in run state, not silently forgotten. A
  rejected item should not re-enter stage 1 on the next hourly snapshot.
- Each stage-1 pass item may enter the expensive path at most once: one full
  item-page collection, one primary-image download/cache pass, one visual
  feature extraction pass, and one stage-2 score.
- After stage 2 has a decision, the item must not be full-scraped, visually
  enriched, or stage-2 scored again in the same run. Only market-status rechecks
  are allowed.
- Items that are tracked for precision or false-negative analysis are rechecked
  only according to the schedule above until they are sold or age out of the
  7-day horizon.
- Rechecks must fetch only the minimum data needed to decide whether the item is
  still on sale or sold. They must not refresh model features or alter prior
  stage scores.
- Reports must separate repeated collection events from unique item counts.
  Model quality and precision decisions should be based on unique items, while
  event counts are only for operational load monitoring.
- Restarting the loop must resume the existing run state before scraping. A
  restart must not create duplicate work for items that already have stage-1 or
  stage-2 decisions.

## Metrics

The main metric is final precision among items that passed both stages.

Show live results with the unique-item, matured-cohort format described in
[`CASCADE_RESULTS_REPORTING.md`](CASCADE_RESULTS_REPORTING.md).

The report also tracks:

- Stage-1 pass count.
- Stage-2 pass count.
- Full item collection success/failure count.
- Precision by checkpoint window.
- False positives: stage-2 pass items that did not sell by an evaluated window.
- False negatives: stage-1 pass items rejected by stage 2 that later sold.

## Commands

Dry-run the plan:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/benchmark_basic_to_full/cascade_runner.py collect-once --dry-run --out-dir data/experiments/benchmark_basic_to_full/live_runs/dry_run_plan
```

Dry-run the old strict basic plan for comparison:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/benchmark_basic_to_full/cascade_runner.py collect-once --dry-run --stage1-source basic_5 --out-dir data/experiments/benchmark_basic_to_full/live_runs/dry_run_old_basic_plan
```

Run one collection pass:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/benchmark_basic_to_full/cascade_runner.py collect-once --out-dir data/experiments/benchmark_basic_to_full/live_runs/cascade_live
```

Recheck due items:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/benchmark_basic_to_full/cascade_runner.py recheck-due --out-dir data/experiments/benchmark_basic_to_full/live_runs/cascade_live
```

Generate report:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/benchmark_basic_to_full/cascade_runner.py report --out-dir data/experiments/benchmark_basic_to_full/live_runs/cascade_live
```

Plot the live stage-1 and stage-2 score distributions for sold vs unsold items
that have matured through a checkpoint:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/benchmark_basic_to_full/report_live_score_distributions.py --run-dir data/experiments/benchmark_basic_to_full/live_runs/cascade_live --window-hours 24
```

The distribution report writes a single figure with both model score columns and
their effective per-search thresholds under the selected run's `reports/`
folder. Use `--all-observed` only when "not sold yet" rows are acceptable
instead of a matured checkpoint cohort.

Run scheduled loop:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/benchmark_basic_to_full/cascade_runner.py run-loop --out-dir data/experiments/benchmark_basic_to_full/live_runs/cascade_live --collect-every-hours 1
```

Drain an existing live run without collecting new items, and stop rechecking
after the `72h` checkpoint has been filled:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/benchmark_basic_to_full/cascade_runner.py run-loop --out-dir data/experiments/benchmark_basic_to_full/live_runs/cascade_live --recheck-only --max-recheck-age-hours 72
```

The default `run-loop` now uses the teacher-student first stage with a precision objective. To restart with the precision student explicitly, use:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/benchmark_basic_to_full/cascade_runner.py run-loop --out-dir data/experiments/benchmark_basic_to_full/live_runs/cascade_live --student-run student_fullvisual_score_precision_20260516_222723 --student-objective precision --student-precision-target 0.95 --collect-every-hours 1
```

To compare with the older recall-targeted first stage:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/benchmark_basic_to_full/cascade_runner.py run-loop --out-dir data/experiments/benchmark_basic_to_full/live_runs/cascade_live_recall_smoke --student-run student_fullvisual_score_20260515_154011 --student-objective recall --student-recall-target 0.95 --collect-every-hours 1
```

## Notes

- All stage-1 pass items are tracked, not only final stage-2 pass items. This is needed for false-negative analysis.
- The full+visual stage can be slower because it collects item pages and runs photo-quality methods.
- DINO features are computed from the primary local image for the live item.
- The final proof should come from live checkpoint precision, not only offline sold-vs-unsold metrics.
- If full collection volume becomes too high, try the 90% student-recall target before going back to the old strict basic models.
