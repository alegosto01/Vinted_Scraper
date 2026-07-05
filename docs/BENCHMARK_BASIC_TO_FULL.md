> ⚠️ **Caveman-compressed** — terse/fragment style to save tokens. Technical substance, code, commands, URLs kept verbatim. Original backed up under `~/.local/share/caveman-compress/backups/`.

# Benchmark Basic To Full

## Goal

Experiment test two-stage deal rank cascade:

1. Run cheap teacher-student basic model first.
2. Fully collect item/seller details only for items pass basic model.
3. Add photo-arbitrage visual features.
4. Run full-scrape-plus-visual model as final precision filter.

Experiment isolated under:

```text
experiments/old/benchmark_basic_to_full/data/
```

No write to production `old_df.csv`, `sold_df.csv`, `unsold_df.csv`, or normal full-scrape files.

## Models

Default model source:

```text
experiments/old/full_scrape_model/data/offline_runs/sold_status_feature_modalities_20260515_full_visual/
```

Default first-stage student source now precision-trained run:

```text
experiments/old/teacher_student_basic_filter/data/offline_runs/student_fullvisual_score_precision_20260516_222723/
```

Prior recall-trained student kept for compare at:

```text
experiments/old/teacher_student_basic_filter/data/offline_runs/student_fullvisual_score_20260515_154011/
```

Cascade use:

- Stage 1: best teacher-student basic model per search, trained imitate full+visual teacher score. Threshold chosen on validation for **teacher precision** (P(teacher pass | student pass) ≥ target), not recall.
- Stage 2: best `full_scrape_plus_visual` model per search.

Old `basic_5` models still usable for compare with:

```bash
--stage1-source basic_5
```

Recall-trained student re-select explicitly with:

```bash
--student-run student_fullvisual_score_20260515_154011 --student-objective recall --student-recall-target 0.95
```

Current default stage plan use 95% teacher-precision target:

| Search | Stage 1 | Stage 1 Threshold | Stage 2 | Stage 2 Threshold |
| --- | --- | ---: | --- | ---: |
| `griffati_donna_all` | `extra_trees_basic_student_v1` | `0.9569` | `numeric_tree_v1` | `0.9728` |
| `griffati_uomo_all` | `sgd_huber_basic_student_v1` | `0.5677` | `numeric_tree_v1` | `0.9680` |
| `gucci` | `sgd_huber_basic_student_v1` | `0.8759` | `logistic_v1_baseline` | `0.9643` |
| `nike` | `sgd_huber_basic_student_v1` | `0.3399` | `numeric_tree_v1` | `0.9917` |
| `prada` | `sgd_huber_basic_student_v1` | `0.9561` | `linear_svm_calibrated_v1` | `0.9955` |
| `ps4` | `extra_trees_basic_student_v1` | `0.9651` | `linear_svm_calibrated_v1` | `0.9542` |

Why switch: recall-trained first stage kept ~94% teacher-approved test items but pushed many items into expensive full-scrape stage teacher later reject. Switch first stage to precision objective make tight filter that only forward items full+visual teacher likely also approve. `gucci`, `ps4`, `prada` hit 95% teacher-precision floor on validation cleanly. `griffati_uomo_all` and `nike` no hit — no student threshold on those searches reached 95% teacher precision on validation, so threshold selector fall back to best-achievable precision row.

## Recheck Schedule

Tracked items rechecked with this schedule:

- First 24 hours: every 1 hour.
- 24 to 48 hours: every 3 hours.
- After 48 hours: every 12 hours until 7 days.

Precision reported for checkpoint windows:

```text
1h, 2h, 3h, ..., 24h,
27h, 30h, 33h, ..., 48h,
60h, 72h, ..., 168h
```

## Non-Skippable Rules

Rules part of cascade contract, must not be skipped by any runner, restart, repair script, or manual collect cmd:

- Item identity = `(SearchName, item_id)`, with `item_id` normalized from `Dataid` or Vinted item URL. Key decide if item already seen.
- Each item scored by stage 1 at most once per run. After stage-1 decision exists, later hourly catalog snapshots must not score that item again.
- Stage-1 rejects must be remembered in run state, not silently forgotten. Rejected item must not re-enter stage 1 on next hourly snapshot.
- Each stage-1 pass item may enter expensive path at most once: one full item-page collection, one primary-image download/cache pass, one visual feature extraction pass, one stage-2 score.
- After stage 2 decision, item must not be full-scraped, visually enriched, or stage-2 scored again same run. Only market-status rechecks allowed.
- Items tracked for precision or false-negative analysis rechecked only per schedule above until sold or age out 7-day horizon.
- Rechecks must fetch only minimum data needed decide if item still on sale or sold. Must not refresh model features or alter prior stage scores.
- Reports must separate repeated collect events from unique item counts. Model quality and precision decisions based on unique items; event counts only for op load monitor.
- Restart loop must resume existing run state before scrape. Restart must not create duplicate work for items already have stage-1 or stage-2 decisions.

## Metrics

Main metric = final precision among items passed both stages.

Show live results with unique-item, matured-cohort format in
[`CASCADE_RESULTS_REPORTING.md`](CASCADE_RESULTS_REPORTING.md).

Report also track:

- Stage-1 pass count.
- Stage-2 pass count.
- Full item collection success/failure count.
- Precision by checkpoint window.
- False positives: stage-2 pass items that did not sell by an evaluated window.
- False negatives: stage-1 pass items rejected by stage 2 that later sold.

## Commands

Dry-run plan:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python experiments/old/benchmark_basic_to_full/cascade_runner.py collect-once --dry-run --out-dir experiments/old/benchmark_basic_to_full/data/live_runs/dry_run_plan
```

Dry-run old strict basic plan for compare:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python experiments/old/benchmark_basic_to_full/cascade_runner.py collect-once --dry-run --stage1-source basic_5 --out-dir experiments/old/benchmark_basic_to_full/data/live_runs/dry_run_old_basic_plan
```

Run one collect pass:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python experiments/old/benchmark_basic_to_full/cascade_runner.py collect-once --out-dir experiments/old/benchmark_basic_to_full/data/live_runs/cascade_live
```

Recheck due items:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python experiments/old/benchmark_basic_to_full/cascade_runner.py recheck-due --out-dir experiments/old/benchmark_basic_to_full/data/live_runs/cascade_live
```

Gen report:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python experiments/old/benchmark_basic_to_full/cascade_runner.py report --out-dir experiments/old/benchmark_basic_to_full/data/live_runs/cascade_live
```

Plot live stage-1 and stage-2 score distributions for sold vs unsold items matured through checkpoint:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python experiments/old/benchmark_basic_to_full/report_live_score_distributions.py --run-dir experiments/old/benchmark_basic_to_full/data/live_runs/cascade_live --window-hours 24
```

Distribution report write single figure with both model score columns and their effective per-search thresholds under selected run `reports/` folder. Use `--all-observed` only when "not sold yet" rows acceptable instead of matured checkpoint cohort.

Run scheduled loop:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python experiments/old/benchmark_basic_to_full/cascade_runner.py run-loop --out-dir experiments/old/benchmark_basic_to_full/data/live_runs/cascade_live --collect-every-hours 1
```

Drain existing live run without collect new items, stop recheck after `72h` checkpoint filled:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python experiments/old/benchmark_basic_to_full/cascade_runner.py run-loop --out-dir experiments/old/benchmark_basic_to_full/data/live_runs/cascade_live --recheck-only --max-recheck-age-hours 72
```

Default `run-loop` now use teacher-student first stage with precision objective. To restart with precision student explicitly:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python experiments/old/benchmark_basic_to_full/cascade_runner.py run-loop --out-dir experiments/old/benchmark_basic_to_full/data/live_runs/cascade_live --student-run student_fullvisual_score_precision_20260516_222723 --student-objective precision --student-precision-target 0.95 --collect-every-hours 1
```

Compare with older recall-targeted first stage:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python experiments/old/benchmark_basic_to_full/cascade_runner.py run-loop --out-dir experiments/old/benchmark_basic_to_full/data/live_runs/cascade_live_recall_smoke --student-run student_fullvisual_score_20260515_154011 --student-objective recall --student-recall-target 0.95 --collect-every-hours 1
```

## Notes

- All stage-1 pass items tracked, not only final stage-2 pass items. Needed for false-negative analysis.
- Full+visual stage can be slower because it collect item pages and run photo-quality methods.
- DINO features computed from primary local image for live item.
- Final proof should come from live checkpoint precision, not only offline sold-vs-unsold metrics.
- If full collect volume too high, try 90% student-recall target before go back to old strict basic models.