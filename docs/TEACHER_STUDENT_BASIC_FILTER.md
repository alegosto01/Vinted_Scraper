# Teacher-Student Basic Filter

This experiment tests a cheaper first-stage filter for the deal pipeline.

The idea is:

- The **teacher** is the best `full_scrape_plus_visual` model for each search.
- The **student** sees only first-page fields: `Title`, `Brand`, `Size`, `Price`, and `Likes`.
- The student learns the teacher's continuous full+visual score.
- The student threshold is chosen via a configurable objective: **precision** (default) or **recall**.

## Objective

Both objectives train the same student regressor; only the threshold-selection rule differs.

- `precision` (default, 2026-05-16): on validation, pick the lowest threshold (= largest selected_count) whose `teacher_precision_among_selected >= target`. Fallback to the highest achievable precision row if no threshold meets the target. The cascade default target is `0.95`. This makes the first stage a tight filter that mostly forwards items the full+visual teacher will also approve.
- `recall`: on validation, pick the threshold giving the smallest selected_count whose `teacher_recall >= target`. The first stage acts as a recall filter and the expensive second stage makes the final precision decision.

## Why The Switch (2026-05-16)

The recall-trained first stage from `student_fullvisual_score_20260515_154011` kept ~94% of teacher-approved test items but sent many items into the expensive full-scrape stage that the teacher would later reject. The current goal of the cascade is per-window precision, and the disagreement analysis showed full+visual already does most of the heavy lifting, so the cheap stage was changed to also push for precision rather than recall.

The recall-trained run is preserved at `data/experiments/teacher_student_basic_filter/offline_runs/student_fullvisual_score_20260515_154011/` for comparison.

## Current CLI

```bash
# Precision objective (current default for the cascade)
python scripts/experiments/current/teacher_student_basic_filter/train_student.py --all-searches --objective precision --target 0.95

# Recall objective (the older mode, still supported)
python scripts/experiments/current/teacher_student_basic_filter/train_student.py --all-searches --objective recall --target 0.95
```

The `--objective` flag defaults to `recall` so older invocation lines keep working; the cascade however explicitly defaults to the precision-trained student. The trainer also writes a `target_tradeoff_by_search.csv` over `(0.85, 0.90, 0.95, 0.98)` for precision and `(0.90, 0.95, 0.98, 0.99)` for recall.

Outputs are written under:

```text
data/experiments/teacher_student_basic_filter/
```

Main output files:

- `metrics_long.csv`: every student/search/threshold/split metric.
- `best_student_by_search.csv`: best student per search at the default 95% teacher-recall target.
- `target_tradeoff_by_search.csv`: best student per search for 90%, 95%, 98%, and 99% teacher-recall targets.
- `test_scored_items.csv`: row-level test predictions for teacher, student, and old basic model.
- `teacher_student_report.md`: human-readable summary.

## How To Read Results

Important columns:

- `test_teacher_recall`: how many teacher-approved items the student keeps.
- `test_teacher_precision` (precision objective): share of student-kept items the teacher also approves.
- `test_selected_count`: how many items would move to the expensive second stage.
- `test_sold_precision`: offline sold-label precision among items kept by the student.
- `test_cascade_final_sold_precision`: precision after applying the teacher as the second stage.
- `baseline_test_teacher_recall` / `baseline_test_teacher_precision`: same metrics for the old basic_5 baseline filter.

Good behavior depends on the objective:
- **Precision objective** (default): high `test_teacher_precision` with the largest selected count that still clears the precision floor.
- **Recall objective**: high `test_teacher_recall` with the smallest selected count that still meets the recall floor.

## Decision Rule

**For the current cascade default (precision objective):** pick the largest student threshold that still keeps teacher precision above target on validation. If no threshold reaches the target, the trainer falls back to the highest-achievable precision row — be aware that those searches (currently `griffati_uomo_all` and `nike`) are effectively running at "best the student can do" rather than a real 95% guarantee.

**For recall objective:** prefer the lowest threshold that keeps enough teacher recall on validation. Suggested starting points: 90% if full scraping volume is too high, 95% if missing good items is more costly than doing extra full scrapes, avoid 98–99% unless full scraping capacity is not a problem.

The full+visual model remains the final high-precision filter regardless of the first-stage objective.
