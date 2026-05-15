# Teacher-Student Basic Filter

This experiment tests a cheaper first-stage filter for the deal pipeline.

The idea is:

- The **teacher** is the best `full_scrape_plus_visual` model for each search.
- The **student** sees only first-page fields: `Title`, `Brand`, `Size`, `Price`, and `Likes`.
- The student learns the teacher's continuous full+visual score.
- The student threshold is chosen to keep high recall of teacher-approved items.

This is different from the old basic model, which was tuned mostly for precision. Here the first stage should avoid rejecting items that the expensive second stage would like.

## Why This Exists

The disagreement analysis showed that many items approved by the full+visual model were rejected by the basic model. That means the first model should probably act like a broad recall filter, not the final decision-maker.

Teacher-student training is a clean way to do that:

- Train the strong full+visual model first.
- Score historical rows with the full+visual model.
- Train a cheap basic model to predict that score.
- Choose a threshold that keeps most teacher-approved rows.
- Let the full+visual model make the final high-precision decision.

## Current CLI

```bash
python scripts/experiments/teacher_student_basic_filter/train_student.py --all-searches
```

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
- `test_selected_count`: how many items would move to the expensive second stage.
- `test_sold_precision`: offline sold-label precision among items kept by the student.
- `test_cascade_final_sold_precision`: precision after applying the teacher as the second stage.
- `baseline_test_teacher_recall`: how many teacher-approved items the old first filter kept.

Good behavior for this stage is not maximum precision. Good behavior is high teacher recall with a manageable number of selected items.

## Decision Rule

For live use, prefer the lowest student threshold that keeps enough teacher recall on validation.

Suggested starting points:

- Use 90% teacher recall if full scraping volume is too high.
- Use 95% teacher recall if missing good items is more costly than doing extra full scrapes.
- Avoid 98-99% unless full scraping capacity is not a problem, because those settings usually keep most listings.

The full+visual model should remain the final high-precision filter.
