> ⚠️ **Caveman-compressed** — terse/fragment style to save tokens. Technical substance, code, commands, URLs kept verbatim. Original backed up under `~/.local/share/caveman-compress/backups/`.

# Cascade Results Reporting

Doc define preferred way show live cascade results. Use for
`benchmark_basic_to_full` cascade + any future equivalent cascade
runner.

## Goal

Report answer these clear:

- How many **unique** items pass model 1?
- How many **unique** items pass model 2?
- How many unique model-2 pass items sold?
- How many unique model-2 reject items sold?
- Thresholds both models per search?
- Precision after 1h, 2h, 3h, 6h, 12h, 18h, 24h, later windows?
- Which elapsed-hour buckets sold items actual sell?

## Core Rule

Always report model perf on **unique items**, not repeat scrape events.

Unique item key:

```text
(SearchName, item_id)
```

Use `tracking_key` if present. If missing, rebuild from:

```text
SearchName.lower() + "::" + item_id
```

Repeat hourly page-1 snapshots good for ops monitoring, but
must not inflate model perf counts.

## Time Window Rule

Use **matured cohorts** for precision windows.

For `N` hour row, denominator =:

```text
unique model-2 pass items that are at least N hours old
```

Numerator =:

```text
those denominator items that sold within N hours of first_stage1_pass_at
```

Mean denominators shrink at later hours. Expected:
new items excluded til old enough for checkpoint.

No count young unsold items as fails for windows not reach
yet.

## Required Tables

### Unique Funnel

One row per search:

| Column | Meaning |
| --- | --- |
| `Search` | Search folder/name. |
| `S1 thr` | Effective stage-1 threshold used in live run. |
| `S2 thr` | Effective stage-2 threshold used in live run. |
| `S1 pass` | Unique items pass model 1. |
| `S2 pass` | Unique items pass model 2. |
| `S2 sold` | Unique model-2 pass items later sold. |
| `Reject sold` | Unique stage-1 pass/model-2 reject items later sold. |

Also include total row across searches.

Preferred Markdown shape:

```text
| Search | S1 thr | S2 thr | S1 pass | S2 pass | S2 sold | Reject sold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gucci` | 0.4569 | 0.8213 | 71 | 66 | 18 | 1 |
| **Total** |  |  | 338 | 263 | 69 | 19 |
```

### Overall Matured Precision

Show final-model precision across all searches:

| Column | Meaning |
| --- | --- |
| `Hour` | Elapsed hour checkpoint. |
| `Sold` | Unique model-2 pass items sold within hour window. |
| `Matured items` | Unique model-2 pass items old enough for window. |
| `Precision` | `Sold / Matured items`. |

Preferred checkpoints:

```text
1h, 2h, 3h, 6h, 12h, 18h, 24h, 27h, 30h
```

Add later checkpoints only when matured items exist.

### Per-Search Checkpoints

Compact cells for final-model precision per search:

```text
sold/matured (precision%)
```

Preferred columns:

```text
1h, 6h, 12h, 24h, 30h
```

Example:

```text
| Search | 1h | 6h | 12h | 24h | 30h |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ps4` | 1/73 | 10/72 | 15/49 | 7/15 | 6/13 |
```

### Sold Hour Buckets

Show when sold items sold after first tracking:

| Column | Meaning |
| --- | --- |
| `Search` | Search folder/name. |
| `Hour bucket` | Elapsed bucket, e.g. `1-2h`. |
| `Sold in hour` | Unique final-model pass items sold in bucket. |

Table answer "how many got sold each hour".

## Required CSV Outputs

When gen files, save under run's `reports/` folder:

```text
cascade_unique_latest_summary.csv
cascade_unique_matured_precision_per_search_hour.csv
cascade_unique_sold_elapsed_hour_buckets.csv
cascade_unique_overall_matured_precision.csv
cascade_unique_latest_report.md
```

## Data Sources

Use:

```text
<run_dir>/tracked_items.csv
<run_dir>/cascade_plan.csv
```

Optional ops event counts can use:

```text
<run_dir>/events.jsonl
<run_dir>/reports/cascade_collection_by_search_hour.csv
```

No use event counts for perf tables.

## Implementation Notes

Normalize booleans w/ values like:

```text
true, 1, 1.0, yes
```

Important timestamps:

| Field | Use |
| --- | --- |
| `first_stage1_pass_at` | Start time for elapsed-hour windows. |
| `sold_at` | Sold detect time. |
| `last_seen_at` | Useful for dedup to newest state. |

Important model fields:

| Field | Use |
| --- | --- |
| `Stage1Threshold` | Effective threshold actual stored on tracked items. |
| `Stage2Threshold` | Effective threshold actual stored on tracked items. |
| `Stage2Passed` | Final model pass flag. |
| `Stage2Score` | Whether stage 2 scored item. |

If thresholds missing from tracked rows, fall back to `cascade_plan.csv`.
If stage 1 had configured offset in runner, report effective threshold
that appears on `tracked_items.csv`.

## Minimal Reproducible Algorithm

1. Read `tracked_items.csv`.
2. Build or use `tracking_key`.
3. Sort by `last_seen_at` + keep last row per `tracking_key`.
4. Parse `Stage2Passed`, `Stage2Score`, `first_stage1_pass_at`, `sold_at`.
5. For funnel:
   - `S1 pass` = count unique rows.
   - `S2 pass` = count unique rows w/ `Stage2Passed == true`.
   - `S2 sold` = count unique rows w/ `Stage2Passed == true` + `sold_at`.
   - `Reject sold` = count unique rows w/ `Stage2Passed != true` + `sold_at`.
6. For each hour checkpoint:
   - Filter to final-model pass rows.
   - Denominator = rows where `now - first_stage1_pass_at >= checkpoint`.
   - Numerator = denom rows where `sold_at - first_stage1_pass_at <= checkpoint`.
7. For sold-hour buckets:
   - Compute `sold_elapsed_hours = sold_at - first_stage1_pass_at`.
   - Bucket into `0-1h`, `1-2h`, `2-3h`, etc.
8. Print three Markdown tables + write CSV outputs.

## Interpretation

When explaining report, use these phrases:

- "Unique" = dedup by `(SearchName, item_id)`.
- "Matured items" = items old enough for hour checkpoint.
- Denominators shrink at later hours cuz fewer items aged into those
  windows.
- Row can include items first seen across many different hourly snapshots, but
  each item count once.