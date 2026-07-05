# Legacy Clustering Approach

This folder contains the older product/variant clustering deal-ranking approach.

The implementation was moved here so the active scraper, deal-finder models, and
shared utilities are easier to distinguish from the clustering experiment.

## Contents

- `vinted_pipeline_batch.py`: batch product/variant clustering and deal scoring.
- `vinted_pipeline_incremental.py`: incremental assignment against an existing clustering index.
- `vinted_index_score.py`: SQLite-backed product/variant index storage.
- `clustering_products_from_csv.py`: older standalone clustering script.
- `tests/`: focused tests for this approach.

## Compatibility

These old entry points still exist as wrappers:

```bash
python3 scripts/workflow_runner.py batch --folder ps4 --autotune_variants
python3 scripts/workflow_runner.py incremental --folder ps4 --new_items new.csv
python3 scripts/clustering_products_from_csv.py --input input.csv --out_dir out
```

New code should import from:

```python
experiments.old.clustering_approach
```

instead of:

```python
analysis_pipeline.scoring.vinted_pipeline_batch
analysis_pipeline.scoring.vinted_pipeline_incremental
analysis_pipeline.scoring.vinted_index_score
```

