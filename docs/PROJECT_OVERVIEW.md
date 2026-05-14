# Vinted New Version

This project scrapes Vinted searches, tracks item status over time, ranks potential deals, evaluates whether the ranking worked, and optionally performs a deeper final scrape for buy candidates.

The project currently stores most per-search data under:

```text
data/simple_scrape/<search_name>/
```

Example:

```text
data/simple_scrape/ps4/
data/simple_scrape/prada/
data/simple_scrape/gucci/
data/simple_scrape/griffati_donna_all/
```

## Main Ideas

- The normal scraper collects listings and updates market status files.
- The ranking pipeline creates deal scores from scraped listings.
- The eventual-sale checker revisits promising items later to see whether they sold.
- The evaluator compares deal scores against later sale outcomes.
- The final buy filter performs deeper item/seller scraping on shortlisted candidates.

## Important Scripts

| Script | Purpose |
|---|---|
| `scripts/main.py` | Long-running scraper entry point |
| `scripts/workflow_runner.py` | Main command wrapper for pipeline steps |
| `scripts/simple_scraper.py` | Core simple scraping and CSV update logic |
| `scripts/scraping_options.py` | Item recheck/eventual-sale helper logic |
| `scripts/daily_eventual_sales.py` | Background eventual-sale checker |
| `experiments/clustering_approach/vinted_pipeline_batch.py` | Legacy clustering-based deal ranking pipeline |
| `scripts/analysis_pipeline/evaluation/update_eventual_sales.py` | Eventual-sale labeling CLI |
| `scripts/analysis_pipeline/scoring/final_buy_filter.py` | Final deeper buy-candidate filtering |

## Recommended Working Style

Keep project memory in Markdown files inside `docs/`. When you discover a bug, pipeline rule, command, or data meaning, write it down here so the project becomes easier to resume later.

Useful files:

- `docs/PIPELINE.md`
- `docs/DATA_FILES.md`
- `docs/EVENTUAL_SALES.md`
- `docs/REMOTE_ACCESS.md`
- `docs/TROUBLESHOOTING.md`
- `docs/CODEX_WORKFLOW.md`
- `docs/IDEAS.md`

Legacy clustering code now lives under `experiments/clustering_approach/`.
