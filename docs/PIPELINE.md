# Pipeline

This is the practical order for running the project.

## 1. Run The Scraper

```bash
python3 scripts/workflow_runner.py main
```

Alternative long-running entry point:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/main.py
```

This updates the per-search files under:

```text
data/simple_scrape/<search_name>/
```

Important outputs:

- `big_raw.csv`
- `old_df.csv`
- `sold_df.csv`
- `unsold_df.csv`

## 2. Rank Deals With The Legacy Clustering Approach

This is the older product/variant clustering pipeline. It is kept for reference
and reproducibility under `experiments/clustering_approach/`.

```bash
python3 scripts/workflow_runner.py batch --folder ps4 --autotune_variants
```

Replace `ps4` with the search folder you want.

Important output:

```text
data/simple_scrape/ps4/pipeline_out/deals_ranked.csv
```

## 3. Check Eventual Sales

This checks whether previously scraped/ranked items sold later.

```bash
python3 scripts/workflow_runner.py update-eventual-sales --folder ps4 --use_pipeline_out --min_deal_score 2.0 --min_deal_confidence 0.7 --require_deal_eligible --top_n 100 --allow_residential_fallback
```

Important outputs:

- `eventual_sale_check/big_raw_eventual_sale_labeled.csv`
- `eventual_sale_check/sold_eventually.csv`
- `eventual_sale_check/not_sold_yet.csv`
- `eventual_sale_check/priority_check_queue.csv`

Important rule:

Rows already in `sold_df.csv` should not also appear in `eventual_sale_check/sold_eventually.csv`.

## 4. Evaluate Deal Ranking

```bash
python3 scripts/workflow_runner.py evaluate --folder ps4 --use_eventual_sales
```

Important output:

```text
data/simple_scrape/ps4/deal_score_eval/
```

This tells you whether high deal scores were actually connected with later sale outcomes.

## 5. Run Final Buy Filter

```bash
python3 scripts/workflow_runner.py final-buy-filter --folder ps4 --require_deal_eligible
```

This is where full item/seller scraping happens for shortlisted candidates.

Important outputs:

- `final_buy_filter/buy_candidates_input.csv`
- `final_buy_filter/buy_candidates_enriched.csv`
- `final_buy_filter/buy_candidates_recommended.csv`

## 6. Evaluate Buy Decisions

```bash
python3 scripts/workflow_runner.py evaluate-buy-decisions --folder ps4 --use_eventual_sales
```

Important output:

```text
data/simple_scrape/ps4/buy_eval/
```

## Typical Search Folders

Examples currently used in this project:

- `ps4`
- `prada`
- `gucci`
- `griffati_donna_all`
- `griffati_uomo_all`
- `Borse_Griffate`
- `Scarpe_Griffate`
- `nike`
