# Eventual Sales

Eventual-sale checking answers this question:

Did an item that looked interesting or unsold at scrape time sell later?

This is useful because the best deals may disappear after some time. If a high-ranked item sells later, that can be evidence that the ranking was meaningful.

## Normal Meaning

| File | Meaning |
|---|---|
| `sold_df.csv` | Item was confirmed sold during normal scraping or priority recheck |
| `sold_eventually.csv` | Item was not initially sold, but sold later |
| `not_sold_yet.csv` | Item was checked later and still had not sold |

## No-Overlap Rule

The same item should not appear in both:

```text
sold_df.csv
eventual_sale_check/sold_eventually.csv
```

Why:

If an item is already confirmed sold in `sold_df.csv`, it is not an eventual sale. It is already a known sold item.

## Important Identifiers

The safest columns for overlap/duplicate checks are usually:

- `Dataid`
- `Link`

Duplicate-looking rows can still be legitimate if they have different `Dataid` values. For example, many sellers can list the same product title, brand, price, and size.

## Priority Rechecks

The background eventual-sale checker can prioritize certain items. If a priority recheck confirms that an item is sold, the fixed behavior is:

```text
write to sold_df.csv
do not duplicate into sold_eventually.csv
```

## Useful Commands

Run eventual-sale checking for one search:

```bash
python3 scripts/workflow_runner.py update-eventual-sales --folder ps4 --use_pipeline_out --min_deal_score 2.0 --min_deal_confidence 0.7 --require_deal_eligible --top_n 100 --allow_residential_fallback
```

Evaluate using eventual-sale labels:

```bash
python3 scripts/workflow_runner.py evaluate --folder ps4 --use_eventual_sales
```

## Recent Bug Rule

If eventual-sale checking seems stuck, inspect:

```text
data/simple_scrape/eventual_sale.log
```

A previous bug happened when `Price` had pandas string dtype and the code tried to write a float price into it. The code now casts mutable status/price columns to object before updating them.

