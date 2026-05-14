# Troubleshooting

This file collects practical checks for common project problems.

## Eventual Sales Did Not Save

First check whether the process is still running:

```bash
ps -ef | rg 'scripts/main.py|workflow_runner.py main|daily_eventual_sales|python'
```

Then inspect the log:

```bash
sed -n '1,200p' data/simple_scrape/eventual_sale.log
```

Likely places to check:

- `data/simple_scrape/<search>/eventual_sale_check/sold_eventually.csv`
- `data/simple_scrape/<search>/eventual_sale_check/not_sold_yet.csv`
- `data/simple_scrape/<search>/eventual_sale_check/priority_check_queue.csv`
- `data/simple_scrape/<search>/sold_df.csv`

Important:

Priority confirmed-sold rows should go to `sold_df.csv`, not `sold_eventually.csv`.

## Duplicate Rows In sold_df.csv

Not every repeated title is a duplicate. Many rows can have the same title, brand, price, and size while still being different listings.

The most important duplicate keys are:

- `Dataid`
- `Link`

If `Dataid` and `Link` are different, the rows may be legitimate different listings.

## Overlap Between sold_df.csv And sold_eventually.csv

This should not happen.

Bad overlap:

```text
data/simple_scrape/<search>/sold_df.csv
data/simple_scrape/<search>/eventual_sale_check/sold_eventually.csv
```

Expected behavior:

- Known sold items stay in `sold_df.csv`.
- Later-sold items go into `sold_eventually.csv`.
- The same `Dataid` should not be in both.

## Full Seller Data Missing

Full seller/item scraping usually happens during:

```bash
python3 scripts/workflow_runner.py final-buy-filter --folder ps4 --require_deal_eligible
```

Check:

```text
data/simple_scrape/ps4/final_buy_filter/buy_candidates_enriched.csv
```

Important seller/item columns:

- `SellerName`
- `SellerId`
- `Location`
- `ReviewsCount`
- `Stars`
- `Description`
- `Condition`
- `Interested_count`
- `View_count`

## Tailscale Says Logged Out

If:

```bash
tailscale status
```

returns:

```text
Logged out.
```

then run:

```bash
sudo tailscale up --ssh
```

Open the login URL and approve the machine.

Then verify:

```bash
tailscale status
tailscale ip -4
```

## Useful Test Commands

Run the tests touched by scraper/eventual-sale fixes:

```bash
python3 -m unittest tests.test_daily_eventual_sales tests.test_update_eventual_sales tests.test_sold_csv_recheck tests.test_simple_scraper
```

Run Python compile checks:

```bash
python3 -m py_compile scripts/simple_scraper.py scripts/scraping_options.py scripts/daily_eventual_sales.py scripts/workflow_runner.py
```

