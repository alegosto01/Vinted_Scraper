# Data Files

Most project data is organized per search.

Example:

```text
data/simple_scrape/ps4/
```

## Core CSV Files

| File | Meaning |
|---|---|
| `big_raw.csv` | Main scraped listings for a search |
| `old_df.csv` | Listings already known from previous scraper runs |
| `sold_df.csv` | Items confirmed sold during normal scraping or priority recheck |
| `unsold_df.csv` | Items seen as still available/unsold |
| `pipeline_out/deals_ranked.csv` | Listings ranked by the legacy clustering deal-scoring pipeline |
| `deal_score_eval/evaluation_report.json` | Evaluation report for deal score performance |
| `final_buy_filter/buy_candidates_enriched.csv` | Shortlisted candidates after full item/seller scraping |
| `full_scrape/items_enriched.csv` | Full item/seller rows collected from score extremes, sold backfill, or newly confirmed sold items |
| `full_scrape/full_scrape_failures.csv` | Item pages that could not be loaded during full item/seller collection |
| `full_scrape/full_scrape_events.csv` | Audit trail of full item/seller collection attempts |
| `buy_eval/` | Evaluation outputs for buy-decision performance |

## Eventual-Sale Files

These live under:

```text
data/simple_scrape/<search_name>/eventual_sale_check/
```

| File | Meaning |
|---|---|
| `big_raw_eventual_sale_labeled.csv` | Eventual-sale labeled dataset |
| `sold_eventually.csv` | Items that were not initially sold but sold later |
| `not_sold_yet.csv` | Items checked later and still not sold |
| `priority_check_queue.csv` | Items waiting for priority status recheck |

Important rule:

An item should not be in both `sold_df.csv` and `sold_eventually.csv`.

## Image Cache

Listing images are cached under:

```text
data/simple_scrape/<search_name>/image_cache/
```

The current cache mostly stores `.webp` images. These are local image files on disk. When a model opens an image, it decodes the compressed file into raw pixels in RAM.

## Full Scrape Seller Data

The newer final-buy-filter flow stores seller/item details inline in:

```text
data/simple_scrape/<search_name>/final_buy_filter/buy_candidates_enriched.csv
```

Important enriched columns include:

- `Description`
- `Condition`
- `Upload_date`
- `Upload_date_days`
- `Interested_count`
- `View_count`
- `SellerName`
- `SellerId`
- `Location`
- `ReviewsCount`
- `Stars`

Normal live scraping can also write reusable full item/seller data under:

```text
data/simple_scrape/<search_name>/full_scrape/
```

This folder is used for:

- `score_high`: new live rows with `DealFinderScore >= 0.95`
- `score_low`: new live rows with `DealFinderScore <= 0.05`
- `sold_backfill`: historical rows from per-search `sold_df.csv`
- `sold_confirmed_live`: rows newly confirmed sold by the priority status checker

Additional image-count columns include:

- `PrimaryImageUrl`
- `FullImageUrls`
- `VisiblePictureCount`
- `HiddenPictureCount`
- `PictureCount`

`HiddenPictureCount` is parsed from carousel overlay text such as `+4`, including newer item-page structures where the overlay appears inside a `figure > button > div` photo tile. `PictureCount` is the best available total from extracted image URLs, visible photo tiles, and hidden overlay count.

Older full-scrape data may also exist under:

```text
data/full_scrape/
```

Legacy files include:

- `old_df.csv`
- `sold_df.csv`
- `unsold_df.csv`
- `sellers_df.csv`
