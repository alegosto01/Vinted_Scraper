import logging

from config.logging_config import configure_logging
from config.project_config import settings
from config.search_loader import load_searches
from daily_eventual_sales import start_eventual_sales_background_worker


LOGGER = logging.getLogger(__name__)


def main() -> int:
    configure_logging()
    settings.ensure_runtime_dirs()

    searches = load_searches(str(settings.paths.searches_yaml))
    programmed_searches = [search for search in searches.values() if search.enabled]

    from scraping_options import preflight_parallel_scrape

    validation = preflight_parallel_scrape(programmed_searches, mode='collect', app_settings=settings)
    validation.log(LOGGER)
    if not validation.ok:
        return 1

    workers, _stop_event = start_eventual_sales_background_worker(programmed_searches)
    LOGGER.info(
        "Started eventual-sale background workers threads=%s",
        ",".join(worker.name for worker in workers),
    )

    from scraping_options import scrape_specific_items_parallel

    scrape_specific_items_parallel(
        programmed_searches,
        pages_to_scrape=10,
        delay_between_batch_of_searches=3600,
        max_search_counts=10000,
        search_workers=6,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
