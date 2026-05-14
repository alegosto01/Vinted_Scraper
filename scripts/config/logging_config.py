from __future__ import annotations

from contextlib import contextmanager
import logging
import sys
import threading
from pathlib import Path

from config.project_config import settings


ANSI_RESET = "[0m"
ANSI_BOLD = "[1m"
ANSI_DIM = "[2m"
LOGGER_COLOR = "[94m"
LEVEL_STYLES = {
    logging.DEBUG: ("[36m", "[DEBUG]", "..."),
    logging.INFO: ("[32m", "[INFO ]", "==>"),
    logging.WARNING: ("[33m", "[WARN ]", "!!!"),
    logging.ERROR: ("[31m", "[ERROR]", "XXX"),
    logging.CRITICAL: ("[35m", "[CRIT ]", "###"),
}
THIRD_PARTY_LOGGERS = (
    'httpx',
    'httpcore',
    'huggingface_hub',
    'sentence_transformers',
    'urllib3',
    'requests_html',
    'selenium',
    'transformers',
)
EVENTUAL_SALES_LOG_FILENAME = "eventual_sale.log"
_EVENTUAL_SALES_HANDLER_NAME = "eventual-sales-file-handler"
_EVENTUAL_SALES_CONTEXT = threading.local()


def _is_eventual_sales_context() -> bool:
    return bool(getattr(_EVENTUAL_SALES_CONTEXT, "enabled", False))


class SkipEventualSalesFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _is_eventual_sales_context()


class OnlyEventualSalesFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _is_eventual_sales_context()


def _eventual_sales_log_path() -> Path:
    path = Path(str(settings.paths.simple_scrape_dir)) / EVENTUAL_SALES_LOG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_eventual_sales_file_logging(root_logger: logging.Logger | None = None) -> logging.Handler:
    root_logger = root_logger or logging.getLogger()
    for handler in root_logger.handlers:
        if getattr(handler, "name", "") == _EVENTUAL_SALES_HANDLER_NAME:
            return handler

    handler = logging.FileHandler(_eventual_sales_log_path(), encoding="utf-8")
    handler.name = _EVENTUAL_SALES_HANDLER_NAME
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    handler.addFilter(OnlyEventualSalesFilter())
    root_logger.addHandler(handler)
    return handler


@contextmanager
def eventual_sales_log_context():
    ensure_eventual_sales_file_logging()
    previous = _is_eventual_sales_context()
    _EVENTUAL_SALES_CONTEXT.enabled = True
    try:
        yield
    finally:
        _EVENTUAL_SALES_CONTEXT.enabled = previous


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base_message = super().format(record)
        if not sys.stderr.isatty():
            return base_message

        color, level_label, marker = LEVEL_STYLES.get(
            record.levelno,
            ("", f"[{record.levelname.upper():<5}]", ">>>"),
        )
        timestamp = self.formatTime(record, self.datefmt)
        logger_name = f"{LOGGER_COLOR}{ANSI_BOLD}{record.name}{ANSI_RESET}"
        level_text = f"{color}{ANSI_BOLD}{level_label}{ANSI_RESET}"
        marker_text = f"{color}{ANSI_BOLD}{marker}{ANSI_RESET}"
        time_text = f"{ANSI_DIM}{timestamp}{ANSI_RESET}"
        return f"{time_text} {marker_text} {level_text} {logger_name} {base_message}"


def _coerce_level(value: str) -> int:
    candidate = getattr(logging, str(value).upper(), None)
    return candidate if isinstance(candidate, int) else logging.INFO


def configure_logging(force: bool = False) -> logging.Logger:
    root_logger = logging.getLogger()
    if force:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

    root_logger.setLevel(_coerce_level(settings.logging.app_level))

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter('%(message)s', datefmt='%H:%M:%S'))
        handler.addFilter(SkipEventualSalesFilter())
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(ColorFormatter('%(message)s', datefmt='%H:%M:%S'))
                has_skip_filter = any(isinstance(filter_, SkipEventualSalesFilter) for filter_ in handler.filters)
                if not has_skip_filter:
                    handler.addFilter(SkipEventualSalesFilter())

    ensure_eventual_sales_file_logging(root_logger)

    third_party_level = _coerce_level(settings.logging.third_party_level)
    for logger_name in THIRD_PARTY_LOGGERS:
        logging.getLogger(logger_name).setLevel(third_party_level)

    return root_logger
