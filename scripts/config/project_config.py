from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


CONFIG_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = CONFIG_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_ENV_PATH = PROJECT_ROOT / '.env'
DEFAULT_TELEGRAM_ENV_PATH = SCRIPT_DIR / 'telegram_scripts' / 'bot_env.env'

load_dotenv(DEFAULT_ENV_PATH)
load_dotenv(DEFAULT_TELEGRAM_ENV_PATH)


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / 'data'
    simple_scrape_dir: Path = PROJECT_ROOT / 'data' / 'simple_scrape'
    full_scrape_dir: Path = PROJECT_ROOT / 'data' / 'full_scrape'
    models_dir: Path = PROJECT_ROOT / 'models'
    searches_yaml: Path = PROJECT_ROOT / 'data' / 'searches.yaml'
    brand_ids_csv: Path = PROJECT_ROOT / 'data' / 'brand_ids.csv'
    telegram_env_file: Path = DEFAULT_TELEGRAM_ENV_PATH

    def ensure_runtime_dirs(self) -> None:
        for path in (self.data_dir, self.simple_scrape_dir, self.full_scrape_dir, self.models_dir):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str | None = os.getenv('BOT_TOKEN')
    chat_id: str | None = os.getenv('CHAT_ID')

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)


@dataclass(frozen=True)
class ProxyConfig:
    api_token: str | None = os.getenv('API_TOKEN')
    scraping_browser_auth: str | None = os.getenv('BRIGHTDATA_SCRAPING_BROWSER_AUTH')
    scraping_browser_host: str = os.getenv('BRIGHTDATA_SCRAPING_BROWSER_HOST', 'zproxy.lum-superproxy.io:9515')
    datacenter_proxy_url_override: str | None = os.getenv('BRIGHTDATA_DATACENTER_PROXY_URL')
    datacenter_host: str = os.getenv('BRIGHTDATA_DATACENTER_HOST', 'brd.superproxy.io')
    datacenter_port: int = int(os.getenv('BRIGHTDATA_DATACENTER_PORT', '33335'))
    datacenter_username: str | None = os.getenv('BRIGHTDATA_DATACENTER_USERNAME')
    datacenter_password: str | None = os.getenv('BRIGHTDATA_DATACENTER_PASSWORD')
    residential_host: str = os.getenv('BRIGHTDATA_RESIDENTIAL_HOST', 'brd.superproxy.io')
    residential_port: int = int(os.getenv('BRIGHTDATA_RESIDENTIAL_PORT', '33335'))
    residential_username: str | None = os.getenv('BRIGHTDATA_RESIDENTIAL_USERNAME')
    residential_password: str | None = os.getenv('BRIGHTDATA_RESIDENTIAL_PASSWORD')
    web_unlocker_proxy: str | None = os.getenv('BRIGHTDATA_WEB_UNLOCKER_PROXY')

    @property
    def scraping_browser_url(self) -> str | None:
        if not self.scraping_browser_auth:
            return None
        return f'https://{self.scraping_browser_auth}@{self.scraping_browser_host}'

    @property
    def datacenter_proxy_url(self) -> str | None:
        if self.datacenter_proxy_url_override:
            return self.datacenter_proxy_url_override
        if not self.datacenter_username or not self.datacenter_password:
            return None
        return (
            f'http://{self.datacenter_username}:{self.datacenter_password}'
            f'@{self.datacenter_host}:{self.datacenter_port}'
        )

    @property
    def residential_proxy_url(self) -> str | None:
        if not self.residential_username or not self.residential_password:
            return None
        return (
            f'http://{self.residential_username}:{self.residential_password}'
            f'@{self.residential_host}:{self.residential_port}'
        )

    @property
    def has_residential_proxy(self) -> bool:
        return bool(self.residential_proxy_url)

    @property
    def has_datacenter_proxy(self) -> bool:
        return bool(self.datacenter_proxy_url)

    @property
    def has_scraping_browser(self) -> bool:
        return bool(self.scraping_browser_url)


@dataclass(frozen=True)
class LoggingConfig:
    app_level: str = os.getenv('VINTED_LOG_LEVEL', 'INFO').upper()
    third_party_level: str = os.getenv('VINTED_THIRD_PARTY_LOG_LEVEL', 'WARNING').upper()


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def log(self, logger: logging.Logger) -> None:
        for warning in self.warnings:
            logger.warning(warning)
        for error in self.errors:
            logger.error(error)


@dataclass(frozen=True)
class Settings:
    paths: PathsConfig = PathsConfig()
    telegram: TelegramConfig = TelegramConfig()
    proxy: ProxyConfig = ProxyConfig()
    logging: LoggingConfig = LoggingConfig()

    def ensure_runtime_dirs(self) -> None:
        self.paths.ensure_runtime_dirs()

    def validate_for_simple_scrape(self, programmed_searches: Iterable[object], require_proxy: bool = True) -> ValidationResult:
        result = ValidationResult()
        searches = list(programmed_searches)

        if not self.paths.searches_yaml.exists():
            result.errors.append(f'Missing searches YAML file: {self.paths.searches_yaml}')
        if not self.paths.brand_ids_csv.exists():
            result.warnings.append(f'Brand IDs CSV not found yet: {self.paths.brand_ids_csv}')
        if not searches:
            result.errors.append('No enabled searches were loaded. Check data/searches.yaml.')
        if require_proxy and not self.proxy.has_residential_proxy:
            result.errors.append(
                'Missing Bright Data residential proxy configuration. '
                'Set BRIGHTDATA_RESIDENTIAL_USERNAME and BRIGHTDATA_RESIDENTIAL_PASSWORD in .env.'
            )
        if not self.telegram.is_configured:
            result.warnings.append('Telegram notifications are not fully configured; scraper notifications may be skipped.')
        if not os.getenv('HF_TOKEN'):
            result.warnings.append('HF_TOKEN is not set; Hugging Face downloads may be slower and more rate-limited.')

        return result


settings = Settings()
