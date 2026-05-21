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
    allowed_user_id: str | None = os.getenv('TELEGRAM_ALLOWED_USER_ID')
    image_bot_token: str | None = os.getenv('IMAGE_BOT_TOKEN')
    image_allowed_user_id: str | None = os.getenv('IMAGE_ALLOWED_USER_ID')

    # Accountability pipeline chat IDs
    accountability_enabled: bool = os.getenv('ACCOUNTABILITY_ENABLED', 'false').lower() in ('1', 'true', 'yes')
    recommended_deals_chat_id: str | None = os.getenv('RECOMMENDED_DEALS_CHAT_ID')
    bought_items_chat_id: str | None = os.getenv('BOUGHT_ITEMS_CHAT_ID')
    drafting_items_chat_id: str | None = os.getenv('DRAFTING_ITEMS_CHAT_ID')
    selling_items_chat_id: str | None = os.getenv('SELLING_ITEMS_CHAT_ID')
    resold_chat_id: str | None = os.getenv('RESOLD_CHAT_ID')
    tracking_chat_id: str | None = os.getenv('TRACKING_CHAT_ID')

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    @property
    def resolved_allowed_user_id(self) -> int | None:
        return self._resolve_user_id(self.allowed_user_id, self.chat_id)

    @property
    def resolved_image_allowed_user_id(self) -> int | None:
        # falls back to the main bot's allowed user / chat id
        return self._resolve_user_id(
            self.image_allowed_user_id or self.allowed_user_id,
            self.chat_id,
        )

    @property
    def resolved_recommended_deals_chat_id(self) -> int | None:
        return self._resolve_chat_id(self.recommended_deals_chat_id, self.chat_id)

    @property
    def resolved_bought_items_chat_id(self) -> int | None:
        return self._resolve_chat_id(self.bought_items_chat_id, self.chat_id)

    @property
    def resolved_drafting_items_chat_id(self) -> int | None:
        return self._resolve_chat_id(self.drafting_items_chat_id, self.chat_id)

    @property
    def resolved_selling_items_chat_id(self) -> int | None:
        return self._resolve_chat_id(self.selling_items_chat_id, self.chat_id)

    @property
    def resolved_resold_chat_id(self) -> int | None:
        return self._resolve_chat_id(self.resold_chat_id, self.chat_id)

    @property
    def resolved_tracking_chat_id(self) -> int | None:
        return self._resolve_chat_id(self.tracking_chat_id, self.chat_id)

    @staticmethod
    def _resolve_user_id(explicit_value: str | None, chat_id: str | None) -> int | None:
        explicit = (explicit_value or '').strip()
        if explicit:
            try:
                return int(explicit)
            except ValueError:
                return None

        chat = (chat_id or '').strip()
        if not chat:
            return None
        try:
            resolved = int(chat)
        except ValueError:
            return None
        if resolved > 0:
            return resolved
        return None

    @staticmethod
    def _resolve_chat_id(explicit_value: str | None, fallback_chat_id: str | None) -> int | None:
        explicit = (explicit_value or '').strip()
        if explicit:
            try:
                return int(explicit)
            except ValueError:
                return None
        fallback = (fallback_chat_id or '').strip()
        if fallback:
            try:
                return int(fallback)
            except ValueError:
                return None
        return None


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
        if require_proxy and not self.proxy.has_datacenter_proxy:
            result.errors.append(
                'Missing Bright Data datacenter proxy configuration. '
                'Set BRIGHTDATA_DATACENTER_USERNAME and BRIGHTDATA_DATACENTER_PASSWORD in .env, '
                'or set BRIGHTDATA_DATACENTER_PROXY_URL.'
            )
        if not self.telegram.is_configured:
            result.warnings.append('Telegram notifications are not fully configured; scraper notifications may be skipped.')
        if not os.getenv('HF_TOKEN'):
            result.warnings.append('HF_TOKEN is not set; Hugging Face downloads may be slower and more rate-limited.')

        return result


settings = Settings()
