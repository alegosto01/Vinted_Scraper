import time
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import Remote, ChromeOptions
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection
from config.project_config import settings
from utils_lib.retry_utils import sleep_with_backoff

SBR_WEBDRIVER = settings.proxy.scraping_browser_url
def make_driver():
    chrome_options = Options()
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
        "profile.managed_default_content_settings.fonts": 2,
        "profile.managed_default_content_settings.media_stream": 2,
        "profile.default_content_setting_values.notifications": 2,
        "profile.default_content_setting_values.popups": 2,
        "profile.default_content_setting_values.geolocation": 2,
        "profile.managed_default_content_settings.javascript": 1,
    }
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--window-size=375,667")

    for attempt in range(3):
        try:
            if not SBR_WEBDRIVER:
                raise RuntimeError("Missing Bright Data scraping browser configuration")
            print('Connecting to Scraping Browser...')
            sbr_connection = ChromiumRemoteConnection(SBR_WEBDRIVER, 'goog', 'chrome')
            driver = Remote(sbr_connection, options=ChromeOptions())
            driver.set_page_load_timeout(120)
            print("Connected successfully")
            return driver
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            sleep_with_backoff(attempt + 1, base_delay=15)
    print("Failed to load the page after multiple attempts.")
    return None
