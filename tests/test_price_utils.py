import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

sys.modules.setdefault('openpyxl', types.SimpleNamespace(load_workbook=lambda *args, **kwargs: None))
selenium_exc = types.ModuleType('selenium.common.exceptions')
selenium_exc.WebDriverException = Exception
sys.modules.setdefault('selenium.common.exceptions', selenium_exc)
sys.modules.setdefault('Scraper', types.SimpleNamespace(Scraper=object))

from utils_lib.utils import extract_listing_price_text, parse_price_text
sys.modules.pop('Scraper', None)


class PriceUtilsTests(unittest.TestCase):
    def test_extract_listing_price_prefers_fee_included_amount_when_present(self):
        details = " brand: Nike, condizioni: Nuovo con cartellino, taglia: 45, €110.00, €116.20 include la Protezione acquisti"
        self.assertEqual(extract_listing_price_text(details), "116.20")

    def test_extract_listing_price_falls_back_to_single_listing_price(self):
        details = " brand: Nike, condizioni: Nuovo con cartellino, taglia: 45, €110.00"
        self.assertEqual(extract_listing_price_text(details), "110.00")

    def test_parse_price_text_handles_european_thousands_and_decimals(self):
        self.assertEqual(parse_price_text("1.234,30€"), 1234.30)

    def test_parse_price_text_handles_simple_comma_decimals(self):
        self.assertEqual(parse_price_text("273,70€"), 273.70)

    def test_parse_price_text_treats_single_dot_with_three_trailing_digits_as_thousands(self):
        self.assertEqual(parse_price_text("1.071"), 1071.0)


if __name__ == "__main__":
    unittest.main()
