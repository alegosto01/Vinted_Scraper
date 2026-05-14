import sys
import unittest
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from repair_big_raw_prices import decide_price_repair, recover_separator_collapsed_price


class RepairBigRawPricesTests(unittest.TestCase):
    def test_separator_recovery_examples(self):
        self.assertEqual(recover_separator_collapsed_price('1.9852'), Decimal('1985.20'))
        self.assertEqual(recover_separator_collapsed_price('3.67465'), Decimal('3674.65'))
        self.assertIsNone(recover_separator_collapsed_price('578.2'))

    def test_decision_prefers_fee_inversion_for_protection_total(self):
        decision = decide_price_repair('37.450')
        self.assertEqual(decision.final_price, Decimal('35.00'))
        self.assertEqual(decision.reason, 'fee_inverted_from_normalized')

    def test_decision_recovers_separator_corruption(self):
        decision = decide_price_repair('1.071')
        self.assertEqual(decision.final_price, Decimal('1071.00'))
        self.assertEqual(decision.reason, 'separator_recovered')

    def test_decision_can_recover_separator_then_fee(self):
        decision = decide_price_repair('1.9852')
        self.assertEqual(decision.final_price, Decimal('1890.00'))
        self.assertEqual(decision.reason, 'separator_recovered_then_fee_inverted')

    def test_short_decimal_is_only_normalized(self):
        decision = decide_price_repair('578.2')
        self.assertEqual(decision.final_price, Decimal('578.20'))
        self.assertEqual(decision.reason, 'normalize_only')


if __name__ == '__main__':
    unittest.main()
