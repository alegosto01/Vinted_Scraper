import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / 'scripts'
ANALYSIS_DIR = SCRIPTS_DIR / 'analysis_pipeline'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from analysis_pipeline.evaluation.evaluate_buy_decisions import add_sold_labels, build_buy_report, ensure_buy_columns


class EvaluateBuyDecisionsTests(unittest.TestCase):
    def test_build_buy_report_computes_precision_and_profit_metrics(self):
        buys = pd.DataFrame([
            {'Dataid': '1', 'WorthBuying': True, 'BuyDecisionScore': 0.9, 'ExpectedProfit': 20, 'Price': 100},
            {'Dataid': '2', 'WorthBuying': True, 'BuyDecisionScore': 0.8, 'ExpectedProfit': 15, 'Price': 50},
            {'Dataid': '3', 'WorthBuying': False, 'BuyDecisionScore': 0.2, 'ExpectedProfit': 8, 'Price': 40},
            {'Dataid': '4', 'WorthBuying': False, 'BuyDecisionScore': 0.1, 'ExpectedProfit': 5, 'Price': 30},
        ])
        sold = pd.DataFrame([{'Dataid': '1'}, {'Dataid': '3'}])
        sold_eventually = pd.DataFrame([{'Dataid': '4'}])

        labeled = add_sold_labels(buys, sold, sold_eventually, 'Dataid', no_dedupe=False)
        labeled = ensure_buy_columns(labeled, 'WorthBuying', 'BuyDecisionScore', 0.62)
        report, summary_by_flag, confusion = build_buy_report(labeled, 'WorthBuying', 'ExpectedProfit', 'Price')

        self.assertEqual(report['true_positives'], 1)
        self.assertEqual(report['false_positives'], 1)
        self.assertEqual(report['false_negatives'], 2)
        self.assertAlmostEqual(report['precision_buy'], 0.5)
        self.assertAlmostEqual(report['recall_buy'], 1.0 / 3.0)
        self.assertEqual(report['selected_expected_profit_sum'], 35.0)
        self.assertEqual(report['selected_expected_profit_sum_sold_only'], 20.0)
        self.assertEqual(report['selected_capital_required'], 150.0)
        self.assertEqual(int(summary_by_flag.loc[summary_by_flag['Decision'] == 'buy', 'Count'].iloc[0]), 2)
        self.assertEqual(int(confusion.loc[confusion['Outcome'] == 'true_positive_buy_and_sold', 'Count'].iloc[0]), 1)


if __name__ == '__main__':
    unittest.main()
