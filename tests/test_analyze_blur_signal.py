import unittest
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
ANALYSIS_DIR = SCRIPTS_DIR / "analysis_pipeline"
for path in (SCRIPTS_DIR, ANALYSIS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis_pipeline.evaluation.analyze_blur_signal import common_language_effect, cohens_d, summarize_signal


class AnalyzeBlurSignalTests(unittest.TestCase):
    def test_common_language_effect_is_one_when_all_sold_scores_higher(self):
        sold = np.asarray([10.0, 11.0, 12.0])
        unsold = np.asarray([1.0, 2.0, 3.0])
        self.assertEqual(common_language_effect(sold, unsold), 1.0)

    def test_cohens_d_is_negative_when_sold_scores_lower(self):
        sold = np.asarray([1.0, 2.0, 3.0])
        unsold = np.asarray([4.0, 5.0, 6.0])
        self.assertLess(cohens_d(sold, unsold), 0.0)

    def test_summarize_signal_reports_blurry_rate_difference(self):
        sold = np.asarray([10.0, 20.0, 30.0])
        unsold = np.asarray([40.0, 50.0, 60.0])
        summary = summarize_signal(
            sold,
            unsold,
            blur_threshold=25.0,
            permutations=100,
            bootstrap_samples=100,
            seed=0,
        )
        self.assertEqual(summary["sold_blurry_rate"], 2 / 3)
        self.assertEqual(summary["unsold_blurry_rate"], 0.0)
        self.assertLess(summary["mean_diff_sold_minus_unsold"], 0.0)


if __name__ == "__main__":
    unittest.main()
