from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from experiments.old.teacher_student_basic_filter.train_student import choose_threshold_for_teacher_recall


class TeacherStudentThresholdTests(unittest.TestCase):
    def test_choose_highest_threshold_that_hits_teacher_recall(self):
        scores = np.array([0.95, 0.80, 0.60, 0.40, 0.20])
        teacher_pass = np.array([True, True, True, False, False])

        chosen = choose_threshold_for_teacher_recall(scores, teacher_pass, target_recall=2 / 3)

        self.assertEqual(chosen["threshold"], 0.8)
        self.assertEqual(chosen["selected_count"], 2)
        self.assertAlmostEqual(chosen["teacher_recall"], 2 / 3)

    def test_falls_back_to_best_available_recall(self):
        scores = np.array([0.90, 0.70, 0.50])
        teacher_pass = np.array([False, False, True])

        chosen = choose_threshold_for_teacher_recall(scores, teacher_pass, target_recall=1.0)

        self.assertEqual(chosen["threshold"], 0.5)
        self.assertEqual(chosen["selected_count"], 3)
        self.assertEqual(chosen["teacher_recall"], 1.0)

    def test_handles_no_teacher_pass_rows(self):
        scores = np.array([0.90, 0.70, 0.50])
        teacher_pass = np.array([False, False, False])

        chosen = choose_threshold_for_teacher_recall(scores, teacher_pass, target_recall=0.95)

        self.assertTrue(np.isnan(chosen["teacher_recall"]))
        self.assertEqual(chosen["teacher_selected_count"], 0)


if __name__ == "__main__":
    unittest.main()
