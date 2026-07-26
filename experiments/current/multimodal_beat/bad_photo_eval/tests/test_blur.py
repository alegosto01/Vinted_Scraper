"""Tests for the texture-normalized blur_score and blur-batch determinism."""
from __future__ import annotations
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import blur_score


class BlurScore(unittest.TestCase):
    def _tbl(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "laplacian_variance": rng.random(n) * 0.01 + 1e-4,
            "edge_density": rng.random(n) * 0.2 + 0.01,
            "dynamic_range": rng.random(n) * 0.5 + 0.3})

    def test_requires_columns(self):
        with self.assertRaises(ValueError):
            blur_score.add_blur_score(pd.DataFrame({"laplacian_variance": [1.0]}))

    def test_lower_sharpness_scores_blurrier(self):
        # hold texture fixed, vary only laplacian: lower lap -> higher blur_score
        t = pd.DataFrame({"laplacian_variance": [0.02, 0.002, 0.0002],
                          "edge_density": [0.1, 0.1, 0.1],
                          "dynamic_range": [0.5, 0.5, 0.5]})
        s = blur_score.add_blur_score(t)["blur_score"].to_numpy()
        self.assertTrue(s[0] < s[1] < s[2])          # monotonically blurrier

    def test_feature_error_is_nan(self):
        t = self._tbl(5)
        t["feature_error"] = [0, 1, 0, 0, 1]
        s = blur_score.add_blur_score(t)["blur_score"]
        self.assertTrue(np.isnan(s.iloc[1]) and np.isnan(s.iloc[4]))
        self.assertFalse(np.isnan(s.iloc[0]))

    def test_texture_normalization_beats_raw_on_plain_sharp(self):
        # a plain-but-sharp image (high lap? no: low edge, moderate lap) should not be
        # ranked as blurry as a truly soft image (low lap, high texture context).
        t = pd.DataFrame({
            "laplacian_variance": [0.001, 0.001],   # same raw sharpness
            "edge_density":       [0.02, 0.20],     # left = plain, right = textured-but-soft
            "dynamic_range":      [0.5, 0.5]})
        s = blur_score.add_blur_score(t)["blur_score"].to_numpy()
        # the textured-but-soft image (more edges yet same low lap) reads blurrier
        self.assertGreater(s[1], s[0])


class BlurBatchDeterminism(unittest.TestCase):
    SCORES = HERE / "data" / "scored_with_blur.csv"

    @unittest.skipUnless((HERE / "data" / "scored_with_blur.csv").exists(), "scored file")
    def test_deterministic(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            cmd = [sys.executable, str(HERE / "build_blur_batch.py"), "--scores",
                   str(self.SCORES), "--n", "30", "--seed", "20260726", "--out-dir"]
            subprocess.run(cmd + [d1], check=True, capture_output=True, cwd=HERE)
            subprocess.run(cmd + [d2], check=True, capture_output=True, cwd=HERE)
            a = pd.read_csv(Path(d1) / "blur_candidates_private.csv")
            b = pd.read_csv(Path(d2) / "blur_candidates_private.csv")
            self.assertTrue(a.equals(b))


if __name__ == "__main__":
    unittest.main()
