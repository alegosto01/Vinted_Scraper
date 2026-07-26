"""Tests for hybrid v4: question margins, rules-only gates, gate screening, holdout."""
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
sys.path.insert(0, str(HERE.parent))
import hybrid
import develop_gates


def _toy_pairs(dims=512):
    rng = np.random.default_rng(0)
    def unit(k):
        v = rng.standard_normal((k, dims)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    return {"crop": (unit(3), unit(3)), "clutter": (unit(3), unit(3))}


class QuestionMargins(unittest.TestCase):
    def test_paired_margin_and_disagreement(self):
        pairs = _toy_pairs()
        img = np.random.default_rng(1).standard_normal((6, 512)).astype(np.float32)
        img /= np.linalg.norm(img, axis=1, keepdims=True)
        cols = hybrid.question_margins(img, pairs)
        ge, be = pairs["crop"]
        pm = (img @ be.T) - (img @ ge.T)
        self.assertTrue(np.allclose(cols["q_crop_margin"], pm.mean(1), atol=1e-6))
        self.assertTrue(np.allclose(cols["q_crop_disagreement"], pm.std(1), atol=1e-6))

    def test_prompt_count_invariance(self):
        pairs = _toy_pairs()
        img = np.random.default_rng(2).standard_normal((5, 512)).astype(np.float32)
        img /= np.linalg.norm(img, axis=1, keepdims=True)
        base = hybrid.question_margins(img, pairs)
        dbl = {q: (np.vstack([g, g]), np.vstack([b, b])) for q, (g, b) in pairs.items()}
        d = hybrid.question_margins(img, dbl)
        self.assertTrue(np.allclose(base["q_crop_margin"], d["q_crop_margin"], atol=1e-6))


def _toy_table(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "simple_blur_score": rng.random(n), "simple_exposure_score": rng.random(n),
        "simple_resolution_score": rng.random(n), "simple_glare_score": rng.random(n),
        "q_crop_margin": rng.standard_normal(n), "q_clutter_margin": rng.standard_normal(n),
        "q_item_visibility_margin": rng.standard_normal(n),
        "q_tilt_margin": rng.standard_normal(n), "q_glare_margin": rng.standard_normal(n)})


class Gates(unittest.TestCase):
    def test_gate_table_in_unit_range(self):
        g = hybrid.defect_gate_table(_toy_table())
        for c in g.columns:
            self.assertGreaterEqual(g[c].min(), 0.0)
            self.assertLessEqual(g[c].max(), 1.0)

    def test_v4_is_max_over_kept_gates(self):
        t = _toy_table()
        kept = ["gate_blur", "gate_clutter"]
        score, reason, g = hybrid.v4_score(t, kept)
        self.assertTrue(np.allclose(score, g[kept].max(axis=1)))
        self.assertTrue(set(reason.dropna().unique()).issubset({"blur", "clutter"}))

    def test_v4_all_nan_row_is_nan(self):
        t = _toy_table(5)
        for c in ["simple_blur_score", "q_clutter_margin"]:
            t.loc[0, c] = np.nan
        score, reason, _ = hybrid.v4_score(t, ["gate_blur", "gate_clutter"])
        self.assertTrue(np.isnan(score.iloc[0]))
        self.assertIsNone(reason.iloc[0])


class GateScreening(unittest.TestCase):
    def _dev_files(self, d):
        # 30 rows, 2 searches; gate_blur (via simple_blur_score) perfectly separates,
        # gate_crop (q_crop_margin) is anti-predictive.
        n = 30
        rng = np.random.default_rng(3)
        y = np.array([1, 0] * 15)
        t = pd.DataFrame({
            "item_id": [str(i) for i in range(n)],
            "primary_search": (["gucci"] * 15 + ["nike"] * 15),
            "search_names": (["gucci"] * 15 + ["nike"] * 15),
            "simple_blur_score": np.where(y == 1, 0.8 + rng.random(n) * 0.2,
                                          rng.random(n) * 0.3),
            "simple_exposure_score": rng.random(n),
            "simple_resolution_score": rng.random(n),
            "simple_glare_score": rng.random(n),
            "q_crop_margin": np.where(y == 1, -1.0, 1.0) + rng.random(n) * 0.1,
            "q_clutter_margin": rng.standard_normal(n),
            "q_item_visibility_margin": rng.standard_normal(n),
            "q_tilt_margin": rng.standard_normal(n),
            "q_glare_margin": rng.standard_normal(n)})
        human = pd.DataFrame({"item_id": t.item_id,
                              "human_technical_quality": np.where(y == 1, "bad", "good")})
        t.to_csv(Path(d) / "scored.csv", index=False)
        human.to_csv(Path(d) / "human.csv", index=False)
        return Path(d) / "scored.csv", Path(d) / "human.csv"

    def test_keeps_predictive_drops_antipredictive(self):
        with tempfile.TemporaryDirectory() as d:
            sc, hu = self._dev_files(d)
            surviving, rep = develop_gates.screen(sc, hu, Path(d) / "out")
            self.assertIn("gate_blur", surviving["kept_gates"])
            self.assertNotIn("gate_crop", surviving["kept_gates"])
            # crop should show anti-predictive AUROC (< 0.5)
            crop = rep[rep.gate == "gate_crop"].iloc[0]
            self.assertLess(crop.auroc, 0.5)


class HoldoutBuild(unittest.TestCase):
    SCORES = HERE / "data" / "scored_with_v4.csv"
    EXCL = HERE / "data" / "evaluation" / "eval_candidates_private.csv"

    @unittest.skipUnless((HERE / "data" / "scored_with_v4.csv").exists(),
                         "scored_with_v4 present")
    def test_excludes_used_and_deterministic(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            cmd = [sys.executable, str(HERE / "build_holdout.py"),
                   "--scores", str(self.SCORES), "--exclude", str(self.EXCL),
                   "--seed", "20260725", "--out-dir"]
            subprocess.run(cmd + [d1], check=True, capture_output=True, cwd=HERE)
            subprocess.run(cmd + [d2], check=True, capture_output=True, cwd=HERE)
            a = pd.read_csv(Path(d1) / "holdout_candidates_private.csv")
            b = pd.read_csv(Path(d2) / "holdout_candidates_private.csv")
            excl = set(pd.read_csv(self.EXCL, usecols=["item_id"]).item_id.astype(str))
            self.assertEqual(len(set(a.item_id.astype(str)) & excl), 0)   # no reuse
            self.assertTrue(a.equals(b))                                  # deterministic
            lab = pd.read_csv(Path(d1) / "holdout_label_sheet.csv")
            self.assertNotIn("item_id", lab.columns)


if __name__ == "__main__":
    unittest.main()
