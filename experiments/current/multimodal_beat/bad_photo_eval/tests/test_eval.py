"""Tests for the blind evaluation-set builder and the metrics script (unittest).

Uses data/smoke_scored.csv as a small fixture. Metrics tests build a tiny synthetic
private+labels pair so precision math is hand-checkable.
"""
from __future__ import annotations
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
PY = sys.executable
SMOKE = HERE / "data" / "smoke_scored.csv"
BUILD = HERE / "build_bad_photo_eval.py"
EVAL = HERE / "evaluate_bad_photo_eval.py"
FORBIDDEN = ("item_id", "score", "margin", "selection", "v1", "v2", "v3",
             "product", "type", "rank")


def _build(out_dir, seed=24072026):
    subprocess.run([PY, str(BUILD), "--scores", str(SMOKE), "--per-method-search", "3",
                    "--random-per-search", "3", "--seed", str(seed),
                    "--out-dir", str(out_dir)], check=True, capture_output=True)


@unittest.skipUnless(SMOKE.exists(), "smoke_scored.csv fixture missing")
class EvalSetTests(unittest.TestCase):
    def test_deterministic_same_seed(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            _build(a); _build(b)
            ha = hashlib.sha256((Path(a) / "eval_candidates_private.csv").read_bytes()).hexdigest()
            hb = hashlib.sha256((Path(b) / "eval_candidates_private.csv").read_bytes()).hexdigest()
            self.assertEqual(ha, hb)

    def test_blind_sheet_has_no_method_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            _build(d)
            cols = pd.read_csv(Path(d) / "blind_label_sheet.csv").columns
            for c in cols:
                lc = c.lower()
                self.assertFalse(any(f in lc for f in FORBIDDEN),
                                 f"blind sheet leaks column {c}")

    def test_random_rows_outside_top_region(self):
        with tempfile.TemporaryDirectory() as d:
            _build(d)
            priv = pd.read_csv(Path(d) / "eval_candidates_private.csv")
            reasons = priv["selection_reasons"].fillna("").astype(str)
            has_random = reasons.str.contains(":random")
            has_method = reasons.str.contains(r":(?:v1|v2|v3|simple)", regex=True)
            # no item is simultaneously a random pick AND a top-method pick
            self.assertEqual(int((has_random & has_method).sum()), 0)

    def test_repeats_excluded_from_unique_and_reuse_images(self):
        with tempfile.TemporaryDirectory() as d:
            _build(d)
            priv = pd.read_csv(Path(d) / "eval_candidates_private.csv")
            reps = priv[priv["is_repeat"] == 1]
            orig = priv[priv["is_repeat"] == 0]
            self.assertGreater(len(reps), 0)
            self.assertTrue(orig["blind_id"].is_unique and priv["blind_id"].is_unique)
            # every repeat reuses an original image and an original item
            self.assertTrue(set(reps["image_path"]).issubset(set(orig["image_path"])))
            self.assertTrue(set(reps["item_id"]).issubset(set(orig["item_id"])))


@unittest.skipUnless(EVAL.exists(), "evaluate_bad_photo_eval.py not present yet")
class MetricsTests(unittest.TestCase):
    def _fixture(self, d):
        """8 items, one search. v3=clip_overall_margin ranks bad-first perfectly for
        top-4; build so precision@3 == 1.0 and known @5/@8."""
        rows = []
        # blind_id, item_id, score (clip_overall_margin), label
        # top-3 by score are all bad -> p@3 = 1.0
        data = [("B1", "1", 0.9, "bad"), ("B2", "2", 0.8, "bad"), ("B3", "3", 0.7, "bad"),
                ("B4", "4", 0.6, "good"), ("B5", "5", 0.5, "bad"),
                ("B6", "6", 0.4, "good"), ("B7", "7", 0.3, "uncertain"), ("B8", "8", 0.2, "good")]
        priv = pd.DataFrame([{
            "blind_id": b, "item_id": i, "search_names": "gucci", "primary_search": "gucci",
            "title": "x", "image_path": f"/i/{i}.webp", "is_repeat": 0,
            "selection_reasons": "gucci:v3" if b != "B8" else "gucci:random",
            "v1_generic_clip_score": s, "v2_typed_clip_score": s,
            "clip_overall_margin": s, "simple_overall_score": s,
            "clip_top_defect": "blur", "simple_top_defect": "blur"} for b, i, s, _ in data])
        labels = pd.DataFrame([{
            "blind_id": b, "technical_quality": lab,
            "hurts_listing_presentation": "yes" if lab == "bad" else "no",
            "fixable_by_retake": "yes" if lab == "bad" else "no",
            "defect_tags": "blur" if lab == "bad" else "", "notes": ""}
            for b, i, s, lab in data])
        p = Path(d) / "priv.csv"; l = Path(d) / "labels.csv"
        priv.to_csv(p, index=False); labels.to_csv(l, index=False)
        return p, l

    def test_precision_at_k_handcheck(self):
        with tempfile.TemporaryDirectory() as d:
            p, l = self._fixture(d)
            subprocess.run([PY, str(EVAL), "--private-candidates", str(p),
                            "--labels", str(l), "--out-dir", d], check=True,
                           capture_output=True)
            m = pd.read_csv(Path(d) / "metrics_by_method.csv").set_index("method")
            # top-3 by clip_overall_margin = items 1,2,3 all bad -> pooled p@3 == 1.0.
            # excluding uncertain (B7): top-5 = 1,2,3(bad),4(good),5(bad) -> 4/5 = 0.8
            self.assertAlmostEqual(m.loc["v3", "pooled_p@3"], 1.0, places=6)
            self.assertAlmostEqual(m.loc["v3", "pooled_p@5"], 0.8, places=6)
            self.assertTrue((Path(d) / "results.md").exists())

    def test_runs_on_blank_labels(self):
        with tempfile.TemporaryDirectory() as d:
            p, l = self._fixture(d)
            lab = pd.read_csv(l)
            for c in ("technical_quality", "hurts_listing_presentation",
                      "fixable_by_retake", "defect_tags"):
                lab[c] = ""
            lab.to_csv(l, index=False)
            r = subprocess.run([PY, str(EVAL), "--private-candidates", str(p),
                                "--labels", str(l), "--out-dir", d], capture_output=True)
            self.assertEqual(r.returncode, 0, r.stderr.decode()[-500:])
            self.assertTrue((Path(d) / "results.md").exists())


if __name__ == "__main__":
    unittest.main()
