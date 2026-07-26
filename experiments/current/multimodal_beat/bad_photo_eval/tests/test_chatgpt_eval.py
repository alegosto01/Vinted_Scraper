"""Tests for analyze_chatgpt_labels.py (frozen evaluation on pseudo-labels)."""
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
import analyze_chatgpt_labels as A

PRIV = HERE / "data" / "evaluation" / "eval_candidates_private.csv"
LAB = HERE / "data" / "evaluation" / "blind_label_sheet_chatgpt.csv"


def _mk(tqs, searches=None, is_repeat=None, item_ids=None, scores=None, random_idx=()):
    n = len(tqs)
    searches = searches or ["gucci"] * n
    is_repeat = is_repeat or [0] * n
    item_ids = item_ids or [str(i) for i in range(n)]
    scores = scores if scores is not None else list(range(n, 0, -1))
    priv = pd.DataFrame({
        "blind_id": [f"B{i:03d}" for i in range(n)], "item_id": item_ids,
        "search_names": searches, "primary_search": [s.split("|")[0] for s in searches],
        "title": ["t"] * n, "image_path": [f"/i/{i}.webp" for i in range(n)],
        "is_repeat": is_repeat,
        "selection_reasons": ["gucci:random" if i in random_idx else "gucci:v3"
                              for i in range(n)],
        "v1_generic_clip_score": scores, "v2_typed_clip_score": scores,
        "clip_overall_margin": scores, "simple_overall_score": scores,
        "clip_top_defect": ["crop"] * n, "simple_top_defect": ["blur"] * n})
    lab = pd.DataFrame({"blind_id": priv.blind_id, "technical_quality": tqs,
                        "hurts_listing_presentation": ["no"] * n,
                        "fixable_by_retake": ["no"] * n, "defect_tags": [""] * n,
                        "notes": [""] * n})
    return priv, lab


def _merge(priv, lab):
    with tempfile.TemporaryDirectory() as d:
        p, l = Path(d) / "p.csv", Path(d) / "l.csv"
        priv.to_csv(p, index=False); lab.to_csv(l, index=False)
        return A.load_merged(p, l)


class MergeAndTargets(unittest.TestCase):
    def test_merge_mismatch_raises(self):
        priv, lab = _mk(["good", "bad"])
        lab.loc[0, "blind_id"] = "ZZZ"
        with self.assertRaises(SystemExit):
            _merge(priv, lab)

    def test_t1_excludes_uncertain_notitem_repeats(self):
        priv, lab = _mk(["bad", "good", "uncertain", "not_item_photo", "bad"],
                        is_repeat=[0, 0, 0, 0, 1])
        m = _merge(priv, lab)
        e = A.eligible(m, "T1")
        self.assertEqual(set(e.tq), {"bad", "good"})
        self.assertTrue((e.is_repeat == 0).all())
        self.assertEqual(len(e), 2)                 # one bad + one good (repeat dropped)

    def test_t2_positive_is_not_item_photo(self):
        priv, lab = _mk(["bad", "good", "not_item_photo"])
        m = _merge(priv, lab)
        e = A.eligible(m, "T2")
        self.assertEqual(int(e["pos"].sum()), 1)
        self.assertTrue(e.loc[e.tq == "not_item_photo", "pos"].all())
        self.assertFalse(e.loc[e.tq == "bad", "pos"].any())   # bad is negative here

    def test_pooled_dedups_to_nonrepeat(self):
        priv, lab = _mk(["bad", "bad"], item_ids=["9", "9"], is_repeat=[0, 1])
        m = _merge(priv, lab)
        self.assertEqual(len(A.eligible(m, "T1")), 1)


class Metrics(unittest.TestCase):
    def test_precision_at_k(self):
        pool = pd.DataFrame({"s": [5, 4, 3, 2, 1], "pos": [1, 1, 0, 1, 0]})
        p, tp, fp = A.precision_at_k(pool, "s", 3)
        self.assertAlmostEqual(p, 2 / 3); self.assertEqual((tp, fp), (2, 1))
        self.assertEqual(A.precision_at_k(pool, "s", 9), (None, None, None))  # < k

    def test_macro_excludes_small_searches(self):
        # gucci has 8 rows, nike has 3 -> macro P@8 uses only gucci
        tqs = ["bad", "good"] * 4 + ["bad", "good", "bad"]
        searches = ["gucci"] * 8 + ["nike"] * 3
        priv, lab = _mk(tqs, searches=searches)
        m = _merge(priv, lab)
        out, _ = A.evaluate_target(m, "T1")
        self.assertEqual(out["macro"]["v3"]["macro_p@8_n_searches"], 1)

    def test_random_prevalence_and_lift_unavailable_when_small(self):
        # only 3 random rows -> lift unavailable
        tqs = ["bad", "good", "good", "bad", "good"]
        priv, lab = _mk(tqs, random_idx=(0, 1, 2))
        m = _merge(priv, lab)
        out, _ = A.evaluate_target(m, "T1")
        self.assertEqual(out["random_eligible"], 3)
        self.assertFalse(out["lift_available"])
        self.assertIsNone(out["per_method"]["v3"]["lift@8"])


@unittest.skipUnless(PRIV.exists() and LAB.exists(), "real eval files present")
class RealDataIntegration(unittest.TestCase):
    def test_audit_queue_deterministic(self):
        m = A.load_merged(PRIV, LAB)
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            A.build_audit_queue(m, Path(d1), seed=24072026)
            A.build_audit_queue(m, Path(d2), seed=24072026)
            a = (Path(d1) / "human_audit_queue.csv").read_bytes()
            b = (Path(d2) / "human_audit_queue.csv").read_bytes()
            self.assertEqual(a, b)

    def test_repeats_100pct_excluded_from_pool(self):
        m = A.load_merged(PRIV, LAB)
        e = A.eligible(m, "T1")
        self.assertTrue((e.is_repeat == 0).all())


class NoTrainingOrPromptTuning(unittest.TestCase):
    def test_source_has_no_model_fit_or_prompts(self):
        src = (HERE / "analyze_chatgpt_labels.py").read_text()
        for banned in ("LogisticRegression", ".fit(", "DEFECT_PROMPTS", "encode_prompts"):
            self.assertNotIn(banned, src, f"analysis must not {banned}")
        # only the four frozen score columns are ranked
        self.assertEqual(set(A.METHODS.values()),
                         {"v1_generic_clip_score", "v2_typed_clip_score",
                          "clip_overall_margin", "simple_overall_score"})


if __name__ == "__main__":
    unittest.main()
