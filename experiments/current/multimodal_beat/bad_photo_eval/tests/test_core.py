"""Tests for alignment, CLIP defect margins, and deduplication (stdlib unittest).

Run: /home/ale/miniconda3/envs/vinted_scraper/bin/python -m unittest discover -s tests -p 'test_*.py'
(from experiments/current/multimodal_beat/bad_photo_eval/)
Eval-set and metrics tests live in test_eval.py.
"""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))       # for embed_blocks

import alignment
import clip_margins
import score_bad_photos as sbp
import simple_features as sf


def _fake_idx(existing_ids):
    idx = {i: f"/dir/{i}" for i in existing_ids}
    alignment.build_id_index = lambda: idx
    alignment.first_image = lambda d: (d if d and d.split("/")[-1] in
                                       {str(x) for x in existing_ids} else None)


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self._b, self._f = alignment.build_id_index, alignment.first_image
        self._load = alignment.np.load

    def tearDown(self):
        alignment.build_id_index, alignment.first_image = self._b, self._f
        alignment.np.load = self._load

    def test_correct_passes(self):
        _fake_idx({"100", "102"})
        df = pd.DataFrame({"item_id": ["100", "101", "102", "103"]})
        ok, rep = alignment.verify_alignment(df, np.array([1, 0, 1, 0]), "item_id")
        self.assertTrue(ok)
        self.assertEqual(rep["positions_disagree"], 0)

    def test_mismatched_order_fails(self):
        _fake_idx({"100", "102"})
        df = pd.DataFrame({"item_id": ["101", "100", "103", "102"]})   # shuffled
        ok, rep = alignment.verify_alignment(df, np.array([1, 0, 1, 0]), "item_id", tolerance=0.0)
        self.assertFalse(ok)
        self.assertGreater(rep["positions_disagree"], 0)

    def test_row_count_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "c.csv"
            pd.DataFrame({"item_id": ["1", "2", "3"]}).to_csv(csv, index=False)
            alignment.np.load = lambda p: np.zeros((5, 512), np.float32)
            with self.assertRaisesRegex(SystemExit, "ROW COUNT MISMATCH"):
                alignment.recover(csv, write=False)

    def test_missing_ids_and_dotzero_normalization(self):
        out = alignment.normalize_ids(pd.Series([" 100.0 ", "101", 102.0])).tolist()
        self.assertEqual(out, ["100", "101", "102"])


def _toy_text(dims=512):
    rng = np.random.default_rng(0)
    def unit(k):
        v = rng.standard_normal((k, dims)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    return {"blur": {"good": unit(3), "bad": unit(3)},
            "tilt": {"good": unit(2), "bad": unit(2)}}


def _unit_img(n, seed):
    img = np.random.default_rng(seed).standard_normal((n, 512)).astype(np.float32)
    return img / np.linalg.norm(img, axis=1, keepdims=True)


class ClipMarginTests(unittest.TestCase):
    def setUp(self):
        self._defects = clip_margins.DEFECTS
        clip_margins.DEFECTS = ["blur", "tilt"]

    def tearDown(self):
        clip_margins.DEFECTS = self._defects

    def test_margin_is_mean_bad_minus_mean_good(self):
        text = _toy_text(); img = _unit_img(7, 1)
        cols = clip_margins.score_margins(img, text)
        exp = (img @ text["blur"]["bad"].T).mean(1) - (img @ text["blur"]["good"].T).mean(1)
        self.assertTrue(np.allclose(cols["clip_blur_margin"], exp, atol=1e-6))

    def test_overall_is_max_of_margins(self):
        text = _toy_text(); img = _unit_img(5, 2)
        cols = clip_margins.score_margins(img, text)
        stacked = np.vstack([cols["clip_blur_margin"], cols["clip_tilt_margin"]]).max(0)
        self.assertTrue(np.allclose(cols["clip_overall_margin"], stacked, atol=1e-6))
        self.assertTrue(np.all(cols["clip_top_margin"] >= cols["clip_second_margin"]))

    def test_no_softmax_or_sigmoid(self):
        cols = clip_margins.score_margins(_unit_img(200, 3), _toy_text())
        m = cols["clip_blur_margin"]
        self.assertLess(m.min(), 0.0)         # spans zero -> not a bounded probability
        self.assertGreater(m.max(), 0.0)

    def test_prompt_count_invariance(self):
        text = _toy_text(); img = _unit_img(6, 4)
        base = clip_margins.score_margins(img, text)
        doubled = {d: {"good": np.vstack([text[d]["good"]] * 2),
                       "bad": np.vstack([text[d]["bad"]] * 2)} for d in text}
        dbl = clip_margins.score_margins(img, doubled)
        self.assertTrue(np.allclose(base["clip_blur_margin"], dbl["clip_blur_margin"], atol=1e-6))

    def test_text_embeddings_normalized_and_cached(self):
        with tempfile.TemporaryDirectory() as d:
            clip_margins.CACHE = Path(d)
            try:
                t1 = clip_margins.encode_prompts()
            except Exception:
                self.skipTest("CLIP model unavailable")
            for defect in t1:
                for kind in ("good", "bad"):
                    self.assertTrue(np.allclose(
                        np.linalg.norm(t1[defect][kind], axis=1), 1.0, atol=1e-4))
            cached = list(Path(d).glob("defect_text_*.npz"))
            self.assertTrue(cached, "cache file not written")
            t2 = clip_margins.encode_prompts()   # from cache
            self.assertTrue(np.allclose(t1["blur"]["bad"], t2["blur"]["bad"]))


class DedupTests(unittest.TestCase):
    def setUp(self):
        self._resolve = sf.resolve_first_image

    def tearDown(self):
        sf.resolve_first_image = self._resolve

    def test_one_item_two_searches_one_row(self):
        sf.resolve_first_image = lambda iid, idx: f"/img/{iid}.webp"
        df = pd.DataFrame({"item_id": ["9090394550", "9090394550", "7"],
                           "SearchName": ["gucci", "griffati_uomo_all", "nike"],
                           "Title": ["bag", "bag", "shoe"]})
        ids = np.array(["9090394550", "9090394550", "7"])
        uniq = sbp.deduplicate(df, ids, np.array([1, 1, 1]), {})
        self.assertEqual(len(uniq), 2)
        row = uniq[uniq["item_id"] == "9090394550"].iloc[0]
        self.assertEqual(set(row["search_names"]), {"gucci", "griffati_uomo_all"})
        self.assertEqual(set(row["original_row_indices"]), {0, 1})

    def test_no_duplicate_item_ids(self):
        sf.resolve_first_image = lambda iid, idx: ""
        n = 50
        ids = np.array([str(i % 10) for i in range(n)])
        df = pd.DataFrame({"item_id": ids, "SearchName": ["s"] * n, "Title": ["t"] * n})
        uniq = sbp.deduplicate(df, ids, np.ones(n, int), {})
        self.assertTrue(uniq["item_id"].is_unique)
        self.assertEqual(len(uniq), 10)


if __name__ == "__main__":
    unittest.main()
