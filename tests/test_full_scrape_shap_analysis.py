import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.old.full_scrape_model import shap_analysis


class FullScrapeShapAnalysisTests(unittest.TestCase):
    def test_dino_embedding_dimensions_are_detected(self):
        self.assertTrue(shap_analysis.is_dino_embedding_feature("numeric__DinoEmbedding_0001"))
        self.assertTrue(shap_analysis.is_dino_embedding_feature("DinoEmbedding_0383"))
        self.assertFalse(shap_analysis.is_dino_embedding_feature("numeric__DinoEmbeddingNorm"))
        self.assertFalse(shap_analysis.is_dino_embedding_feature("numeric__DinoOutlierScore"))

    def test_readable_visual_features_are_grouped_separately(self):
        original, display, group = shap_analysis.clean_transformed_feature_name("numeric__PhotoPrimaryBrightness")
        self.assertEqual(original, "PhotoPrimaryBrightness")
        self.assertEqual(display, "PhotoPrimaryBrightness")
        self.assertEqual(group, "visual_readable")

        original, display, group = shap_analysis.clean_transformed_feature_name("numeric__DinoOutlierScore")
        self.assertEqual(original, "DinoOutlierScore")
        self.assertEqual(display, "DinoOutlierScore")
        self.assertEqual(group, "visual_readable")

    def test_feature_importance_can_exclude_raw_dino_embeddings(self):
        values = np.array([[0.5, 2.0, -0.25], [0.25, -1.0, 0.75]])
        feature_names = [
            "numeric__Price",
            "numeric__DinoEmbedding_0000",
            "numeric__PhotoPrimarySharpness",
        ]
        frame = shap_analysis.feature_importance_frame(
            values=values,
            feature_names=feature_names,
            search_name="nike",
            feature_mode="full_scrape_plus_visual",
            approach="numeric_tree_v1",
            explainer_method="test",
            exclude_dino_embeddings=True,
        )
        self.assertEqual(set(frame["original_feature"]), {"Price", "PhotoPrimarySharpness"})
        self.assertNotIn("DinoEmbedding_0000", set(frame["original_feature"]))


if __name__ == "__main__":
    unittest.main()
