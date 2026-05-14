import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.clustering_approach.vinted_pipeline_batch import (
    combine_feature_blocks,
    select_primary_image_source,
    variant_cluster,
)
import experiments.clustering_approach.vinted_pipeline_batch as batch_module


class VintedPipelineBatchTests(unittest.TestCase):
    def test_select_primary_image_source_prefers_local_primary_path(self):
        row = {
            "LocalPrimaryImagePath": "/tmp/item/main.webp",
            "LocalImagePaths": '["/tmp/item/other.webp"]',
            "Images": '["https://example.com/remote.webp"]',
        }
        self.assertEqual(select_primary_image_source(row), "/tmp/item/main.webp")

    def test_select_primary_image_source_falls_back_to_images_list(self):
        row = {
            "LocalPrimaryImagePath": "",
            "LocalImagePaths": "[]",
            "Images": '["https://example.com/first.webp", "https://example.com/second.webp"]',
        }
        self.assertEqual(select_primary_image_source(row), "https://example.com/first.webp")

    def test_combine_feature_blocks_returns_row_normalized_matrix(self):
        text = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        image = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        combined = combine_feature_blocks(text, 0.5 * image)

        self.assertEqual(combined.shape, (2, 4))
        norms = np.linalg.norm(combined, axis=1)
        self.assertTrue(np.allclose(norms, 1.0))

    def test_variant_cluster_keeps_base_dim_and_adds_image_dim_for_clustering(self):
        df = pd.DataFrame(
            {
                "Title_norm": ["prada nylon bag black", "prada nylon bag red"],
                "Price": [100.0, 120.0],
            }
        )

        def fake_embed(texts):
            return np.asarray([[float(idx + 1), float(len(text))] for idx, text in enumerate(texts)], dtype=np.float32)

        image_vectors = np.asarray([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32)
        original_cluster = batch_module.cluster_agglomerative_cosine
        batch_module.cluster_agglomerative_cosine = lambda X, distance_threshold: np.arange(len(X), dtype=int)
        try:
            _, labs, _, _, Xv_base, Xv_cluster = variant_cluster(
                df,
                fake_embed,
                core_frac=0.7,
                vthr=0.33,
                pwt=0.35,
                image_vectors=image_vectors,
                image_weight=0.2,
            )
        finally:
            batch_module.cluster_agglomerative_cosine = original_cluster

        self.assertEqual(len(labs), 2)
        self.assertEqual(Xv_base.shape, (2, 3))
        self.assertEqual(Xv_cluster.shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
