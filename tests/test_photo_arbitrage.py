import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.photo_arbitrage import dataset, features, modeling, quality_methods
from experiments.photo_arbitrage.paths import assert_photo_path


class PhotoArbitrageTests(unittest.TestCase):
    def test_candidate_building_uses_search_data_and_local_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            search = root / "gucci"
            image_dir = search / "image_cache" / "123"
            image_dir.mkdir(parents=True)
            Image.new("RGB", (20, 20), color=(90, 90, 90)).save(image_dir / "image_01.webp")
            pd.DataFrame(
                [
                    {
                        "Dataid": "123",
                        "Title": "Gucci item",
                        "Brand": "Gucci",
                        "Price": 30,
                        "Link": "https://www.vinted.it/items/123-test",
                        "Images": "['https://images.example/a.webp']",
                    }
                ]
            ).to_csv(search / "big_raw.csv", index=False)
            (search / "full_scrape").mkdir()
            pd.DataFrame(
                [
                    {
                        "Dataid": "123",
                        "Description": "desc",
                        "Condition": "new",
                        "PictureCount": 3,
                    }
                ]
            ).to_csv(search / "full_scrape" / "items_enriched.csv", index=False)

            out = dataset.build_candidate_dataset(all_searches=True, simple_root=root)

            self.assertEqual(len(out), 1)
            self.assertEqual(out.iloc[0]["SearchName"], "gucci")
            self.assertIn("image_01.webp", out.iloc[0]["LocalPrimaryImagePath"])
            self.assertEqual(float(out.iloc[0]["PictureCount"]), 3.0)

    def test_missing_images_do_not_break_feature_extraction(self):
        frame = pd.DataFrame([{"Dataid": "1", "LocalImagePaths": "[]", "PictureCount": 1}])

        out = features.add_photo_features(frame)

        self.assertEqual(float(out.iloc[0]["PhotoMissingImages"]), 1.0)
        self.assertIn("PhotoLowQualityFraction", out.columns)

    def test_feature_extraction_reads_local_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "img.png"
            Image.new("RGB", (40, 20), color=(120, 80, 40)).save(path)
            frame = pd.DataFrame([{"Dataid": "1", "LocalImagePaths": f"['{path}']", "PictureCount": 1}])

            out = features.add_photo_features(frame)

            self.assertEqual(float(out.iloc[0]["PhotoPrimaryWidth"]), 40.0)
            self.assertEqual(float(out.iloc[0]["PhotoPrimaryHeight"]), 20.0)
            self.assertEqual(float(out.iloc[0]["PhotoMissingImages"]), 0.0)

    def test_label_loading_keeps_only_trainable_labels(self):
        labels = pd.DataFrame(
            [
                {"manual_label": "photo_quality_bad", "LocalImagePaths": "[]"},
                {"manual_label": "photo_quality_good", "LocalImagePaths": "[]"},
                {"manual_label": "unclear", "LocalImagePaths": "[]"},
                {"manual_label": "", "LocalImagePaths": "[]"},
            ]
        )

        out = modeling.prepare_labeled_frame(labels)

        self.assertEqual(out["PhotoQualityTarget"].tolist(), [1, 0])

    def test_tiny_model_training_and_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            dark = Path(tmp) / "dark.png"
            bright = Path(tmp) / "bright.png"
            Image.new("RGB", (40, 40), color=(15, 15, 15)).save(dark)
            Image.new("RGB", (40, 40), color=(180, 180, 180)).save(bright)
            labels = pd.DataFrame(
                [
                    {"manual_label": "photo_quality_bad", "LocalImagePaths": f"['{dark}']", "PictureCount": 1, "Price": 20, "Brand": "A"},
                    {"manual_label": "photo_quality_bad", "LocalImagePaths": f"['{dark}']", "PictureCount": 1, "Price": 25, "Brand": "A"},
                    {"manual_label": "photo_quality_good", "LocalImagePaths": f"['{bright}']", "PictureCount": 3, "Price": 30, "Brand": "A"},
                    {"manual_label": "photo_quality_good", "LocalImagePaths": f"['{bright}']", "PictureCount": 3, "Price": 35, "Brand": "A"},
                ]
            )

            model, metadata = modeling.train_photo_quality_model(labels)
            scored = features.add_photo_features(labels)
            probs, version = modeling.score_bad_photo_probability(scored, model=model)

            self.assertEqual(metadata["status"], "trained")
            self.assertEqual(version, "photo_quality_v1")
            self.assertEqual(len(probs), 4)

    def test_business_score_output_columns(self):
        frame = pd.DataFrame([{"PhotoMissingImages": 1, "PhotoLowQualityFraction": 1, "PhotoOnlyOneImage": 1, "PriceNum": 20, "BrandPresent": 1}])
        frame["BadPhotoProbability"] = 0.9

        out = features.add_business_scores(frame)

        self.assertIn("PhotoOpportunityScore", out.columns)
        self.assertIn("PhotoOpportunityNotes", out.columns)

    def test_quality_method_simple_score_and_combined_score(self):
        frame = pd.DataFrame(
            [
                {
                    "PhotoMissingImages": 1,
                    "PhotoLowQualityFraction": 1,
                    "PhotoOnlyOneImage": 1,
                    "PhotoDuplicateImageCount": 0,
                    "PhotoPrimaryContrast": 0.05,
                    "PhotoPrimarySharpness": 10,
                }
            ]
        )

        out = quality_methods.add_quality_method_scores(
            frame,
            config=quality_methods.MethodConfig(methods=("simple",)),
        )

        self.assertIn("SimpleBadPhotoScore", out.columns)
        self.assertIn("CombinedBadPhotoScore", out.columns)
        self.assertGreater(float(out.iloc[0]["SimpleBadPhotoScore"]), 0.5)

    def test_quality_method_missing_images_do_not_break_table(self):
        frame = pd.DataFrame([{"LocalImagePaths": "[]"}])

        out = quality_methods.add_quality_method_scores(
            frame,
            config=quality_methods.MethodConfig(methods=("pyiqa",)),
        )

        self.assertIn("PyiqaStatus", out.columns)
        self.assertIn("CombinedBadPhotoScore", out.columns)

    def test_dino_load_error_does_not_use_fallback(self):
        err = OSError("You are trying to access a gated repo. 401 Unauthorized")

        status = quality_methods.summarize_dino_load_error(err, "facebook/dinov3-vits16-pretrain-lvd1689m")

        self.assertEqual(status, "load_failed_gated_repo:facebook/dinov3-vits16-pretrain-lvd1689m")
        self.assertNotIn("fallback", status.lower())

    def test_dino_token_scope_error_is_specific(self):
        err = RuntimeError("403 Forbidden: Please enable access to public gated repositories in your fine-grained token settings")

        status = quality_methods.summarize_dino_load_error(err, "facebook/dinov3-vits16-pretrain-lvd1689m")

        self.assertEqual(status, "load_failed_token_scope_public_gated:facebook/dinov3-vits16-pretrain-lvd1689m")

    def test_dino_token_scope_error_is_detected_from_cause(self):
        cause = RuntimeError("403 Forbidden: Please enable access to public gated repositories in your fine-grained token settings")
        err = RuntimeError("Cannot find the requested files")
        err.__cause__ = cause

        status = quality_methods.summarize_dino_load_error(err, "facebook/dinov3-vits16-pretrain-lvd1689m")

        self.assertEqual(status, "load_failed_token_scope_public_gated:facebook/dinov3-vits16-pretrain-lvd1689m")

    def test_photo_path_guard_rejects_outside_paths(self):
        with self.assertRaises(ValueError):
            assert_photo_path(Path("/tmp/not_photo_arbitrage.csv"))


if __name__ == "__main__":
    unittest.main()
