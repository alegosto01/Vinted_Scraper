import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.old.photo_arbitrage import dataset, features, modeling, quality_methods
from experiments.old.photo_arbitrage.compare_quality_methods import (
    build_fashionclip_pseudo_label_review,
    render_fashionclip_review_html,
    restore_existing_review_annotations,
    summarize_quality_status_counts,
    write_fashionclip_review_html,
)
from experiments.old.photo_arbitrage.paths import assert_photo_path
from experiments.old.photo_arbitrage.report import (
    fashionclip_failure_examples,
    fashionclip_pseudo_agreement,
    quality_method_score_metrics,
    usefulness_sentence,
)


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

    def test_fashionclip_pseudo_labels_are_opt_in_for_training(self):
        labels = pd.DataFrame(
            [
                {"manual_label": "", "FashionClipPseudoLabel": "photo_quality_bad", "LocalImagePaths": "[]"},
                {"manual_label": "", "FashionClipPseudoLabel": "photo_quality_good", "LocalImagePaths": "[]"},
            ]
        )

        out = modeling.prepare_labeled_frame(labels)

        self.assertEqual(len(out), 0)

    def test_label_readiness_summary_reports_missing_manual_classes(self):
        labels = pd.DataFrame(
            [
                {"manual_label": "photo_quality_bad"},
                {"manual_label": "photo_quality_bad"},
                {"manual_label": "unclear"},
                {"manual_label": ""},
            ]
        )

        readiness = modeling.label_readiness_summary(labels)

        self.assertEqual(readiness["status"], "not_ready")
        self.assertEqual(readiness["bad_rows"], 2)
        self.assertEqual(readiness["good_rows"], 0)
        self.assertEqual(readiness["blank_rows"], 1)
        self.assertIn("photo_quality_good", readiness["message"])

    def test_label_readiness_summary_can_use_fashionclip_pseudo_source(self):
        labels = pd.DataFrame(
            [
                {"manual_label": "", "FashionClipPseudoLabel": "photo_quality_bad"},
                {"manual_label": "", "FashionClipPseudoLabel": "photo_quality_bad"},
                {"manual_label": "", "FashionClipPseudoLabel": "photo_quality_good"},
                {"manual_label": "", "FashionClipPseudoLabel": "photo_quality_good"},
            ]
        )

        readiness = modeling.label_readiness_summary(labels, label_source="fashionclip_pseudo")

        self.assertEqual(readiness["status"], "ready")
        self.assertEqual(readiness["trainable_rows"], 4)
        self.assertEqual(readiness["label_source"], "fashionclip_pseudo")

    def test_fashionclip_pseudo_label_source_can_train_weak_baseline(self):
        labels = pd.DataFrame(
            [
                {"manual_label": "", "FashionClipPseudoLabel": "photo_quality_bad", "LocalImagePaths": "[]", "FashionClipBadPhotoScore": 0.9, "FashionClipScoreMargin": -0.8},
                {"manual_label": "", "FashionClipPseudoLabel": "photo_quality_bad", "LocalImagePaths": "[]", "FashionClipBadPhotoScore": 0.8, "FashionClipScoreMargin": -0.6},
                {"manual_label": "", "FashionClipPseudoLabel": "photo_quality_good", "LocalImagePaths": "[]", "FashionClipBadPhotoScore": 0.1, "FashionClipScoreMargin": 0.8},
                {"manual_label": "", "FashionClipPseudoLabel": "photo_quality_good", "LocalImagePaths": "[]", "FashionClipBadPhotoScore": 0.2, "FashionClipScoreMargin": 0.6},
            ]
        )
        config = quality_methods.MethodConfig(methods=("fashionclip",))

        frame = modeling.prepare_labeled_frame(labels, method_config=config, label_source="fashionclip_pseudo")
        _, metadata = modeling.train_photo_quality_model(labels, method_config=config, label_source="fashionclip_pseudo")

        self.assertEqual(frame["TrainingLabel"].tolist(), ["photo_quality_bad", "photo_quality_bad", "photo_quality_good", "photo_quality_good"])
        self.assertEqual(metadata["status"], "trained")
        self.assertEqual(metadata["label_source"], "fashionclip_pseudo")
        self.assertEqual(metadata["training_rows"], 4)
        self.assertEqual(metadata["training_metrics"]["status"], "evaluated")
        self.assertIn(metadata["evaluation"]["status"], {"cross_validated", "skipped"})

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
            self.assertEqual(metadata["training_metrics"]["rows"], 4)
            self.assertEqual(metadata["evaluation"]["status"], "cross_validated")
            self.assertEqual(metadata["evaluation"]["folds"], 2)

    def test_training_requires_two_labels_per_class_for_evaluation_ready_model(self):
        labels = pd.DataFrame(
            [
                {"manual_label": "photo_quality_bad", "LocalImagePaths": "[]"},
                {"manual_label": "photo_quality_bad", "LocalImagePaths": "[]"},
                {"manual_label": "photo_quality_bad", "LocalImagePaths": "[]"},
                {"manual_label": "photo_quality_good", "LocalImagePaths": "[]"},
            ]
        )

        model, metadata = modeling.train_photo_quality_model(labels)

        self.assertIsNone(model)
        self.assertEqual(metadata["status"], "not_trained")
        self.assertIn("photo_quality_good", metadata["reason"])

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

    def test_fashionclip_method_missing_images_do_not_touch_model(self):
        frame = pd.DataFrame([{"LocalImagePaths": "[]"}])

        out = quality_methods.add_quality_method_scores(
            frame,
            config=quality_methods.MethodConfig(methods=("fashionclip",)),
        )

        self.assertIn("FashionClipBadPhotoScore", out.columns)
        self.assertEqual(out.iloc[0]["FashionClipStatus"], "missing_image")
        self.assertEqual(out.iloc[0]["QualityMethodStatus"], "missing_image")

    def test_fashionclip_scoring_populates_scores_with_cached_model(self):
        class FakeProcessor:
            def __call__(self, *, text, images, return_tensors, padding):
                import torch

                self.text = text
                self.images = images
                self.return_tensors = return_tensors
                self.padding = padding
                return {"pixel_values": torch.zeros((1, 3, 2, 2))}

        class FakeModel:
            def __call__(self, **_inputs):
                import torch

                return type("Output", (), {"logits_per_image": torch.tensor([[3.0, 2.0, 1.0, -2.0, -3.0, -4.0]])})()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "listing.png"
            Image.new("RGB", (20, 20), color=(150, 150, 150)).save(path)
            frame = pd.DataFrame([{"LocalImagePaths": f"['{path}']"}])
            processor = FakeProcessor()

            with patch.object(quality_methods, "package_available", return_value=True), patch.object(
                quality_methods,
                "_load_fashionclip_model",
                return_value=(processor, FakeModel(), "cpu"),
            ):
                out = quality_methods.add_quality_method_scores(
                    frame,
                    config=quality_methods.MethodConfig(methods=("fashionclip",)),
                )

        self.assertEqual(out.iloc[0]["FashionClipStatus"], "ok_cached")
        self.assertGreater(float(out.iloc[0]["FashionClipGoodScore"]), float(out.iloc[0]["FashionClipBadScore"]))
        self.assertAlmostEqual(
            float(out.iloc[0]["FashionClipGoodScore"]) + float(out.iloc[0]["FashionClipBadScore"]),
            1.0,
            places=6,
        )
        self.assertGreater(float(out.iloc[0]["FashionClipScoreMargin"]), 0.0)
        self.assertEqual(len(processor.text), len(quality_methods.FASHIONCLIP_GOOD_PROMPTS) + len(quality_methods.FASHIONCLIP_BAD_PROMPTS))

    def test_fashionclip_method_alias_and_feature_columns(self):
        self.assertEqual(quality_methods.normalize_methods("fashionclip"), ("fashionclip",))
        self.assertEqual(quality_methods.normalize_methods("fclip"), ("fashionclip",))
        self.assertEqual(quality_methods.normalize_methods("clip"), ("aesthetic",))

        columns = quality_methods.model_feature_columns(("fashionclip",))

        self.assertIn("FashionClipBadPhotoScore", columns)
        self.assertIn("FashionClipScoreMargin", columns)

    def test_fashionclip_uncached_load_error_is_specific(self):
        err = OSError(
            "Can't load image processor for 'patrickjohncyh/fashion-clip'. "
            "If you were trying to load it from 'https://huggingface.co/models', "
            "make sure you don't have a local directory with the same name."
        )

        status = quality_methods.summarize_fashionclip_load_error(
            err,
            "patrickjohncyh/fashion-clip",
            local_files_only=True,
        )

        self.assertEqual(status, "load_failed_model_not_cached:patrickjohncyh/fashion-clip")

    def test_fashionclip_pseudo_label_review_keeps_confident_rows(self):
        scored = pd.DataFrame(
            [
                {
                    "Title": "good",
                    "FashionClipGoodScore": 0.91,
                    "FashionClipBadScore": 0.09,
                    "FashionClipBadPhotoScore": 0.09,
                    "FashionClipStatus": "ok_cached",
                    "CombinedBadPhotoScore": 0.1,
                },
                {
                    "Title": "bad",
                    "FashionClipGoodScore": 0.04,
                    "FashionClipBadScore": 0.96,
                    "FashionClipBadPhotoScore": 0.96,
                    "FashionClipStatus": "ok_cached",
                    "CombinedBadPhotoScore": 0.9,
                },
                {
                    "Title": "unclear",
                    "FashionClipGoodScore": 0.6,
                    "FashionClipBadScore": 0.4,
                    "FashionClipBadPhotoScore": 0.4,
                    "FashionClipStatus": "ok_cached",
                    "CombinedBadPhotoScore": 0.4,
                },
            ]
        )

        review = build_fashionclip_pseudo_label_review(scored, threshold=0.85, top_n=10)

        self.assertEqual(review["Title"].tolist(), ["bad", "good"])
        self.assertEqual(review["FashionClipPseudoLabel"].tolist(), ["photo_quality_bad", "photo_quality_good"])
        self.assertIn("manual_label", review.columns)
        self.assertIn("manual_notes", review.columns)

    def test_fashionclip_review_html_renders_local_images_and_escapes_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "listing.png"
            Image.new("RGB", (20, 20), color=(150, 150, 150)).save(path)
            review = pd.DataFrame(
                [
                    {
                        "Title": "Bad <photo>",
                        "FashionClipPseudoLabel": "photo_quality_bad",
                        "FashionClipPseudoConfidence": 0.96,
                        "FashionClipGoodScore": 0.04,
                        "FashionClipBadScore": 0.96,
                        "SearchName": "gucci",
                        "Price": 20,
                        "Link": "https://example.test/item",
                        "LocalPrimaryImagePath": str(path),
                    }
                ]
            )

            html = render_fashionclip_review_html(review)

        self.assertIn("photo_quality_bad: 1", html)
        self.assertIn("Bad &lt;photo&gt;", html)
        self.assertIn(path.resolve().as_uri(), html)
        self.assertIn("fashionclip_pseudo_label_review_queue.csv", html)

    def test_fashionclip_review_html_writer_rejects_outside_paths(self):
        with self.assertRaises(ValueError):
            write_fashionclip_review_html(pd.DataFrame(), Path("/tmp/fashionclip_review.html"))

    def test_existing_fashionclip_review_annotations_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing_path = Path(tmp) / "fashionclip_pseudo_label_review_queue.csv"
            pd.DataFrame(
                [
                    {
                        "SearchName": "Gucci",
                        "Dataid": "123",
                        "manual_label": "photo_quality_bad",
                        "manual_notes": "dark crop",
                    },
                    {
                        "SearchName": "Prada",
                        "Dataid": "456",
                        "manual_label": "",
                        "manual_notes": "skip later",
                    },
                ]
            ).to_csv(existing_path, index=False)
            review = pd.DataFrame(
                [
                    {
                        "SearchName": "gucci",
                        "Dataid": "123",
                        "FashionClipPseudoLabel": "photo_quality_good",
                        "manual_label": "",
                        "manual_notes": "",
                    },
                    {
                        "SearchName": "prada",
                        "Dataid": "456",
                        "FashionClipPseudoLabel": "photo_quality_good",
                        "manual_label": "photo_quality_good",
                        "manual_notes": "",
                    },
                ]
            )

            restored, count = restore_existing_review_annotations(review, existing_path)

        self.assertEqual(count, 2)
        self.assertEqual(restored.iloc[0]["manual_label"], "photo_quality_bad")
        self.assertEqual(restored.iloc[0]["manual_notes"], "dark crop")
        self.assertEqual(restored.iloc[1]["manual_label"], "photo_quality_good")
        self.assertEqual(restored.iloc[1]["manual_notes"], "skip later")

    def test_quality_status_summary_counts_optional_methods(self):
        scored = pd.DataFrame(
            [
                {"FashionClipStatus": "ok_cached", "PyiqaStatus": "pyiqa_not_installed"},
                {"FashionClipStatus": "load_failed_model_not_cached:patrickjohncyh/fashion-clip", "PyiqaStatus": "pyiqa_not_installed"},
                {"FashionClipStatus": "ok_cached", "PyiqaStatus": "missing_image"},
                {"FashionClipStatus": "", "PyiqaStatus": ""},
            ]
        )

        summary = summarize_quality_status_counts(scored)

        self.assertEqual(summary["FashionClipStatus"]["ok_cached"], 2)
        self.assertEqual(summary["FashionClipStatus"]["load_failed_model_not_cached:patrickjohncyh/fashion-clip"], 1)
        self.assertEqual(summary["PyiqaStatus"]["pyiqa_not_installed"], 2)
        self.assertNotIn("", summary["FashionClipStatus"])
        self.assertNotIn("DinoStatus", summary)

    def test_training_metadata_preserves_fashionclip_config(self):
        labels = pd.DataFrame(
            [
                {"manual_label": "photo_quality_bad", "LocalImagePaths": "[]", "FashionClipBadPhotoScore": 0.9, "FashionClipScoreMargin": -0.8},
                {"manual_label": "photo_quality_bad", "LocalImagePaths": "[]", "FashionClipBadPhotoScore": 0.8, "FashionClipScoreMargin": -0.6},
                {"manual_label": "photo_quality_good", "LocalImagePaths": "[]", "FashionClipBadPhotoScore": 0.1, "FashionClipScoreMargin": 0.8},
                {"manual_label": "photo_quality_good", "LocalImagePaths": "[]", "FashionClipBadPhotoScore": 0.2, "FashionClipScoreMargin": 0.6},
            ]
        )
        config = quality_methods.MethodConfig(
            methods=("simple", "fashionclip"),
            fashionclip_model="local-fashionclip-test",
            fashionclip_local_files_only=True,
        )

        _, metadata = modeling.train_photo_quality_model(labels, method_config=config)

        self.assertEqual(metadata["status"], "trained")
        self.assertIn("FashionClipBadPhotoScore", metadata["features"])
        self.assertEqual(metadata["quality_method_config"]["fashionclip_model"], "local-fashionclip-test")
        self.assertTrue(metadata["quality_method_config"]["fashionclip_local_files_only"])

    def test_fashionclip_report_agreement_compares_pseudo_to_manual_labels(self):
        review = pd.DataFrame(
            [
                {"manual_label": "photo_quality_bad", "FashionClipPseudoLabel": "photo_quality_bad"},
                {"manual_label": "photo_quality_good", "FashionClipPseudoLabel": "photo_quality_bad"},
                {"manual_label": "photo_quality_good", "FashionClipPseudoLabel": "photo_quality_good"},
                {"manual_label": "unclear", "FashionClipPseudoLabel": "photo_quality_good"},
                {"manual_label": "", "FashionClipPseudoLabel": "photo_quality_bad"},
            ]
        )

        metrics = fashionclip_pseudo_agreement(review)

        self.assertEqual(metrics["manual_reviewed_rows"], 3)
        self.assertEqual(metrics["pending_manual_rows"], 1)
        self.assertEqual(metrics["comparable_rows"], 3)
        self.assertEqual(metrics["agreement"], 0.6667)
        self.assertEqual(metrics["bad_precision"], 0.5)
        self.assertEqual(metrics["bad_recall"], 1.0)
        self.assertIn("mixed", usefulness_sentence(metrics))

    def test_fashionclip_report_compares_method_scores_against_manual_labels(self):
        review = pd.DataFrame(
            [
                {"manual_label": "photo_quality_bad", "SimpleBadPhotoScore": 0.9, "FashionClipBadPhotoScore": 0.9, "CombinedBadPhotoScore": 0.9},
                {"manual_label": "photo_quality_bad", "SimpleBadPhotoScore": 0.8, "FashionClipBadPhotoScore": 0.2, "CombinedBadPhotoScore": 0.7},
                {"manual_label": "photo_quality_good", "SimpleBadPhotoScore": 0.2, "FashionClipBadPhotoScore": 0.8, "CombinedBadPhotoScore": 0.3},
                {"manual_label": "photo_quality_good", "SimpleBadPhotoScore": 0.1, "FashionClipBadPhotoScore": 0.1, "CombinedBadPhotoScore": 0.2},
            ]
        )

        metrics = quality_method_score_metrics(review)
        simple = metrics[metrics["method"] == "simple"].iloc[0]
        fashionclip = metrics[metrics["method"] == "fashionclip"].iloc[0]

        self.assertEqual(simple["auc_bad_vs_good"], 1.0)
        self.assertEqual(simple["threshold_0_5_accuracy"], 1.0)
        self.assertEqual(simple["mean_delta_bad_minus_good"], 0.7)
        self.assertEqual(fashionclip["threshold_0_5_accuracy"], 0.5)

    def test_fashionclip_report_failure_examples_show_pseudo_manual_mismatches(self):
        review = pd.DataFrame(
            [
                {
                    "Title": "confident mismatch",
                    "manual_label": "photo_quality_good",
                    "FashionClipPseudoLabel": "photo_quality_bad",
                    "FashionClipPseudoConfidence": 0.98,
                },
                {
                    "Title": "match",
                    "manual_label": "photo_quality_bad",
                    "FashionClipPseudoLabel": "photo_quality_bad",
                    "FashionClipPseudoConfidence": 0.99,
                },
                {
                    "Title": "lower mismatch",
                    "manual_label": "photo_quality_bad",
                    "FashionClipPseudoLabel": "photo_quality_good",
                    "FashionClipPseudoConfidence": 0.9,
                },
            ]
        )

        examples = fashionclip_failure_examples(review, limit=1)

        self.assertEqual(examples["Title"].tolist(), ["confident mismatch"])
        self.assertEqual(examples.iloc[0]["manual_label"], "photo_quality_good")

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
