import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
ANALYSIS_DIR = SCRIPTS_DIR / "analysis_pipeline"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from analysis_pipeline.scoring.visual_rerank import analyze_listing_images, normalize_image_sources


class VisualRerankTests(unittest.TestCase):
    def create_image(self, size=(320, 430), color=(120, 120, 120), path=None, variant="horizontal"):
        image = Image.new("RGB", size, color=color)
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, size[0] - 20, size[1] - 20), outline=(255, 255, 255), width=6)
        if variant == "horizontal":
            draw.line((20, size[1] // 2, size[0] - 20, size[1] // 2), fill=(0, 0, 0), width=4)
        else:
            draw.line((20, 20, size[0] - 20, size[1] - 20), fill=(0, 0, 0), width=4)
            draw.line((20, size[1] - 20, size[0] - 20, 20), fill=(255, 255, 0), width=4)
        image.save(path)

    def test_normalize_image_sources_handles_stringified_list(self):
        raw = "['https://example.com/a.webp', 'https://example.com/b.webp']"
        self.assertEqual(
            normalize_image_sources(raw),
            ["https://example.com/a.webp", "https://example.com/b.webp"],
        )

    def test_analyze_listing_images_penalizes_duplicate_low_coverage_luxury_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "img1.png"
            self.create_image(path=path)
            result = analyze_listing_images([str(path), str(path)], title="Prada bag", search_name="prada", enable_clip=False)

        self.assertLess(result["VisualCompletenessScore"], 0.7)
        self.assertGreater(result["VisualRiskPenalty"], 0.1)
        self.assertIn("duplicate_images", result["VisualAnalysisNotes"])

    def test_analyze_listing_images_rewards_multiple_distinct_game_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            path1 = Path(tmp) / "front.png"
            path2 = Path(tmp) / "back.png"
            self.create_image(path=path1, color=(40, 110, 210), variant="horizontal")
            self.create_image(path=path2, color=(180, 60, 40), variant="diagonal")
            result = analyze_listing_images([str(path1), str(path2)], title="Detroit become human ps4", search_name="ps4", enable_clip=False)

        self.assertGreater(result["VisualCompletenessScore"], 0.8)
        self.assertEqual(result["VisualUniqueImageCount"], 2)
        self.assertLess(result["VisualRiskPenalty"], 0.2)


if __name__ == "__main__":
    unittest.main()
