import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
ANALYSIS_DIR = SCRIPTS_DIR / "analysis_pipeline"
for path in (SCRIPTS_DIR, ANALYSIS_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis_pipeline.scoring.foreground_blur import compute_foreground_blur_metrics, heuristic_foreground_mask


class ForegroundBlurTests(unittest.TestCase):
    def create_scene(self, path: Path, *, blur_object: bool) -> None:
        width, height = 320, 320
        image = Image.new("RGB", (width, height), (210, 205, 195))
        draw = ImageDraw.Draw(image)
        for x in range(0, width, 14):
            color = (120 + (x % 40), 95 + (x % 30), 70 + (x % 20))
            draw.line((x, 0, x, height), fill=color, width=3)
        for y in range(0, height, 18):
            color = (80 + (y % 50), 120 + (y % 40), 150 + (y % 30))
            draw.line((0, y, width, y), fill=color, width=2)

        obj = Image.new("RGBA", (170, 170), (0, 0, 0, 0))
        obj_draw = ImageDraw.Draw(obj)
        obj_draw.rounded_rectangle((15, 18, 155, 150), radius=18, fill=(25, 35, 45, 255), outline=(255, 255, 255, 255), width=4)
        obj_draw.line((25, 85, 145, 85), fill=(245, 210, 80, 255), width=6)
        obj_draw.line((38, 38, 132, 132), fill=(255, 255, 255, 255), width=4)
        if blur_object:
            obj = obj.filter(ImageFilter.GaussianBlur(radius=4.5))

        image.paste(obj.convert("RGB"), (75, 75))
        image.save(path)

    def test_heuristic_mask_prefers_center_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.png"
            self.create_scene(path, blur_object=False)
            image = Image.open(path).convert("RGB")
            mask, confidence = heuristic_foreground_mask(image)

        self.assertTrue(mask[mask.shape[0] // 2, mask.shape[1] // 2])
        self.assertGreater(mask.mean(), 0.08)
        self.assertLess(mask.mean(), 0.7)
        self.assertGreater(confidence, 0.2)

    def test_foreground_metrics_drop_when_object_is_blurred(self):
        with tempfile.TemporaryDirectory() as tmp:
            sharp_path = Path(tmp) / "sharp.png"
            blur_path = Path(tmp) / "blur.png"
            self.create_scene(sharp_path, blur_object=False)
            self.create_scene(blur_path, blur_object=True)
            sharp = compute_foreground_blur_metrics(str(sharp_path), backend="heuristic")
            blur = compute_foreground_blur_metrics(str(blur_path), backend="heuristic")

        self.assertGreater(sharp.foreground_sharpness_score, blur.foreground_sharpness_score)
        self.assertGreater(sharp.foreground_laplacian_variance, blur.foreground_laplacian_variance)
        self.assertGreater(sharp.foreground_tenengrad, blur.foreground_tenengrad)


if __name__ == "__main__":
    unittest.main()
