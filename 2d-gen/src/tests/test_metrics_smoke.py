from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from eval.metrics import frechet_distance, list_images, pair_clip_i_paths


class MetricsSmokeTest(unittest.TestCase):
    def test_list_images_filters_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            Image.new("RGB", (16, 16), color=(255, 0, 0)).save(tmpdir_path / "a.png")
            Image.new("RGB", (16, 16), color=(0, 255, 0)).save(tmpdir_path / "b.jpg")
            (tmpdir_path / "ignore.txt").write_text("x", encoding="utf-8")
            images = list_images(tmpdir_path)
            self.assertEqual([path.name for path in images], ["a.png", "b.jpg"])

    def test_frechet_distance_is_zero_for_identical_statistics(self) -> None:
        mu = np.array([0.0, 1.0], dtype=np.float64)
        sigma = np.array([[1.0, 0.2], [0.2, 2.0]], dtype=np.float64)
        distance = frechet_distance(mu, sigma, mu, sigma)
        self.assertAlmostEqual(distance, 0.0, places=6)

    def test_pair_clip_i_paths_requires_matching_names(self) -> None:
        real_paths = [Path("a.png"), Path("b.png")]
        generated_paths = [Path("a.png"), Path("c.png")]
        with self.assertRaises(ValueError):
            pair_clip_i_paths(real_paths, generated_paths)


if __name__ == "__main__":
    unittest.main()
