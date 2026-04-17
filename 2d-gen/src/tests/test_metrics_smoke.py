from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from eval.metrics import (
    _build_biomedclip_cache_path,
    _build_inception_cache_path,
    _load_manifest_records,
    evaluate_generation_quality,
    evaluate_unconditional_generation_quality,
    frechet_distance,
    list_images,
    pair_clip_i_paths,
    validate_aligned_image_dirs,
)


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

    def test_validate_aligned_image_dirs_rejects_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            real_dir = tmpdir_path / "real"
            generated_dir = tmpdir_path / "generated"
            real_dir.mkdir()
            generated_dir.mkdir()
            Image.new("RGB", (16, 16), color=(255, 0, 0)).save(real_dir / "a.png")
            Image.new("RGB", (16, 16), color=(0, 255, 0)).save(real_dir / "b.png")
            Image.new("RGB", (16, 16), color=(255, 255, 0)).save(generated_dir / "a.png")
            with self.assertRaises(ValueError):
                validate_aligned_image_dirs(real_dir, generated_dir)

    def test_validate_aligned_image_dirs_rejects_filename_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            real_dir = tmpdir_path / "real"
            generated_dir = tmpdir_path / "generated"
            real_dir.mkdir()
            generated_dir.mkdir()
            Image.new("RGB", (16, 16), color=(255, 0, 0)).save(real_dir / "a.png")
            Image.new("RGB", (16, 16), color=(0, 255, 0)).save(generated_dir / "b.png")
            with self.assertRaises(ValueError):
                validate_aligned_image_dirs(real_dir, generated_dir)

    def test_load_manifest_records_skips_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "metadata.jsonl"
            manifest_path.write_text(
                '\n{"image_path": "/tmp/sample.png", "prompt": "slice"}\n\n',
                encoding="utf-8",
            )
            records = _load_manifest_records(manifest_path)
            self.assertEqual(records, [{"image_path": "/tmp/sample.png", "prompt": "slice"}])

    def test_build_inception_cache_path_changes_when_source_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            image_path = tmpdir_path / "sample.png"
            weights_path = tmpdir_path / "weights.pth"
            cache_dir = tmpdir_path / "cache"
            Image.new("RGB", (16, 16), color=(255, 0, 0)).save(image_path)
            weights_path.write_bytes(b"weights")
            first_cache_path = _build_inception_cache_path([image_path], weights_path, cache_dir)
            image_path.write_bytes(image_path.read_bytes() + b"changed")
            second_cache_path = _build_inception_cache_path([image_path], weights_path, cache_dir)
            self.assertNotEqual(first_cache_path, second_cache_path)

    def test_build_biomedclip_cache_path_changes_when_source_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            image_path = tmpdir_path / "sample.png"
            model_path = tmpdir_path / "biomedclip"
            cache_dir = tmpdir_path / "cache"
            Image.new("RGB", (16, 16), color=(255, 0, 0)).save(image_path)
            model_path.mkdir()
            (model_path / "open_clip_config.json").write_text("{}", encoding="utf-8")
            (model_path / "open_clip_pytorch_model.bin").write_bytes(b"weights")
            first_cache_path = _build_biomedclip_cache_path([image_path], model_path, cache_dir)
            image_path.write_bytes(image_path.read_bytes() + b"changed")
            second_cache_path = _build_biomedclip_cache_path([image_path], model_path, cache_dir)
            self.assertNotEqual(first_cache_path, second_cache_path)

    def test_evaluate_generation_quality_returns_expected_metric_fields(self) -> None:
        real_features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        generated_features = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        generated_probs = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float64)
        real_biomedclip_features = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
        generated_biomedclip_features = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64)
        with (
            patch("eval.metrics._load_inception_backbone", return_value=(MagicMock(), torch.device("cpu"))),
            patch(
                "eval.metrics.compute_inception_features_and_probs",
                side_effect=[
                    (real_features, np.zeros((2, 2), dtype=np.float64)),
                    (generated_features, generated_probs),
                ],
            ) as mock_inception,
            patch("eval.metrics.compute_fid", side_effect=[12.345, 45.678]),
            patch("eval.metrics.compute_inception_score", return_value=(1.234, 0.567)),
            patch("eval.metrics.compute_clip_i", return_value=(0.876, 0.123)),
            patch("eval.metrics.compute_clip_t", return_value=(0.765, 0.234)),
            patch(
                "eval.metrics._load_biomedclip",
                return_value=(MagicMock(), MagicMock(), MagicMock(), torch.device("cpu")),
            ),
            patch(
                "eval.metrics.compute_biomedclip_features",
                side_effect=[real_biomedclip_features, generated_biomedclip_features],
            ),
            patch("eval.metrics.compute_biomedclip_i", return_value=(0.654, 0.321)),
            patch("eval.metrics.compute_biomedclip_t", return_value=(0.543, 0.432)),
        ):
            result = evaluate_generation_quality(
                real_image_dir="/tmp/real",
                generated_image_dir="/tmp/generated",
                generated_manifest_path="/tmp/generated/metadata.jsonl",
                batch_size=4,
                num_workers=2,
                inception_weights_path="/tmp/inception.pth",
                clip_model_path="/tmp/clip",
                biomedclip_model_path="/tmp/biomedclip",
                real_inception_cache_dir="/tmp/cache",
                real_biomedclip_cache_dir="/tmp/cache",
            )
        self.assertEqual(mock_inception.call_count, 2)
        self.assertEqual(
            result.__dict__,
            {
                "fid": 12.35,
                "inception_score_mean": 1.23,
                "inception_score_std": 0.57,
                "clip_i_mean": 0.88,
                "clip_i_std": 0.12,
                "clip_t_mean": 0.77,
                "clip_t_std": 0.23,
                "med_fid": 45.68,
                "biomedclip_i_mean": 0.65,
                "biomedclip_i_std": 0.32,
                "biomedclip_t_mean": 0.54,
                "biomedclip_t_std": 0.43,
            },
        )

    def test_evaluate_unconditional_generation_quality_uses_image_only_metrics(self) -> None:
        real_features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        generated_features = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        generated_probs = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float64)
        real_biomedclip_features = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
        generated_biomedclip_features = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float64)
        with (
            patch("eval.metrics.validate_aligned_image_dirs", return_value=[]),
            patch("eval.metrics._load_inception_backbone", return_value=(MagicMock(), torch.device("cpu"))),
            patch(
                "eval.metrics.compute_inception_features_and_probs",
                side_effect=[
                    (real_features, np.zeros((2, 2), dtype=np.float64)),
                    (generated_features, generated_probs),
                ],
            ) as mock_inception,
            patch("eval.metrics.compute_fid", side_effect=[12.345, 45.678]),
            patch("eval.metrics.compute_inception_score", return_value=(1.234, 0.567)),
            patch("eval.metrics.compute_clip_i", return_value=(0.876, 0.123)),
            patch("eval.metrics.compute_clip_t", side_effect=AssertionError("CLIP-T should not run")),
            patch(
                "eval.metrics._load_biomedclip",
                return_value=(MagicMock(), MagicMock(), MagicMock(), torch.device("cpu")),
            ),
            patch(
                "eval.metrics.compute_biomedclip_features",
                side_effect=[real_biomedclip_features, generated_biomedclip_features],
            ),
            patch("eval.metrics.compute_biomedclip_i", return_value=(0.654, 0.321)),
            patch("eval.metrics.compute_biomedclip_t", side_effect=AssertionError("BiomedCLIP-T should not run")),
        ):
            result = evaluate_unconditional_generation_quality(
                real_image_dir="/tmp/real",
                generated_image_dir="/tmp/generated",
                batch_size=4,
                num_workers=2,
                inception_weights_path="/tmp/inception.pth",
                clip_model_path="/tmp/clip",
                biomedclip_model_path="/tmp/biomedclip",
                real_inception_cache_dir="/tmp/cache",
                real_biomedclip_cache_dir="/tmp/cache",
            )
        self.assertEqual(mock_inception.call_count, 2)
        self.assertEqual(
            result.__dict__,
            {
                "fid": 12.35,
                "inception_score_mean": 1.23,
                "inception_score_std": 0.57,
                "clip_i_mean": 0.88,
                "clip_i_std": 0.12,
                "med_fid": 45.68,
                "biomedclip_i_mean": 0.65,
                "biomedclip_i_std": 0.32,
            },
        )


if __name__ == "__main__":
    unittest.main()
