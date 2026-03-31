from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from common.types import MetricResult
from eval.run_evaluate import main


class RunEvaluateTest(unittest.TestCase):
    def test_main_forwards_num_workers_and_cache_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.json"
            timestamp = datetime(2026, 3, 31, 12, 34, 56)
            timestamped_output_path = output_path.with_name("metrics_20260331_123456.json")
            config = {
                "eval": {
                    "real_image_dir": "/tmp/real",
                    "generated_image_dir": "/tmp/generated",
                    "generated_manifest": "/tmp/generated/metadata.jsonl",
                    "output_path": str(output_path),
                    "batch_size": 8,
                    "num_workers": 3,
                    "inception_weights_path": "/tmp/inception.pth",
                    "clip_model_path": "/tmp/clip",
                    "real_inception_cache_dir": str(Path(tmpdir) / "cache"),
                }
            }
            with (
                patch.object(sys, "argv", ["run_evaluate.py", "--config", "/tmp/eval.yaml"]),
                patch("eval.run_evaluate.datetime") as mock_datetime,
                patch("eval.run_evaluate.load_yaml_config", return_value=config),
                patch(
                    "eval.run_evaluate.evaluate_generation_quality",
                    return_value=MetricResult(
                        fid=1.0,
                        inception_score_mean=2.0,
                        inception_score_std=3.0,
                        clip_i_mean=4.0,
                        clip_i_std=5.0,
                        clip_t_mean=6.0,
                        clip_t_std=7.0,
                    ),
                ) as mock_evaluate,
                patch("eval.run_evaluate.write_json", return_value=timestamped_output_path),
            ):
                mock_datetime.now.return_value = timestamp
                main()
        mock_evaluate.assert_called_once_with(
            real_image_dir="/tmp/real",
            generated_image_dir="/tmp/generated",
            generated_manifest_path="/tmp/generated/metadata.jsonl",
            batch_size=8,
            num_workers=3,
            inception_weights_path="/tmp/inception.pth",
            clip_model_path="/tmp/clip",
            real_inception_cache_dir=str(Path(tmpdir) / "cache"),
        )

    def test_main_defaults_real_inception_cache_dir_next_to_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "metrics.json"
            timestamp = datetime(2026, 3, 31, 12, 34, 56)
            timestamped_output_path = output_path.with_name("metrics_20260331_123456.json")
            config = {
                "eval": {
                    "real_image_dir": "/tmp/real",
                    "generated_image_dir": "/tmp/generated",
                    "generated_manifest": "/tmp/generated/metadata.jsonl",
                    "output_path": str(output_path),
                    "batch_size": 8,
                    "num_workers": 0,
                    "inception_weights_path": "/tmp/inception.pth",
                    "clip_model_path": "/tmp/clip",
                }
            }
            with (
                patch.object(sys, "argv", ["run_evaluate.py", "--config", "/tmp/eval.yaml"]),
                patch("eval.run_evaluate.datetime") as mock_datetime,
                patch("eval.run_evaluate.load_yaml_config", return_value=config),
                patch(
                    "eval.run_evaluate.evaluate_generation_quality",
                    return_value=MetricResult(
                        fid=1.0,
                        inception_score_mean=2.0,
                        inception_score_std=3.0,
                        clip_i_mean=4.0,
                        clip_i_std=5.0,
                        clip_t_mean=6.0,
                        clip_t_std=7.0,
                    ),
                ) as mock_evaluate,
                patch("eval.run_evaluate.write_json", return_value=timestamped_output_path),
            ):
                mock_datetime.now.return_value = timestamp
                main()
        mock_evaluate.assert_called_once_with(
            real_image_dir="/tmp/real",
            generated_image_dir="/tmp/generated",
            generated_manifest_path="/tmp/generated/metadata.jsonl",
            batch_size=8,
            num_workers=0,
            inception_weights_path="/tmp/inception.pth",
            clip_model_path="/tmp/clip",
            real_inception_cache_dir=timestamped_output_path.parent.resolve() / "cache",
        )


if __name__ == "__main__":
    unittest.main()
