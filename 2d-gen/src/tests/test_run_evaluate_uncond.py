from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from common.constants import DEFAULT_BIOMEDCLIP_MODEL_PATH, DEFAULT_CLIP_MODEL_PATH, DEFAULT_INCEPTION_WEIGHTS_PATH
from common.types import UnconditionalMetricResult
from eval.run_evaluate_uncond import main


class RunEvaluateUncondTest(unittest.TestCase):
    def test_main_forwards_unconditional_eval_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.json"
            timestamp = datetime(2026, 4, 16, 12, 34, 56)
            timestamped_output_path = output_path.with_name("metrics_20260416_123456.json")
            config = {
                "eval": {
                    "real_image_dir": "/tmp/real",
                    "generated_image_dir": "/tmp/generated",
                    "output_path": str(output_path),
                    "batch_size": 8,
                    "num_workers": 3,
                    "inception_weights_path": "/tmp/inception.pth",
                    "clip_model_path": "/tmp/clip",
                    "biomedclip_model_path": "/tmp/biomedclip",
                    "real_inception_cache_dir": str(Path(tmpdir) / "cache"),
                    "real_biomedclip_cache_dir": str(Path(tmpdir) / "cache"),
                }
            }
            with (
                patch.object(sys, "argv", ["run_evaluate_uncond.py", "--config", "/tmp/eval.yaml"]),
                patch("eval.run_evaluate_uncond.datetime") as mock_datetime,
                patch("eval.run_evaluate_uncond.load_yaml_config", return_value=config),
                patch(
                    "eval.run_evaluate_uncond.evaluate_unconditional_generation_quality",
                    return_value=UnconditionalMetricResult(
                        fid=1.0,
                        inception_score_mean=2.0,
                        inception_score_std=3.0,
                        clip_i_mean=4.0,
                        clip_i_std=5.0,
                        med_fid=6.0,
                        biomedclip_i_mean=7.0,
                        biomedclip_i_std=8.0,
                    ),
                ) as mock_evaluate,
                patch("eval.run_evaluate_uncond.write_json", return_value=timestamped_output_path),
            ):
                mock_datetime.now.return_value = timestamp
                main()
        mock_evaluate.assert_called_once_with(
            real_image_dir="/tmp/real",
            generated_image_dir="/tmp/generated",
            batch_size=8,
            num_workers=3,
            inception_weights_path="/tmp/inception.pth",
            clip_model_path="/tmp/clip",
            biomedclip_model_path="/tmp/biomedclip",
            real_inception_cache_dir=str(Path(tmpdir) / "cache"),
            real_biomedclip_cache_dir=str(Path(tmpdir) / "cache"),
        )

    def test_main_defaults_model_paths_and_cache_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "metrics.json"
            timestamp = datetime(2026, 4, 16, 12, 34, 56)
            timestamped_output_path = output_path.with_name("metrics_20260416_123456.json")
            config = {
                "eval": {
                    "real_image_dir": "/tmp/real",
                    "generated_image_dir": "/tmp/generated",
                    "output_path": str(output_path),
                }
            }
            with (
                patch.object(sys, "argv", ["run_evaluate_uncond.py", "--config", "/tmp/eval.yaml"]),
                patch("eval.run_evaluate_uncond.datetime") as mock_datetime,
                patch("eval.run_evaluate_uncond.load_yaml_config", return_value=config),
                patch(
                    "eval.run_evaluate_uncond.evaluate_unconditional_generation_quality",
                    return_value=UnconditionalMetricResult(
                        fid=1.0,
                        inception_score_mean=2.0,
                        inception_score_std=3.0,
                        clip_i_mean=4.0,
                        clip_i_std=5.0,
                        med_fid=6.0,
                        biomedclip_i_mean=7.0,
                        biomedclip_i_std=8.0,
                    ),
                ) as mock_evaluate,
                patch("eval.run_evaluate_uncond.write_json", return_value=timestamped_output_path),
            ):
                mock_datetime.now.return_value = timestamp
                main()
        mock_evaluate.assert_called_once_with(
            real_image_dir="/tmp/real",
            generated_image_dir="/tmp/generated",
            batch_size=8,
            num_workers=0,
            inception_weights_path=DEFAULT_INCEPTION_WEIGHTS_PATH,
            clip_model_path=DEFAULT_CLIP_MODEL_PATH,
            biomedclip_model_path=DEFAULT_BIOMEDCLIP_MODEL_PATH,
            real_inception_cache_dir=timestamped_output_path.parent.resolve() / "cache",
            real_biomedclip_cache_dir=timestamped_output_path.parent.resolve() / "cache",
        )


if __name__ == "__main__":
    unittest.main()
