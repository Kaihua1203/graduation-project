from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from common.config import load_yaml_config
from data.dataset import ManifestImagePromptDataset


class ConfigAndDatasetSmokeTest(unittest.TestCase):
    def test_load_yaml_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("model:\n  family: stable_diffusion\n", encoding="utf-8")
            config = load_yaml_config(config_path)
            self.assertEqual(config["model"]["family"], "stable_diffusion")

    def test_manifest_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            image_path = tmpdir_path / "sample.png"
            Image.new("RGB", (32, 32), color=(128, 64, 32)).save(image_path)

            manifest_path = tmpdir_path / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps({"image_path": str(image_path), "prompt": "toy prompt"}) + "\n",
                encoding="utf-8",
            )

            dataset = ManifestImagePromptDataset(manifest_path, image_size=32)
            sample = dataset[0]
            self.assertEqual(sample["prompt"], "toy prompt")
            self.assertEqual(tuple(sample["pixel_values"].shape), (3, 32, 32))


if __name__ == "__main__":
    unittest.main()
