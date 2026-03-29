from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from infer import generator


class _FakeStableDiffusionPipeline:
    last_instance: "_FakeStableDiffusionPipeline | None" = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "_FakeStableDiffusionPipeline":
        instance = cls()
        instance.from_pretrained_args = args
        instance.from_pretrained_kwargs = kwargs
        cls.last_instance = instance
        return instance

    def to(self, device):
        self.device = device
        return self

    def load_lora_weights(self, path):
        self.lora_path = path

    def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        color = (len(self.prompts), 0, 0)
        image = Image.new("RGB", (4, 4), color=color)
        return types.SimpleNamespace(images=[image])

    def __init__(self) -> None:
        self.prompts: list[str] = []


class InferGeneratorTest(unittest.TestCase):
    def _run_fake_inference(self, prompts_path: Path) -> tuple[list[dict[str, object]], list[str]]:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "generated"
            config = {
                "model": {
                    "family": "stable_diffusion",
                    "pretrained_path": "/tmp/model",
                    "lora_path": "/tmp/lora",
                    "local_files_only": True,
                },
                "infer": {
                    "prompts_path": str(prompts_path),
                    "output_dir": str(output_dir),
                    "num_inference_steps": 2,
                    "guidance_scale": 7.5,
                    "height": 32,
                    "width": 32,
                    "seed": 3407,
                },
            }

            with (
                mock.patch.object(generator, "prepare_diffusers_import", lambda: None),
                mock.patch.object(generator.torch.cuda, "is_available", return_value=False),
                mock.patch.dict(
                    sys.modules,
                    {"diffusers": types.SimpleNamespace(StableDiffusionPipeline=_FakeStableDiffusionPipeline)},
                ),
            ):
                generator.run_stable_diffusion_inference(config)

            metadata_path = output_dir / "metadata.jsonl"
            records = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
            output_files = sorted(path.name for path in output_dir.glob("*.png"))
            return records, output_files

    def test_single_file_inference_uses_line_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "prompts.txt"
            prompts_file.write_text("first prompt\n\nsecond prompt\n", encoding="utf-8")

            records, output_files = self._run_fake_inference(prompts_file)

        self.assertEqual(output_files, ["sample_00000.png", "sample_00001.png"])
        self.assertEqual([record["prompt"] for record in records], ["first prompt", "second prompt"])
        self.assertEqual([record["prompt_source_kind"] for record in records], ["single_file", "single_file"])
        self.assertEqual([record["prompt_source_index"] for record in records], [1, 3])
        self.assertEqual([record["prompt_source_path"] for record in records], [str(prompts_file.resolve())] * 2)
        self.assertEqual(_FakeStableDiffusionPipeline.last_instance.prompts, ["first prompt", "second prompt"])

    def test_directory_inference_is_sorted_by_prompt_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir) / "prompts"
            prompts_dir.mkdir()
            (prompts_dir / "b.txt").write_text("second prompt\n", encoding="utf-8")
            (prompts_dir / "a.txt").write_text("first prompt\n", encoding="utf-8")

            records, output_files = self._run_fake_inference(prompts_dir)

        self.assertEqual(output_files, ["sample_00000.png", "sample_00001.png"])
        self.assertEqual([record["prompt"] for record in records], ["first prompt", "second prompt"])
        self.assertEqual([record["prompt_source_kind"] for record in records], ["directory_file", "directory_file"])
        self.assertEqual([record["prompt_source_index"] for record in records], [1, 2])
        self.assertEqual(
            [record["prompt_source_path"] for record in records],
            [str((prompts_dir / "a.txt").resolve()), str((prompts_dir / "b.txt").resolve())],
        )
        self.assertEqual(_FakeStableDiffusionPipeline.last_instance.prompts, ["first prompt", "second prompt"])


if __name__ == "__main__":
    unittest.main()
