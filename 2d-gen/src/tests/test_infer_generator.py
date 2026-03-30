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


class _FakeStableDiffusion3Pipeline(_FakeStableDiffusionPipeline):
    last_instance: "_FakeStableDiffusion3Pipeline | None" = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "_FakeStableDiffusion3Pipeline":
        instance = cls()
        instance.from_pretrained_args = args
        instance.from_pretrained_kwargs = kwargs
        cls.last_instance = instance
        return instance


def _build_fake_partial_state_class(
    *,
    split_indices: list[int] | None = None,
    num_processes: int = 1,
    process_index: int = 0,
):
    class _FakePartialState:
        last_instance: "_FakePartialState | None" = None

        def __init__(self) -> None:
            self.device = generator.torch.device("cpu")
            self.num_processes = num_processes
            self.process_index = process_index
            self.is_main_process = process_index == 0
            self.split_inputs: list[list[int]] = []
            _FakePartialState.last_instance = self

        class _SplitContext:
            def __init__(self, indices: list[int]) -> None:
                self.indices = indices

            def __enter__(self) -> list[int]:
                return self.indices

            def __exit__(self, exc_type, exc, tb) -> None:
                del exc_type, exc, tb
                return None

        def split_between_processes(self, indices: list[int]):
            self.split_inputs.append(list(indices))
            if split_indices is None:
                selected_indices = list(indices)
            else:
                selected_indices = [index for index in split_indices if index in indices]
            return self._SplitContext(selected_indices)

        def wait_for_everyone(self) -> None:
            return None

    return _FakePartialState


class InferGeneratorTest(unittest.TestCase):
    def _run_fake_inference(
        self,
        prompts_path: Path,
        *,
        resume: bool = False,
        existing_indices: list[int] | None = None,
        split_indices: list[int] | None = None,
        num_processes: int = 1,
        process_index: int = 0,
        gpu_ids: list[int] | None = None,
        family: str = "stable_diffusion",
    ) -> tuple[list[dict[str, object]], list[str], list[list[int]]]:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "generated"
            config = {
                "model": {
                    "family": family,
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
            if gpu_ids is not None:
                config["infer"]["gpu_ids"] = gpu_ids

            output_dir.mkdir(parents=True, exist_ok=True)
            for index in existing_indices or []:
                prompt_records = generator.collect_prompt_records(prompts_path)
                image_path = generator._build_output_image_path(output_dir, index, prompt_records[index])
                Image.new("RGB", (4, 4), color=(255, 255, 255)).save(image_path)

            fake_partial_state = _build_fake_partial_state_class(
                split_indices=split_indices,
                num_processes=num_processes,
                process_index=process_index,
            )

            with (
                mock.patch.object(generator, "prepare_diffusers_import", lambda: None),
                mock.patch.object(generator.torch.cuda, "is_available", return_value=False),
                mock.patch.dict(
                    sys.modules,
                    {
                        "diffusers": types.SimpleNamespace(
                            StableDiffusionPipeline=_FakeStableDiffusionPipeline,
                            StableDiffusion3Pipeline=_FakeStableDiffusion3Pipeline,
                        ),
                        "accelerate": types.SimpleNamespace(PartialState=fake_partial_state),
                    },
                ),
            ):
                generator.run_stable_diffusion_inference(config, resume=resume)

            metadata_path = output_dir / "metadata.jsonl"
            if metadata_path.exists():
                records = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
            else:
                records = []
            output_files = sorted(path.name for path in output_dir.glob("*.png"))
            split_inputs = fake_partial_state.last_instance.split_inputs
            return records, output_files, split_inputs

    def test_single_file_inference_uses_line_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "prompts.txt"
            prompts_file.write_text("first prompt\n\nsecond prompt\n", encoding="utf-8")

            records, output_files, split_inputs = self._run_fake_inference(prompts_file)

        self.assertEqual(output_files, ["sample_00000.png", "sample_00001.png"])
        self.assertEqual([record["prompt"] for record in records], ["first prompt", "second prompt"])
        self.assertEqual([record["sample_index"] for record in records], [0, 1])
        self.assertEqual([record["prompt_source_kind"] for record in records], ["single_file", "single_file"])
        self.assertEqual([record["prompt_source_index"] for record in records], [1, 3])
        self.assertEqual([record["prompt_source_path"] for record in records], [str(prompts_file.resolve())] * 2)
        self.assertEqual([record["seed"] for record in records], [3407, 3408])
        self.assertEqual(split_inputs, [[0, 1]])
        self.assertEqual(_FakeStableDiffusionPipeline.last_instance.prompts, ["first prompt", "second prompt"])

    def test_directory_inference_is_sorted_by_prompt_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir) / "prompts"
            prompts_dir.mkdir()
            (prompts_dir / "b.txt").write_text("second prompt\n", encoding="utf-8")
            (prompts_dir / "a.txt").write_text("first prompt\n", encoding="utf-8")

            records, output_files, split_inputs = self._run_fake_inference(prompts_dir)

        self.assertEqual(output_files, ["a.png", "b.png"])
        self.assertEqual([record["prompt"] for record in records], ["first prompt", "second prompt"])
        self.assertEqual([record["sample_index"] for record in records], [0, 1])
        self.assertEqual([Path(record["image_path"]).name for record in records], ["a.png", "b.png"])
        self.assertEqual([record["prompt_source_kind"] for record in records], ["directory_file", "directory_file"])
        self.assertEqual([record["prompt_source_index"] for record in records], [1, 2])
        self.assertEqual(
            [record["prompt_source_path"] for record in records],
            [str((prompts_dir / "a.txt").resolve()), str((prompts_dir / "b.txt").resolve())],
        )
        self.assertEqual(split_inputs, [[0, 1]])
        self.assertEqual(_FakeStableDiffusionPipeline.last_instance.prompts, ["first prompt", "second prompt"])

    def test_resume_skips_existing_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "prompts.txt"
            prompts_file.write_text("first prompt\nsecond prompt\nthird prompt\n", encoding="utf-8")

            records, output_files, split_inputs = self._run_fake_inference(
                prompts_file,
                resume=True,
                existing_indices=[0, 1],
            )

        self.assertEqual(output_files, ["sample_00000.png", "sample_00001.png", "sample_00002.png"])
        self.assertEqual([record["sample_index"] for record in records], [2])
        self.assertEqual([record["prompt"] for record in records], ["third prompt"])
        self.assertEqual(split_inputs, [[2]])
        self.assertEqual(_FakeStableDiffusionPipeline.last_instance.prompts, ["third prompt"])

    def test_directory_resume_skips_existing_prompt_basename_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir) / "prompts"
            prompts_dir.mkdir()
            (prompts_dir / "a.txt").write_text("first prompt\n", encoding="utf-8")
            (prompts_dir / "b.txt").write_text("second prompt\n", encoding="utf-8")
            (prompts_dir / "c.txt").write_text("third prompt\n", encoding="utf-8")

            records, output_files, split_inputs = self._run_fake_inference(
                prompts_dir,
                resume=True,
                existing_indices=[0, 1],
            )

        self.assertEqual(output_files, ["a.png", "b.png", "c.png"])
        self.assertEqual([record["sample_index"] for record in records], [2])
        self.assertEqual([Path(record["image_path"]).name for record in records], ["c.png"])
        self.assertEqual(split_inputs, [[2]])
        self.assertEqual(_FakeStableDiffusionPipeline.last_instance.prompts, ["third prompt"])

    def test_inference_uses_process_split_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "prompts.txt"
            prompts_file.write_text("first prompt\nsecond prompt\nthird prompt\n", encoding="utf-8")

            records, output_files, split_inputs = self._run_fake_inference(
                prompts_file,
                split_indices=[1],
                gpu_ids=[0],
            )

        self.assertEqual(split_inputs, [[0, 1, 2]])
        self.assertEqual(output_files, ["sample_00001.png"])
        self.assertEqual([record["sample_index"] for record in records], [1])
        self.assertEqual(_FakeStableDiffusionPipeline.last_instance.prompts, ["second prompt"])

    def test_sd3_inference_uses_sd3_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = Path(tmpdir) / "prompts.txt"
            prompts_file.write_text("sd3 prompt\n", encoding="utf-8")

            records, output_files, split_inputs = self._run_fake_inference(
                prompts_file,
                family="stable_diffusion_3",
            )

        self.assertEqual(output_files, ["sample_00000.png"])
        self.assertEqual([record["prompt"] for record in records], ["sd3 prompt"])
        self.assertEqual(split_inputs, [[0]])
        self.assertEqual(_FakeStableDiffusion3Pipeline.last_instance.prompts, ["sd3 prompt"])


if __name__ == "__main__":
    unittest.main()
