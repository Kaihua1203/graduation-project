from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from infer import run_infer_uncond_ldm
from uncond_ldm import checkpointing
from uncond_ldm.pipeline import UnconditionalLatentDiffusionPipeline


class _FakeComponentBase:
    def __init__(self, name: str) -> None:
        self.name = name
        self.loaded_from: str | None = None
        self.local_files_only: bool | None = None

    @classmethod
    def from_pretrained(cls, path, local_files_only=True):
        instance = cls(cls.__name__)
        instance.loaded_from = str(Path(path).resolve())
        instance.local_files_only = local_files_only
        cls.last_instance = instance
        return instance


class _FakeAutoencoderKL(_FakeComponentBase):
    last_instance: "_FakeAutoencoderKL | None" = None


class _FakeUNet2DModel(_FakeComponentBase):
    last_instance: "_FakeUNet2DModel | None" = None


class _FakeDDPMScheduler(_FakeComponentBase):
    last_instance: "_FakeDDPMScheduler | None" = None


class _FakeDDIMScheduler(_FakeComponentBase):
    last_instance: "_FakeDDIMScheduler | None" = None


class _FakeUNet:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(in_channels=4)
        self.dtype = torch.float32

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, latents, timestep):
        del timestep
        return types.SimpleNamespace(sample=torch.zeros_like(latents))


class _FakeScheduler:
    def __init__(self) -> None:
        self.init_noise_sigma = 1.0
        self.timesteps = []

    def set_timesteps(self, num_inference_steps, device=None):
        del device
        self.timesteps = list(range(num_inference_steps, 0, -1))

    def scale_model_input(self, latents, timestep):
        del timestep
        return latents

    def step(self, noise_pred, timestep, latents):
        del noise_pred, timestep
        return types.SimpleNamespace(prev_sample=latents * 0.5)


class _FakeVAE:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(scaling_factor=0.5, block_out_channels=[1, 2, 4, 8])

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def decode(self, latents):
        sample = torch.nn.functional.interpolate(latents[:, :3], scale_factor=8, mode="nearest")
        return types.SimpleNamespace(sample=sample)


class UncondLdmCheckpointingTest(unittest.TestCase):
    def test_load_export_bundle_paths_requires_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "bundle"
            bundle_dir.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "Export bundle is missing required artifacts"):
                checkpointing.load_export_bundle_paths(bundle_dir)

    def test_load_inference_components_uses_metadata_vae_and_scheduler_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "bundle"
            (bundle_dir / "unet").mkdir(parents=True)
            (bundle_dir / "scheduler").mkdir()
            (bundle_dir / "scheduler" / "scheduler_config.json").write_text(
                json.dumps({"_class_name": "DDPMScheduler"}),
                encoding="utf-8",
            )
            (bundle_dir / "model_metadata.json").write_text(
                json.dumps({"vae": {"pretrained_model_name_or_path": str(Path(tmpdir) / "vae")}}),
                encoding="utf-8",
            )
            (bundle_dir / "training_summary.json").write_text(json.dumps({"global_step": 10}), encoding="utf-8")

            bundle = checkpointing.load_export_bundle_paths(bundle_dir)
            with (
                mock.patch.object(checkpointing, "prepare_diffusers_import", lambda: None),
                mock.patch.dict(
                    sys.modules,
                    {
                        "diffusers": types.SimpleNamespace(
                            AutoencoderKL=_FakeAutoencoderKL,
                            DDIMScheduler=_FakeDDIMScheduler,
                            DDPMScheduler=_FakeDDPMScheduler,
                            UNet2DModel=_FakeUNet2DModel,
                        )
                    },
                ),
            ):
                checkpointing.load_inference_components(bundle, scheduler_type="ddim", local_files_only=False)

        self.assertEqual(_FakeUNet2DModel.last_instance.loaded_from, str((bundle_dir / "unet").resolve()))
        self.assertEqual(_FakeDDIMScheduler.last_instance.loaded_from, str((bundle_dir / "scheduler").resolve()))
        self.assertEqual(_FakeAutoencoderKL.last_instance.loaded_from, str((Path(tmpdir) / "vae").resolve()))
        self.assertFalse(_FakeAutoencoderKL.last_instance.local_files_only)


class UncondLdmPipelineTest(unittest.TestCase):
    def test_pipeline_generates_expected_number_of_images(self) -> None:
        pipeline = UnconditionalLatentDiffusionPipeline(
            unet=_FakeUNet(),
            scheduler=_FakeScheduler(),
            vae=_FakeVAE(),
            device=torch.device("cpu"),
            image_size=64,
        )

        images = pipeline.generate(seeds=[3407, 3408], num_inference_steps=3)

        self.assertEqual(len(images), 2)
        self.assertEqual(images[0].size, (64, 64))
        self.assertEqual(images[1].mode, "RGB")


class RunInferUncondLdmTest(unittest.TestCase):
    def test_run_unconditional_inference_aligns_filenames_and_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            reference_dir = tmpdir_path / "reference"
            reference_dir.mkdir()
            from PIL import Image

            for name in ("b.png", "a.png", "c.png"):
                Image.new("RGB", (8, 8), color=(255, 255, 255)).save(reference_dir / name)

            output_dir = tmpdir_path / "outputs"
            config = {
                "model": {
                    "export_bundle_path": str(tmpdir_path / "bundle"),
                    "vae_path": str(tmpdir_path / "vae"),
                    "local_files_only": True,
                },
                "infer": {
                    "reference_image_dir": str(reference_dir),
                    "output_dir": str(output_dir),
                    "batch_size": 2,
                    "num_inference_steps": 5,
                    "seed": 100,
                    "image_size": 16,
                    "scheduler": {"type": "ddim"},
                },
            }

            class _StubPipeline:
                def __init__(self):
                    self.calls: list[tuple[list[int], int]] = []

                def generate(self, *, seeds, num_inference_steps):
                    from PIL import Image

                    self.calls.append((list(seeds), num_inference_steps))
                    return [Image.new("RGB", (16, 16), color=(seed, 0, 0)) for seed in seeds]

            stub_pipeline = _StubPipeline()
            stub_bundle = types.SimpleNamespace(
                bundle_dir=(tmpdir_path / "bundle").resolve(),
                training_summary_path=(tmpdir_path / "bundle" / "training_summary.json").resolve(),
            )

            with mock.patch.object(run_infer_uncond_ldm, "build_pipeline", return_value=(stub_pipeline, stub_bundle)):
                result = run_infer_uncond_ldm.run_unconditional_inference(config)

            output_names = sorted(path.name for path in output_dir.glob("*.png"))
            metadata_records = [
                json.loads(line) for line in (output_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(output_names, ["a.png", "b.png", "c.png"])
        self.assertEqual([Path(record["generated_image_path"]).name for record in metadata_records], output_names)
        self.assertEqual([Path(record["reference_image_path"]).name for record in metadata_records], output_names)
        self.assertEqual([record["sample_index"] for record in metadata_records], [0, 1, 2])
        self.assertEqual([record["seed"] for record in metadata_records], [100, 101, 102])
        self.assertEqual(stub_pipeline.calls, [([100, 101], 5), ([102], 5)])
        self.assertEqual(result["num_generated"], 3)

    def test_resume_preserves_existing_outputs_in_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            from PIL import Image

            tmpdir_path = Path(tmpdir)
            reference_dir = tmpdir_path / "reference"
            output_dir = tmpdir_path / "outputs"
            reference_dir.mkdir()
            output_dir.mkdir()
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(reference_dir / "a.png")
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(reference_dir / "b.png")
            Image.new("RGB", (8, 8), color=(0, 0, 0)).save(output_dir / "a.png")
            (output_dir / "metadata.jsonl").write_text(
                json.dumps(
                    {
                        "generated_image_path": str((output_dir / "a.png").resolve()),
                        "reference_image_path": str((reference_dir / "a.png").resolve()),
                        "sample_index": 0,
                        "seed": 77,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = {
                "model": {
                    "export_bundle_path": str(tmpdir_path / "bundle"),
                    "local_files_only": True,
                },
                "infer": {
                    "reference_image_dir": str(reference_dir),
                    "output_dir": str(output_dir),
                    "batch_size": 4,
                    "num_inference_steps": 5,
                    "seed": 77,
                    "image_size": 16,
                },
            }

            class _StubPipeline:
                def generate(self, *, seeds, num_inference_steps):
                    del num_inference_steps
                    return [Image.new("RGB", (16, 16), color=(seed, 0, 0)) for seed in seeds]

            stub_bundle = types.SimpleNamespace(
                bundle_dir=(tmpdir_path / "bundle").resolve(),
                training_summary_path=(tmpdir_path / "bundle" / "training_summary.json").resolve(),
            )
            with mock.patch.object(run_infer_uncond_ldm, "build_pipeline", return_value=(_StubPipeline(), stub_bundle)):
                run_infer_uncond_ldm.run_unconditional_inference(config, resume=True)

            records = [
                json.loads(line) for line in (output_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([Path(record["generated_image_path"]).name for record in records], ["a.png", "b.png"])
        self.assertEqual([record["seed"] for record in records], [77, 78])


if __name__ == "__main__":
    unittest.main()
