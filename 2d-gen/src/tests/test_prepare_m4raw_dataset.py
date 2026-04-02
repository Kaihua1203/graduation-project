from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from data.prepare_m4raw_dataset import (
    PROMPT_TEMPLATE,
    SOURCE_MODALITY_TO_OUTPUT,
    build_manifests,
    iter_selected_files,
    normalize_to_uint8,
    process_file,
    validate_selected_files,
)


class PrepareM4RawDatasetTest(unittest.TestCase):
    def test_normalize_to_uint8_handles_constant_image(self) -> None:
        image = np.ones((4, 4), dtype=np.float32)
        normalized = normalize_to_uint8(image)
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertTrue(np.array_equal(normalized, np.zeros((4, 4), dtype=np.uint8)))

    def test_process_file_writes_pngs_prompts_and_manifest(self) -> None:
        rng = np.random.default_rng(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "source"
            output_root = root / "output"
            source_root.mkdir()

            file_path = source_root / "2022061001_T101.h5"
            kspace = (
                rng.standard_normal((2, 4, 8, 8)) + 1j * rng.standard_normal((2, 4, 8, 8))
            ).astype(np.complex64)

            with h5py.File(file_path, "w") as handle:
                handle.create_dataset("kspace", data=kspace)

            slice_count = process_file(file_path, output_root, image_size=32, slice_index_start=0)
            self.assertEqual(slice_count, 2)

            modality = SOURCE_MODALITY_TO_OUTPUT["T101"]
            images_dir = output_root / modality / "images"
            prompts_dir = output_root / modality / "prompts"
            image_paths = sorted(images_dir.glob("*.png"))
            prompt_paths = sorted(prompts_dir.glob("*.txt"))
            self.assertEqual([path.name for path in image_paths], ["2022061001_T1_00.png", "2022061001_T1_01.png"])
            self.assertEqual([path.name for path in prompt_paths], ["2022061001_T1_00.txt", "2022061001_T1_01.txt"])

            with Image.open(image_paths[0]) as image:
                self.assertEqual(image.mode, "L")
                self.assertEqual(image.size, (32, 32))
                image_array = np.asarray(image)
                self.assertGreaterEqual(int(image_array.min()), 0)
                self.assertLessEqual(int(image_array.max()), 255)

            prompt = prompt_paths[1].read_text(encoding="utf-8")
            self.assertEqual(prompt, PROMPT_TEMPLATE.format(modality="T1", slice_id=1))

            manifest_paths = build_manifests(output_root)
            self.assertEqual(manifest_paths, [output_root / "train_T1.jsonl"])

            records = [json.loads(line) for line in (output_root / "train_T1.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["image_path"], str((images_dir / "2022061001_T1_00.png").resolve()))
            self.assertEqual(records[0]["prompt"], PROMPT_TEMPLATE.format(modality="T1", slice_id=0))

    def test_iter_selected_files_filters_only_requested_modalities(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir)
            for name in [
                "2022061001_T101.h5",
                "2022061001_T201.h5",
                "2022061001_FLAIR01.h5",
                "2022061001_T102.h5",
            ]:
                (source_root / name).write_bytes(b"")

            selected = iter_selected_files(source_root)
            self.assertEqual(
                [path.name for path in selected],
                ["2022061001_FLAIR01.h5", "2022061001_T101.h5", "2022061001_T201.h5"],
            )

    def test_validate_selected_files_requires_expected_count_per_modality(self) -> None:
        selected_files = [Path(f"/tmp/2022061001_{modality}.h5") for modality in SOURCE_MODALITY_TO_OUTPUT]

        with self.assertRaisesRegex(ValueError, "Unexpected selected file counts"):
            validate_selected_files(selected_files, expected_files_per_modality=2)


if __name__ == "__main__":
    unittest.main()
