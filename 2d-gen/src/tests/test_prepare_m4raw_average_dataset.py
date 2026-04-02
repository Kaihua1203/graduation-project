from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from data.prepare_m4raw_average_dataset import (
    AVERAGED_MODALITY_GROUPS,
    average_reconstruction_rss,
    group_case_files,
    parse_args,
    process_case_group,
    validate_grouped_files,
)
from data.prepare_m4raw_dataset import PROMPT_TEMPLATE, build_manifests


class PrepareM4RawAverageDatasetTest(unittest.TestCase):
    def test_average_reconstruction_rss_computes_elementwise_mean(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.h5"
            second = root / "second.h5"

            with h5py.File(first, "w") as handle:
                handle.create_dataset("reconstruction_rss", data=np.full((2, 4, 4), 2.0, dtype=np.float32))
            with h5py.File(second, "w") as handle:
                handle.create_dataset("reconstruction_rss", data=np.full((2, 4, 4), 6.0, dtype=np.float32))

            averaged = average_reconstruction_rss((first, second))
            self.assertTrue(np.allclose(averaged, np.full((2, 4, 4), 4.0, dtype=np.float32)))

    def test_process_case_group_writes_pngs_prompts_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "output"
            repetitions = []
            volumes = [
                np.arange(2 * 6 * 6, dtype=np.float32).reshape(2, 6, 6),
                np.arange(2 * 6 * 6, dtype=np.float32).reshape(2, 6, 6) + 10.0,
                np.arange(2 * 6 * 6, dtype=np.float32).reshape(2, 6, 6) + 20.0,
            ]

            for index, repetition in enumerate(("T101", "T102", "T103")):
                file_path = root / f"2022061001_{repetition}.h5"
                with h5py.File(file_path, "w") as handle:
                    handle.create_dataset("reconstruction_rss", data=volumes[index])
                repetitions.append(file_path)

            slice_count = process_case_group(
                case_id="2022061001",
                modality="T1",
                repetition_paths=tuple(repetitions),
                output_root=output_root,
                image_size=32,
                slice_index_start=0,
            )
            self.assertEqual(slice_count, 2)

            images_dir = output_root / "T1" / "images"
            prompts_dir = output_root / "T1" / "prompts"
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

            manifest_paths = build_manifests(output_root, ("T1",))
            self.assertEqual(manifest_paths, [output_root / "train_T1.jsonl"])

            records = [json.loads(line) for line in (output_root / "train_T1.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["image_path"], str((images_dir / "2022061001_T1_00.png").resolve()))
            self.assertEqual(records[0]["prompt"], PROMPT_TEMPLATE.format(modality="T1", slice_id=0))

    def test_group_case_files_collects_expected_repetitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir)
            for name in [
                "2022061001_T101.h5",
                "2022061001_T102.h5",
                "2022061001_T103.h5",
                "2022061001_T201.h5",
                "2022061001_T202.h5",
                "2022061001_T203.h5",
                "2022061001_FLAIR01.h5",
                "2022061001_FLAIR02.h5",
                "2022061001_FLAIR03.h5",
            ]:
                (source_root / name).write_bytes(b"")

            grouped = group_case_files(source_root)
            self.assertEqual(
                tuple(path.name for path in grouped["T1"]["2022061001"]),
                tuple(f"2022061001_{rep}.h5" for rep in AVERAGED_MODALITY_GROUPS["T1"]),
            )
            self.assertEqual(
                tuple(path.name for path in grouped["T2"]["2022061001"]),
                tuple(f"2022061001_{rep}.h5" for rep in AVERAGED_MODALITY_GROUPS["T2"]),
            )
            self.assertEqual(
                tuple(path.name for path in grouped["FLAIR"]["2022061001"]),
                tuple(f"2022061001_{rep}.h5" for rep in AVERAGED_MODALITY_GROUPS["FLAIR"]),
            )

    def test_validate_grouped_files_requires_complete_case_groups(self) -> None:
        grouped_files = {
            "T1": {"2022061001": tuple(Path(f"/tmp/2022061001_{rep}.h5") for rep in AVERAGED_MODALITY_GROUPS["T1"])},
            "T2": {"2022061001": tuple(Path(f"/tmp/2022061001_{rep}.h5") for rep in AVERAGED_MODALITY_GROUPS["T2"])},
            "FLAIR": {
                "2022061001": tuple(Path(f"/tmp/2022061001_{rep}.h5") for rep in AVERAGED_MODALITY_GROUPS["FLAIR"])
            },
        }

        with self.assertRaisesRegex(ValueError, "Unexpected averaged file groups"):
            validate_grouped_files(grouped_files, expected_cases_per_modality=2)

    def test_parse_args_supports_skip_manifests_flag(self) -> None:
        import sys

        argv = sys.argv
        try:
            sys.argv = ["prepare_m4raw_average_dataset.py", "--skip-manifests"]
            args = parse_args()
        finally:
            sys.argv = argv

        self.assertTrue(args.skip_manifests)


if __name__ == "__main__":
    unittest.main()
