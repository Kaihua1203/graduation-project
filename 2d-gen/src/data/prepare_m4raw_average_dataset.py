from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from data.prepare_m4raw_dataset import (
    PROMPT_TEMPLATE,
    build_manifests,
    get_output_directories,
    normalize_to_uint8,
    reset_output_root,
    resize_grayscale_image,
)


AVERAGED_MODALITY_GROUPS = {
    "T1": ("T101", "T102", "T103"),
    "T2": ("T201", "T202", "T203"),
    "FLAIR": ("FLAIR01", "FLAIR02"),
}
DEFAULT_SOURCE_ROOT = Path("/NAS_data/M4RawV1.6/multicoil_train")
DEFAULT_OUTPUT_ROOT = Path("/home/jupyter-wenkaihua/data3_link/kaihua.wen/dataset/M4Raw-average")
DEFAULT_IMAGE_SIZE = 512
EXPECTED_CASES_PER_MODALITY = 128
OUTPUT_MODALITIES = tuple(sorted(AVERAGED_MODALITY_GROUPS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert averaged M4Raw repetitions into per-slice grayscale PNGs, prompts, and JSONL manifests."
    )
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--slice-index-start", type=int, default=0)
    parser.add_argument("--expected-cases-per-modality", type=int, default=EXPECTED_CASES_PER_MODALITY)
    parser.add_argument(
        "--skip-manifests",
        action="store_true",
        help="Skip writing train_<modality>.jsonl files. Useful for test-only dataset exports.",
    )
    return parser.parse_args()


def group_case_files(source_root: Path) -> dict[str, dict[str, tuple[Path, ...]]]:
    grouped_files: dict[str, dict[str, tuple[Path, ...]]] = {}
    for modality, repetitions in AVERAGED_MODALITY_GROUPS.items():
        case_groups: dict[str, tuple[Path, ...]] = {}
        first_repetition = repetitions[0]
        for first_path in sorted(source_root.glob(f"*_{first_repetition}.h5")):
            case_id = first_path.stem.rsplit("_", maxsplit=1)[0]
            repetition_paths = tuple(source_root / f"{case_id}_{repetition}.h5" for repetition in repetitions)
            case_groups[case_id] = repetition_paths
        grouped_files[modality] = case_groups
    return grouped_files


def validate_grouped_files(
    grouped_files: dict[str, dict[str, tuple[Path, ...]]],
    expected_cases_per_modality: int,
) -> None:
    errors: list[str] = []

    for modality, case_groups in grouped_files.items():
        case_count = len(case_groups)
        if case_count != expected_cases_per_modality:
            errors.append(f"{modality} cases={case_count}")

        repetitions = AVERAGED_MODALITY_GROUPS[modality]
        for case_id, repetition_paths in sorted(case_groups.items()):
            missing = [
                repetition
                for repetition, repetition_path in zip(repetitions, repetition_paths)
                if not repetition_path.is_file()
            ]
            if missing:
                errors.append(f"{case_id}_{modality} missing {','.join(missing)}")

    if errors:
        raise ValueError(
            "Unexpected averaged file groups. "
            f"Expected {expected_cases_per_modality} complete cases per modality but found: {', '.join(errors)}"
        )


def average_reconstruction_rss(repetition_paths: tuple[Path, ...]) -> np.ndarray:
    averaged_image: np.ndarray | None = None

    for repetition_path in repetition_paths:
        with h5py.File(repetition_path, "r") as handle:
            reconstruction_rss = np.asarray(handle["reconstruction_rss"], dtype=np.float32)
        if averaged_image is None:
            averaged_image = reconstruction_rss
        else:
            if reconstruction_rss.shape != averaged_image.shape:
                raise ValueError(
                    f"Shape mismatch while averaging {repetition_path}: "
                    f"expected {averaged_image.shape}, got {reconstruction_rss.shape}"
                )
            averaged_image += reconstruction_rss

    if averaged_image is None:
        raise ValueError("No repetitions were provided for averaging")

    return averaged_image / float(len(repetition_paths))


def process_case_group(
    case_id: str,
    modality: str,
    repetition_paths: tuple[Path, ...],
    output_root: Path,
    image_size: int,
    slice_index_start: int,
) -> int:
    averaged_volume = average_reconstruction_rss(repetition_paths)
    images_dir, prompts_dir = get_output_directories(output_root, modality)
    written_count = 0

    for slice_offset in range(averaged_volume.shape[0]):
        slice_id = slice_index_start + slice_offset
        stem = f"{case_id}_{modality}_{slice_id:02d}"
        normalized = normalize_to_uint8(averaged_volume[slice_offset])
        output_image = resize_grayscale_image(normalized, image_size)
        output_image.save(images_dir / f"{stem}.png")

        prompt = PROMPT_TEMPLATE.format(modality=modality, slice_id=slice_id)
        (prompts_dir / f"{stem}.txt").write_text(prompt, encoding="utf-8")
        written_count += 1

    return written_count


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not source_root.is_dir():
        raise ValueError(f"Source directory does not exist: {source_root}")

    grouped_files = group_case_files(source_root)
    validate_grouped_files(grouped_files, args.expected_cases_per_modality)
    reset_output_root(output_root, OUTPUT_MODALITIES)

    total_cases = 0
    total_slices = 0
    for modality in OUTPUT_MODALITIES:
        for case_id, repetition_paths in sorted(grouped_files[modality].items()):
            total_cases += 1
            total_slices += process_case_group(
                case_id=case_id,
                modality=modality,
                repetition_paths=repetition_paths,
                output_root=output_root,
                image_size=args.image_size,
                slice_index_start=args.slice_index_start,
            )

    print(f"Processed {total_cases} averaged case groups into {total_slices} PNG/TXT pairs under {output_root}")
    if args.skip_manifests:
        print("Skipped manifest generation")
        return

    manifest_paths = build_manifests(output_root, OUTPUT_MODALITIES)
    for manifest_path in manifest_paths:
        print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
