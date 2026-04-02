from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import fastmri
import fastmri.data.transforms as T
import h5py
import numpy as np
from PIL import Image

from data.manifest_builder import write_manifest


SOURCE_MODALITY_TO_OUTPUT = {
    "T101": "T1",
    "T201": "T2",
    "FLAIR01": "FLAIR",
}
PROMPT_TEMPLATE = "a {modality} brain MRI slice at slice position {slice_id}, acquired with a low-field 0.3T scanner"
DEFAULT_SOURCE_ROOT = Path("/NAS_data/M4RawV1.6/multicoil_train")
DEFAULT_OUTPUT_ROOT = Path("/home/jupyter-wenkaihua/data3_link/kaihua.wen/dataset/M4raw/train")
DEFAULT_IMAGE_SIZE = 512
EXPECTED_FILES_PER_MODALITY = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert selected M4Raw multicoil files into per-slice grayscale PNGs, prompts, and JSONL manifests."
    )
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--slice-index-start", type=int, default=0)
    parser.add_argument("--expected-files-per-modality", type=int, default=EXPECTED_FILES_PER_MODALITY)
    return parser.parse_args()


def reconstruct_rss_image(slice_kspace: np.ndarray) -> np.ndarray:
    slice_kspace_tensor = T.to_tensor(slice_kspace)
    slice_image = fastmri.ifft2c(slice_kspace_tensor)
    slice_image_abs = fastmri.complex_abs(slice_image)
    slice_image_rss = fastmri.rss(slice_image_abs, dim=0)
    return slice_image_rss.numpy()


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image_min = float(image.min())
    image_max = float(image.max())
    if image_max <= image_min:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (image - image_min) / (image_max - image_min)
    return np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)


def resize_grayscale_image(image: np.ndarray, image_size: int) -> Image.Image:
    pil_image = Image.fromarray(image, mode="L")
    return pil_image.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)


def get_output_directories(output_root: Path, modality: str) -> tuple[Path, Path]:
    modality_root = output_root / modality
    images_dir = modality_root / "images"
    prompts_dir = modality_root / "prompts"
    images_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, prompts_dir


def iter_selected_files(source_root: Path) -> list[Path]:
    selected_files: list[Path] = []
    for source_modality in SOURCE_MODALITY_TO_OUTPUT:
        selected_files.extend(sorted(source_root.glob(f"*_{source_modality}.h5")))
    return sorted(selected_files)


def validate_selected_files(selected_files: list[Path], expected_files_per_modality: int) -> None:
    counts_by_modality = {source_modality: 0 for source_modality in SOURCE_MODALITY_TO_OUTPUT}
    for file_path in selected_files:
        _, source_modality = file_path.stem.split("_", maxsplit=1)
        counts_by_modality[source_modality] += 1

    invalid_modalities = [
        f"{source_modality}={count}"
        for source_modality, count in sorted(counts_by_modality.items())
        if count != expected_files_per_modality
    ]
    if invalid_modalities:
        raise ValueError(
            "Unexpected selected file counts. "
            f"Expected {expected_files_per_modality} per modality but found: {', '.join(invalid_modalities)}"
        )


def reset_output_root(output_root: Path) -> None:
    for modality in sorted(set(SOURCE_MODALITY_TO_OUTPUT.values())):
        modality_root = output_root / modality
        if modality_root.exists():
            shutil.rmtree(modality_root)
        manifest_path = output_root / f"train_{modality}.jsonl"
        manifest_path.unlink(missing_ok=True)


def process_file(
    file_path: Path,
    output_root: Path,
    image_size: int,
    slice_index_start: int,
) -> int:
    case_id, source_modality = file_path.stem.split("_", maxsplit=1)
    modality = SOURCE_MODALITY_TO_OUTPUT[source_modality]
    images_dir, prompts_dir = get_output_directories(output_root, modality)
    written_count = 0

    with h5py.File(file_path, "r") as handle:
        kspace = handle["kspace"]
        for slice_offset in range(kspace.shape[0]):
            slice_id = slice_index_start + slice_offset
            stem = f"{case_id}_{modality}_{slice_id:02d}"
            slice_rss = reconstruct_rss_image(kspace[slice_offset])
            normalized = normalize_to_uint8(slice_rss)
            output_image = resize_grayscale_image(normalized, image_size)
            output_image.save(images_dir / f"{stem}.png")

            prompt = PROMPT_TEMPLATE.format(modality=modality, slice_id=slice_id)
            (prompts_dir / f"{stem}.txt").write_text(prompt, encoding="utf-8")
            written_count += 1

    return written_count


def build_manifests(output_root: Path) -> list[Path]:
    manifest_paths: list[Path] = []
    for modality in sorted(set(SOURCE_MODALITY_TO_OUTPUT.values())):
        images_dir, prompts_dir = get_output_directories(output_root, modality)
        if not any(images_dir.iterdir()):
            continue
        output_path = output_root / f"train_{modality}.jsonl"
        manifest_paths.append(write_manifest(images_dir, prompts_dir, output_path))
    return manifest_paths


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not source_root.is_dir():
        raise ValueError(f"Source directory does not exist: {source_root}")

    selected_files = iter_selected_files(source_root)
    if not selected_files:
        raise ValueError(f"No matching M4Raw files found in {source_root}")
    validate_selected_files(selected_files, args.expected_files_per_modality)
    reset_output_root(output_root)

    total_slices = 0
    for file_path in selected_files:
        total_slices += process_file(file_path, output_root, args.image_size, args.slice_index_start)

    manifest_paths = build_manifests(output_root)

    print(f"Processed {len(selected_files)} files into {total_slices} PNG/TXT pairs under {output_root}")
    for manifest_path in manifest_paths:
        print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
