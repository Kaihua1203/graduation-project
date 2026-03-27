from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _collect_single_level_files(directory: str | Path) -> list[Path]:
    base_dir = Path(directory).expanduser().resolve()
    if not base_dir.is_dir():
        raise ValueError(f"Directory does not exist: {base_dir}")
    return sorted(path for path in base_dir.iterdir() if path.is_file())


def _build_stem_map(paths: list[Path], kind: str) -> dict[str, Path]:
    stem_map: dict[str, Path] = {}
    duplicate_stems: list[str] = []
    for path in paths:
        if path.stem in stem_map:
            duplicate_stems.append(path.stem)
            continue
        stem_map[path.stem] = path.resolve()
    if duplicate_stems:
        unique_stems = ", ".join(sorted(set(duplicate_stems)))
        raise ValueError(f"Duplicate {kind} stems found: {unique_stems}")
    return stem_map


def build_manifest_records(
    images_dir: str | Path,
    prompts_dir: str | Path,
) -> list[dict[str, str]]:
    image_paths = [
        path for path in _collect_single_level_files(images_dir) if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    prompt_paths = [
        path for path in _collect_single_level_files(prompts_dir) if path.suffix.lower() == ".txt"
    ]

    if not image_paths:
        raise ValueError(f"No supported image files found in {Path(images_dir).expanduser().resolve()}")
    image_map = _build_stem_map(image_paths, "image")
    prompt_map = _build_stem_map(prompt_paths, "prompt")

    missing_prompts = sorted(stem for stem in image_map if stem not in prompt_map)
    if missing_prompts:
        raise ValueError(f"Missing prompt files for image stems: {', '.join(missing_prompts)}")

    missing_images = sorted(stem for stem in prompt_map if stem not in image_map)
    if missing_images:
        raise ValueError(f"Missing image files for prompt stems: {', '.join(missing_images)}")

    records: list[dict[str, str]] = []
    for stem in sorted(image_map):
        prompt = prompt_map[stem].read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"Prompt file is empty: {prompt_map[stem]}")
        records.append(
            {
                "image_path": str(image_map[stem]),
                "prompt": prompt,
            }
        )
    return records


def write_manifest(
    images_dir: str | Path,
    prompts_dir: str | Path,
    output_path: str | Path,
) -> Path:
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    records = build_manifest_records(images_dir, prompts_dir)
    with output_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 2d-gen JSONL manifest from images/ and prompts/.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--prompts-dir", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = write_manifest(args.images_dir, args.prompts_dir, args.output_path)
    print(f"Wrote manifest to {output_path}")


if __name__ == "__main__":
    main()
