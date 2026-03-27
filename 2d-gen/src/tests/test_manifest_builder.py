from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from data.manifest_builder import build_manifest_records, write_manifest


class ManifestBuilderTest(unittest.TestCase):
    def test_build_manifest_records_pairs_by_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            prompts_dir = root / "prompts"
            images_dir.mkdir()
            prompts_dir.mkdir()

            Image.new("RGB", (16, 16), color=(1, 2, 3)).save(images_dir / "0002.png")
            Image.new("RGB", (16, 16), color=(4, 5, 6)).save(images_dir / "0001.png")
            (prompts_dir / "0001.txt").write_text("first prompt\n", encoding="utf-8")
            (prompts_dir / "0002.txt").write_text("second prompt\n", encoding="utf-8")

            records = build_manifest_records(images_dir, prompts_dir)
            self.assertEqual(
                records,
                [
                    {"image_path": str((images_dir / "0001.png").resolve()), "prompt": "first prompt"},
                    {"image_path": str((images_dir / "0002.png").resolve()), "prompt": "second prompt"},
                ],
            )

    def test_write_manifest_raises_for_missing_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            prompts_dir = root / "prompts"
            output_path = root / "manifest.jsonl"
            images_dir.mkdir()
            prompts_dir.mkdir()

            Image.new("RGB", (16, 16), color=(1, 2, 3)).save(images_dir / "0001.png")

            with self.assertRaisesRegex(ValueError, "Missing prompt files"):
                write_manifest(images_dir, prompts_dir, output_path)

    def test_write_manifest_raises_for_missing_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            prompts_dir = root / "prompts"
            output_path = root / "manifest.jsonl"
            images_dir.mkdir()
            prompts_dir.mkdir()

            (prompts_dir / "0001.txt").write_text("prompt only", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "No supported image files found"):
                write_manifest(images_dir, prompts_dir, output_path)

    def test_write_manifest_raises_for_empty_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            prompts_dir = root / "prompts"
            output_path = root / "manifest.jsonl"
            images_dir.mkdir()
            prompts_dir.mkdir()

            Image.new("RGB", (16, 16), color=(1, 2, 3)).save(images_dir / "0001.png")
            (prompts_dir / "0001.txt").write_text("\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Prompt file is empty"):
                write_manifest(images_dir, prompts_dir, output_path)

    def test_write_manifest_raises_for_extra_prompt_without_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            prompts_dir = root / "prompts"
            output_path = root / "manifest.jsonl"
            images_dir.mkdir()
            prompts_dir.mkdir()

            Image.new("RGB", (16, 16), color=(1, 2, 3)).save(images_dir / "0001.png")
            (prompts_dir / "0001.txt").write_text("matched prompt", encoding="utf-8")
            (prompts_dir / "0002.txt").write_text("unmatched prompt", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Missing image files"):
                write_manifest(images_dir, prompts_dir, output_path)

    def test_write_manifest_raises_for_duplicate_image_stems(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            prompts_dir = root / "prompts"
            output_path = root / "manifest.jsonl"
            images_dir.mkdir()
            prompts_dir.mkdir()

            Image.new("RGB", (16, 16), color=(1, 2, 3)).save(images_dir / "0001.png")
            Image.new("RGB", (16, 16), color=(4, 5, 6)).save(images_dir / "0001.jpg")
            (prompts_dir / "0001.txt").write_text("duplicate image stem", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Duplicate image stems"):
                write_manifest(images_dir, prompts_dir, output_path)

    def test_write_manifest_creates_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_dir = root / "images"
            prompts_dir = root / "prompts"
            output_path = root / "manifest.jsonl"
            images_dir.mkdir()
            prompts_dir.mkdir()

            Image.new("RGB", (16, 16), color=(1, 2, 3)).save(images_dir / "case.png")
            (prompts_dir / "case.txt").write_text("portal venous phase", encoding="utf-8")

            write_manifest(images_dir, prompts_dir, output_path)
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(
                json.loads(lines[0]),
                {
                    "image_path": str((images_dir / "case.png").resolve()),
                    "prompt": "portal venous phase",
                },
            )


if __name__ == "__main__":
    unittest.main()
