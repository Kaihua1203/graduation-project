from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageOnlyDataset(Dataset):
    def __init__(
        self,
        resolution: int,
        center_crop: bool = False,
        random_flip: bool = False,
        image_interpolation_mode: str = "bilinear",
        max_train_samples: int | None = None,
        manifest_path: str | Path | None = None,
        image_dir: str | Path | None = None,
        image_column: str = "image_path",
        allowed_extensions: list[str] | None = None,
    ) -> None:
        if bool(manifest_path) == bool(image_dir):
            raise ValueError("Exactly one of manifest_path or image_dir must be provided.")

        self.image_column = image_column
        self.allowed_extensions = tuple((allowed_extensions or [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]))
        self.records = self._load_records(
            manifest_path=Path(manifest_path).expanduser().resolve() if manifest_path else None,
            image_dir=Path(image_dir).expanduser().resolve() if image_dir else None,
        )
        if max_train_samples is not None:
            self.records = self.records[:max_train_samples]

        interpolation = getattr(transforms.InterpolationMode, image_interpolation_mode.upper(), None)
        if interpolation is None:
            raise ValueError(f"Unsupported interpolation mode: {image_interpolation_mode}")

        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=interpolation),
                transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda image: image),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def _load_records(self, manifest_path: Path | None, image_dir: Path | None) -> list[dict[str, Any]]:
        if manifest_path is not None:
            return self._load_manifest_records(manifest_path)
        assert image_dir is not None
        return self._load_directory_records(image_dir)

    def _load_manifest_records(self, manifest_path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                image_path = record.get(self.image_column)
                if not isinstance(image_path, str) or not image_path.strip():
                    raise ValueError(f"Manifest line {line_number} must include a non-empty {self.image_column}.")
                records.append({"image_path": str(Path(image_path).expanduser().resolve())})
        if not records:
            raise ValueError(f"No training records found in {manifest_path}")
        return records

    def _load_directory_records(self, image_dir: Path) -> list[dict[str, Any]]:
        if not image_dir.exists() or not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        records = [
            {"image_path": str(path.resolve())}
            for path in sorted(image_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in self.allowed_extensions
        ]
        if not records:
            raise ValueError(f"No supported image files found in {image_dir}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.records[index]["image_path"]
        image = Image.open(image_path).convert("RGB")
        return {
            "pixel_values": self.transform(image),
            "image_path": image_path,
        }


def collate_image_only_batch(examples: list[dict[str, Any]]) -> dict[str, Any]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples], dim=0)
    return {
        "pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float(),
        "image_path": [example["image_path"] for example in examples],
    }
