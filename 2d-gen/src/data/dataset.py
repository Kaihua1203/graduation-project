from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ManifestImagePromptDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        resolution: int,
        image_column: str = "image_path",
        caption_column: str = "prompt",
        center_crop: bool = False,
        random_flip: bool = False,
        image_interpolation_mode: str = "bilinear",
        max_train_samples: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.image_column = image_column
        self.caption_column = caption_column
        self.records = self._load_records(self.manifest_path, image_column, caption_column)
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

    @staticmethod
    def _load_records(path: Path, image_column: str, caption_column: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if image_column not in record or caption_column not in record:
                    raise ValueError(
                        f"Manifest line {line_number} must include {image_column} and {caption_column}."
                    )
                records.append(record)
        if not records:
            raise ValueError(f"No training records found in {path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = Image.open(record[self.image_column]).convert("RGB")
        pixel_values = self.transform(image)
        return {
            "pixel_values": pixel_values,
            "prompt": record[self.caption_column],
            "image_path": record[self.image_column],
        }
