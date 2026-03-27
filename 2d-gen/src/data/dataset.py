from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ManifestImagePromptDataset(Dataset):
    def __init__(self, manifest_path: str | Path, image_size: int) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.records = self._load_records(self.manifest_path)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
            ]
        )

    @staticmethod
    def _load_records(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if "image_path" not in record or "prompt" not in record:
                    raise ValueError(
                        f"Manifest line {line_number} must include image_path and prompt."
                    )
                records.append(record)
        if not records:
            raise ValueError(f"No training records found in {path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("RGB")
        image = self.transform(image)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        pixel_values = torch.tensor(image_array.tolist(), dtype=torch.float32).permute(2, 0, 1)
        pixel_values = (pixel_values - 0.5) / 0.5
        return {
            "pixel_values": pixel_values,
            "prompt": record["prompt"],
            "image_path": record["image_path"],
        }
