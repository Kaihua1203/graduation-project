from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import AutoTokenizer, CLIPModel

from common.constants import DEFAULT_CLIP_MODEL_PATH, DEFAULT_INCEPTION_WEIGHTS_PATH
from common.types import MetricResult


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
INCEPTION_CACHE_VERSION = "v1"


def list_images(directory: str | Path) -> list[Path]:
    image_dir = Path(directory).expanduser().resolve()
    images = sorted(
        [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    )
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    return images


def _log_progress(message: str) -> None:
    print(f"[eval] {message}", flush=True)


class ImageTensorDataset(Dataset[torch.Tensor]):
    def __init__(self, paths: list[Path], preprocess_fn: Callable[[Image.Image], torch.Tensor]) -> None:
        self.paths = paths
        self.preprocess_fn = preprocess_fn

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.paths[index]).convert("RGB")
        return self.preprocess_fn(image)


class ClipTPairDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(self, records: list[dict[str, str]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("RGB")
        return _preprocess_for_clip(image), record["prompt"]


class ClipIImagePairDataset(Dataset[torch.Tensor]):
    def __init__(self, paired_paths: list[tuple[Path, Path]]) -> None:
        self.paired_paths = paired_paths

    def __len__(self) -> int:
        return len(self.paired_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        real_path, generated_path = self.paired_paths[index]
        real_image = Image.open(real_path).convert("RGB")
        generated_image = Image.open(generated_path).convert("RGB")
        return torch.stack(
            [_preprocess_for_clip(real_image), _preprocess_for_clip(generated_image)],
            dim=0,
        )


def pair_clip_i_paths(real_paths: list[Path], generated_paths: list[Path]) -> list[tuple[Path, Path]]:
    if len(real_paths) != len(generated_paths):
        raise ValueError("CLIP-I requires the same number of real and generated images.")
    real_names = [path.name for path in real_paths]
    generated_names = [path.name for path in generated_paths]
    if real_names != generated_names:
        raise ValueError(
            "CLIP-I requires matching filenames between real and generated directories."
        )
    return list(zip(real_paths, generated_paths))


class InceptionBackbone(torch.nn.Module):
    def __init__(self, weights_path: str | Path) -> None:
        super().__init__()
        self.model = models.inception_v3(aux_logits=False, transform_input=False)
        state_dict = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self._captured_features: torch.Tensor | None = None
        self.model.avgpool.register_forward_hook(self._capture_avgpool)

    def _capture_avgpool(self, module, inputs, outputs) -> None:
        self._captured_features = torch.flatten(outputs, start_dim=1)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(images)
        if self._captured_features is None:
            raise RuntimeError("Inception hook did not capture pooled features.")
        features = self._captured_features
        self._captured_features = None
        return logits, features


def _preprocess_for_inception(image: Image.Image) -> torch.Tensor:
    resized = transforms.Resize((299, 299))(image.convert("RGB"))
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).to(dtype=torch.float32)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def _build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _iter_progress_steps(total_items: int, batch_size: int) -> tuple[int, int]:
    total_batches = max(1, (total_items + batch_size - 1) // batch_size)
    log_every = max(1, total_batches // 10)
    return total_batches, log_every


def _load_inception_backbone(
    inception_weights_path: str | Path,
) -> tuple[InceptionBackbone, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionBackbone(inception_weights_path).to(device)
    return model, device


def _build_inception_cache_path(
    image_paths: list[Path],
    inception_weights_path: str | Path,
    cache_dir: str | Path,
) -> Path:
    resolved_weights_path = Path(inception_weights_path).expanduser().resolve()
    signature = hashlib.sha256()
    signature.update(INCEPTION_CACHE_VERSION.encode("utf-8"))
    signature.update(str(resolved_weights_path).encode("utf-8"))
    for path in image_paths:
        resolved_path = path.resolve()
        stat = resolved_path.stat()
        signature.update(str(resolved_path).encode("utf-8"))
        signature.update(str(stat.st_size).encode("utf-8"))
        signature.update(str(stat.st_mtime_ns).encode("utf-8"))
    resolved_cache_dir = Path(cache_dir).expanduser().resolve()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    return resolved_cache_dir / f"real_inception_{signature.hexdigest()}.npz"


def _load_cached_inception_outputs(
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not cache_path.exists():
        return None
    _log_progress(f"Loading cached Inception outputs from {cache_path}")
    with np.load(cache_path) as data:
        return data["features"], data["probs"]


def _write_cached_inception_outputs(
    cache_path: Path,
    features: np.ndarray,
    probs: np.ndarray,
) -> None:
    np.savez_compressed(cache_path, features=features, probs=probs)
    _log_progress(f"Saved cached Inception outputs to {cache_path}")


@torch.no_grad()
def _extract_inception_features_and_probs(
    image_paths: list[Path],
    batch_size: int,
    num_workers: int,
    model: InceptionBackbone,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dataloader = _build_dataloader(
        ImageTensorDataset(image_paths, preprocess_fn=_preprocess_for_inception),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    total_batches, log_every = _iter_progress_steps(len(image_paths), batch_size)
    all_features = []
    all_probs = []
    for batch_index, batch in enumerate(dataloader, start=1):
        batch = batch.to(device, non_blocking=device.type == "cuda")
        logits, features = model(batch)
        probs = F.softmax(logits, dim=1)
        all_features.append(features.detach().cpu().to(dtype=torch.float64).numpy())
        all_probs.append(probs.detach().cpu().to(dtype=torch.float64).numpy())
        if batch_index == 1 or batch_index % log_every == 0 or batch_index == total_batches:
            _log_progress(f"Inception batches: {batch_index}/{total_batches}")
    return np.concatenate(all_features, axis=0), np.concatenate(all_probs, axis=0)


def compute_inception_features_and_probs(
    image_dir: str | Path,
    batch_size: int,
    num_workers: int,
    model: InceptionBackbone,
    device: torch.device,
    inception_weights_path: str | Path = DEFAULT_INCEPTION_WEIGHTS_PATH,
    cache_dir: str | Path | None = None,
    use_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    image_paths = list_images(image_dir)
    _log_progress(
        f"Inception feature extraction from {image_dir} on {device.type} for {len(image_paths)} images "
        f"({max(1, (len(image_paths) + batch_size - 1) // batch_size)} batches, batch_size={batch_size})"
    )
    if use_cache and cache_dir is not None:
        cache_path = _build_inception_cache_path(image_paths, inception_weights_path, cache_dir)
        cached_outputs = _load_cached_inception_outputs(cache_path)
        if cached_outputs is not None:
            return cached_outputs
        features, probs = _extract_inception_features_and_probs(
            image_paths=image_paths,
            batch_size=batch_size,
            num_workers=num_workers,
            model=model,
            device=device,
        )
        _write_cached_inception_outputs(cache_path, features, probs)
        return features, probs
    return _extract_inception_features_and_probs(
        image_paths=image_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        model=model,
        device=device,
    )


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    sigma1 = (sigma1 + sigma1.T) / 2.0
    sigma2 = (sigma2 + sigma2.T) / 2.0
    sigma1_eigvals, sigma1_eigvecs = np.linalg.eigh(sigma1)
    sigma1_eigvals = np.clip(sigma1_eigvals, a_min=0.0, a_max=None)
    sigma1_sqrt = sigma1_eigvecs @ np.diag(np.sqrt(sigma1_eigvals)) @ sigma1_eigvecs.T

    middle = sigma1_sqrt @ sigma2 @ sigma1_sqrt
    middle = (middle + middle.T) / 2.0
    middle_eigvals = np.linalg.eigvalsh(middle)
    middle_eigvals = np.clip(middle_eigvals, a_min=0.0, a_max=None)
    trace_covmean = float(np.sum(np.sqrt(middle_eigvals)))
    diff = mu1 - mu2
    fid = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * trace_covmean
    return float(fid)


def compute_fid(
    real_features: np.ndarray,
    generated_features: np.ndarray,
) -> float:
    real_mu = np.mean(real_features, axis=0)
    real_sigma = np.cov(real_features, rowvar=False)
    generated_mu = np.mean(generated_features, axis=0)
    generated_sigma = np.cov(generated_features, rowvar=False)
    return frechet_distance(real_mu, real_sigma, generated_mu, generated_sigma)


def compute_inception_score(
    probs: np.ndarray,
    num_splits: int = 5,
) -> tuple[float, float]:
    effective_splits = max(1, min(num_splits, len(probs)))
    probs = np.clip(probs, 1e-12, 1.0)
    split_scores = []
    for split in np.array_split(probs, effective_splits):
        if len(split) == 0:
            continue
        marginal = np.mean(split, axis=0, keepdims=True)
        kl = split * (np.log(split) - np.log(marginal))
        score = float(np.exp(np.mean(np.sum(kl, axis=1))))
        split_scores.append(score)
    return float(np.mean(split_scores)), float(np.std(split_scores))


def _preprocess_for_clip(image: Image.Image) -> torch.Tensor:
    resized = transforms.Resize((224, 224))(image.convert("RGB"))
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).to(dtype=torch.float32)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def _load_clip(clip_model_path: str | Path) -> tuple[CLIPModel, AutoTokenizer, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(
        str(clip_model_path),
        local_files_only=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(clip_model_path),
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer, device


def _load_manifest_records(manifest_path: str | Path) -> list[dict[str, str]]:
    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    records = []
    with resolved_manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No prompt/image pairs found in {resolved_manifest_path}")
    return records


@torch.no_grad()
def compute_clip_t(
    generated_manifest_path: str | Path,
    batch_size: int,
    num_workers: int,
    clip_model_path: str | Path = DEFAULT_CLIP_MODEL_PATH,
) -> tuple[float, float]:
    manifest_path = Path(generated_manifest_path).expanduser().resolve()
    model, tokenizer, device = _load_clip(clip_model_path)
    similarities = []
    records = _load_manifest_records(manifest_path)
    dataloader = _build_dataloader(
        ClipTPairDataset(records),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    total_records = len(records)
    total_batches, log_every = _iter_progress_steps(total_records, batch_size)
    _log_progress(f"CLIP-T from {manifest_path} on {device.type} for {total_records} prompt/image pairs")
    for batch_index, (pixel_values, prompts) in enumerate(dataloader, start=1):
        pixel_values = pixel_values.to(device, non_blocking=device.type == "cuda")
        text_inputs = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        image_embeds = F.normalize(model.get_image_features(pixel_values=pixel_values), dim=-1)
        text_embeds = F.normalize(model.get_text_features(**text_inputs), dim=-1)
        similarities.extend(torch.sum(image_embeds * text_embeds, dim=-1).detach().cpu().tolist())
        if batch_index == 1 or batch_index % log_every == 0 or batch_index == total_batches:
            _log_progress(f"CLIP-T batches: {batch_index}/{total_batches}")
    return float(np.mean(similarities)), float(np.std(similarities))


@torch.no_grad()
def compute_clip_i(
    real_image_dir: str | Path,
    generated_image_dir: str | Path,
    batch_size: int,
    num_workers: int,
    clip_model_path: str | Path = DEFAULT_CLIP_MODEL_PATH,
) -> tuple[float, float]:
    real_paths = list_images(real_image_dir)
    generated_paths = list_images(generated_image_dir)
    model, _, device = _load_clip(clip_model_path)
    similarities = []
    paired_paths = pair_clip_i_paths(real_paths, generated_paths)
    total_pairs = len(paired_paths)
    dataloader = _build_dataloader(
        ClipIImagePairDataset(paired_paths),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    total_batches, log_every = _iter_progress_steps(total_pairs, batch_size)
    _log_progress(f"CLIP-I on {device.type} for {total_pairs} aligned image pairs")
    for batch_index, pixel_values in enumerate(dataloader, start=1):
        pixel_values = pixel_values.to(device, non_blocking=device.type == "cuda")
        batch_size_current = pixel_values.shape[0]
        flattened_pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
        image_embeds = F.normalize(model.get_image_features(pixel_values=flattened_pixel_values), dim=-1)
        image_embeds = image_embeds.view(batch_size_current, 2, -1)
        similarities.extend(
            torch.sum(image_embeds[:, 0] * image_embeds[:, 1], dim=-1).detach().cpu().tolist()
        )
        if batch_index == 1 or batch_index % log_every == 0 or batch_index == total_batches:
            _log_progress(f"CLIP-I batches: {batch_index}/{total_batches}")
    return float(np.mean(similarities)), float(np.std(similarities))


def evaluate_generation_quality(
    real_image_dir: str | Path,
    generated_image_dir: str | Path,
    generated_manifest_path: str | Path,
    batch_size: int,
    num_workers: int = 0,
    inception_weights_path: str | Path = DEFAULT_INCEPTION_WEIGHTS_PATH,
    clip_model_path: str | Path = DEFAULT_CLIP_MODEL_PATH,
    real_inception_cache_dir: str | Path | None = None,
) -> MetricResult:
    if num_workers < 0:
        raise ValueError("eval.num_workers must be non-negative.")
    inception_model, inception_device = _load_inception_backbone(inception_weights_path)
    _log_progress("Starting FID")
    real_features, _ = compute_inception_features_and_probs(
        image_dir=real_image_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        model=inception_model,
        device=inception_device,
        inception_weights_path=inception_weights_path,
        cache_dir=real_inception_cache_dir,
        use_cache=real_inception_cache_dir is not None,
    )
    generated_features, generated_probs = compute_inception_features_and_probs(
        image_dir=generated_image_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        model=inception_model,
        device=inception_device,
        inception_weights_path=inception_weights_path,
    )
    fid = compute_fid(real_features, generated_features)
    _log_progress("Starting Inception Score")
    is_mean, is_std = compute_inception_score(generated_probs)
    _log_progress("Starting CLIP-I")
    clip_i_mean, clip_i_std = compute_clip_i(
        real_image_dir,
        generated_image_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        clip_model_path=clip_model_path,
    )
    _log_progress("Starting CLIP-T")
    clip_t_mean, clip_t_std = compute_clip_t(
        generated_manifest_path,
        batch_size=batch_size,
        num_workers=num_workers,
        clip_model_path=clip_model_path,
    )
    _log_progress("Finished evaluation")
    return MetricResult(
        fid=round(fid, 2),
        inception_score_mean=round(is_mean, 2),
        inception_score_std=round(is_std, 2),
        clip_i_mean=round(clip_i_mean, 2),
        clip_i_std=round(clip_i_std, 2),
        clip_t_mean=round(clip_t_mean, 2),
        clip_t_std=round(clip_t_std, 2),
    )
