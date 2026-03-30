from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from transformers import AutoTokenizer, CLIPModel

from common.constants import DEFAULT_CLIP_MODEL_PATH, DEFAULT_INCEPTION_WEIGHTS_PATH
from common.types import MetricResult


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


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
    tensor = torch.tensor(image_array.tolist(), dtype=torch.float32).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def _iter_image_batches(paths: list[Path], batch_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = torch.stack(
            [_preprocess_for_inception(Image.open(path)) for path in batch_paths],
            dim=0,
        )
        yield batch


def _iter_progress_steps(total_items: int, batch_size: int) -> tuple[int, int]:
    total_batches = max(1, (total_items + batch_size - 1) // batch_size)
    log_every = max(1, total_batches // 10)
    return total_batches, log_every


def compute_inception_features_and_probs(
    image_dir: str | Path,
    batch_size: int,
    inception_weights_path: str | Path = DEFAULT_INCEPTION_WEIGHTS_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionBackbone(inception_weights_path).to(device)
    image_paths = list_images(image_dir)
    total_batches, log_every = _iter_progress_steps(len(image_paths), batch_size)
    _log_progress(
        f"Inception feature extraction from {image_dir} on {device.type} for {len(image_paths)} images "
        f"({total_batches} batches, batch_size={batch_size})"
    )
    all_features = []
    all_probs = []
    for batch_index, batch in enumerate(_iter_image_batches(image_paths, batch_size=batch_size), start=1):
        batch = batch.to(device)
        logits, features = model(batch)
        probs = F.softmax(logits, dim=1)
        all_features.append(np.asarray(features.detach().cpu().float().tolist(), dtype=np.float64))
        all_probs.append(np.asarray(probs.detach().cpu().float().tolist(), dtype=np.float64))
        if batch_index == 1 or batch_index % log_every == 0 or batch_index == total_batches:
            _log_progress(f"Inception batches: {batch_index}/{total_batches}")
    return np.concatenate(all_features, axis=0), np.concatenate(all_probs, axis=0)


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
    real_image_dir: str | Path,
    generated_image_dir: str | Path,
    batch_size: int,
    inception_weights_path: str | Path = DEFAULT_INCEPTION_WEIGHTS_PATH,
) -> float:
    real_features, _ = compute_inception_features_and_probs(
        real_image_dir, batch_size, inception_weights_path
    )
    generated_features, _ = compute_inception_features_and_probs(
        generated_image_dir, batch_size, inception_weights_path
    )
    real_mu = np.mean(real_features, axis=0)
    real_sigma = np.cov(real_features, rowvar=False)
    generated_mu = np.mean(generated_features, axis=0)
    generated_sigma = np.cov(generated_features, rowvar=False)
    return frechet_distance(real_mu, real_sigma, generated_mu, generated_sigma)


def compute_inception_score(
    generated_image_dir: str | Path,
    batch_size: int,
    inception_weights_path: str | Path = DEFAULT_INCEPTION_WEIGHTS_PATH,
    num_splits: int = 5,
) -> tuple[float, float]:
    _, probs = compute_inception_features_and_probs(
        generated_image_dir, batch_size, inception_weights_path
    )
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
    tensor = torch.tensor(image_array.tolist(), dtype=torch.float32).permute(2, 0, 1)
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


@torch.no_grad()
def compute_clip_t(
    generated_manifest_path: str | Path,
    clip_model_path: str | Path = DEFAULT_CLIP_MODEL_PATH,
) -> float:
    manifest_path = Path(generated_manifest_path).expanduser().resolve()
    model, tokenizer, device = _load_clip(clip_model_path)
    similarities = []
    lines = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total_records = len(lines)
    log_every = max(1, total_records // 10) if total_records else 1
    _log_progress(f"CLIP-T from {manifest_path} on {device.type} for {total_records} prompt/image pairs")
    with manifest_path.open("r", encoding="utf-8") as handle:
        for record_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            image = Image.open(record["image_path"]).convert("RGB")
            pixel_values = _preprocess_for_clip(image).unsqueeze(0).to(device)
            text_inputs = tokenizer(
                [record["prompt"]],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
            image_embeds = F.normalize(model.get_image_features(pixel_values=pixel_values), dim=-1)
            text_embeds = F.normalize(model.get_text_features(**text_inputs), dim=-1)
            similarities.append(
                float(torch.sum(image_embeds * text_embeds, dim=-1).mean().cpu().item())
            )
            if record_index == 1 or record_index % log_every == 0 or record_index == total_records:
                _log_progress(f"CLIP-T pairs: {record_index}/{total_records}")
    if not similarities:
        raise ValueError(f"No prompt/image pairs found in {manifest_path}")
    return float(np.mean(similarities))


@torch.no_grad()
def compute_clip_i(
    real_image_dir: str | Path,
    generated_image_dir: str | Path,
    clip_model_path: str | Path = DEFAULT_CLIP_MODEL_PATH,
) -> float:
    real_paths = list_images(real_image_dir)
    generated_paths = list_images(generated_image_dir)
    model, tokenizer, device = _load_clip(clip_model_path)
    similarities = []
    paired_paths = pair_clip_i_paths(real_paths, generated_paths)
    total_pairs = len(paired_paths)
    log_every = max(1, total_pairs // 10) if total_pairs else 1
    _log_progress(f"CLIP-I on {device.type} for {total_pairs} aligned image pairs")
    for pair_index, (real_path, generated_path) in enumerate(paired_paths, start=1):
        real_image = Image.open(real_path).convert("RGB")
        generated_image = Image.open(generated_path).convert("RGB")
        pixel_values = torch.stack(
            [_preprocess_for_clip(real_image), _preprocess_for_clip(generated_image)],
            dim=0,
        ).to(device)
        image_embeds = F.normalize(model.get_image_features(pixel_values=pixel_values), dim=-1)
        similarities.append(
            float(torch.sum(image_embeds[0] * image_embeds[1], dim=-1).cpu().item())
        )
        if pair_index == 1 or pair_index % log_every == 0 or pair_index == total_pairs:
            _log_progress(f"CLIP-I pairs: {pair_index}/{total_pairs}")
    return float(np.mean(similarities))


def evaluate_generation_quality(
    real_image_dir: str | Path,
    generated_image_dir: str | Path,
    generated_manifest_path: str | Path,
    batch_size: int,
    inception_weights_path: str | Path = DEFAULT_INCEPTION_WEIGHTS_PATH,
    clip_model_path: str | Path = DEFAULT_CLIP_MODEL_PATH,
) -> MetricResult:
    _log_progress("Starting FID")
    fid = compute_fid(real_image_dir, generated_image_dir, batch_size, inception_weights_path)
    _log_progress("Starting Inception Score")
    is_mean, is_std = compute_inception_score(
        generated_image_dir, batch_size, inception_weights_path
    )
    _log_progress("Starting CLIP-I")
    clip_i = compute_clip_i(real_image_dir, generated_image_dir, clip_model_path)
    _log_progress("Starting CLIP-T")
    clip_t = compute_clip_t(generated_manifest_path, clip_model_path)
    _log_progress("Finished evaluation")
    return MetricResult(
        fid=fid,
        inception_score_mean=is_mean,
        inception_score_std=is_std,
        clip_i=clip_i,
        clip_t=clip_t,
    )
