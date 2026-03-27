from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TimeState:
    timesteps: torch.Tensor | None
    sigmas: torch.Tensor | None = None
    guidance: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Conditioning:
    prompt_embeds: torch.Tensor | None = None
    pooled_prompt_embeds: torch.Tensor | None = None
    prompt_mask: torch.Tensor | None = None
    add_time_ids: torch.Tensor | None = None
    text_ids: torch.Tensor | None = None
    img_ids: torch.Tensor | None = None
    img_shapes: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    prompt: str
    seed: int


@dataclass
class MetricResult:
    fid: float
    inception_score_mean: float
    inception_score_std: float
    clip_i: float
    clip_t: float
