# Config Parameter Guide

This file explains all tunable parameters under `configs/`.

## 1) SSL Config: `ssl/vicreg_lits.yaml`

### Core method

- `name`: experiment name shown in logs/checkpoints.
- `method`: SSL method, currently `vicreg`.
- `backbone.name`: encoder backbone (`resnet18` or `resnet50` recommended).

### VICReg loss

- `method_kwargs.proj_hidden_dim`: projector hidden dimension.
- `method_kwargs.proj_output_dim`: projector output dimension.
- `method_kwargs.sim_loss_weight`: invariance term weight.
- `method_kwargs.var_loss_weight`: variance term weight.
- `method_kwargs.cov_loss_weight`: covariance term weight.

### Data

- `data.train_path`: unlabeled image folder path for SSL.
- `data.val_path`: optional validation folder; keep `null` for no validation.
- `data.format`: `image_folder` for regular folder-based data.
- `data.no_labels`: `true` for unlabeled custom dataset.
- `data.num_workers`: dataloader workers.

### Optimizer / Scheduler

- `optimizer.batch_size`: per-GPU batch size.
- `optimizer.lr`: base learning rate (auto-scaled by total batch size in solo-learn).
- `optimizer.weight_decay`: weight decay.
- `scheduler.name`: LR scheduler, default `warmup_cosine`.

### Checkpointing

- `checkpoint.enabled`: enable saving checkpoints.
- `checkpoint.dir`: output root directory.
- `checkpoint.frequency`: save every N epochs (currently 20).
- `checkpoint.keep_prev`: keep all periodic checkpoints when `true`.

### Logging

- `wandb.enabled`: W&B switch (kept disabled here).
- `swanlab.enabled`: enable SwanLab logger.
- `swanlab.project`: SwanLab project name.
- `swanlab.experiment_name`: experiment/run name.
- `swanlab.description`: run description.
- `swanlab.logdir`: local swan log directory.

### Runtime

- `max_epochs`: total SSL epochs.
- `devices`: GPU list, e.g. `[0]` or `[0,1]`.
- `strategy`: `auto` (single GPU) or `ddp` (multi GPU).
- `sync_batchnorm`: recommended `true` when multi-GPU DDP.
- `precision`: mixed precision mode.

## 2) SSL Augmentations: `ssl/augmentations/lits_asymmetric.yaml`

Two augmentation branches are defined for multi-view SSL.

Adjustable groups:

- `rrc`: crop enable + scale range.
- `color_jitter`: probability and intensity.
- `gaussian_blur`: blur probability.
- `solarization`: usually 0 for medical grayscale-like data.
- `horizontal_flip`: flip probability.
- `crop_size`: input crop resolution.
- `num_crops`: number of views from this branch.

## 3) Seg Configs: `seg/train_ssl.yaml` and `seg/train_random.yaml`

### Data paths and task definition

- `data.train_images_dir`, `data.train_masks_dir`
- `data.val_images_dir`, `data.val_masks_dir`
- `data.image_suffix`, `data.mask_suffix`
- `data.in_channels`: input channels (LiTS PNG is RGB, set to 3).
- `data.num_classes`: number of segmentation classes (LiTS: 3).
- `data.roi_size`: patch size for random crop training.
- `data.train_num_samples`: crops sampled per image in each training iteration.

### Model

- `model.name`:
  - `flexible_unet` (default): MONAI `FlexibleUNet` with configurable backbone.
  - `unet`: MONAI plain `UNet` (no backbone-style encoder mapping).
- `model.backbone_name`: used by `flexible_unet` (e.g. `resnet18`, `resnet50`).
- `model.unet_channels`: used by `unet`, default `[32, 64, 128, 256, 512]`.
- `model.unet_strides`: used by `unet`, default `[2, 2, 2, 2]`.
- `model.unet_num_res_units`: used by `unet`, default `2`.

Notes:
- SSL encoder loading requires a model with `.encoder` (e.g. `flexible_unet`).
- `vit`/`unetr`/`swinunetr` are reserved names in script but not implemented yet.

### Encoder initialization

- `pretrained_encoder.enabled`: whether to load SSL encoder.
- `pretrained_encoder.path`: path to extracted encoder `.pth`.

### Training control

- `train.epochs`: total epochs (currently 100).
- `train.freeze_encoder_epochs`: stage-A frozen-encoder epochs.
- `train.batch_size`: mini-batch size.
- `train.num_workers`: dataloader workers.
- `train.amp`: mixed precision training.
- `train.val_interval`: validate every N epochs.
- `train.lr_encoder`: LR for encoder (stage-B).
- `train.lr_decoder`: LR for decoder/head.
- `train.weight_decay`: optimizer weight decay.
- `train.gpu_ids`: GPU ids for DataParallel, e.g. `[0]` or `[0,1]`.

### SwanLab for segmentation

- `swanlab.enabled`: turn SwanLab logging on/off.
- `swanlab.project`, `swanlab.experiment_name`, `swanlab.description`, `swanlab.logdir`.

## 4) Recommended safe tuning order

1. `train.batch_size`, `train.gpu_ids`, `data.roi_size` (fit memory first).
2. `train.lr_encoder`, `train.lr_decoder`, `train.freeze_encoder_epochs`.
3. SSL `backbone.name` and `max_epochs`.
4. SSL augmentation strengths (`color_jitter`, `rrc` scale, `gaussian_blur`).

Keep only one major variable change per run for fair comparisons.
