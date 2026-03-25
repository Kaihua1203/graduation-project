import argparse
import json
import random
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import monai
import numpy as np
import torch
import torch.nn as nn
import yaml
from monai.data import DataLoader, Dataset, decollate_batch, list_data_collate
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
from monai.networks.nets import FlexibleUNet, UNet
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandFlipd,
    ScaleIntensityd,
)
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sanitize_experiment_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    cleaned = cleaned.strip(".-")
    return cleaned or "seg-training"


def list_pairs(images_dir: Path, masks_dir: Path, image_suffix: str, mask_suffix: str) -> List[Dict[str, str]]:
    images = sorted(images_dir.glob(f"*{image_suffix}"))
    pairs: List[Dict[str, str]] = []
    for image_path in images:
        stem = image_path.stem
        mask_path = masks_dir / f"{stem}{mask_suffix}"
        if mask_path.exists():
            pairs.append({"image": str(image_path), "label": str(mask_path)})
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in {images_dir} and {masks_dir}.")
    return pairs


def parse_gpu_ids(cfg: Dict, cli_gpus: Optional[str]) -> List[int]:
    if cli_gpus:
        return [int(x.strip()) for x in cli_gpus.split(",") if x.strip()]
    return list(cfg["train"].get("gpu_ids", [0]))


def build_transforms(cfg: Dict) -> Tuple[Compose, Compose]:
    roi_size = tuple(cfg["data"]["roi_size"])
    train_num_samples = int(cfg["data"]["train_num_samples"])

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=train_num_samples,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return train_transforms, val_transforms


def get_encoder(module: nn.Module) -> nn.Module:
    if hasattr(module, "encoder"):
        return module.encoder
    if hasattr(module, "module") and hasattr(module.module, "encoder"):
        return module.module.encoder
    raise RuntimeError("Model encoder attribute not found.")


def has_encoder(module: nn.Module) -> bool:
    if hasattr(module, "encoder"):
        return True
    if hasattr(module, "module") and hasattr(module.module, "encoder"):
        return True
    return False


def build_model(cfg: Dict, device: torch.device) -> nn.Module:
    model_cfg = cfg["model"]
    model_name = model_cfg.get("name", "flexible_unet").lower()
    in_channels = int(cfg["data"]["in_channels"])
    out_channels = int(cfg["data"]["num_classes"])

    if model_name == "flexible_unet":
        backbone_name = model_cfg["backbone_name"]
        model = FlexibleUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            backbone=backbone_name,
            pretrained=False,
            spatial_dims=2,
        )
    elif model_name == "unet":
        channels = tuple(model_cfg.get("unet_channels", [32, 64, 128, 256, 512]))
        strides = tuple(model_cfg.get("unet_strides", [2, 2, 2, 2]))
        model = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=int(model_cfg.get("unet_num_res_units", 2)),
        )
    elif model_name in {"vit", "unetr", "swinunetr"}:
        raise ValueError(
            f"model.name={model_name} is reserved but not implemented in this script yet. "
            "Use flexible_unet/unet now, or extend build_model() for ViT-based segmentation."
        )
    else:
        raise ValueError(f"Unsupported model.name: {model_name}")

    return model.to(device)


def prepare_optimizer(model: nn.Module, cfg: Dict, train_encoder: bool) -> torch.optim.Optimizer:
    lr_encoder = float(cfg["train"]["lr_encoder"])
    lr_decoder = float(cfg["train"]["lr_decoder"])
    weight_decay = float(cfg["train"]["weight_decay"])

    if not has_encoder(model):
        lr = float(cfg["train"].get("lr", lr_decoder))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    encoder = get_encoder(model)
    encoder_ids = {id(p) for p in encoder.parameters()}
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]

    if train_encoder:
        return torch.optim.AdamW(
            [
                {"params": list(encoder.parameters()), "lr": lr_encoder},
                {"params": decoder_params, "lr": lr_decoder},
            ],
            weight_decay=weight_decay,
        )

    return torch.optim.AdamW(decoder_params, lr=lr_decoder, weight_decay=weight_decay)


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    if not has_encoder(model):
        return
    encoder = get_encoder(model)
    for p in encoder.parameters():
        p.requires_grad = trainable


def load_pretrained_encoder(model: nn.Module, ckpt_path: Path) -> None:
    if not has_encoder(model):
        raise RuntimeError("Current model has no encoder attribute; cannot load pretrained encoder.")
    payload = torch.load(ckpt_path, map_location="cpu")
    if isinstance(payload, dict) and "encoder_state_dict" in payload:
        state = payload["encoder_state_dict"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    elif isinstance(payload, dict):
        state = payload
    else:
        raise RuntimeError("Unsupported pretrained encoder file format.")

    clean_state = {}
    for key, value in state.items():
        k = key
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("backbone."):
            k = k[len("backbone.") :]
        if k.startswith("encoder."):
            k = k[len("encoder.") :]
        clean_state[k] = value

    encoder = get_encoder(model)
    missing, unexpected = encoder.load_state_dict(clean_state, strict=False)
    print(f"Loaded pretrained encoder from: {ckpt_path}")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")


def init_swanlab(cfg: Dict):
    sw_cfg = cfg.get("swanlab", {})
    if not sw_cfg.get("enabled", False):
        return None, None

    try:
        import swanlab
    except ImportError:
        print("swanlab is not installed. Continue without swanlab logging.")
        return None, None

    kwargs = {
        "project": sw_cfg.get("project", "2d-ssl-seg"),
        "experiment_name": sw_cfg.get("experiment_name", "seg-training"),
        "description": sw_cfg.get("description", ""),
        "config": cfg,
    }
    if "logdir" in sw_cfg:
        kwargs["logdir"] = sw_cfg["logdir"]
    run = swanlab.init(**kwargs)
    return run, swanlab


def save_epoch_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: Dict,
    epoch: int,
    best_dice: float,
    history: List[Dict[str, Any]],
    row: Dict[str, Any],
) -> None:
    save_model = model.module if isinstance(model, nn.DataParallel) else model
    payload: Dict[str, Any] = {
        "model_state_dict": save_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": cfg,
        "epoch": epoch,
        "best_dice": best_dice,
        "history": history,
    }
    payload.update(row)
    torch.save(payload, checkpoint_path)


def load_resume_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> Tuple[int, float, List[Dict[str, Any]], bool]:
    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError(f"Invalid resume checkpoint: {ckpt_path}")

    save_model = model.module if isinstance(model, nn.DataParallel) else model
    save_model.load_state_dict(payload["model_state_dict"], strict=True)

    optimizer_loaded = False
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        optimizer_loaded = True
    if "scaler_state_dict" in payload:
        scaler.load_state_dict(payload["scaler_state_dict"])

    start_epoch = int(payload.get("epoch", 0))
    best_dice = float(payload.get("best_dice", payload.get("val_dice", -1.0)))
    raw_history = payload.get("history", [])
    history: List[Dict[str, Any]] = raw_history if isinstance(raw_history, list) else []
    print(f"Resumed from checkpoint: {ckpt_path}")
    print(f"Resume epoch: {start_epoch}, best_dice: {best_dice:.6f}, history_len: {len(history)}")
    return start_epoch, best_dice, history, optimizer_loaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MONAI 2D segmentation with SSL or random init.")
    parser.add_argument("--config", required=True, type=str, help="Path to segmentation YAML config.")
    parser.add_argument("--pretrained-encoder", default=None, type=str, help="Optional override path.")
    parser.add_argument("--output-dir", default=None, type=str, help="Optional override output directory.")
    parser.add_argument("--random-init", action="store_true", help="Force random initialization.")
    parser.add_argument("--gpus", default=None, type=str, help='GPU ids, e.g. "0" or "0,1".')
    parser.add_argument("--experiment-name", default=None, type=str, help="Override swanlab experiment name.")
    parser.add_argument("--save-every", default=None, type=int, help="Save epoch checkpoint every N epochs.")
    parser.add_argument("--resume-from", default=None, type=str, help="Resume checkpoint path, e.g. epoch-20.pt.")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    set_seed(int(cfg["seed"]))

    if args.experiment_name is not None:
        cfg.setdefault("swanlab", {})["experiment_name"] = args.experiment_name

    experiment_name = sanitize_experiment_name(
        str(cfg.get("swanlab", {}).get("experiment_name", "seg-training"))
    )
    output_root = Path(args.output_dir or cfg["output_dir"])
    output_dir = output_root / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    sw_run, swanlab_mod = init_swanlab(cfg)

    train_pairs = list_pairs(
        Path(cfg["data"]["train_images_dir"]),
        Path(cfg["data"]["train_masks_dir"]),
        cfg["data"]["image_suffix"],
        cfg["data"]["mask_suffix"],
    )
    val_pairs = list_pairs(
        Path(cfg["data"]["val_images_dir"]),
        Path(cfg["data"]["val_masks_dir"]),
        cfg["data"]["image_suffix"],
        cfg["data"]["mask_suffix"],
    )

    train_tfms, val_tfms = build_transforms(cfg)
    train_ds = Dataset(train_pairs, transform=train_tfms)
    val_ds = Dataset(val_pairs, transform=val_tfms)

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    gpu_ids = parse_gpu_ids(cfg, args.gpus)
    use_cuda = torch.cuda.is_available() and len(gpu_ids) > 0
    device = torch.device(f"cuda:{gpu_ids[0]}" if use_cuda else "cpu")

    model = build_model(cfg, device)

    use_pretrained = bool(cfg["pretrained_encoder"]["enabled"]) and not args.random_init
    if args.pretrained_encoder is not None:
        use_pretrained = True
        cfg["pretrained_encoder"]["path"] = args.pretrained_encoder

    if use_pretrained:
        load_pretrained_encoder(model, Path(cfg["pretrained_encoder"]["path"]))
    else:
        print("Using random initialization for encoder.")

    if use_cuda and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"Using DataParallel on GPUs: {gpu_ids}")

    freeze_epochs = int(cfg["train"]["freeze_encoder_epochs"]) if has_encoder(model) else 0
    set_encoder_trainable(model, trainable=False if freeze_epochs > 0 else True)
    optimizer = prepare_optimizer(model, cfg, train_encoder=(freeze_epochs == 0))

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
    num_classes = int(cfg["data"]["num_classes"])
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    miou_metric = MeanIoU(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(
        include_background=False,
        percentile=95.0,
        reduction="mean",
    )
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)

    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"]["amp"]) and device.type == "cuda")
    max_epochs = int(cfg["train"]["epochs"])
    val_interval = int(cfg["train"]["val_interval"])

    history: List[Dict[str, Any]] = []
    best_dice = -1.0
    best_path = output_dir / "best_model.pt"
    if args.save_every is not None:
        save_every = max(1, int(args.save_every))
    else:
        save_every = max(1, int(cfg["train"].get("save_every", 20)))
    start_epoch = 0

    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        start_epoch, best_dice, history, optimizer_loaded = load_resume_checkpoint(
            resume_path, model, optimizer, scaler
        )
        if start_epoch < freeze_epochs:
            set_encoder_trainable(model, trainable=False)
        else:
            set_encoder_trainable(model, trainable=True)
            if freeze_epochs > 0 and not optimizer_loaded:
                optimizer = prepare_optimizer(model, cfg, train_encoder=True)

    epoch_bar = tqdm(range(start_epoch, max_epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        if epoch == freeze_epochs and freeze_epochs > 0:
            tqdm.write(f"Epoch {epoch}: unfreezing encoder and rebuilding optimizer.")
            set_encoder_trainable(model, trainable=True)
            optimizer = prepare_optimizer(model, cfg, train_encoder=True)

        model.train()
        train_loss = 0.0
        n_steps = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1:>4}/{max_epochs} [train]", leave=False, unit="batch") as batch_bar:
            for batch in batch_bar:
                images = batch["image"].to(device)
                labels = batch["label"].long().to(device)
                if labels.ndim == 4 and labels.shape[1] >= 1:
                    labels = labels[:, 0]

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=bool(cfg["train"]["amp"]) and device.type == "cuda"):
                    logits = model(images)
                    loss = loss_fn(logits, labels.unsqueeze(1))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += float(loss.item())
                n_steps += 1
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / max(1, n_steps)
        row = {"epoch": epoch + 1, "train_loss": avg_train_loss}

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="the ground truth of class")
                    warnings.filterwarnings("ignore", message="the prediction of class")
                    for batch in val_loader:
                        images = batch["image"].to(device)
                        labels = batch["label"].long().to(device)
                        if labels.ndim == 4 and labels.shape[1] >= 1:
                            labels = labels[:, 0]

                        logits = model(images)
                        y_pred = [post_pred(p) for p in decollate_batch(logits)]
                        y_true = [post_label(t) for t in decollate_batch(labels.unsqueeze(1))]

                        dice_metric(y_pred=y_pred, y=y_true)
                        miou_metric(y_pred=y_pred, y=y_true)
                        hd95_metric(y_pred=y_pred, y=y_true)

                mean_dice = float(dice_metric.aggregate().item())
                mean_miou = float(miou_metric.aggregate().item())
                mean_hd95 = float(hd95_metric.aggregate().item())
                dice_metric.reset()
                miou_metric.reset()
                hd95_metric.reset()

                row.update({"val_dice": mean_dice, "val_miou": mean_miou, "val_hd95": mean_hd95})

                if mean_dice > best_dice:
                    best_dice = mean_dice
                    save_model = model.module if isinstance(model, nn.DataParallel) else model
                    torch.save(
                        {
                            "model_state_dict": save_model.state_dict(),
                            "config": cfg,
                            "epoch": epoch + 1,
                            "val_dice": mean_dice,
                            "val_miou": mean_miou,
                            "val_hd95": mean_hd95,
                        },
                        best_path,
                    )

        history.append(row)

        if (epoch + 1) % save_every == 0:
            epoch_ckpt = output_dir / f"epoch-{epoch + 1}.pt"
            save_epoch_checkpoint(
                checkpoint_path=epoch_ckpt,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                cfg=cfg,
                epoch=epoch + 1,
                best_dice=best_dice,
                history=history,
                row=row,
            )
            tqdm.write(f"Saved epoch checkpoint: {epoch_ckpt}")

        postfix: Dict = {"loss": f"{avg_train_loss:.4f}"}
        if "val_dice" in row:
            postfix["dice"] = f"{row['val_dice']:.4f}"
            postfix["hd95"] = f"{row['val_hd95']:.1f}"
        epoch_bar.set_postfix(postfix)
        tqdm.write(
            f"Epoch {epoch+1:>4}/{max_epochs}  loss={avg_train_loss:.4f}"
            + (f"  val_dice={row['val_dice']:.4f}  val_miou={row['val_miou']:.4f}  val_hd95={row['val_hd95']:.2f}" if "val_dice" in row else "")
        )

        if sw_run is not None and swanlab_mod is not None:
            sw_row = {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in row.items()}
            swanlab_mod.log(sw_row, step=epoch + 1)

    with (output_dir / "metrics_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with (output_dir / "train_config_dump.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Training complete. Best Dice: {best_dice:.6f}")
    print(f"Best model path: {best_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Num foregrounds 0")
    monai.config.print_config()
    main()
