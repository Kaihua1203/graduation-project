import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import monai
import torch
from monai.data import DataLoader, Dataset, decollate_batch, list_data_collate
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
from monai.networks.nets import FlexibleUNet, UNet
from monai.transforms import AsDiscrete, Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, ScaleIntensityd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmentation checkpoint on a test split.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to best_model.pt.")
    parser.add_argument("--test-images-dir", required=True, type=str, help="Test images directory.")
    parser.add_argument("--test-masks-dir", required=True, type=str, help="Test masks directory.")
    parser.add_argument("--image-suffix", default=".png", type=str)
    parser.add_argument("--mask-suffix", default=".png", type=str)
    parser.add_argument("--gpus", default="0", type=str, help='GPU ids, e.g. "0" or "0,1".')
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--output-json", default="", type=str, help="Optional output JSON path.")
    parser.add_argument("--log-file", default="", type=str, help="Optional append-only log file path.")
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Optional base output dir. Writes JSONL under output-dir/logs/<run-name>.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        type=str,
        help="Optional run name for log folder, e.g. seg-ssl or seg-random. If omitted, tries to infer from checkpoint path.",
    )
    return parser.parse_args()


def normalize_run_name(name: str) -> str:
    lowered = name.strip().lower().replace("_", "-")
    if lowered in {"seg-ssl", "ssl", "segssl", "seg_ssl"}:
        return "seg-ssl"
    if lowered in {"seg-random", "random", "segrandom", "seg_random"}:
        return "seg-random"
    return lowered or "seg-unknown"


def infer_run_name_from_checkpoint(checkpoint: Path) -> Optional[str]:
    for parent in [checkpoint.parent, *checkpoint.parents]:
        name = parent.name.strip().lower()
        if name in {"seg_ssl", "seg-ssl"}:
            return "seg-ssl"
        if name in {"seg_random", "seg-random"}:
            return "seg-random"
    return None


def infer_output_root_from_checkpoint(checkpoint: Path) -> Optional[Path]:
    for parent in [checkpoint.parent, *checkpoint.parents]:
        name = parent.name.strip().lower()
        if name in {"seg_ssl", "seg-ssl", "seg_random", "seg-random"}:
            return parent.parent
    return None


def resolve_default_log_base(args: argparse.Namespace, checkpoint: Path) -> Optional[Path]:
    explicit_output_dir = Path(args.output_dir) if args.output_dir else None
    explicit_run_name = normalize_run_name(args.run_name) if args.run_name else None
    inferred_run_name = infer_run_name_from_checkpoint(checkpoint)
    run_name = explicit_run_name or inferred_run_name

    if explicit_output_dir is not None and run_name is not None:
        return explicit_output_dir / "logs" / run_name

    inferred_root = infer_output_root_from_checkpoint(checkpoint)
    if inferred_root is not None and run_name is not None:
        return inferred_root / "logs" / run_name

    return None


def map_image_to_mask_name(image_stem: str) -> str:
    if image_stem.endswith("_0000"):
        return image_stem[: -len("_0000")]
    return image_stem


def list_pairs(images_dir: Path, masks_dir: Path, image_suffix: str, mask_suffix: str) -> List[Dict[str, str]]:
    images = sorted(images_dir.glob(f"*{image_suffix}"))
    pairs: List[Dict[str, str]] = []
    missing_masks: List[str] = []
    for image_path in images:
        mask_stem = map_image_to_mask_name(image_path.stem)
        mask_path = masks_dir / f"{mask_stem}{mask_suffix}"
        if mask_path.exists():
            pairs.append({"image": str(image_path), "label": str(mask_path)})
        else:
            missing_masks.append(image_path.name)
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in {images_dir} and {masks_dir}.")
    if missing_masks:
        print(f"Warning: {len(missing_masks)} images have no matching mask. First 5: {missing_masks[:5]}")
    return pairs


def build_model_from_cfg(cfg: Dict, device: torch.device) -> torch.nn.Module:
    model_cfg = cfg["model"]
    model_name = model_cfg.get("name", "flexible_unet").lower()
    in_channels = int(cfg["data"]["in_channels"])
    out_channels = int(cfg["data"]["num_classes"])

    if model_name == "flexible_unet":
        model = FlexibleUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            backbone=model_cfg["backbone_name"],
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
    else:
        raise ValueError(f"Unsupported model.name: {model_name}")

    return model.to(device)


def select_device(gpus: str) -> torch.device:
    gpu_ids = [int(x.strip()) for x in gpus.split(",") if x.strip()]
    use_cuda = torch.cuda.is_available() and len(gpu_ids) > 0
    return torch.device(f"cuda:{gpu_ids[0]}" if use_cuda else "cpu")


def evaluate(checkpoint: Path, test_images_dir: Path, test_masks_dir: Path, image_suffix: str, mask_suffix: str, num_workers: int, gpus: str) -> Dict:
    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload or "config" not in payload:
        raise RuntimeError("Invalid checkpoint format, expecting keys: model_state_dict/config.")

    cfg = payload["config"]
    device = select_device(gpus)
    model = build_model_from_cfg(cfg, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    pairs = list_pairs(test_images_dir, test_masks_dir, image_suffix, mask_suffix)
    tfm = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    ds = Dataset(pairs, transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=(device.type == "cuda"),
    )

    num_classes = int(cfg["data"]["num_classes"])
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    miou_metric = MeanIoU(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")

    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="the ground truth of class")
            warnings.filterwarnings("ignore", message="the prediction of class")
            for batch in loader:
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
    mean_iou = float(miou_metric.aggregate().item())
    mean_hd95 = float(hd95_metric.aggregate().item())
    dice_metric.reset()
    miou_metric.reset()
    hd95_metric.reset()

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "num_pairs": len(pairs),
        "test_images_dir": str(test_images_dir),
        "test_masks_dir": str(test_masks_dir),
        "dice": round(mean_dice, 3),
        "iou": round(mean_iou, 3),
        "hd95": round(mean_hd95, 3),
    }


def append_log(log_file: Path, result: Dict) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    line = (
        f"[{result['timestamp']}] checkpoint={result['checkpoint']} "
        f"num_pairs={result['num_pairs']} dice={result['dice']:.3f} "
        f"iou={result['iou']:.3f} hd95={result['hd95']:.3f} "
        f"images={result['test_images_dir']} masks={result['test_masks_dir']}"
    )
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_jsonl(jsonl_file: Path, result: Dict) -> None:
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    result = evaluate(
        checkpoint=Path(args.checkpoint),
        test_images_dir=Path(args.test_images_dir),
        test_masks_dir=Path(args.test_masks_dir),
        image_suffix=args.image_suffix,
        mask_suffix=args.mask_suffix,
        num_workers=args.num_workers,
        gpus=args.gpus,
    )

    print(json.dumps(result, indent=2))

    default_base = resolve_default_log_base(args, Path(args.checkpoint))
    if default_base is not None:
        base = default_base
        base.mkdir(parents=True, exist_ok=True)
        default_jsonl = base / "evaluate_history.jsonl"
        append_jsonl(default_jsonl, result)
        print(f"Appended JSONL to: {default_jsonl}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved JSON to: {out}")

    if args.log_file:
        append_log(Path(args.log_file), result)
        print(f"Appended log to: {args.log_file}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Num foregrounds 0")
    monai.config.print_config()
    main()
