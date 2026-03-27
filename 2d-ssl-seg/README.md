# 2D SSL Seg (LiTS2017)

End-to-end 2D pipeline for:

1. SSL pretraining with `solo-learn` (VICReg + ResNet backbone).
2. Backbone extraction from SSL checkpoint.
3. MONAI-based downstream segmentation fine-tuning on LiTS2017.
4. Fair comparison between `SSL-pretrained` and `Random-init`.

## Project Layout

- `configs/ssl/vicreg_lits.yaml`: solo-learn VICReg config for unlabeled pretraining.
- `configs/ssl/augmentations/lits_asymmetric.yaml`: conservative medical-style augmentations.
- `configs/seg/train_ssl.yaml`: segmentation config that loads extracted encoder weights.
- `configs/seg/train_random.yaml`: segmentation config with random initialization baseline.
- `src/extract_backbone.py`: extract `backbone.*` / `encoder.*` into an encoder-only file.
- `src/train_segmentation.py`: MONAI `FlexibleUNet` training script with two-stage fine-tuning.
- `src/run_ssl_pretrain.py`: swanlab-compatible SSL launcher for solo-learn.
- `scripts/*.sh`: one-command launchers.
- `configs/CONFIG_GUIDE.md`: detailed config parameter reference.

## Data Assumptions

LiTS data is expected at:

- Unlabeled SSL data:
  - `/home/jupyter-wenkaihua/data3_link/kaihua.wen/dataset/MedSegFactory/train-Seg/LiTS2017/pretrain/images`
- Labeled segmentation train:
  - `/home/jupyter-wenkaihua/data3_link/kaihua.wen/dataset/MedSegFactory/train-Seg/LiTS2017/train/images`
  - `/home/jupyter-wenkaihua/data3_link/kaihua.wen/dataset/MedSegFactory/train-Seg/LiTS2017/train/masks`
- Labeled segmentation val:
  - `/home/jupyter-wenkaihua/data3_link/kaihua.wen/dataset/MedSegFactory/train-Seg/LiTS2017/val/images`
  - `/home/jupyter-wenkaihua/data3_link/kaihua.wen/dataset/MedSegFactory/train-Seg/LiTS2017/val/masks`

Label setting follows LiTS metadata:

- `0`: background
- `1`: liver
- `2`: liver tumor

## 0) Environment

**Recommended: use the project venv** at `/home/jupyter-wenkaihua/data3_link/kaihua.wen/venv/solo-learn` (already aligned for both SSL and seg).

Activate and set paths (optional; run scripts from project root):

```bash
source /home/jupyter-wenkaihua/data3_link/kaihua.wen/code/graduation-project/2d-ssl-seg/scripts/run_with_venv.sh
```

Or run scripts directly: they use `VENV_DIR` if set, else the default venv path above (see each script).

**Reinstall / recreate venv:** in the venv, install deps so that `pkg_resources` is available (needed by lightning/torchmetrics):

```bash
/path/to/venv/solo-learn/bin/python -m pip install --upgrade pip
/path/to/venv/solo-learn/bin/python -m pip install "setuptools>=58,<66"
# Then install solo-learn (editable) and 2d-ssl-seg deps:
/path/to/venv/solo-learn/bin/python -m pip install -e /path/to/solo-learn
/path/to/venv/solo-learn/bin/python -m pip install -r /path/to/2d-ssl-seg/requirements.txt
```

`torch` / `torchvision` / CUDA should be installed in the venv beforehand.

## 1) SSL Pretraining (solo-learn)

Run from this project:

```bash
bash scripts/run_ssl_pretrain.sh
```

This uses the project SSL launcher (which internally uses solo-learn modules) with:

- `method=vicreg`
- `data.dataset=custom`
- `data.no_labels=True`
- `data.format=image_folder`
- `backbone.name=resnet18` (change to `resnet50` if desired)
- `checkpoint.frequency=20` (save every 20 epochs by default)

Outputs are saved under:

- `outputs/ssl/trained_models/vicreg/<run_id>/...`

## 2) Extract Encoder Weights

After pretraining, pick the final `.ckpt` and run:

```bash
bash scripts/run_extract_encoder.sh /path/to/ssl_checkpoint.ckpt
```

Output:

- `outputs/encoders/lits_vicreg_encoder.pth`

Multi-GPU example:

```bash
GPU_IDS=0,1 bash scripts/run_ssl_pretrain.sh
```

## 3) Segmentation Fine-tuning (SSL)

```bash
bash scripts/run_seg_train_ssl.sh
```

Two-stage schedule:

1. Stage A: freeze encoder, train decoder/head.
2. Stage B: unfreeze full model, lower encoder LR for joint optimization.

Multi-GPU example:

```bash
GPU_IDS=0,1 bash scripts/run_seg_train_ssl.sh
```

## 4) Random Init Baseline

```bash
bash scripts/run_seg_train_random.sh
```

## 5) Compare Results

Each run writes:

- `metrics_history.json`
- `best_model.pt`

under `output_dir/experiment_name` (e.g. `outputs/seg_ssl/seg-ssl` and `outputs/seg_random/seg-random`).

Evaluation writes JSONL logs into separate folders:

- `outputs/logs/seg-ssl/evaluate_history.jsonl`
- `outputs/logs/seg-random/evaluate_history.jsonl`

Compare best validation metrics:

- Mean Dice (foreground classes)
- Mean IoU (foreground classes)
- HD95 (foreground classes)

## Notes

- The segmentation model uses MONAI native `FlexibleUNet` (no custom backbone re-implementation).
- SSL encoder loading is non-strict and compatible with `backbone.*` or `encoder.*` key styles.
- Keep data split fixed (`LiTS2017/splits/split_seed3407_n200.json`) for fair experiments.
- Full config field explanations and editable knobs are documented in `configs/CONFIG_GUIDE.md`.

## Build, Test, and Development Commands
Run commands from `2d-ssl-seg/` unless noted.

- `source scripts/run_with_venv.sh`: setup environment.
- `bash scripts/run_ssl_pretrain.sh`: run SSL pretraining.
- `bash scripts/run_extract_encoder.sh /path/to/ssl_checkpoint.ckpt`: export encoder weights.
- `bash scripts/run_seg_train_ssl.sh` / `bash scripts/run_seg_train_random.sh`: train segmentation (SSL vs random baseline).
- `bash scripts/run_seg_eval_lits.sh ssl` / `bash scripts/run_seg_eval_lits.sh random`: evaluate on LiTS.