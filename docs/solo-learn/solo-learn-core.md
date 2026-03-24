# solo-learn Core (for graduation-project)

## Scope

In this project, `solo-learn` has one job only:
**run self-supervised pretraining and produce encoder/backbone weights for downstream segmentation.**

## Main Pipeline

1. Entry point: `main_pretrain.py`
2. Config parsing: `solo/args/pretrain.py`
3. Method instantiation: `METHODS[cfg.method]` in `solo/methods/__init__.py`
4. Shared training skeleton: `solo/methods/base.py`
5. Data augmentation + dataloader: `solo/data/pretrain_dataloader.py`
6. Training execution: `Trainer.fit(...)`

## Most Important Parameters for This Project

- `method`
- `backbone.name`
- `data.train_path`
- `optimizer` (batch size / lr / weight decay)
- `max_epochs`

## Outputs You Actually Care About

- pretrained checkpoints
- backbone/encoder weights that can be loaded by downstream segmentation

## Can Be Ignored for Now

- `main_linear.py` / `main_knn.py` / `main_umap.py`
- object detection downstream
- new-method extension tutorials
