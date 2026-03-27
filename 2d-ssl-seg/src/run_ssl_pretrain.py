import inspect
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

try:
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import Callback, LearningRateMonitor
    from lightning.pytorch.loggers.wandb import WandbLogger
    from lightning.pytorch.strategies.ddp import DDPStrategy
except ImportError:
    import pytorch_lightning as pl

    # Compatibility shim for codebases importing lightning.pytorch.
    lightning_pkg = types.ModuleType("lightning")
    lightning_pkg.pytorch = pl
    sys.modules.setdefault("lightning", lightning_pkg)
    sys.modules.setdefault("lightning.pytorch", pl)
    sys.modules.setdefault("lightning.pytorch.callbacks", pl.callbacks)
    sys.modules.setdefault("lightning.pytorch.loggers", pl.loggers)
    sys.modules.setdefault("lightning.pytorch.strategies", pl.strategies)

    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import Callback, LearningRateMonitor
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.strategies.ddp import DDPStrategy


def _setup_solo_path() -> None:
    solo_root = os.environ.get(
        "SOLO_LEARN_DIR",
        "/home/jupyter-wenkaihua/data3_link/kaihua.wen/code/solo-learn",
    )
    solo_root_path = Path(solo_root).resolve()
    if str(solo_root_path) not in sys.path:
        sys.path.insert(0, str(solo_root_path))


_setup_solo_path()

from solo.args.pretrain import parse_cfg  # noqa: E402
from solo.data.classification_dataloader import prepare_data as prepare_data_classification  # noqa: E402
from solo.data.pretrain_dataloader import (  # noqa: E402
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
    prepare_dataloader,
    prepare_datasets,
)
from solo.methods import METHODS  # noqa: E402
from solo.utils.auto_resumer import AutoResumer  # noqa: E402
from solo.utils.checkpointer import Checkpointer  # noqa: E402
from solo.utils.misc import make_contiguous, omegaconf_select  # noqa: E402

try:
    from solo.data.dali_dataloader import PretrainDALIDataModule, build_transform_pipeline_dali  # noqa: E402
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from solo.utils.auto_umap import AutoUMAP  # noqa: E402
except ImportError:
    _umap_available = False
else:
    _umap_available = True

try:
    from swanlab.integration.pytorch_lightning import SwanLabLogger
except ImportError:
    SwanLabLogger = None


_SwanLabBase = SwanLabLogger if SwanLabLogger is not None else object


class FilteredSwanLabLogger(_SwanLabBase):  # type: ignore[misc]
    """SwanLab logger with metric-level filtering and epoch-based x-axis for epoch metrics."""

    _BLOCKED_METRICS = {
        "train_acc1",
        "train_acc5",
        "train_class_loss",
        "train_acc1_step",
        "train_acc1_epoch",
        "train_acc5_step",
        "train_acc5_epoch",
        "train_class_loss_step",
        "train_class_loss_epoch",
    }

    def __init__(
        self,
        *args: Any,
        steps_per_epoch: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        kwargs = {k: v for k, v in kwargs.items() if k != "steps_per_epoch"}
        super().__init__(*args, **kwargs)
        self._steps_per_epoch = steps_per_epoch

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        filtered = {k: v for k, v in metrics.items() if k not in self._BLOCKED_METRICS}
        if not filtered:
            return
        epoch_keys = [k for k in filtered if k.endswith("_epoch")]
        step_keys = [k for k in filtered if k not in epoch_keys]
        epoch_metrics = {k: filtered[k] for k in epoch_keys}
        step_metrics = {k: filtered[k] for k in step_keys}
        if self._steps_per_epoch is not None and step is not None and epoch_metrics:
            epoch_step = step // self._steps_per_epoch
            super().log_metrics(epoch_metrics, step=epoch_step)
        elif epoch_metrics:
            super().log_metrics(epoch_metrics, step=step)
        if step_metrics:
            super().log_metrics(step_metrics, step=step)


class SaveFinalCheckpoint(Callback):
    """Forces one final checkpoint save at train end when using solo Checkpointer."""

    def on_train_end(self, trainer: Trainer, _) -> None:
        for callback in trainer.callbacks:
            if isinstance(callback, Checkpointer):
                callback.save(trainer)
                return


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)
    seed_everything(cfg.seed)

    assert cfg.method in METHODS, f"Choose from {METHODS.keys()}"
    if cfg.data.num_large_crops != 2:
        assert cfg.method in ["wmse", "mae"]

    model = METHODS[cfg.method](cfg)
    make_contiguous(model)
    if not cfg.performance.disable_channel_last:
        model = model.to(memory_format=torch.channels_last)

    if cfg.data.dataset == "custom" and (cfg.data.no_labels or cfg.data.val_path is None):
        val_loader = None
    elif cfg.data.dataset in ["imagenet100", "imagenet"] and cfg.data.val_path is None:
        val_loader = None
    else:
        if cfg.data.format == "dali":
            val_data_format = "image_folder"
        else:
            val_data_format = cfg.data.format
        _, val_loader = prepare_data_classification(
            cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            val_data_path=cfg.data.val_path,
            data_format=val_data_format,
            batch_size=cfg.optimizer.batch_size,
            num_workers=cfg.data.num_workers,
        )

    steps_per_epoch: Optional[int] = None
    if cfg.data.format == "dali":
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with pip3 install .[dali]."
        pipelines = []
        for aug_cfg in cfg.augmentations:
            pipelines.append(
                NCropAugmentation(
                    build_transform_pipeline_dali(
                        cfg.data.dataset, aug_cfg, dali_device=cfg.dali.device
                    ),
                    aug_cfg.num_crops,
                )
            )
        transform = FullTransformPipeline(pipelines)
        dali_datamodule = PretrainDALIDataModule(
            dataset=cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            transforms=transform,
            num_large_crops=cfg.data.num_large_crops,
            num_small_crops=cfg.data.num_small_crops,
            num_workers=cfg.data.num_workers,
            batch_size=cfg.optimizer.batch_size,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
            dali_device=cfg.dali.device,
            encode_indexes_into_labels=cfg.dali.encode_indexes_into_labels,
        )
        dali_datamodule.val_dataloader = lambda: val_loader
    else:
        pipelines = []
        for aug_cfg in cfg.augmentations:
            pipelines.append(
                NCropAugmentation(
                    build_transform_pipeline(cfg.data.dataset, aug_cfg), aug_cfg.num_crops
                )
            )
        transform = FullTransformPipeline(pipelines)
        if cfg.debug_augmentations:
            print("Transforms:")
            print(transform)
        train_dataset = prepare_datasets(
            cfg.data.dataset,
            transform,
            train_data_path=cfg.data.train_path,
            data_format=cfg.data.format,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
        )
        train_loader = prepare_dataloader(
            train_dataset, batch_size=cfg.optimizer.batch_size, num_workers=cfg.data.num_workers
        )
        steps_per_epoch = len(train_loader)

    ckpt_path = None
    if cfg.auto_resume.enabled and cfg.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(cfg.checkpoint.dir, cfg.method),
            max_hours=cfg.auto_resume.max_hours,
        )
        resume_from_checkpoint, _ = auto_resumer.find_checkpoint(cfg)
        if resume_from_checkpoint is not None:
            print("Resuming from previous checkpoint:", f"'{resume_from_checkpoint}'")
            ckpt_path = resume_from_checkpoint
    elif cfg.resume_from_checkpoint is not None:
        ckpt_path = cfg.resume_from_checkpoint
        del cfg.resume_from_checkpoint

    callbacks = []
    if cfg.checkpoint.enabled:
        ckpt = Checkpointer(
            cfg,
            logdir=os.path.join(cfg.checkpoint.dir, cfg.method),
            frequency=cfg.checkpoint.frequency,
            keep_prev=cfg.checkpoint.keep_prev,
        )
        callbacks.append(ckpt)
        callbacks.append(SaveFinalCheckpoint())

    if omegaconf_select(cfg, "auto_umap.enabled", False):
        assert _umap_available, "UMAP is not currently avaiable, please install it first with [umap]."
        auto_umap = AutoUMAP(
            cfg.name,
            logdir=os.path.join(cfg.auto_umap.dir, cfg.method),
            frequency=cfg.auto_umap.frequency,
        )
        callbacks.append(auto_umap)

    logger = None
    if omegaconf_select(cfg, "swanlab.enabled", False):
        if SwanLabLogger is None:
            print("swanlab is not installed, fallback to no external logger.")
        else:
            logger = FilteredSwanLabLogger(
                project=omegaconf_select(cfg, "swanlab.project", "2d-ssl-seg"),
                experiment_name=omegaconf_select(cfg, "swanlab.experiment_name", cfg.name),
                description=omegaconf_select(cfg, "swanlab.description", ""),
                logdir=omegaconf_select(cfg, "swanlab.logdir", "outputs/swanlab/ssl"),
                config=OmegaConf.to_container(cfg),
                steps_per_epoch=steps_per_epoch,
            )
            callbacks.append(LearningRateMonitor(logging_interval="step"))
    elif cfg.wandb.enabled:
        logger = WandbLogger(
            name=cfg.name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            offline=cfg.wandb.offline,
        )
        logger.watch(model, log="gradients", log_freq=100)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer_kwargs = OmegaConf.to_container(cfg)
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
    trainer_kwargs.update(
        {
            "logger": logger,
            "callbacks": callbacks,
            "enable_checkpointing": False,
            "strategy": DDPStrategy(find_unused_parameters=False)
            if cfg.strategy == "ddp"
            else cfg.strategy,
        }
    )
    trainer = Trainer(**trainer_kwargs)

    if cfg.data.format == "dali":
        trainer.fit(model, ckpt_path=ckpt_path, datamodule=dali_datamodule)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
