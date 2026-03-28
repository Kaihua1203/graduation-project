from __future__ import annotations

import argparse

from common.config import load_yaml_config, normalize_training_config
from common.runtime import set_seed
from train.base_trainer import BaseDiffusionTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 2d-gen LoRA adapters.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = normalize_training_config(load_yaml_config(args.config))
    set_seed(config["train"].get("seed", 3407))
    trainer = BaseDiffusionTrainer(config)
    final_checkpoint = trainer.train()
    print(f"Training finished. Final LoRA checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
