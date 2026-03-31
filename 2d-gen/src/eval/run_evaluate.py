from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from common.config import load_yaml_config, ensure_section
from common.runtime import write_json
from eval.metrics import evaluate_generation_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated images.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    eval_cfg = ensure_section(config, "eval")
    num_workers = int(eval_cfg.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("eval.num_workers must be non-negative.")
    output_path = Path(eval_cfg["output_path"]).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
    real_inception_cache_dir = eval_cfg.get("real_inception_cache_dir") or output_path.parent / "cache"
    result = evaluate_generation_quality(
        real_image_dir=eval_cfg["real_image_dir"],
        generated_image_dir=eval_cfg["generated_image_dir"],
        generated_manifest_path=eval_cfg["generated_manifest"],
        batch_size=eval_cfg.get("batch_size", 8),
        num_workers=num_workers,
        inception_weights_path=eval_cfg["inception_weights_path"],
        clip_model_path=eval_cfg["clip_model_path"],
        real_inception_cache_dir=real_inception_cache_dir,
    )
    output_path = write_json(result.__dict__, output_path)
    print(f"Saved generation metrics to {output_path}")


if __name__ == "__main__":
    main()
