import argparse
from pathlib import Path
from typing import Dict

import torch


def extract_encoder_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    encoder_state: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]

        if clean_key.startswith("backbone."):
            clean_key = clean_key[len("backbone.") :]
            encoder_state[clean_key] = value
        elif clean_key.startswith("encoder."):
            clean_key = clean_key[len("encoder.") :]
            encoder_state[clean_key] = value

    return encoder_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract encoder/backbone weights from SSL checkpoint.")
    parser.add_argument("--input", required=True, type=str, help="Path to SSL checkpoint (.ckpt/.pt/.pth).")
    parser.add_argument("--output", required=True, type=str, help="Path to extracted encoder file (.pth).")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(in_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint does not contain a valid state_dict.")

    encoder_state = extract_encoder_state(state_dict)
    if not encoder_state:
        raise RuntimeError("No encoder/backbone keys found in checkpoint.")

    payload = {
        "encoder_state_dict": encoder_state,
        "source_checkpoint": str(in_path),
        "num_tensors": len(encoder_state),
    }
    torch.save(payload, out_path)

    print(f"Saved encoder weights: {out_path}")
    print(f"Total tensors: {len(encoder_state)}")


if __name__ == "__main__":
    main()
