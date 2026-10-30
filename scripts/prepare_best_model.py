"""Strip a training checkpoint down to inference weights (no optimizer, no FT branch)."""

import argparse
from pathlib import Path

import torch

from src.minifasv2.config import get_kernel
from src.minifasv2.model import MultiFTNet, load_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to training checkpoint")
    parser.add_argument("--output", type=Path, default=Path("models/best_model.pth"))
    parser.add_argument("--input_size", type=int, default=128)
    args = parser.parse_args()

    model = MultiFTNet(conv6_kernel=get_kernel(args.input_size, args.input_size))
    load_weights(model, args.checkpoint, include_ft=False)
    state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith("FTGenerator.")}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state_dict,
            "input_size": args.input_size,
            "num_classes": 2,
            "architecture": "MiniFASNetV2SE",
        },
        args.output,
    )
    print(f"Saved {args.output} ({args.output.stat().st_size / 2**20:.2f} MB)")
