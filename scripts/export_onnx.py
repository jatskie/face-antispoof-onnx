"""Export a PyTorch checkpoint to ONNX, optionally with an INT8-quantized copy."""

import argparse
import os
from pathlib import Path

import onnx
import onnxsim
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.minifasv2.config import get_kernel
from src.minifasv2.model import MultiFTNet, load_weights


def load_model(checkpoint_path, input_size=128):
    model = MultiFTNet(conv6_kernel=get_kernel(input_size, input_size))
    load_weights(model, checkpoint_path)
    return model.eval()


def export_to_onnx(model, output_path, input_size=128):
    torch.onnx.export(
        model,
        torch.randn(1, 3, input_size, input_size),
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,  # TorchScript exporter: the one the released models were built with
    )
    simplified, ok = onnxsim.simplify(onnx.load(output_path))
    if not ok:
        raise RuntimeError("Simplified ONNX model failed validation")
    onnx.save(simplified, output_path)


def size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_path", type=Path, help="Path to .pth checkpoint")
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, help="Output .onnx (default: checkpoint path with .onnx)"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Also write an INT8 model next to the output, with a _quantized suffix",
    )
    args = parser.parse_args()

    output = args.output or args.checkpoint_path.with_suffix(".onnx")
    export_to_onnx(load_model(args.checkpoint_path, args.input_size), output, args.input_size)
    print(f"ONNX model: {output} ({size_mb(output):.2f} MB)")

    if args.quantize:
        quantized = output.with_name(f"{output.stem}_quantized.onnx")
        quantize_dynamic(output, quantized, weight_type=QuantType.QUInt8)
        print(f"Quantized model: {quantized} ({size_mb(quantized):.2f} MB)")
