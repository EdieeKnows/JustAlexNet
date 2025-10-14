from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import torch

from model_factory import build_model as build_train_model
from transfer_learning import build_model as build_transfer_model

logger = logging.getLogger("export")


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(not key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def _load_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in checkpoint:
                state = checkpoint[key]
                break
        else:
            state = checkpoint
    else:
        state = checkpoint
    return _clean_state_dict(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a PyTorch checkpoint to ONNX format.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the .pt/.pth checkpoint file.")
    parser.add_argument("--output", type=Path, required=True, help="Destination path for the exported ONNX model.")
    parser.add_argument(
        "--checkpoint-type",
        choices=["train", "transfer"],
        default="train",
        help="Specify which training pipeline created the checkpoint. "
        "'train' corresponds to train.py, 'transfer' corresponds to transfer_learning.py.",
    )
    parser.add_argument("--model", default="alexnet", help="Model architecture name when exporting checkpoints from train.py.")
    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=["resnet18", "resnet152"],
        help="Backbone used when exporting checkpoints from transfer_learning.py.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        required=True,
        help="Number of output classes. Must match the model configuration used during training.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Spatial resolution of the dummy input used for export (default: 224).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy batch size used during export. Exported model supports dynamic batch dimension.",
    )
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version (default: 13).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("export")

    args = parse_args()

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading checkpoint from %s", checkpoint_path)
    state_dict = _load_state_dict(checkpoint_path)

    if args.checkpoint_type == "train":
        model = build_train_model(args.model, num_classes=args.num_classes)
    else:
        model = build_transfer_model(args.num_classes, args.backbone)

    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(args.batch_size, 3, args.image_size, args.image_size, dtype=torch.float32)

    logger.info("Exporting model to %s (opset %d)", output_path, args.opset)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    logger.info("ONNX model saved to %s", output_path)


if __name__ == "__main__":
    main()

