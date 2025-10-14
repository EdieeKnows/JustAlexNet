from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_factory import build_model
from transfer_learning import build_model as build_transfer_model

logger = logging.getLogger("validator")


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip ``module.`` prefixes introduced by ``DataParallel`` if present."""
    if not state_dict:
        return state_dict
    if all(not key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def _load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    state_dict = _clean_state_dict(state_dict)
    model.load_state_dict(state_dict)


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def _build_dataloader(
    data_root: Path,
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, Sequence[str]]:
    transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    if len(dataset.classes) == 0:
        raise ValueError(f"No classes found at {data_root}. Ensure the directory follows the ImageFolder structure.")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataloader, dataset.classes


def _compute_topk_correct(predictions: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...]) -> List[int]:
    if predictions.ndim != 2:
        raise ValueError("Predictions tensor is expected to be 2-dimensional")

    max_k = min(max(topk), predictions.size(1))
    _, pred_topk = predictions.topk(max_k, 1, True, True)
    pred_topk = pred_topk.t()

    expanded_targets = targets.view(1, -1).expand_as(pred_topk)
    correct = pred_topk.eq(expanded_targets)

    results: List[int] = []
    for k in topk:
        k = min(k, predictions.size(1))
        if k <= 0:
            results.append(0)
            continue
        correct_k = correct[:k].any(dim=0).sum().item()
        results.append(int(correct_k))
    return results


def _load_class_names(class_map: Path | None, fallback: Sequence[str]) -> List[str]:
    if class_map is None:
        return list(fallback)
    class_map = class_map.expanduser().resolve()
    if not class_map.is_file():
        logger.warning("Class mapping file %s does not exist. Falling back to dataset classes.", class_map)
        return list(fallback)
    with class_map.open("r", encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    if not names:
        logger.warning("Class mapping file %s is empty. Falling back to dataset classes.", class_map)
        return list(fallback)
    return names


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()

    total_samples = 0
    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            top1_correct, top5_correct = _compute_topk_correct(outputs, targets, (1, 5))

            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_top1 += top1_correct
            total_top5 += top5_correct

    if total_samples == 0:
        raise ValueError("The validation dataloader did not yield any samples.")

    avg_loss = total_loss / total_samples
    top1_acc = total_top1 / total_samples
    top5_acc = total_top5 / total_samples

    return {"loss": avg_loss, "top1": top1_acc, "top5": top5_acc}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a trained model against an ImageFolder dataset."
    )
    parser.add_argument(
        "--checkpoint-type",
        choices=["train", "transfer"],
        default="train",
        help="Specify which training script produced the checkpoint. "
        "'train' expects checkpoints from train.py, 'transfer' expects checkpoints from transfer_learning.py.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the ImageFolder dataset root.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--model",
        default="alexnet",
        help="Model architecture to instantiate for checkpoints produced by train.py (default: alexnet).",
    )
    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=["resnet18", "resnet152"],
        help="Backbone architecture used when the checkpoint was produced by transfer_learning.py.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of output classes. If omitted, inferred from the dataset.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size used for validation transforms (default: 224).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Validation batch size (default: 64).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for the dataloader (default: 4).",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable pinned memory for faster host-to-device transfers.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run validation on. 'auto' selects the best available.",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="Optional text file containing class names (one per line).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("validator")

    args = parse_args()

    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found at {data_root}")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    device = _select_device(args.device)
    logger.info("Using %s for validation", device)

    dataloader, dataset_classes = _build_dataloader(
        data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    inferred_num_classes = len(dataset_classes)
    if args.num_classes is None:
        num_classes = inferred_num_classes
    else:
        num_classes = args.num_classes
        if num_classes != inferred_num_classes:
            logger.warning(
                "Dataset reports %d classes but --num-classes was set to %d. "
                "Make sure they match the checkpoint.",
                inferred_num_classes,
                num_classes,
            )

    if args.checkpoint_type == "train":
        model = build_model(args.model, num_classes=num_classes)
    else:
        model = build_transfer_model(num_classes, args.backbone)
    model.to(device)

    logger.info("Loading checkpoint from %s", checkpoint_path)
    _load_checkpoint(model, checkpoint_path, device)

    class_names = _load_class_names(args.class_map, dataset_classes)
    logger.info("Class count: %d", len(class_names))

    metrics = evaluate(model, dataloader, device)
    logger.info("Validation results: loss=%.4f, top1=%.2f%%, top5=%.2f%%", metrics["loss"], metrics["top1"] * 100, metrics["top5"] * 100)


if __name__ == "__main__":
    main()
