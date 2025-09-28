from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model_factory import build_model

logger = logging.getLogger("inference")


def _clean_state_dict(state_dict: dict) -> dict:
    if all(not key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
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


def _load_class_names(class_map: Path | None) -> List[str]:
    if class_map is None:
        return []
    class_map = class_map.expanduser().resolve()
    if not class_map.is_file():
        logger.warning("Class mapping file %s does not exist. Falling back to index labels.", class_map)
        return []
    with class_map.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _format_class_name(class_names: Sequence[str], index: int) -> str:
    if 0 <= index < len(class_names):
        return class_names[index]
    return f"class_{index}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("inference")

    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("image", type=Path, help="Path to the input image")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--model", default="alexnet", help="Model architecture to use")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of output classes")
    parser.add_argument("--class-map", type=Path, default=None, help="Optional text file containing class names (one per line)")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to display")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run inference on. 'auto' selects the best available",
    )
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info("Using %s for inference", device)

    model = build_model(args.model, args.num_classes)
    model.to(device)
    model.eval()

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    logger.info("Loading checkpoint from %s", checkpoint_path)
    _load_checkpoint(model, checkpoint_path, device)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found at {image_path}")

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)

    topk = min(args.topk, probabilities.size(1))
    top_probabilities, top_indices = torch.topk(probabilities, topk)

    class_names = _load_class_names(args.class_map)

    logger.info("Top-%d predictions for %s:", topk, image_path)
    for rank, (probability, class_index) in enumerate(zip(top_probabilities[0], top_indices[0]), start=1):
        class_id = class_index.item()
        class_name = _format_class_name(class_names, class_id)
        logger.info("%d. %s (id=%d): %.2f%%", rank, class_name, class_id, probability.item() * 100)


if __name__ == "__main__":
    main()
