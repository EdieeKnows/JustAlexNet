"""Transfer learning script for tiny classification datasets.

This module demonstrates how to fine-tune a model that was pre-trained on
ImageNet when only a handful of labeled samples are available.  The code keeps
most of the backbone frozen and only learns a lightweight classification head
while applying aggressive regularisation and data augmentation techniques that
are well-suited for extremely small datasets.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms


@dataclass
class TrainingConfig:
    """Configuration parameters for the transfer learning run."""

    data_root: Path
    train_split: float = 0.8
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 25
    patience: int = 5
    output_dir: Path = Path("artifacts")
    image_size: int = 224
    seed: int = 42
    backbone: str = "resnet18"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return training and validation transforms.

    Strong augmentation on the training set helps combat overfitting when the
    dataset only contains a few dozen examples.
    """

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_tf, eval_tf


def build_datasets(config: TrainingConfig) -> Tuple[Dataset, Dataset]:
    train_tf, eval_tf = build_transforms(config.image_size)
    full_ds = datasets.ImageFolder(config.data_root, transform=train_tf)
    eval_ds = datasets.ImageFolder(config.data_root, transform=eval_tf)

    num_samples = len(full_ds)
    if num_samples < 4:
        raise ValueError("The dataset is too small. At least 4 samples are required.")

    indices = list(range(num_samples))
    random.shuffle(indices)
    train_cutoff = int(num_samples * config.train_split)

    train_idx = indices[:train_cutoff]
    val_idx = indices[train_cutoff:]

    return Subset(full_ds, train_idx), Subset(eval_ds, val_idx)


def build_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """Create a sampler that balances classes even when the dataset is tiny."""

    targets = [dataset.dataset.targets[i] for i in dataset.indices]  # type: ignore[attr-defined]
    counts = Counter(targets)
    sample_weights = torch.tensor([1.0 / counts[label] for label in targets], dtype=torch.float)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def build_dataloaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds, val_ds = build_datasets(config)

    sampler = build_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def build_model(num_classes: int, backbone: str = "resnet18") -> nn.Module:
    backbone = backbone.lower()
    if backbone == "resnet152":
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
    elif backbone in {"resnet", "resnet18"}:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Unsupported backbone. Choose between 'resnet18' and 'resnet152'.")

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, in_features // 2),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(in_features // 2, num_classes),
    )

    return model


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * inputs.size(0)

        preds = outputs.argmax(dim=1)
        running_correct += (preds == targets).sum().item()

    loss = running_loss / len(dataloader.dataset)
    accuracy = running_correct / len(dataloader.dataset)
    return loss, accuracy


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def run_training(config: TrainingConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(config)
    num_classes = len(train_loader.dataset.dataset.classes)  # type: ignore[attr-defined]

    model = build_model(num_classes, config.backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.max_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{config.max_epochs} - "
            f"train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, config.output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping triggered.")
                break

    save_checkpoint(model, config.output_dir / "last_model.pt")


def parse_args(args: Iterable[str] | None = None) -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Transfer learning on a tiny dataset")
    parser.add_argument("data_root", type=Path, help="Path to the dataset structured like ImageFolder")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the classifier head")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay strength")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--output", type=Path, default=Path("artifacts"), help="Where to store checkpoints")
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet152"],
        help="Which pre-trained backbone to fine-tune",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parsed = parser.parse_args(args)

    return TrainingConfig(
        data_root=parsed.data_root,
        batch_size=parsed.batch_size,
        max_epochs=parsed.epochs,
        patience=parsed.patience,
        lr=parsed.lr,
        weight_decay=parsed.weight_decay,
        image_size=parsed.image_size,
        num_workers=parsed.workers,
        output_dir=parsed.output,
        seed=parsed.seed,
        backbone=parsed.backbone,
    )


def main() -> None:
    config = parse_args()
    run_training(config)


if __name__ == "__main__":
    main()
