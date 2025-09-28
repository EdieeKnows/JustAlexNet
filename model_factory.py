"""Utility helpers for constructing supported classification models."""
from __future__ import annotations

import torch.nn as nn

from alexnet import AlexNet
from resnet import resnet18
from densenet import densenet201


def build_model(name: str, num_classes: int) -> nn.Module:
    """Factory method to construct a supported classification model."""
    name = name.lower()
    if name == "alexnet":
        return AlexNet(num_classes=num_classes)
    if name in {"resnet", "resnet18"}:
        return resnet18(num_classes=num_classes)
    if name in {"densenet", "densenet201"}:
        return densenet201(num_classes=num_classes)

    raise ValueError(
        f"Unsupported model '{name}'. Available options: alexnet, resnet18, densenet201"
    )
