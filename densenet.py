"""DenseNet model definitions with a DenseNet-201 factory."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class _DenseLayer(nn.Module):
    """Single layer within a DenseBlock."""

    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
    ) -> None:
        super().__init__()

        inter_channels = bn_size * growth_rate

        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            num_input_features, inter_channels, kernel_size=1, stride=1, bias=False
        )

        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(p=drop_rate) if drop_rate > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))

        if self.dropout is not None:
            new_features = self.dropout(new_features)

        return torch.cat([x, new_features], dim=1)


class _DenseBlock(nn.Module):
    """Dense block made of stacked dense layers."""

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
    ) -> None:
        super().__init__()

        layers = []
        num_features = num_input_features
        for _ in range(num_layers):
            layer = _DenseLayer(
                num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate
            )
            layers.append(layer)
            num_features += growth_rate

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Module):
    """Transition layer reducing feature map size and channels."""

    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()

        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            num_input_features, num_output_features, kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.relu(self.norm(x)))
        return self.pool(x)


class DenseNet(nn.Module):
    """Densely connected convolutional network."""

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Iterable[int] = (6, 12, 48, 32),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(OrderedDict([
            (
                "conv0",
                nn.Conv2d(
                    3,
                    num_init_features,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
            ),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers,
                num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                out_features = num_features // 2
                transition = _Transition(num_features, out_features)
                self.features.add_module(f"transition{i + 1}", transition)
                num_features = out_features

        self.features.add_module("norm_final", nn.BatchNorm2d(num_features))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def densenet201(num_classes: int = 1000) -> DenseNet:
    """Construct a DenseNet-201 model."""

    return DenseNet(
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        num_init_features=64,
        bn_size=4,
        drop_rate=0.0,
        num_classes=num_classes,
    )


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = densenet201().to(device)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    output = model(dummy_input)
    print("Output shape:", output.shape)
