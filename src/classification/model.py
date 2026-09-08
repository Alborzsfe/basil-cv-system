"""Disease-classifier factory with small and transfer-learning backbones."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class TinyClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(tensor).flatten(1))


def build_classifier(
    architecture: str, *, num_classes: int, pretrained: bool = True, in_channels: int = 3
) -> nn.Module:
    if num_classes < 2:
        raise ValueError("num_classes must be at least two.")
    if architecture == "tiny_cnn":
        return TinyClassifier(in_channels, num_classes)
    from torchvision import models

    if architecture == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        )
        model.features[0][0] = _adapt_first_conv(model.features[0][0], in_channels)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif architecture == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.conv1 = _adapt_first_conv(model.conv1, in_channels)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "convnext_tiny":
        model = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        )
        model.features[0][0] = _adapt_first_conv(model.features[0][0], in_channels)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(
            "Supported classifiers: efficientnet_v2_s, resnet50, convnext_tiny, tiny_cnn."
        )
    return model


def freeze_backbone(model: nn.Module, architecture: str, frozen: bool = True) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = not frozen
    for parameter in _head(model, architecture).parameters():
        parameter.requires_grad = True


def find_last_convolution(model: nn.Module) -> nn.Conv2d:
    layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not layers:
        raise ValueError("Classifier has no convolutional layer for Grad-CAM.")
    return layers[-1]


def _head(model: nn.Module, architecture: str) -> nn.Module:
    accessors: dict[str, Callable[[nn.Module], nn.Module]] = {
        "tiny_cnn": lambda item: item.classifier,
        "efficientnet_v2_s": lambda item: item.classifier,
        "resnet50": lambda item: item.fc,
        "convnext_tiny": lambda item: item.classifier,
    }
    if architecture not in accessors:
        raise ValueError(f"Unsupported classifier '{architecture}'.")
    return accessors[architecture](model)


def _adapt_first_conv(layer: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if in_channels == layer.in_channels:
        return layer
    if in_channels <= 0:
        raise ValueError("in_channels must be positive.")
    replacement = nn.Conv2d(
        in_channels,
        layer.out_channels,
        layer.kernel_size,
        layer.stride,
        layer.padding,
        bias=layer.bias is not None,
    )
    with torch.no_grad():
        repeated = layer.weight.repeat(
            1, (in_channels + layer.in_channels - 1) // layer.in_channels, 1, 1
        )[:, :in_channels]
        replacement.weight.copy_(repeated * (layer.in_channels / in_channels))
        if layer.bias is not None and replacement.bias is not None:
            replacement.bias.copy_(layer.bias)
    return replacement
