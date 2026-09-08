"""U-Net implementations for fast tests and pretrained ResNet encoders."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, tensor: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        tensor = F.interpolate(tensor, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((tensor, skip), dim=1))


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        one = self.enc1(tensor)
        two = self.enc2(self.pool(one))
        three = self.enc3(self.pool(two))
        bottleneck = self.bottleneck(self.pool(three))
        return self.head(self.up1(self.up2(self.up3(bottleneck, three), two), one))


class ResNetUNet(nn.Module):
    def __init__(
        self, *, encoder_name: str, pretrained: bool, in_channels: int, num_classes: int
    ) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34

        builders = {
            "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
            "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
        }
        if encoder_name not in builders:
            raise ValueError(f"Unsupported U-Net encoder '{encoder_name}'.")
        builder, weights = builders[encoder_name]
        encoder = builder(weights=weights if pretrained else None)
        if in_channels != 3:
            encoder.conv1 = _adapt_first_conv(encoder.conv1, in_channels)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4,
        )
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.final = nn.Sequential(ConvBlock(64, 32), nn.Conv2d(32, num_classes, 1))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        size = tensor.shape[-2:]
        stem = self.stem(tensor)
        one = self.layer1(self.pool(stem))
        two = self.layer2(one)
        three = self.layer3(two)
        four = self.layer4(three)
        decoded = self.up1(self.up2(self.up3(self.up4(four, three), two), one), stem)
        return self.final(F.interpolate(decoded, size=size, mode="bilinear", align_corners=False))


def _adapt_first_conv(layer: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if in_channels <= 0:
        raise ValueError("in_channels must be positive.")
    replacement = nn.Conv2d(
        in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding, bias=False
    )
    with torch.no_grad():
        if in_channels == 1:
            replacement.weight.copy_(layer.weight.mean(dim=1, keepdim=True))
        else:
            repeated = layer.weight.repeat(1, (in_channels + 2) // 3, 1, 1)[:, :in_channels]
            replacement.weight.copy_(repeated * (3 / in_channels))
    return replacement


def build_segmentation_model(
    architecture: str = "unet",
    *,
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    num_classes: int = 1,
    base_channels: int = 16,
    **_: Any,
) -> nn.Module:
    if architecture not in {"unet", "tiny_unet"}:
        raise ValueError("Supported architectures are 'unet' and 'tiny_unet'.")
    if architecture == "tiny_unet" or encoder_name == "tiny":
        return TinyUNet(in_channels, num_classes, base_channels)
    if encoder_weights not in {"imagenet", None, "none"}:
        raise ValueError("encoder_weights must be 'imagenet' or null/'none'.")
    return ResNetUNet(
        encoder_name=encoder_name,
        pretrained=encoder_weights == "imagenet",
        in_channels=in_channels,
        num_classes=num_classes,
    )
