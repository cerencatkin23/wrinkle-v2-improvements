from typing import Tuple

import torch
import torch.nn as nn
import torchvision


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet34UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, pretrained: bool = True):
        super().__init__()
        backbone = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool0 = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.center = ConvBlock(512, 512)
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            ConvBlock(32, 32),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.encoder0(x)
        e1 = self.encoder1(self.pool0(e0))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.center(e4)
        d4 = self.up4(center, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)
        d1 = self.up1(d2, e0)
        d0 = self.up0(d1)
        if d0.shape[-2:] != x.shape[-2:]:
            d0 = torch.nn.functional.interpolate(d0, size=x.shape[-2:], mode="bilinear", align_corners=False)
        out = self.final_conv(d0)
        return out


def build_unet(encoder: str, in_channels: int, out_channels: int, pretrained: bool = True) -> nn.Module:
    if encoder.lower() == "resnet34":
        return ResNet34UNet(in_channels=in_channels, out_channels=out_channels, pretrained=pretrained)
    raise ValueError(f"Unsupported encoder: {encoder}")
