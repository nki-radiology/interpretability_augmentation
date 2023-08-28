"""Modifed UNet that has two decoders, one to reconstruct the image multiplied for the inverse of the saliency maps, the other used for segmentation"""

import torch.nn.functional as F
from torch import nn
import torch
from NewSDNet.models.ModifiedUNet.unet_parts import *


class Encoder(nn.Module):
    def __init__(self, n_channels: int, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x5, x4, x3, x2, x1


class Decoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        bilinear: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.n_classes = n_classes

        factor = 2 if self.bilinear else 1
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, skipconnections):
        x5, x4, x3, x2, x1 = skipconnections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ModifiedUNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = True):
        super(ModifiedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.encoder = Encoder(self.n_channels, self.bilinear)
        self.segmentor_decoder = Decoder(self.n_channels, self.n_classes, self.bilinear)
        self.reconstruction_decoder = Decoder(
            self.n_channels, self.n_channels, self.bilinear
        )

    def forward(self, x: torch.Tensor, script_type: str):
        if script_type == "training":
            skipconnections = self.encoder(x)
            segmentation_logits = self.segmentor_decoder(skipconnections)
            reconstruction_logits = self.reconstruction_decoder(skipconnections)

            return (
                segmentation_logits,
                reconstruction_logits,
            )  # NOTE: should I call the reconstruction decoder also in inference or just in training?

        else:
            skipconnections = self.encoder(x)
            segmentation_logits = self.segmentor_decoder(skipconnections)

            return segmentation_logits


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
