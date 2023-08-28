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
        print(
            f"\n AAAAAAAAAAAAAAAA shape di logits of segmentation decoder: {logits.shape}"
        )
        return logits


class ReconstructionDecoder(nn.Module):
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
        self.up1 = Up(1024 * 2, 512 * 2 // factor, self.bilinear)
        self.up2 = Up(512 * 2, 256 * 2 // factor, self.bilinear)
        self.up3 = Up(256 * 2, 128 * 2 // factor, self.bilinear)
        self.up4 = Up(128 * 2, 64 * 2, bilinear)
        self.outc = OutConv(64 * 2, self.n_classes)

    def forward(self, skipconnections):
        x5, x4, x3, x2, x1 = skipconnections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        print(f"\n AAAAAAAAAAAAAAAA shape di logits of decoder: {logits.shape}")
        return logits


class SiHUNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = True):
        super(SiHUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.unbiased_encoder = Encoder(self.n_channels, self.bilinear)
        self.biased_encoder = Encoder(self.n_channels, self.bilinear)
        self.reconstruction_decoder = ReconstructionDecoder(
            self.n_channels, self.n_channels, self.bilinear
        )
        self.unbiased_segmentor = Decoder(
            self.n_channels, self.n_classes, self.bilinear
        )
        self.biased_segmentor = Decoder(self.n_channels, self.n_classes, self.bilinear)

    def forward(self, x: torch.Tensor, script_type: str):
        if script_type == "training":
            unbiased_skipconnections = self.unbiased_encoder(x)
            biased_skipconnections = self.biased_encoder(x)

            unbiased_segmentation_logits = self.unbiased_segmentor(
                unbiased_skipconnections
            )
            biased_segmentation_logits = self.biased_segmentor(biased_skipconnections)

            cat_skip_connections = [
                torch.cat((skip1, skip2), dim=1)
                for skip1, skip2 in zip(
                    unbiased_skipconnections, biased_skipconnections
                )
            ]
            reconstruction_logits = self.reconstruction_decoder(cat_skip_connections)

            return (
                unbiased_segmentation_logits,
                biased_segmentation_logits,
                reconstruction_logits,
            )  # NOTE: should I call the reconstruction decoder also in inference or just in training?

        else:
            skipconnections = self.unbiased_encoder(x)
            segmentation_logits = self.unbiased_segmentor(skipconnections)

            return segmentation_logits
