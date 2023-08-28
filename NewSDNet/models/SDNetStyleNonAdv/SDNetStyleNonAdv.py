import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
from NewSDNet.models.SDNet.adain import *
from NewSDNet.models.SDNet.blocks import *


class AdaINDecoder(nn.Module):
    def __init__(self, anatomy_out_channels):
        super().__init__()
        self.anatomy_out_channels = anatomy_out_channels
        self.conv1 = conv_relu(self.anatomy_out_channels, 128, 3, 1, 1)
        self.conv2 = conv_relu(128, 64, 3, 1, 1)
        self.conv3 = conv_relu(64, 32, 3, 1, 1)
        self.conv4 = conv_no_activ(32, 3, 3, 1, 1)

    def forward(self, a, z):
        out = adaptive_instance_normalization(a, z)
        out = self.conv1(out)
        out = adaptive_instance_normalization(out, z)
        out = self.conv2(out)
        out = adaptive_instance_normalization(out, z)
        out = self.conv3(out)
        out = adaptive_instance_normalization(out, z)
        out = F.tanh(self.conv4(out))

        return out


class Decoder(nn.Module):
    def __init__(self, anatomy_out_channels, z_length, num_mask_channels):
        super(Decoder, self).__init__()
        self.anatomy_out_channels = anatomy_out_channels
        self.z_length = z_length
        self.num_mask_channels = num_mask_channels
        self.decoder = AdaINDecoder(self.anatomy_out_channels)

    def forward(self, a, z):
        out = self.decoder(a, z)

        return out


class Segmentor(nn.Module):
    def __init__(self, num_output_channels, num_classes):
        super(Segmentor, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_classes = num_classes + 1  # background as extra class

        self.conv1 = conv_bn_relu(self.num_output_channels, 64, 3, 1, 1)
        self.conv2 = conv_bn_relu(64, 64, 3, 1, 1)
        self.pred = nn.Conv2d(64, self.num_classes, 1, 1, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pred(out)
        out = F.softmax(out, dim=1)

        return out


class AEncoder(nn.Module):
    def __init__(self, width, height, ndf, num_output_channels, norm, upsample):
        super(AEncoder, self).__init__()
        """
        UNet encoder for the anatomy factors of the image
        num_output_channels: number of spatial (anatomy) factors to encode
        """
        self.width = width
        self.height = height
        self.ndf = ndf
        self.num_output_channels = num_output_channels
        self.norm = norm
        self.upsample = upsample

        self.unet = UNet(
            self.width,
            self.height,
            self.ndf,
            self.num_output_channels,
            self.norm,
            self.upsample,
        )

    def forward(self, x):
        out = self.unet(x)
        out = F.gumbel_softmax(out, hard=True, dim=1)

        return out


class MEncoder(nn.Module):
    def __init__(self, z_length):
        super(MEncoder, self).__init__()
        """
        VAE encoder to extract intensity (modality) information from the image
        z_length: length of the output vector
        """
        self.z_length = z_length

        self.block1 = conv_bn_lrelu(3, 16, 3, 2, 1)  # 11 if anatomy_channels = 8
        self.block2 = conv_bn_lrelu(16, 32, 3, 2, 1)
        self.block3 = conv_bn_lrelu(32, 64, 3, 2, 1)
        self.block4 = conv_bn_lrelu(64, 128, 3, 2, 1)
        self.fc = nn.Linear(
            131072, 32
        )  # original sdnet:(4608, 32), white matter lesion: (25088, 32)
        self.norm = nn.BatchNorm1d(32)
        self.activ = nn.LeakyReLU(0.03, inplace=True)
        self.mu = nn.Linear(32, self.z_length)
        self.logvar = nn.Linear(32, self.z_length)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def encode(self, x):
        return self.mu(x), self.logvar(x)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.fc(out.view(-1, out.shape[1] * out.shape[2] * out.shape[3]))
        out = self.norm(out)
        out = self.activ(out)

        mu, logvar = self.encode(out)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


class UNet(nn.Module):
    def __init__(
        self, width, height, ndf, num_output_channels, normalization, upsample
    ):
        super(UNet, self).__init__()
        """
        UNet autoencoder
        """
        self.h = height
        self.w = width
        self.norm = normalization
        self.ndf = ndf
        self.num_output_channels = num_output_channels
        self.upsample = upsample

        self.encoder_block1 = conv_block_unet(3, self.ndf, 3, 1, 1, self.norm)
        self.encoder_block2 = conv_block_unet(
            self.ndf, self.ndf * 2, 3, 1, 1, self.norm
        )
        self.encoder_block3 = conv_block_unet(
            self.ndf * 2, self.ndf * 4, 3, 1, 1, self.norm
        )
        self.encoder_block4 = conv_block_unet(
            self.ndf * 4, self.ndf * 8, 3, 1, 1, self.norm
        )
        self.maxpool = nn.MaxPool2d(2, 2)

        self.bottleneck = ResConv(self.ndf * 8, self.norm)

        self.decoder_upsample1 = Interpolate(
            (self.h // 8, self.w // 8), mode=self.upsample
        )
        self.decoder_upconv1 = upconv(self.ndf * 16, self.ndf * 8, self.norm)
        self.decoder_block1 = conv_block_unet(
            self.ndf * 16, self.ndf * 8, 3, 1, 1, self.norm
        )
        self.decoder_upsample2 = Interpolate(
            (self.h // 4, self.w // 4), mode=self.upsample
        )
        self.decoder_upconv2 = upconv(self.ndf * 8, self.ndf * 4, self.norm)
        self.decoder_block2 = conv_block_unet(
            self.ndf * 8, self.ndf * 4, 3, 1, 1, self.norm
        )
        self.decoder_upsample3 = Interpolate(
            (self.h // 2, self.w // 2), mode=self.upsample
        )
        self.decoder_upconv3 = upconv(self.ndf * 4, self.ndf * 2, self.norm)
        self.decoder_block3 = conv_block_unet(
            self.ndf * 4, self.ndf * 2, 3, 1, 1, self.norm
        )
        self.decoder_upsample4 = Interpolate((self.h, self.w), mode=self.upsample)
        self.decoder_upconv4 = upconv(self.ndf * 2, self.ndf, self.norm)
        self.decoder_block4 = conv_block_unet(
            self.ndf * 2, self.ndf, 3, 1, 1, self.norm
        )
        self.classifier_conv = nn.Conv2d(self.ndf, self.num_output_channels, 3, 1, 1, 1)

    def forward(self, x):
        # encoder
        s1 = self.encoder_block1(x)
        out = self.maxpool(s1)
        s2 = self.encoder_block2(out)
        out = self.maxpool(s2)
        s3 = self.encoder_block3(out)
        out = self.maxpool(s3)
        s4 = self.encoder_block4(out)
        out = self.maxpool(s4)

        # bottleneck
        out = self.bottleneck(out)

        # decoder
        out = self.decoder_upsample1(out)
        out = self.decoder_upconv1(out)
        out = torch.cat((out, s4), 1)
        out = self.decoder_block1(out)
        out = self.decoder_upsample2(out)
        out = self.decoder_upconv2(out)
        out = torch.cat((out, s3), 1)
        out = self.decoder_block2(out)
        out = self.decoder_upsample3(out)
        out = self.decoder_upconv3(out)
        out = torch.cat((out, s2), 1)
        out = self.decoder_block3(out)
        out = self.decoder_upsample4(out)
        out = self.decoder_upconv4(out)
        out = torch.cat((out, s1), 1)
        out = self.decoder_block4(out)
        out = self.classifier_conv(out)

        return out


class CentresClassifier(nn.Module):
    def __init__(self, style_dim, num_classes):
        super(CentresClassifier, self).__init__()
        self.style_dim = style_dim
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.style_dim, self.num_classes)

    def forward(self, z):
        cls = self.classifier(z)
        pred = F.softmax(cls, dim=1)

        return pred


class SDNetStyleNonAdv(nn.Module):
    def __init__(
        self,
        width,
        height,
        num_classes,
        num_centres,
        ndf,
        z_length,
        norm,
        upsample,
        anatomy_out_channels,
        num_mask_channels,
    ):
        super(SDNetStyleNonAdv, self).__init__()
        """
        Args:
            width: input width
            height: input height
            upsample: upsampling type (nearest | bilateral)
            num_classes: number of semantice segmentation classes
            z_length: number of modality factors
            anatomy_out_channels: number of anatomy factors
            norm: feature normalization method (BatchNorm)
            ndf: number of feature channels
        """
        self.h = height
        self.w = width
        self.ndf = ndf
        self.z_length = z_length
        self.anatomy_out_channels = anatomy_out_channels
        self.norm = norm
        self.upsample = upsample
        self.num_classes = num_classes
        self.num_centres = num_centres
        self.num_mask_channels = num_mask_channels

        self.m_encoder = MEncoder(self.z_length)
        self.domain_style_encoder = MEncoder(self.z_length)
        self.a_encoder = AEncoder(
            self.h,
            self.w,
            self.ndf,
            self.anatomy_out_channels,
            self.norm,
            self.upsample,
        )
        self.segmentor = Segmentor(self.anatomy_out_channels, self.num_classes)
        self.decoder = Decoder(
            self.anatomy_out_channels, self.z_length, self.num_mask_channels
        )
        self.classifier = CentresClassifier(self.z_length, self.num_centres)

    def forward(self, x, script_type):
        a_out = self.a_encoder(x)
        seg_pred = self.segmentor(a_out)
        z_out, mu_out, logvar_out = self.m_encoder(x)
        z_out_domain, mu_out_domain, logvar_out_domain = self.domain_style_encoder(x)
        if script_type == "training":
            cls = self.classifier(z_out_domain)
            z_out_cat = torch.cat(
                [z_out, z_out_domain], dim=1
            )  # concatenate the different modality vectors for reconstruction
            reco = self.decoder(a_out, z_out_cat)
            a_out_tilde = self.a_encoder(reco)
            _, mu_out_tilde_common_style, _ = self.m_encoder(reco)
            _, mu_out_tilde_domain_style, _ = self.domain_style_encoder(reco)
            return (
                reco,
                z_out,
                mu_out_tilde_common_style,
                a_out_tilde,
                a_out,
                seg_pred,
                mu_out,
                logvar_out,
                mu_out,
                logvar_out,
                z_out_domain,
                mu_out_domain,
                logvar_out_domain,
                mu_out_tilde_domain_style,
                cls,
            )
        else:
            mu_out_cat = torch.cat([mu_out, mu_out_domain], dim=1)
            reco = self.decoder(a_out, mu_out_cat)
            a_out_tilde = a_out  # dummy assignment, not needed during validation
            mu_out_tilde = mu_out  # dummy assignment, not needed during validation
            return reco, a_out, seg_pred
